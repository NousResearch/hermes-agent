"""``/api/model/info`` must not probe model metadata against a pinned local endpoint.

The probe is pure cost there: with ``model.context_length`` already set by the
operator it cannot report anything new, and local bridges/proxies commonly serve
only the generation route, so discovery fails and surfaces as provider-down noise.
"""


def _record_probe(monkeypatch):
    """Replace the metadata probe with a recorder.

    ``get_model_info`` imports the symbol inside the call, so patching the
    attribute on the module is what the route actually resolves.
    """
    calls = []
    import agent.model_metadata as _mm

    def _fake(**kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(_mm, "get_model_context_length", _fake)
    return calls


class TestModelInfoProbeSkip:
    def test_pinned_local_endpoint_does_not_probe(self, monkeypatch):
        from hermes_cli.config import save_config
        from hermes_cli.web_routers.models import get_model_info

        calls = _record_probe(monkeypatch)
        save_config({
            "model": {
                "default": "llama3.2",
                "provider": "ollama-local",
                "base_url": "http://localhost:11434/v1",
                "context_length": 200000,
            }
        })

        info = get_model_info()

        assert calls == [], "the probe ran against a pinned local endpoint"
        assert info["config_context_length"] == 200000
        assert info["effective_context_length"] == 200000

    def test_remote_endpoint_still_probes(self, monkeypatch):
        """Control. Without it the test above still passes if the probe is
        deleted outright, which would prove nothing about the skip."""
        from hermes_cli.config import save_config
        from hermes_cli.web_routers.models import get_model_info

        calls = _record_probe(monkeypatch)
        save_config({
            "model": {
                "default": "google/gemini-2.5-flash",
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "context_length": 200000,
            }
        })

        get_model_info()

        assert len(calls) == 1, "a remote endpoint must still be probed"

    def test_local_endpoint_without_override_still_probes(self, monkeypatch):
        """Second control: local alone is not enough. Without an operator-stated
        context_length the probe is the only source of the value, so it must run."""
        from hermes_cli.config import save_config
        from hermes_cli.web_routers.models import get_model_info

        calls = _record_probe(monkeypatch)
        save_config({
            "model": {
                "default": "llama3.2",
                "provider": "ollama-local",
                "base_url": "http://localhost:11434/v1",
            }
        })

        get_model_info()

        assert len(calls) == 1, "a local endpoint without an override must still be probed"
