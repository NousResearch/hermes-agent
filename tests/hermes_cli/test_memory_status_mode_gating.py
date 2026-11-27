"""`hermes memory status` must scope the Missing env-var list to the active provider mode.

Regression coverage for NousResearch/hermes-agent#135771: a Hindsight
``local_embedded`` configuration was told to set the cloud-only
``HINDSIGHT_API_KEY`` — twice, once per foreign mode — while the provider's
own ``unavailable_reason()`` hint was never surfaced.
"""

from hermes_cli import memory_setup


def _hindsight_like_schema():
    # Same shape as the catalog plugin's schema: mutually exclusive modes whose
    # fields reuse one env_var (HINDSIGHT_API_KEY), gated by ``when``.
    return [
        {
            "key": "mode",
            "description": "Connection mode",
            "default": "cloud",
            "choices": ["cloud", "local_embedded", "local_external"],
        },
        {
            "key": "api_key",
            "secret": True,
            "env_var": "HINDSIGHT_API_KEY",
            "url": "https://ui.hindsight.vectorize.io",
            "when": {"mode": "cloud"},
        },
        {
            "key": "api_key",
            "secret": True,
            "env_var": "HINDSIGHT_API_KEY",
            "when": {"mode": "local_external"},
        },
        {
            "key": "llm_api_key",
            "secret": True,
            "env_var": "HINDSIGHT_LLM_API_KEY",
            "when": {"mode": "local_embedded"},
        },
    ]


class _UnavailableProvider:
    def __init__(self, schema, reason=None):
        self._schema = schema
        self._reason = reason

    def is_available(self):
        return False

    def get_config_schema(self):
        return self._schema

    def unavailable_reason(self):
        return self._reason or ""


def _run_status(monkeypatch, capsys, provider, provider_config):
    monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)
    monkeypatch.delenv("HINDSIGHT_LLM_API_KEY", raising=False)
    monkeypatch.setattr(
        memory_setup,
        "_get_available_providers",
        lambda: [("hindsight", "API key / local", provider)],
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"provider": "hindsight", "hindsight": provider_config}},
    )
    memory_setup.cmd_status(object())
    return capsys.readouterr().out


def test_status_lists_only_env_vars_of_the_active_mode(monkeypatch, capsys):
    provider = _UnavailableProvider(
        _hindsight_like_schema(), reason="run `hermes plugins update hindsight`"
    )
    out = _run_status(monkeypatch, capsys, provider, {"mode": "local_embedded"})

    assert "HINDSIGHT_API_KEY" not in out  # cloud/local_external-only var
    assert out.count("HINDSIGHT_LLM_API_KEY") == 1  # the var local_embedded needs
    assert "Reason:" in out  # provider's own failure hint
    assert "hermes plugins update hindsight" in out


def test_status_dedupes_shared_env_var_across_modes(monkeypatch, capsys):
    # No explicit mode: the schema default (cloud) applies, so HINDSIGHT_API_KEY
    # is required — but only once, not once per mode sharing the env_var.
    provider = _UnavailableProvider(_hindsight_like_schema())
    out = _run_status(monkeypatch, capsys, provider, {})

    assert out.count("HINDSIGHT_API_KEY") == 1
    assert "HINDSIGHT_LLM_API_KEY" not in out  # belongs to local_embedded only


def test_status_tolerates_provider_without_reason_channel(monkeypatch, capsys):
    # Older plugins have no unavailable_reason(); status must not regress for them.
    class _LegacyProvider:
        def is_available(self):
            return False

        def get_config_schema(self):
            return _hindsight_like_schema()

    out = _run_status(monkeypatch, capsys, _LegacyProvider(), {})

    assert "not available" in out
    assert "Reason:" not in out
    assert out.count("HINDSIGHT_API_KEY") == 1


def test_status_survives_reason_channel_that_raises(monkeypatch, capsys):
    class _ExplodingReasonProvider(_UnavailableProvider):
        def unavailable_reason(self):
            raise RuntimeError("probe exploded")

    out = _run_status(
        monkeypatch, capsys, _ExplodingReasonProvider(_hindsight_like_schema()), {}
    )

    assert "not available" in out
    assert "Reason:" not in out
