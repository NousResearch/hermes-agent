import importlib.util
import sys
from pathlib import Path


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "probe_custom_endpoint.py"
    spec = importlib.util.spec_from_file_location("probe_custom_endpoint", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _ProbeResult:
    def __init__(self, label, url, status_code, content_type, body_preview, error=""):
        self.label = label
        self.url = url
        self.status_code = status_code
        self.content_type = content_type
        self.body_preview = body_preview
        self.error = error


def test_main_returns_nonzero_for_html_inference_response(monkeypatch, capsys):
    probe = _load_probe_module()

    monkeypatch.setattr(
        probe,
        "_load_from_hermes_config",
        lambda env_key="OPENAI_API_KEY": ("http://relay.example.com", "sk-test", "gpt-4o-mini"),
    )

    def fake_request(method, url, headers, payload, timeout):
        if url.endswith("/models"):
            return _ProbeResult("GET models", url, 200, "application/json", '{"data": []}')
        if url.endswith("/chat/completions"):
            return _ProbeResult("POST chat/completions", url, 200, "text/html; charset=utf-8", "<html>ui</html>")
        if url.endswith("/responses"):
            return _ProbeResult("POST responses", url, 200, "text/html; charset=utf-8", "<html>ui</html>")
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(probe, "_request", fake_request)

    exit_code = probe.main(["--from-hermes-config"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "text/html; charset=utf-8" in out
    assert "<html>ui</html>" in out


def test_main_accepts_root_html_models_when_v1_endpoints_work(monkeypatch, capsys):
    probe = _load_probe_module()

    monkeypatch.setattr(
        probe,
        "_load_from_hermes_config",
        lambda env_key="OPENAI_API_KEY": ("http://relay.example.com", "sk-test", "gpt-5.4"),
    )

    def fake_request(method, url, headers, payload, timeout):
        if url == "http://relay.example.com/models":
            return _ProbeResult("GET models", url, 200, "text/html; charset=utf-8", "<html>ui</html>")
        if url == "http://relay.example.com/v1/models":
            return _ProbeResult("GET models", url, 200, "application/json", '{"data": [{"id": "gpt-5.4"}]}')
        if url == "http://relay.example.com/v1/chat/completions":
            return _ProbeResult("POST chat/completions", url, 200, "application/json", '{"id": "chatcmpl_123"}')
        if url == "http://relay.example.com/v1/responses":
            return _ProbeResult("POST responses", url, 200, "application/json", '{"id": "resp_123"}')
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(probe, "_request", fake_request)

    exit_code = probe.main(["--from-hermes-config"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "http://relay.example.com/v1/models" in out
    assert "http://relay.example.com/v1/responses" in out
