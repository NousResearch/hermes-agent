import importlib.util
from pathlib import Path


def _load_setup_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "skills" / "productivity" / "google-workspace" / "scripts" / "setup.py"
    spec = importlib.util.spec_from_file_location("google_workspace_setup", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_oauth_client_disabled_banner_leads_with_appeal_and_no_retry(capsys):
    setup = _load_setup_module()

    setup._print_oauth_client_disabled_banner(
        "OAUTH_CLIENT_DISABLED",
        RuntimeError("disabled_client: client disabled"),
    )

    out = capsys.readouterr().out
    assert "============================================================" in out
    assert "OAUTH_CLIENT_DISABLED - STOP AND READ" in out
    assert "https://accounts.google.com/signin/recovery" in out
    assert "Do NOT retry API calls" in out
    assert "https://console.cloud.google.com/apis/credentials" in out
    assert out.index("https://accounts.google.com/signin/recovery") < out.index(
        "Do NOT retry API calls"
    )
    assert out.index("Do NOT retry API calls") < out.index("THEN DIAGNOSE:")


def test_live_check_disabled_banner_uses_live_check_heading(capsys):
    setup = _load_setup_module()

    setup._print_oauth_client_disabled_banner(
        "LIVE_CHECK_FAILED: OAuth client or account disabled",
        RuntimeError("invalid_client"),
    )

    out = capsys.readouterr().out
    assert "LIVE_CHECK_FAILED: OAuth client or account disabled - STOP AND READ" in out
    assert "invalid_client" in out
    assert "Check account status:      https://myaccount.google.com" in out
