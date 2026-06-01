from pathlib import Path


SCRIPT = Path("scripts/setup_open_webui.sh")


def test_open_webui_signup_defaults_closed():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'OPEN_WEBUI_ENABLE_SIGNUP="${OPEN_WEBUI_ENABLE_SIGNUP:-false}"' in text


def test_open_webui_lan_signup_requires_explicit_risk_acceptance():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "guard_open_webui_signup" in text
    assert "OPEN_WEBUI_ALLOW_LAN_SIGNUP_RACE" in text
    assert "is_loopback_host" in text


def test_open_webui_launcher_quotes_signup_value():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'quoted_signup="$(shell_quote "$OPEN_WEBUI_ENABLE_SIGNUP")"' in text
    assert "export ENABLE_SIGNUP=${quoted_signup}" in text
