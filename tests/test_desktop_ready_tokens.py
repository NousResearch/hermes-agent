from hermes_cli import web_server


def test_headless_ready_tokens_keep_legacy_desktop_compatibility(capsys):
    web_server._emit_ready_tokens(45678, headless=True)

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == [
        "HERMES_BACKEND_READY port=45678",
        "HERMES_DASHBOARD_READY port=45678",
    ]


def test_dashboard_ready_token_stays_legacy_only(capsys):
    web_server._emit_ready_tokens(45678, headless=False)

    assert capsys.readouterr().out.strip() == "HERMES_DASHBOARD_READY port=45678"
