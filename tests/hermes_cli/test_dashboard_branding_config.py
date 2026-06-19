"""Dashboard branding configuration contract tests."""

from hermes_cli import web_server


def test_dashboard_branding_defaults_match_current_ui():
    branding = web_server._dashboard_branding_from_config({})

    assert branding == {
        "app_name": "Hermes Agent",
        "assistant_name": "Hermes",
        "wordmark_lines": ["Hermes", "Agent"],
        "title": "Hermes Agent - Dashboard",
    }


def test_dashboard_branding_accepts_config_overrides():
    branding = web_server._dashboard_branding_from_config(
        {
            "dashboard": {
                "branding": {
                    "app_name": "Transformation Lab",
                    "assistant_name": "Debra",
                    "wordmark": ["Transformation", "Lab"],
                    "title": "Transformation Lab",
                }
            }
        }
    )

    assert branding == {
        "app_name": "Transformation Lab",
        "assistant_name": "Debra",
        "wordmark_lines": ["Transformation", "Lab"],
        "title": "Transformation Lab",
    }


def test_dashboard_branding_derives_wordmark_from_two_word_app_name():
    branding = web_server._dashboard_branding_from_config(
        {"dashboard": {"branding": {"app_name": "Transformation Lab"}}}
    )

    assert branding["wordmark_lines"] == ["Transformation", "Lab"]


def test_dashboard_branding_rejects_html_injection_and_bad_shapes():
    branding = web_server._dashboard_branding_from_config(
        {
            "dashboard": {
                "branding": {
                    "app_name": "<script>alert(1)</script>",
                    "assistant_name": {"bad": "shape"},
                    "wordmark": ["Good", "<img src=x onerror=alert(1)>", "Ignored"],
                    "title": "Bad\nTitle",
                }
            }
        }
    )

    assert branding["app_name"] == "Hermes Agent"
    assert branding["assistant_name"] == "Hermes"
    assert branding["wordmark_lines"] == ["Good", "Agent"]
    assert branding["title"] == "Hermes Agent - Dashboard"


def test_dashboard_bootstrap_includes_safe_json_branding():
    script = web_server._dashboard_bootstrap_script(
        token="abc123",
        prefix="/lab",
        embedded_chat=False,
        auth_required=False,
        branding={
            "app_name": "Transformation Lab",
            "assistant_name": "Debra",
            "wordmark_lines": ["Transformation", "Lab"],
            "title": "Transformation Lab",
        },
    )

    assert 'window.__HERMES_DASHBOARD_BRANDING__={"app_name":"Transformation Lab"' in script
    assert 'window.__HERMES_BASE_PATH__="/lab";' in script
    assert 'window.__HERMES_SESSION_TOKEN__="abc123";' in script
    assert "</script><script>" not in script
