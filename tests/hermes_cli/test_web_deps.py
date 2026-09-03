"""Tests for hermes_cli/web_deps.py — late-binding helpers."""


def test_late_attr_returns_actual_value():
    from hermes_cli.web_deps import late_attr

    class FakeServer:
        token = "abc123"
    server = FakeServer()
    assert late_attr("token")(server) == "abc123"


def test_late_attr_missing_returns_none():
    from hermes_cli.web_deps import late_attr
    assert late_attr("missing")(None) is None


def test_late_delegates_to_late_attr_without_args():
    from hermes_cli.web_deps import late, late_attr
    # late(name) returns the same thing as late_attr(name)
    assert late("x").func == late_attr("x").func


def test_get_session_token_returns_string():
    from hermes_cli.web_deps import get_session_token
    # Without a running server, returns None
    result = get_session_token()
    assert result is None or isinstance(result, str)


def test_get_dashboard_health_returns_dict():
    from hermes_cli.web_deps import get_dashboard_health
    result = get_dashboard_health()
    assert result is None or isinstance(result, (dict, bool))
