"""Regression tests for the Signal setup guidance (#132554).

The Signal adapter speaks exactly one protocol — signal-cli's native HTTP
daemon endpoints (``/api/v1/check``, ``/api/v1/events``, ``/api/v1/rpc``).
Every setup surface used to point at bbernhard/signal-cli-rest-api, whose
REST surface is incompatible: following that guidance leaves SIGNAL_HTTP_URL
404-ing on every request. These tests pin the catalog entry and the env-field
guidance to the daemon wording and the docs page that matches the adapter.
"""

import inspect

SIGNAL_DOCS_URL = (
    "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/signal"
)


def _signal_entry() -> dict:
    from hermes_cli.web_server_messaging import _messaging_platform_catalog

    return next(e for e in _messaging_platform_catalog() if e["id"] == "signal")


def test_signal_catalog_entry_points_at_the_daemon_docs():
    entry = _signal_entry()

    assert entry["docs_url"] == SIGNAL_DOCS_URL
    assert "REST" not in entry["description"]
    assert "daemon" in entry["description"].lower()


def test_signal_env_field_hints_describe_the_http_daemon():
    from hermes_cli.web_routers.messaging import _MESSAGING_ENV_FALLBACKS

    url_field = _MESSAGING_ENV_FALLBACKS["SIGNAL_HTTP_URL"]

    assert "REST" not in url_field["description"]
    assert "daemon" in url_field["description"].lower()
    assert url_field["url"] == SIGNAL_DOCS_URL

    assert "REST" not in _MESSAGING_ENV_FALLBACKS["SIGNAL_ACCOUNT"]["description"]


def test_setup_wizard_does_not_recommend_the_rest_bridge_image():
    # The wizard prints its install hints as inline literals; asserting the
    # module source is the only seam that catches a stale copy paste.
    import hermes_cli.gateway_setup_wizard as wizard

    assert "bbernhard" not in inspect.getsource(wizard)
