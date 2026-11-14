"""#140051: a hosted dashboard plugin cannot set cookies on the gateway's site.

A plugin host's answer is served from the gateway's own origin, so a ``Set-Cookie`` it sends would
set or overwrite the dashboard's cookies. Both hosted paths (the launch profile's
``plugins.isolation: host`` mount and a secondary profile's own host) drop it, logging once per plugin.
"""
from __future__ import annotations

import logging

import pytest

from hermes_cli.plugin_host import PluginHost
from tests.hermes_cli import test_plugin_secondary_profile_host as secondary

daemon = secondary.daemon


def _answers_with_cookies(monkeypatch):
    from hermes_cli import web_server_plugin_profiles
    monkeypatch.setattr(web_server_plugin_profiles, "_cookie_droppers", set(), raising=False)

    def asgi_request(host, *args, **kwargs):
        return {"status": 200, "body": b'{"hosted": true}', "headers": [
            ("content-type", "application/json"), ("Set-Cookie", "a=1; Path=/"),
            ("set-cookie", "b=2; Path=/"), ("Set-Cookie2", "c=3"), ("x-plugin", "kept")]}

    monkeypatch.setattr(PluginHost, "asgi_request", asgi_request)


def _launch_hosted(daemon):
    secondary._install(daemon.launch, "launch-hosted")
    secondary._plugins(daemon.launch, enabled=["launch-hosted"], isolation="host")
    return "/api/plugins/launch-hosted/ping", lambda: daemon.owner, daemon.launch


def _secondary_hosted(daemon):
    secondary._beta_plugin(daemon)
    return secondary.PING, daemon.beta_ticket, daemon.beta  # a ticket per request


@pytest.mark.parametrize("setup", [_launch_hosted, _secondary_hosted], ids=["launch-mount", "secondary-profile"])
def test_hosted_plugin_answer_reaches_the_browser_without_set_cookie(daemon, monkeypatch, caplog, setup):
    _answers_with_cookies(monkeypatch)
    url, credentials, home = setup(daemon)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.web_server"):
        first = daemon.client.get(url, headers=credentials())
        second = daemon.client.get(url, headers=credentials())

    for response in (first, second):
        assert (response.status_code, response.json()) == (200, {"hosted": True})
        assert response.headers.get_list("set-cookie") == []
        assert response.headers.get_list("set-cookie2") == []
        assert response.headers["x-plugin"] == "kept"
    assert not daemon.client.cookies
    warnings = [r.getMessage() for r in caplog.records if "Set-Cookie" in r.getMessage()]
    assert len(warnings) == 1
    assert f"(profile home {home})" in warnings[0]  # names which profile's copy of the plugin did it
