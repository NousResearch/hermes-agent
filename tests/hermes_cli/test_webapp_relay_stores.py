"""A relay store file that cannot be read is never written back without the records it holds.

``webapp/relay-grants.json`` (the grants a gateway issued) and ``webapp/relay-gateways.json`` (the
gateways a hub reaches) each hold every account's records. The relay app is paired to itself, so one
home holds both.
"""
import urllib.parse

# _relay_app is the relay_app fixture.
from tests.hermes_cli.test_webapp_relay import ORIGIN, WEBAPP, _approve, _relay_app, _sign_in

# What a read catches halfway through another write, or a file a sharing violation cut short.
TORN_GRANTS = '{"grants": {"lab-grant": {"hub_name": "Lab", "subject": "someone-else"'
TORN_GATEWAYS = '{"gateways": [{"id": "lab", "url": "http://lab.lan:9120", "subject": "someone-else"'


def _torn(home, name: str, text: str):
    path = home / "webapp" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _outcome(back) -> str:
    return urllib.parse.unquote_plus(back.headers["location"])


def test_a_pairing_leaves_a_grant_file_it_cannot_read_as_it_is(relay_app):
    grants_file = _torn(relay_app.home, "relay-grants.json", TORN_GRANTS)
    browser = relay_app.browser()
    _sign_in(browser)
    back = browser.get(_approve(browser), follow_redirects=False)
    assert "relay=failed" in _outcome(back) and "(503)" in _outcome(back), _outcome(back)
    assert grants_file.read_text(encoding="utf-8") == TORN_GRANTS
    assert browser.get("/api/webapp/relay/gateways").json()["gateways"] == []


def test_a_pairing_does_not_start_on_a_gateway_list_it_cannot_read(relay_app):
    gateways_file = _torn(relay_app.home, "relay-gateways.json", TORN_GATEWAYS)
    browser = relay_app.browser()
    _sign_in(browser)
    started = browser.post("/api/webapp/relay/gateways", json={"url": WEBAPP, "label": "Mini"}, headers=ORIGIN)
    assert started.status_code == 503, started.text
    assert gateways_file.read_text(encoding="utf-8") == TORN_GATEWAYS


def test_a_pairing_that_cannot_be_saved_revokes_the_grant_it_was_given(relay_app):
    browser = relay_app.browser()
    _sign_in(browser)
    callback = _approve(browser)
    gateways_file = _torn(relay_app.home, "relay-gateways.json", TORN_GATEWAYS)  # unreadable from here on
    back = browser.get(callback, follow_redirects=False)
    assert "relay=failed" in _outcome(back) and "could not save the pairing" in _outcome(back), _outcome(back)
    assert gateways_file.read_text(encoding="utf-8") == TORN_GATEWAYS
    # Nobody here could use or revoke it, so it does not stay behind on the other gateway.
    assert browser.get("/api/webapp/relay/grants").json()["grants"] == []
