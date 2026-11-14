"""The browser credential has one generation check and one rotation that ends every grant.

Every grant made under the Webapp's browser credential records the credential's generation and is
honoured only while ``is_current`` says it is the one in use; ``rotate`` (``hermes webapp
--sign-out``) replaces the credential and runs every revoker registered with ``on_rotate``. The
grant kinds themselves are pinned in test_webapp_sign_out_sockets.py and test_webapp_gated_sign_out.py.
"""
from hermes_cli import browser_credential, web_server


def test_a_generation_is_current_only_while_its_credential_is_in_use(monkeypatch):
    """A grant hands over or binds to the credential only while its recorded generation is current."""
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "a" * 43)
    recorded = browser_credential.generation()
    assert recorded == browser_credential.generation_of("a" * 43) != "a" * 43
    assert browser_credential.is_current(recorded)
    assert not browser_credential.is_current(None)
    assert browser_credential.current_if(recorded) == "a" * 43
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "b" * 43)
    assert not browser_credential.is_current(recorded)
    assert browser_credential.current_if(recorded) is None


def test_rotate_replaces_the_credential_then_runs_every_revoker(monkeypatch):
    """Each revoker runs after the replacement, so it sees the old generation as no longer current,
    and the rotation reports how many grants each one ended."""
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "a" * 43)
    monkeypatch.setattr(browser_credential, "_revokers", {})
    old = browser_credential.generation()
    seen = []

    def revoker(count):
        def run():
            seen.append(browser_credential.is_current(old))
            return count
        return run

    browser_credential.on_rotate("sockets", revoker(2))
    browser_credential.on_rotate("file tickets", revoker(0))
    assert browser_credential.rotate("b" * 43) == {"sockets": 2, "file tickets": 0}
    assert web_server._SESSION_TOKEN == browser_credential.current() == "b" * 43
    assert seen == [False, False]
