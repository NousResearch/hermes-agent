"""The Webapp's browser credential file stays private to the gateway's OS user.

The gateway keeps the browser credential in ``webapp/session-token`` inside its home so open tabs
survive a restart. On POSIX the folder is user-only (0700) and the file owner-read/write (0600). A
folder found open to other users is closed again before the credential is used, and a credential
file other users could read is treated as exposed: it is replaced, which signs the open tabs out,
rather than reused. Windows homes keep their own ACL, so this applies to POSIX only. A gateway that
is not serving the Webapp holds no browser credential in memory: signing out of it replaces the
stored one, and its answer says that nothing else was closed.
"""
import os
import stat

import pytest

from gateway import webapp_listener as webapp

posix_only = pytest.mark.platforms("posix")
STORED = "s" * 43


def _store(home, *, folder_mode, file_mode):
    path = home / webapp.TOKEN_PATH
    path.parent.mkdir(parents=True)
    path.write_text(STORED + "\n", encoding="utf-8")
    os.chmod(path, file_mode)
    os.chmod(path.parent, folder_mode)
    return path


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


@posix_only
@pytest.mark.parametrize("file_mode", [0o644, 0o640, 0o604])
def test_a_credential_other_users_could_read_is_replaced(tmp_path, file_mode):
    path = _store(tmp_path, folder_mode=0o700, file_mode=file_mode)
    token = webapp.browser_session_token(tmp_path)
    assert token != STORED
    assert path.read_text(encoding="utf-8").strip() == token
    assert _mode(path) == 0o600


@posix_only
def test_a_private_credential_in_an_open_folder_is_kept_and_the_folder_closed(tmp_path):
    path = _store(tmp_path, folder_mode=0o755, file_mode=0o600)
    assert webapp.browser_session_token(tmp_path) == STORED
    assert _mode(path.parent) == 0o700
    assert _mode(path) == 0o600


@posix_only
def test_signing_out_closes_an_open_credential_folder_too(tmp_path):
    path = _store(tmp_path, folder_mode=0o755, file_mode=0o600)
    assert webapp.browser_session_token(tmp_path, rotate=True) != STORED
    assert (_mode(path.parent), _mode(path)) == (0o700, 0o600)


def test_a_private_credential_is_reused(tmp_path):
    first = webapp.browser_session_token(tmp_path)
    assert webapp.browser_session_token(tmp_path) == first


def test_a_credential_file_that_is_not_text_is_replaced(tmp_path):
    """A token file that is not UTF-8 (saved as UTF-16 by an editor, damaged on disk) is corrupt: it is
    replaced like a malformed one, never raised to the gateway mounting the Webapp."""
    path = tmp_path / webapp.TOKEN_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes((STORED + "\n").encode("utf-16"))
    token = webapp.browser_session_token(tmp_path)
    assert token != STORED
    assert path.read_text(encoding="utf-8").strip() == token


@pytest.mark.parametrize("block", ["folder-is-a-file", "token-is-a-folder"])
def test_a_credential_that_cannot_be_stored_names_its_file(tmp_path, block):
    """A credential that can be neither read nor written (a stray file or folder at its path, a folder
    another user owns, a full or read-only home) raises an OSError that names the file, so the gateway
    reports that file as why the Webapp is unavailable."""
    if block == "folder-is-a-file":
        (tmp_path / "webapp").write_text("not a folder\n", encoding="utf-8")
    else:
        (tmp_path / webapp.TOKEN_PATH).mkdir(parents=True)
    with pytest.raises(webapp.BrowserCredentialError) as raised:
        webapp.browser_session_token(tmp_path)
    assert isinstance(raised.value, OSError)
    assert str(tmp_path / webapp.TOKEN_PATH) in str(raised.value)


@pytest.mark.parametrize("status", [
    {"state": "off", "url": None, "reason": None},
    {"state": "unavailable", "url": None, "reason": "could not listen on 127.0.0.1:9120 (in use)"},
])
def test_signing_out_of_a_gateway_not_serving_the_webapp_replaces_the_stored_credential(tmp_path, status):
    """No browser reaches that gateway's Webapp, so replacing the stored credential is the whole
    sign-out; it says so instead of claiming it closed anything."""
    import json
    from types import SimpleNamespace

    from gateway.control_socket import GatewayControlServer

    before = webapp.browser_session_token(tmp_path)
    runner = SimpleNamespace(session_control_server=GatewayControlServer(tmp_path), webapp_status=status,
                             session_api=SimpleNamespace(app=SimpleNamespace(web=None)))
    webapp.register_control_verbs(runner, tmp_path)
    raw = json.dumps({"protocol": 1, "verb": "webapp-sign-out", "id": 1, "params": {}}).encode()
    answer = json.loads(runner.session_control_server.handle_request_line(raw, "same-user"))
    assert answer["ok"], answer
    assert answer["result"] == {"signed_out": True, "serving": False, "sign_in_gate": False, "connections_closed": 0}
    assert webapp.browser_session_token(tmp_path) != before
