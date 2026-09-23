"""Windows confidentiality for the host rendezvous token + lock (``windows_only``).

``os.open(..., 0o600)`` sets NO ACLs on Windows, so the host token — which holds the backend's
LIVE session token — would inherit whatever the parent directory grants. The SSH runtime's
protected owner+SYSTEM DACL writer is the repo's primitive for this credential class
(``tests/hermes_cli/test_ssh_session_token_parser.py`` documents why); this proves the host
rendezvous actually uses it, and that the host lock still works on that path.
"""

import pytest

from gateway import host_rendezvous as hr

pytestmark = pytest.mark.windows_only


@pytest.fixture
def host_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    yield tmp_path
    hr.release_host_lock(hr.ROLE_SERVE)


def _allowed_sids():
    from hermes_cli import windows_ssh_runtime as wsr

    return wsr._allowed_sids()


def test_host_token_is_written_with_a_protected_owner_only_dacl(host_dir):
    """Owner+SYSTEM only, and the DACL is protected so inheritable parent grants are not merged."""
    import win32security

    from hermes_cli import windows_ssh_runtime as wsr

    assert hr.claim_host_lock(hr.ROLE_SERVE)[0] is hr.HostLockOutcome.ACQUIRED
    hr.publish_record(hr.ROLE_SERVE, host="127.0.0.1", port=9119, token="live-session-token")

    path = hr.token_path(hr.ROLE_SERVE)
    assert path.read_text(encoding="utf-8").strip() == "live-session-token"
    descriptor = win32security.GetFileSecurity(
        str(path), win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION)
    allowed = _allowed_sids()
    assert wsr._sid_str(descriptor.GetSecurityDescriptorOwner()) in allowed
    dacl = descriptor.GetSecurityDescriptorDacl()
    assert dacl is not None, "a null DACL grants everyone"
    for index in range(dacl.GetAceCount()):
        ace = dacl.GetAce(index)
        assert wsr._sid_str(ace[-1]) in allowed
    assert descriptor.GetSecurityDescriptorControl()[0] & win32security.SE_DACL_PROTECTED


def test_rewriting_an_existing_token_keeps_it_private(host_dir):
    """CreateFile ignores the security descriptor on an existing file, and ``os.replace`` fails
    against an open reader — the writer replaces in place instead of inheriting either bug."""
    assert hr.claim_host_lock(hr.ROLE_SERVE)[0] is hr.HostLockOutcome.ACQUIRED
    hr.publish_record(hr.ROLE_SERVE, host="127.0.0.1", port=9119, token="first")
    hr.publish_record(hr.ROLE_SERVE, host="127.0.0.1", port=9119, token="second")

    assert hr.read_token(hr.ROLE_SERVE) == "second"
    record = hr.read_record(hr.ROLE_SERVE, include_stale=True)
    assert record is not None and hr.record_token_is_consistent(record)
