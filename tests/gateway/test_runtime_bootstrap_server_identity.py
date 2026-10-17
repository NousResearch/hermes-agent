"""#140706: a non-elevated client identifies an elevated gateway from the control pipe itself.

``_peer_subject(handle, server=True)`` is the client half of the control-pipe identity check
(``query_runtime_control`` calls it before sending anything). A filtered (non-elevated) token
cannot ``OpenProcess`` an elevated gateway, so every Desktop/CLI attach failed with
``cannot verify bootstrap peer process`` and discovery reported ``authorization``.

These tests drive that entry on any host by injecting the Win32 calls, never ``sys.platform``:
``ctypes.WinDLL`` answers the ULONG-out queries (server PID and session, own session),
``_process_sid`` stands in for OpenProcess + token (raising for an elevated server),
``_pipe_security`` for ``GetSecurityInfo`` on the pipe handle, and ``process_is_elevated`` for
this client's token. ``test_runtime_bootstrap.py`` reads a real pipe's facts on Windows, and
``tests/e2e/core/windows_update/test_elevated_gateway_attach.py`` runs the real
elevated-server / standard-user-client attach.
"""
import ctypes

import pytest

from gateway import runtime_bootstrap_windows as native

OWN = "S-1-5-21-1-2-3-1001"
OTHER = "S-1-5-21-1-2-3-1002"
ADMINS = "S-1-5-32-544"
EVERYONE = "S-1-1-0"
SERVER_PID = 5_000_000  # above any Linux pid_max, so never this process
HANDLE = object()


class _Kernel32:
    """``WinDLL('kernel32')`` answering only the ULONG-out queries the identity check makes."""

    def __init__(self, values):
        self.values = values

    def __getattr__(self, name):
        if name not in self.values:
            raise AttributeError(name)

        def call(_first, out):
            value = self.values[name]
            if value is None:
                return 0
            out._obj.value = value
            return 1
        return call


@pytest.fixture()
def pipe(monkeypatch):
    """Install one pipe server's facts as the Win32 layer reports them to this client."""
    import hermes_dir_acl

    def install(*, process_sid, owner=OWN, aces=((0, OWN),), readable=True, own_elevated=False,
                own_session=1, server_session=1, client_pid=None):
        kernel = _Kernel32({"GetNamedPipeServerProcessId": SERVER_PID, "GetNamedPipeClientProcessId": client_pid,
                            "GetNamedPipeServerSessionId": server_session, "ProcessIdToSessionId": own_session})
        monkeypatch.setattr(ctypes, "WinDLL", lambda name, use_last_error=False: kernel, raising=False)

        def process_sid_(pid):
            if pid not in (SERVER_PID, client_pid):
                return OWN  # this process
            if process_sid is None:
                raise PermissionError("cannot verify bootstrap peer process")  # OpenProcess err=5
            return process_sid
        monkeypatch.setattr(native, "_process_sid", process_sid_)

        def security(handle):
            assert handle is HANDLE
            if not readable:
                raise PermissionError("cannot read named-pipe owner")
            return owner, tuple(aces)
        monkeypatch.setattr(native, "_pipe_security", security, raising=False)
        monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: own_elevated)
    return install


def _refusal() -> PermissionError:
    with pytest.raises(PermissionError) as info:
        native._peer_subject(HANDLE, server=True)
    return info.value


# -- the bug: an elevated gateway seen from a filtered token ------------------------------------


@pytest.mark.parametrize("process_sid", [None, OWN], ids=["process-unopenable", "process-readable"])
def test_a_filtered_client_names_an_elevated_gateway_from_its_pipe(pipe, process_sid):
    """The pipe is this user's (DACL) and was made by an elevated process (owner Administrators):
    the refusal is the named ``gateway_elevated`` one, with #140490's remedy for the pipe server's PID."""
    pipe(process_sid=process_sid, owner=ADMINS)
    refused = _refusal()
    assert getattr(refused, "reason_code", None) == "gateway_elevated", refused
    assert str(refused) == native.refused_owner_remedy("gateway_elevated", SERVER_PID, 1)
    assert refused.pid == SERVER_PID


def test_an_unopenable_same_user_pipe_is_the_elevation_boundary(pipe):
    """Owner this user (the 'object creator' default-owner policy) but the process is unopenable
    from this filtered token in the same session: the same boundary, as ``classify_refused_owner``
    reads an unopenable owner token."""
    pipe(process_sid=None, owner=OWN)
    assert getattr(_refusal(), "reason_code", None) == "gateway_elevated"


def test_an_unopenable_gateway_in_another_session_is_named_as_such(pipe):
    pipe(process_sid=None, owner=ADMINS, own_session=1, server_session=0)
    refused = _refusal()
    assert getattr(refused, "reason_code", None) == "gateway_other_session"
    assert str(refused) == native.refused_owner_remedy("gateway_other_session", SERVER_PID, 0)


def test_discovery_reports_the_named_refusal_without_any_owner_record(pipe, monkeypatch, tmp_path):
    """The real discovery path: the pipe check raising inside ``query_identify`` reaches
    ``discover_gateway_endpoint``, which reports the pipe-proven reason; no ``gateway.pid`` or
    ``gateway.lock`` record is needed to name it."""
    from hermes_cli import gateway_runtime as runtime
    from hermes_cli import gateway_runtime_discovery as discovery
    from hermes_cli.gateway_runtime_cli import ensure_exit_code
    pipe(process_sid=None, owner=ADMINS)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(discovery, "query_identify",
                        lambda *_a, **_k: native._peer_subject(HANDLE, server=True))
    result = runtime.discover_gateway_endpoint(home, timeout=1)
    assert (result.state, result.reason_code) == ("inaccessible", "gateway_elevated")
    assert result.detail == native.refused_owner_remedy("gateway_elevated", SERVER_PID, 1)
    assert ensure_exit_code(result) == 4


# -- what must stay refused, never attached and never named -------------------------------------


@pytest.mark.parametrize(("owner", "aces", "readable"), [
    (OTHER, ((0, OTHER),), True),                 # another user's pipe squatting the name
    (OTHER, ((0, OWN),), True),                   # ... even one that grants this user
    (ADMINS, ((0, EVERYONE),), True),             # an elevated pipe not made for this user
    (ADMINS, ((0, OWN), (0, EVERYONE)), True),    # ... or opened to everyone besides
    (ADMINS, ((1, OWN), (0, OWN)), True),         # not the gateway's single-ACE shape
    (ADMINS, (), True),                           # NULL DACL: everyone
    (OWN, ((0, OTHER),), True),
    (None, (), False),                            # the pipe's security is unreadable
], ids=["other-owner", "other-owner-own-dacl", "admins-everyone", "admins-extra-ace", "deny-ace",
        "null-dacl", "own-owner-other-dacl", "unreadable"])
def test_an_unprovable_pipe_server_stays_the_generic_refusal(pipe, owner, aces, readable):
    pipe(process_sid=None, owner=owner, aces=aces, readable=readable)
    refused = _refusal()
    assert getattr(refused, "reason_code", None) is None
    assert "cannot verify" in str(refused) or "another user" in str(refused)


def test_an_elevated_client_never_attaches_to_a_server_it_cannot_open(pipe):
    """Not this lane's boundary to name, and no identity read: stays the generic refusal."""
    pipe(process_sid=None, owner=ADMINS, own_elevated=True)
    refused = _refusal()
    assert getattr(refused, "reason_code", None) is None
    assert str(refused) == "cannot verify bootstrap peer process"


def test_another_users_readable_server_is_refused(pipe):
    pipe(process_sid=OTHER, owner=OTHER, aces=((0, OTHER),))
    assert str(_refusal()) == "named-pipe peer belongs to another user"


# -- what keeps working --------------------------------------------------------------------------


@pytest.mark.parametrize(("owner", "own_elevated", "readable"), [
    (OWN, False, True),       # same user, neither elevated
    (ADMINS, True, True),     # both elevated
    (None, False, False),     # pipe security unreadable: the process identity alone, as before
])
def test_a_same_user_server_at_the_clients_level_is_attached(pipe, owner, own_elevated, readable):
    pipe(process_sid=OWN, owner=owner, own_elevated=own_elevated, readable=readable)
    assert native._peer_subject(HANDLE, server=True) == "sid:" + OWN


def test_the_serving_side_never_judges_a_client_by_the_pipe_owner(pipe, monkeypatch):
    """Identifying a CLIENT is unchanged: no token and no readable process stays refused."""
    pipe(process_sid=None, client_pid=SERVER_PID + 1)
    monkeypatch.setattr(native, "_pipe_client_sid",
                        lambda handle: (_ for _ in ()).throw(PermissionError("no token")))
    monkeypatch.setattr(native, "_pipe_security", lambda handle: pytest.fail("pipe owner read for a client"),
                        raising=False)
    with pytest.raises(PermissionError, match="cannot verify bootstrap peer process"):
        native._peer_subject(HANDLE, server=False)
