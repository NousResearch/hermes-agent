"""Black-box contracts for opt-in, headless Bitwarden session files.

Only dummy credentials and real files in pytest's sandbox are used. Explicit
user locks are process-local fences; expiry and automatic invalidation are not.
The Windows contract runs on Windows, never by impersonating another OS.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from queue import Queue
import threading
from unittest.mock import patch

import pytest

from agent.vault_backends import unlock
from agent.vault_backends.base import UnlockRequired
from agent.vault_backends.bitwarden import BitwardenLoginBackend
from tests.waiting import wait_until

EXTERNAL = "dummy-external-session-a"
INTERACTIVE = "dummy-interactive-session"
PASSWORD = "dummy-password-for-page-only"

_FAKE_BW = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
root = Path(__file__).resolve().parent
argv = sys.argv[1:]
token = os.environ.get("BW_SESSION")
with (root / "bw.log").open("a", encoding="utf-8") as log:
    log.write(json.dumps({"argv": argv, "BW_SESSION": token}) + "\n")
accepted = json.loads((root / "accepted").read_text(encoding="utf-8"))
if argv[:2] == ["unlock", "--raw"]:
    name = argv[argv.index("--passwordenv") + 1]
    if os.environ.get(name) != "dummy-master":
        sys.stderr.write("Invalid master password."); sys.exit(1)
    print("dummy-interactive-session"); sys.exit(0)
if argv[:1] == ["status"]:
    mode = (root / "status-mode").read_text() if (root / "status-mode").exists() else ""
    if mode == "malformed":
        print("not JSON"); sys.exit(0)
    print(json.dumps({"status": mode or ("unlocked" if token in accepted else "locked")}))
    sys.exit(0)
if token not in accepted or (root / "reject-actual").exists():
    sys.stderr.write("Vault is locked. Session rejected: " + (token or "")); sys.exit(1)
if argv[:2] == ["list", "items"]:
    print(json.dumps([{"id": "abc", "type": 1, "name": "Example",
                       "creationDate": "2026-01-01T00:00:00Z",
                       "login": {"username": "dummy@example.com", "uris": [{"uri": "https://example.com/login"}]}}]))
    sys.exit(0)
if argv[:2] == ["get", "password"]:
    print("dummy-password-for-page-only"); sys.exit(0)
sys.exit(2)
'''


class FakeBitwarden:
    """A real child executable; its private log is read only for assertions."""

    def __init__(self, root: Path):
        self.root = root
        self.exe = root / "bw"
        self.exe.write_text(_FAKE_BW, encoding="utf-8")
        self.exe.chmod(0o700)
        self.session = root / "session"
        self.write_token(EXTERNAL)
        self.accept(EXTERNAL, INTERACTIVE)

    def write_token(self, token: str, *, atomic: bool = False):
        target = self.root / "replacement" if atomic else self.session
        target.write_text(token, encoding="utf-8")
        target.chmod(0o600)
        if atomic:
            target.replace(self.session)

    def accept(self, *tokens: str):
        (self.root / "accepted").write_text(json.dumps(tokens), encoding="utf-8")

    def backend(self, *, external: bool = True):
        cfg = {"binary_path": str(self.exe)}
        if external:
            cfg["session_file"] = str(self.session)
        return BitwardenLoginBackend(cfg)

    def calls(self):
        log = self.root / "bw.log"
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []

    def statuses(self):
        return [call for call in self.calls() if call["argv"][0] == "status"]


@pytest.fixture
def external_bw(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("BW_SESSION", raising=False)
    yield FakeBitwarden(tmp_path)
    unlock.lock("bitwarden")
    unlock.set_current_session_id(None)


@pytest.mark.platforms("posix")
def test_session_file_is_opt_in_and_unconfigured_backend_never_reads_or_spawns(external_bw, monkeypatch):
    backend = external_bw.backend(external=False)
    # An unrelated BW_SESSION in the launch environment is not a source either.
    monkeypatch.setenv("BW_SESSION", "dummy-unrelated-launch-token")
    with patch.object(Path, "read_bytes", side_effect=AssertionError("unconfigured file read")), \
         patch("os.open", side_effect=AssertionError("unconfigured file open")):
        assert not backend.is_unlocked()
        assert backend.list_items() == []
        with pytest.raises(UnlockRequired):
            backend.resolve_password("bw:abc")
    assert external_bw.calls() == []


@pytest.mark.platforms("posix")
def test_session_file_unlocks_real_list_and_resolve_without_leaking(external_bw, caplog):
    from tools.environments.local import build_subprocess_env

    backend = external_bw.backend()
    assert backend.is_unlocked()
    assert [meta.id for meta in backend.list_items()] == ["bw:abc"]
    assert backend.resolve_password("bw:abc") == PASSWORD
    assert backend.lock_reason is None
    assert EXTERNAL not in os.environ.values()
    assert EXTERNAL not in build_subprocess_env().values()
    assert EXTERNAL not in build_subprocess_env(scrub_secrets=False).values()
    assert EXTERNAL not in caplog.text
    calls = external_bw.calls()
    assert calls and all(call["BW_SESSION"] == EXTERNAL for call in calls)
    assert all(EXTERNAL not in json.dumps(call["argv"]) for call in calls)
    assert all("--nointeraction" in call["argv"] for call in calls)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("status", ["locked", "unauthenticated", "malformed"])
def test_nonempty_token_requires_unlocked_json_status(external_bw, status):
    backend = external_bw.backend()
    (external_bw.root / "status-mode").write_text(status, encoding="utf-8")
    assert not backend.is_unlocked()
    assert backend.lock_reason == "external_session_rejected"
    assert backend.list_items() == []
    assert external_bw.statuses()
    assert all(call["argv"][0] == "status" for call in external_bw.calls())


@pytest.mark.platforms("posix")
def test_unaccepted_nonempty_token_is_not_an_unlocked_session(external_bw):
    external_bw.accept("dummy-other-token")
    backend = external_bw.backend()
    assert not backend.is_unlocked()
    assert backend.lock_reason == "external_session_rejected"
    with pytest.raises(UnlockRequired) as caught:
        backend.resolve_password("bw:abc")
    assert caught.value.reason == "external_session_rejected"
    assert EXTERNAL not in str(caught.value)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(("case", "detail"), [
    ("symlink", "symlink"), ("directory", "not_regular_file"),
    ("group-readable", "insecure_permissions"), ("other-readable", "insecure_permissions"),
    ("too-large", "too_large"), ("empty", "empty"), ("multiline", "malformed"),
    ("double-newline", "malformed"), ("tab", "malformed"), ("nul", "malformed"),
    ("delete-control", "malformed"), ("invalid-utf8", "malformed"),
    ("relative", "not_absolute"), ("missing", "missing"),
])
def test_unsafe_session_files_have_precise_fail_closed_details(tmp_path, case, detail):
    from agent.vault_backends.session_file import SessionFileUnavailable, read_session_file

    path = tmp_path / "session"
    path.write_bytes(b"dummy-token")
    path.chmod(0o600)
    payloads = {"too-large": b"x" * 4097, "empty": b"", "multiline": b"dummy-a\ndummy-b",
                "double-newline": b"dummy-token\n\n", "tab": b"dummy\ttoken", "nul": b"dummy\x00token",
                "delete-control": b"dummy\x7ftoken", "invalid-utf8": b"\xff"}
    if case in payloads:
        path.write_bytes(payloads[case])
    elif case == "symlink":
        link = tmp_path / "link"
        link.symlink_to(path)
        path = link
    elif case == "directory":
        path.unlink()
        path.mkdir(mode=0o700)
    elif case in ("group-readable", "other-readable"):
        path.chmod(0o640 if case == "group-readable" else 0o604)
    elif case == "relative":
        path = Path("relative-session")
    elif case == "missing":
        path.unlink()
    with pytest.raises(SessionFileUnavailable) as caught:
        read_session_file(str(path))
    assert caught.value.detail == detail
    assert "dummy-token" not in str(caught.value)


@pytest.mark.platforms("posix")
def test_session_file_wrong_owner_is_rejected(tmp_path, monkeypatch):
    from agent.vault_backends.session_file import SessionFileUnavailable, read_session_file

    path = tmp_path / "session"
    path.write_text("dummy-token", encoding="utf-8")
    path.chmod(0o600)
    owner = path.stat().st_uid
    monkeypatch.setattr(os, "geteuid", lambda: owner + 1)
    with pytest.raises(SessionFileUnavailable) as caught:
        read_session_file(str(path))
    assert caught.value.detail == "wrong_owner"


@pytest.mark.platforms("posix")
def test_unreadable_session_file_is_classified_without_token_disclosure(tmp_path, monkeypatch):
    from agent.vault_backends.session_file import SessionFileUnavailable, read_session_file

    path = tmp_path / "session"
    path.write_text("dummy-token", encoding="utf-8")
    path.chmod(0o600)
    original = os.open

    def denied(name, *args, **kwargs):
        if os.fspath(name) == str(path):
            raise PermissionError("dummy access denied")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(os, "open", denied)
    with pytest.raises(SessionFileUnavailable) as caught:
        read_session_file(str(path))
    assert caught.value.detail == "unreadable"
    assert "dummy-token" not in str(caught.value)


@pytest.mark.platforms("posix")
def test_fifo_session_file_is_rejected_without_blocking(tmp_path):
    from agent.vault_backends.session_file import SessionFileUnavailable, read_session_file

    path = tmp_path / "fifo"
    os.mkfifo(path, mode=0o600)
    result = Queue()

    def read_fifo():
        try:
            result.put(read_session_file(str(path)))
        except Exception as exc:
            result.put(exc)

    worker = threading.Thread(target=read_fifo, daemon=True)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive(), "opening a FIFO must not wait for a writer"
    caught = result.get_nowait()
    assert isinstance(caught, SessionFileUnavailable)
    assert caught.detail == "not_regular_file"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("content", [b"dummy-token", b"dummy-token\n", b"x" * 4096],
                         ids=["no-newline", "single-newline", "4096-byte-boundary"])
def test_secure_file_accepts_single_trailing_newline_and_size_boundary(tmp_path, content):
    from agent.vault_backends.session_file import SessionFileRead, read_session_file

    path = tmp_path / "session"
    path.write_bytes(content)
    path.chmod(0o600)
    result = read_session_file(str(path))
    assert isinstance(result, SessionFileRead)
    assert result.token == content.removesuffix(b"\n").decode("utf-8")
    assert isinstance(result.fingerprint, tuple)
    assert read_session_file(str(path)).fingerprint == result.fingerprint
    assert result.token not in repr(result.fingerprint)


@pytest.mark.platforms("posix")
def test_fingerprint_changes_when_same_size_content_changes_with_restored_mtime(tmp_path):
    from agent.vault_backends.session_file import read_session_file

    path = tmp_path / "session"
    path.write_text("dummy-token-a", encoding="utf-8")
    path.chmod(0o600)
    before_stat = path.stat()
    before = read_session_file(str(path))
    path.write_text("dummy-token-b", encoding="utf-8")
    os.utime(path, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
    after = read_session_file(str(path))
    assert after.token == "dummy-token-b"
    assert before.fingerprint != after.fingerprint


@pytest.mark.platforms("posix")
def test_backend_maps_unavailable_file_to_lock_reason_without_cli(external_bw):
    external_bw.session.chmod(0o640)
    backend = external_bw.backend()
    assert not backend.is_unlocked()
    assert backend.lock_reason == "external_session_unavailable"
    with pytest.raises(UnlockRequired) as caught:
        backend.resolve_password("bw:abc")
    assert caught.value.reason == "external_session_unavailable"
    assert external_bw.calls() == []


@pytest.mark.platforms("posix")
def test_interactive_token_takes_priority_over_valid_or_broken_external_file(external_bw):
    backend = external_bw.backend()
    backend.unlock("dummy-master")
    assert backend.resolve_password("bw:abc") == PASSWORD
    external_bw.session.write_text("", encoding="utf-8")
    assert backend.is_unlocked()
    assert backend.resolve_password("bw:abc") == PASSWORD
    assert unlock.get_session_token("bitwarden") == INTERACTIVE
    assert not external_bw.statuses(), "interactive token bypasses external status probes"
    assert all(call["BW_SESSION"] == INTERACTIVE for call in external_bw.calls() if call["argv"][0] != "unlock")


@pytest.mark.platforms("posix")
def test_user_lock_fences_external_until_successful_interactive_unlock(external_bw, monkeypatch):
    backend = external_bw.backend()
    assert backend.is_unlocked()
    unlock.lock("bitwarden")
    calls_before = len(external_bw.calls())
    reads = []
    original_open = os.open

    def record_open(name, *args, **kwargs):
        if os.fspath(name) == str(external_bw.session):
            reads.append(name)
        return original_open(name, *args, **kwargs)

    with monkeypatch.context() as fenced:
        fenced.setattr(os, "open", record_open)
        assert unlock.is_fenced("bitwarden")
        assert not backend.is_unlocked()
        assert backend.lock_reason == "locked_by_user"
        with pytest.raises(UnlockRequired) as caught:
            backend.resolve_password("bw:abc")
        assert caught.value.reason == "locked_by_user"
    assert reads == [], "a user fence must suppress session-file reads, not just CLI calls"
    assert len(external_bw.calls()) == calls_before
    backend.unlock("dummy-master")
    assert not unlock.is_fenced("bitwarden")
    assert backend.is_unlocked()
    assert backend.lock_reason is None
    assert backend.resolve_password("bw:abc") == PASSWORD
    monkeypatch.setattr(unlock, "_IDLE_TTL_S", 0)
    assert not unlock.is_unlocked("bitwarden")
    assert backend.resolve_password("bw:abc") == PASSWORD
    assert external_bw.calls()[-1]["BW_SESSION"] == EXTERNAL
    assert all(call["argv"][0] not in ("lock", "logout") for call in external_bw.calls())


@pytest.mark.platforms("posix")
def test_failed_interactive_unlock_does_not_remove_user_fence(external_bw):
    backend = external_bw.backend()
    unlock.lock("bitwarden")
    with pytest.raises(RuntimeError):
        backend.unlock("dummy-wrong-master")
    assert unlock.is_fenced("bitwarden")
    assert not backend.is_unlocked()
    assert backend.lock_reason == "locked_by_user"


@pytest.mark.platforms("posix")
def test_idle_expiry_returns_to_external_source_without_fence(external_bw, monkeypatch):
    monkeypatch.setattr(unlock, "_IDLE_TTL_S", 0.05)
    backend = external_bw.backend()
    backend.unlock("dummy-master")
    wait_until(lambda: not unlock.is_unlocked("bitwarden"), timeout=5, what="interactive token idle expiry")
    assert not unlock.is_fenced("bitwarden")
    assert backend.resolve_password("bw:abc") == PASSWORD
    assert external_bw.calls()[-1]["BW_SESSION"] == EXTERNAL


@pytest.mark.platforms("posix")
def test_session_release_returns_to_external_source_without_fence(external_bw):
    backend = external_bw.backend()
    unlock.set_current_session_id("dummy-owning-session")
    backend.unlock("dummy-master")
    unlock.release_session("dummy-owning-session")
    assert unlock.get_session_token("bitwarden") is None
    assert not unlock.is_fenced("bitwarden")
    assert backend.resolve_password("bw:abc") == PASSWORD
    assert external_bw.calls()[-1]["BW_SESSION"] == EXTERNAL


@pytest.mark.platforms("posix")
def test_rejected_interactive_token_is_invalidated_without_generation_or_fence(external_bw, caplog):
    backend = external_bw.backend()
    backend.unlock("dummy-master")
    external_bw.accept(EXTERNAL)
    generation = unlock.begin_unlock("bitwarden")
    with pytest.raises(UnlockRequired) as caught:
        backend.resolve_password("bw:abc")
    assert unlock.get_session_token("bitwarden") is None
    assert unlock.begin_unlock("bitwarden") == generation
    assert not unlock.is_fenced("bitwarden")
    assert INTERACTIVE not in str(caught.value)
    assert INTERACTIVE not in caplog.text
    assert backend.resolve_password("bw:abc") == PASSWORD
    assert external_bw.calls()[-1]["BW_SESSION"] == EXTERNAL


@pytest.mark.platforms("posix")
def test_external_status_is_cached_but_atomic_rotation_revalidates(external_bw):
    backend = external_bw.backend()
    assert backend.resolve_password("bw:abc") == PASSWORD
    assert len(external_bw.statuses()) == 1
    assert backend.resolve_password("bw:abc") == PASSWORD
    assert len(external_bw.statuses()) == 1
    replacement = "dummy-external-session-b"
    external_bw.write_token(replacement, atomic=True)
    external_bw.accept(replacement)
    assert backend.resolve_password("bw:abc") == PASSWORD
    assert len(external_bw.statuses()) == 2
    assert external_bw.calls()[-1]["BW_SESSION"] == replacement
    assert external_bw.statuses()[-1]["BW_SESSION"] == replacement


@pytest.mark.platforms("posix")
def test_rewritten_same_metadata_token_is_revalidated(external_bw):
    backend = external_bw.backend()
    assert backend.is_unlocked()
    before = external_bw.session.stat()
    replacement = "dummy-external-session-b"
    external_bw.write_token(replacement)
    os.utime(external_bw.session, ns=(before.st_atime_ns, before.st_mtime_ns))
    external_bw.accept(replacement)
    assert backend.resolve_password("bw:abc") == PASSWORD
    assert len(external_bw.statuses()) == 2
    assert external_bw.calls()[-1]["BW_SESSION"] == replacement


@pytest.mark.platforms("posix")
def test_external_rejection_clears_validation_cache_without_fence_or_leaks(external_bw, caplog):
    from tools.environments.local import build_subprocess_env

    backend = external_bw.backend()
    assert backend.is_unlocked()
    generation = unlock.begin_unlock("bitwarden")
    reject = external_bw.root / "reject-actual"
    reject.touch()
    with pytest.raises(UnlockRequired) as caught:
        backend.resolve_password("bw:abc")
    assert caught.value.reason == "external_session_rejected"
    assert EXTERNAL not in str(caught.value)
    assert EXTERNAL not in caplog.text
    assert EXTERNAL not in os.environ.values()
    assert EXTERNAL not in build_subprocess_env(scrub_secrets=False).values()
    assert not unlock.is_fenced("bitwarden")
    assert unlock.begin_unlock("bitwarden") == generation
    assert len(external_bw.statuses()) == 1
    reject.unlink()
    assert backend.resolve_password("bw:abc") == PASSWORD
    assert len(external_bw.statuses()) == 2


@pytest.mark.platforms("posix")
def test_empty_during_rewrite_recovers_without_user_unlock(external_bw):
    backend = external_bw.backend()
    assert backend.is_unlocked()
    external_bw.write_token("")
    assert not backend.is_unlocked()
    assert backend.lock_reason == "external_session_unavailable"
    assert not unlock.is_fenced("bitwarden")
    external_bw.write_token(EXTERNAL)
    assert backend.is_unlocked()
    assert backend.lock_reason is None
    assert backend.resolve_password("bw:abc") == PASSWORD


@pytest.mark.platforms("posix")
def test_user_fences_are_isolated_between_profiles(external_bw, monkeypatch):
    backend = external_bw.backend()
    first = os.environ["HERMES_HOME"]
    second = str(external_bw.root / "profile-b")
    assert backend.is_unlocked()
    with monkeypatch.context() as other:
        other.setenv("HERMES_HOME", second)
        unlock.lock("bitwarden")
        assert not backend.is_unlocked()
        assert backend.lock_reason == "locked_by_user"
    assert os.environ["HERMES_HOME"] == first
    assert backend.is_unlocked()
    with monkeypatch.context() as other:
        other.setenv("HERMES_HOME", second)
        unlock.set_current_session_id("dummy-profile-b-session")
        backend.unlock("dummy-master")
        unlock.release_session("dummy-profile-b-session")
        unlock.set_current_session_id(None)
        assert unlock.get_session_token("bitwarden") is None
    unlock.lock("bitwarden")
    assert not backend.is_unlocked()
    with monkeypatch.context() as other:
        other.setenv("HERMES_HOME", second)
        assert not unlock.is_fenced("bitwarden")
        assert backend.is_unlocked()
        unlock.lock("bitwarden")


def test_session_file_default_is_explicitly_opt_in():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["vault"]["bitwarden"]["session_file"] == ""


@pytest.mark.platforms("windows")
def test_session_file_on_windows_is_unavailable_not_a_crash(tmp_path):
    from agent.vault_backends.session_file import SessionFileUnavailable, read_session_file

    path = tmp_path / "session"
    path.write_text("dummy-windows-token", encoding="utf-8")
    with pytest.raises(SessionFileUnavailable) as caught:
        read_session_file(str(path))
    assert caught.value.detail == "unsupported_platform"
    backend = BitwardenLoginBackend({"session_file": str(path)})
    assert not backend.is_unlocked()
    assert backend.lock_reason == "external_session_unavailable"


def test_unlock_required_carries_optional_public_reason_without_token():
    from agent.vault_backends.base import (
        EXTERNAL_SESSION_REJECTED,
        EXTERNAL_SESSION_UNAVAILABLE,
        LOCKED_BY_USER,
        LoginBackend,
    )

    backend = BitwardenLoginBackend()
    assert LoginBackend.lock_reason is None
    assert UnlockRequired(backend).reason is None
    for reason in (LOCKED_BY_USER, EXTERNAL_SESSION_UNAVAILABLE, EXTERNAL_SESSION_REJECTED):
        error = UnlockRequired(backend, reason=reason)
        assert error.backend is backend
        assert error.reason == reason
        assert EXTERNAL not in str(error)
    assert (LOCKED_BY_USER, EXTERNAL_SESSION_UNAVAILABLE, EXTERNAL_SESSION_REJECTED) == (
        "locked_by_user", "external_session_unavailable", "external_session_rejected")


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("mode", [0o600, 0o640], ids=["secure-fd", "insecure-fd"])
def test_session_file_checks_and_reads_the_open_fd_despite_path_replacement(tmp_path, monkeypatch, mode):
    from agent.vault_backends.session_file import SessionFileUnavailable, read_session_file

    path = tmp_path / "session"
    path.write_text("dummy-opened-token", encoding="utf-8")
    path.chmod(mode)
    replacement = tmp_path / "replacement"
    replacement.write_text("dummy-replacement-token", encoding="utf-8")
    replacement.chmod(0o600)
    original_open = os.open

    def replace_after_open(name, *args, **kwargs):
        fd = original_open(name, *args, **kwargs)
        if os.fspath(name) == str(path):
            replacement.replace(path)
        return fd

    monkeypatch.setattr(os, "open", replace_after_open)
    if mode == 0o600:
        assert read_session_file(str(path)).token == "dummy-opened-token"
    else:
        with pytest.raises(SessionFileUnavailable) as caught:
            read_session_file(str(path))
        assert caught.value.detail == "insecure_permissions"


@pytest.mark.platforms("posix")
def test_session_file_growth_after_fstat_is_still_rejected_as_too_large(tmp_path, monkeypatch):
    from agent.vault_backends.session_file import SessionFileUnavailable, read_session_file

    path = tmp_path / "session"
    path.write_text("dummy-token", encoding="utf-8")
    path.chmod(0o600)
    identity = path.stat()
    original_fstat = os.fstat

    def grow_after_stat(fd):
        result = original_fstat(fd)
        if (result.st_dev, result.st_ino) == (identity.st_dev, identity.st_ino):
            with path.open("ab") as stream:
                stream.write(b"x" * 4097)
        return result

    monkeypatch.setattr(os, "fstat", grow_after_stat)
    with pytest.raises(SessionFileUnavailable) as caught:
        read_session_file(str(path))
    assert caught.value.detail == "too_large"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("source", ["external", "interactive"])
def test_rejected_inflight_call_preserves_newer_interactive_token(external_bw, monkeypatch, source):
    from agent.vault_backends import bitwarden

    backend = external_bw.backend()
    if source == "interactive":
        backend.unlock("dummy-master")
    else:
        assert backend.is_unlocked()
    generation = unlock.begin_unlock("bitwarden")
    (external_bw.root / "reject-actual").touch()
    real_run_cli = bitwarden.run_cli

    def complete_with_new_login(argv, **kwargs):
        result = real_run_cli(argv, **kwargs)
        if argv[1:3] == ["get", "password"]:
            assert unlock.store_session_token("bitwarden", "dummy-newer-interactive", generation)
        return result

    monkeypatch.setattr(bitwarden, "run_cli", complete_with_new_login)
    with pytest.raises(UnlockRequired):
        backend.resolve_password("bw:abc")
    assert unlock.get_session_token("bitwarden") == "dummy-newer-interactive"
    assert unlock.begin_unlock("bitwarden") == generation
    assert not unlock.is_fenced("bitwarden")


@pytest.mark.platforms("posix")
def test_nul_session_file_path_fails_closed_without_cli(external_bw):
    from agent.vault_backends.session_file import SessionFileUnavailable, read_session_file

    path = str(external_bw.root / "ses\x00sion")
    with pytest.raises(SessionFileUnavailable) as caught:
        read_session_file(path)
    assert caught.value.detail == "invalid_path"

    backend = BitwardenLoginBackend({"binary_path": str(external_bw.exe), "session_file": path})
    assert backend.is_unlocked() is False
    assert backend.lock_reason == "external_session_unavailable"
    with pytest.raises(UnlockRequired) as locked:
        backend.resolve_password("bw:abc")
    assert locked.value.reason == "external_session_unavailable"
    assert external_bw.calls() == []


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("session_file", [123, ["x"]], ids=["integer", "list"])
def test_invalid_session_file_config_types_fail_closed_without_cli(external_bw, session_file):
    backend = BitwardenLoginBackend({"binary_path": str(external_bw.exe), "session_file": session_file})
    assert backend.is_unlocked() is False
    assert backend.lock_reason == "external_session_unavailable"
    assert external_bw.calls() == []


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("failure", ["missing-binary", "nul-binary", "not-on-path", "timeout"])
def test_external_status_probe_failures_are_unavailable_without_token_leaks(
    external_bw, monkeypatch, caplog, failure,
):
    import logging
    import subprocess

    import agent.secret_sources.base as secret_base
    from agent.vault_backends import bitwarden

    caplog.set_level(logging.DEBUG)
    cfg = {"binary_path": str(external_bw.exe), "session_file": str(external_bw.session)}
    if failure == "missing-binary":
        cfg["binary_path"] = str(external_bw.root / "missing-bw")
    elif failure == "nul-binary":
        cfg["binary_path"] = str(external_bw.root / "b\x00w")
    elif failure == "not-on-path":
        cfg.pop("binary_path")
        monkeypatch.setattr(bitwarden.shutil, "which", lambda name: None)
    else:
        def status_timeout(cmd, **kwargs):
            assert cmd[1:2] == ["status"]
            raise subprocess.TimeoutExpired(cmd, 30, output=EXTERNAL, stderr=EXTERNAL)

        monkeypatch.setattr(secret_base.subprocess, "run", status_timeout)

    backend = BitwardenLoginBackend(cfg)
    assert backend.is_unlocked() is False
    assert backend.lock_reason == "external_session_unavailable"
    with pytest.raises(UnlockRequired) as caught:
        backend.resolve_password("bw:abc")
    assert caught.value.reason == "external_session_unavailable"
    assert EXTERNAL not in str(caught.value)
    assert EXTERNAL not in repr(caught.value)
    assert EXTERNAL not in caplog.text
    assert external_bw.calls() == []


@pytest.mark.platforms("posix")
def test_invalid_utf8_session_payload_has_no_secret_bearing_exception_chain(tmp_path):
    import traceback

    from agent.vault_backends.session_file import SessionFileUnavailable, read_session_file

    path = tmp_path / "session"
    path.write_bytes(b"dummy-secret-bytes\xff")
    path.chmod(0o600)
    with pytest.raises(SessionFileUnavailable) as caught:
        read_session_file(str(path))
    exc = caught.value
    assert exc.detail == "malformed"
    assert exc.__context__ is None
    assert exc.__cause__ is None
    assert "dummy-secret-bytes" not in "".join(traceback.format_exception(exc))


@pytest.mark.platforms("posix")
def test_cached_status_is_recent_validation_not_provider_revocation_detection(external_bw):
    backend = external_bw.backend()
    assert backend.resolve_password("bw:abc") == PASSWORD
    assert len(external_bw.statuses()) == 1
    external_bw.accept("dummy-other-token")

    # Status reports the most recent validation until an operation detects rejection.
    assert backend.is_unlocked() is True
    assert len(external_bw.statuses()) == 1
    with pytest.raises(UnlockRequired) as caught:
        backend.resolve_password("bw:abc")
    assert caught.value.reason == "external_session_rejected"
    assert EXTERNAL not in str(caught.value)
    assert EXTERNAL not in repr(caught.value)
    assert len(external_bw.statuses()) == 1
    assert backend.is_unlocked() is False
    assert backend.lock_reason == "external_session_rejected"
    assert len(external_bw.statuses()) == 2
    assert unlock.is_fenced("bitwarden") is False


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("source", ["external", "interactive"])
def test_user_lock_allows_started_resolve_but_fences_next_call(external_bw, monkeypatch, source):
    from agent.vault_backends import bitwarden

    backend = external_bw.backend()
    if source == "interactive":
        backend.unlock("dummy-master")
    real_run_cli = bitwarden.run_cli

    def lock_before_get_password(argv, **kwargs):
        if argv[1:3] == ["get", "password"]:
            unlock.lock("bitwarden")
        return real_run_cli(argv, **kwargs)

    with monkeypatch.context() as inflight:
        inflight.setattr(bitwarden, "run_cli", lock_before_get_password)
        assert backend.resolve_password("bw:abc") == PASSWORD
    calls_after_started_operation = external_bw.calls()
    assert calls_after_started_operation[-1]["argv"][:2] == ["get", "password"]
    assert calls_after_started_operation[-1]["BW_SESSION"] == (
        EXTERNAL if source == "external" else INTERACTIVE
    )
    with pytest.raises(UnlockRequired) as caught:
        backend.resolve_password("bw:abc")
    assert caught.value.reason == "locked_by_user"
    assert external_bw.calls() == calls_after_started_operation
    assert unlock.is_fenced("bitwarden") is True


@pytest.mark.platforms("posix")
def test_enabled_backends_loads_each_profiles_own_session_file(external_bw, monkeypatch):
    from agent.vault_backends import enabled_backends
    import agent.vault_backend_registry as registry

    homes = [external_bw.root / "home-a", external_bw.root / "home-b"]
    tokens = ["dummy-profile-a-session", "dummy-profile-b-session"]
    session_paths = [home / "session" for home in homes]
    external_bw.accept(*tokens)
    for home, session, token in zip(homes, session_paths, tokens):
        home.mkdir()
        session.write_text(token, encoding="utf-8")
        session.chmod(0o600)
        config = {"vault": {
            "onepassword": {"enabled": False},
            "bitwarden": {"binary_path": str(external_bw.exe), "session_file": str(session)},
        }}
        (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")

    # Plugin constructors are unrelated to native profile/config/session-file routing.
    monkeypatch.setattr(registry, "enabled_plugin_backends", lambda cfg: [])
    original_open = os.open
    opened_sessions = []
    known_sessions = {str(path) for path in session_paths}

    def record_session_open(name, *args, **kwargs):
        if isinstance(name, (str, bytes, os.PathLike)):
            path = os.fsdecode(name)
            if path in known_sessions:
                opened_sessions.append(path)
        return original_open(name, *args, **kwargs)

    monkeypatch.setattr(os, "open", record_session_open)
    try:
        for current, other in [(1, 0), (0, 1)]:
            monkeypatch.setenv("HERMES_HOME", str(homes[current]))
            opened_sessions.clear()
            backend = next(backend for backend in enabled_backends() if backend.name == "bitwarden")
            assert backend.resolve_password("bw:abc") == PASSWORD
            assert str(session_paths[current]) in opened_sessions
            assert str(session_paths[other]) not in opened_sessions
            assert external_bw.calls()[-1]["BW_SESSION"] == tokens[current]
    finally:
        for home in homes:
            monkeypatch.setenv("HERMES_HOME", str(home))
            unlock.lock("bitwarden")
