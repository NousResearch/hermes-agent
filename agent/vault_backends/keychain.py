"""macOS Keychain login backend for the browser credential vault (``security`` CLI).

A dedicated, passworded keychain file (default ``<HERMES_HOME>/vault/keychain.keychain-db``)
holds internet-password items. The unlock password lives in a 0600 sidecar
(``<HERMES_HOME>/vault/keychain.pw``) when Hermes may unlock unattended, or is asked for
via the surface's masked prompt when it may not. Handles are ``kc:<server>|<account>``.

Why a dedicated file instead of the login keychain: the login keychain is tied to the
GUI security session and prompts — locked/headless reads fail with rc 36/152 (see the
hermes-secret-management skill). A dedicated file created with ``create-keychain``
unlocks with its own password, is headless-safe, profile-scoped under HERMES_HOME, and
never touches the keychain search list (verified live: ``security list-keychains`` is
unchanged after create).

``security`` CLI surface (verified on macOS 27):
- create-keychain <file>                     prompts on a TTY ("password:" / "retype:")
                                              → the backend feeds a pty; rc=0 headless-safe;
                                              never adds to the keychain search list
- add-internet-password -a <acct> -s <srvr> [-l label] [-D origin] -w <pw> [-U] <file>
        headless rc=0 WITHOUT ``-A`` (the ``-A`` flag is INVALID here, rc 48).
        ``-U`` updates an existing item in place — and creates it when missing — so
        replaces never delete first (a failed write cannot lose the old credential).
        ``-D`` persists the item description; the backend stores the exact login
        origin there and fills only that exact origin (no scheme widening).
        The ``-w`` value is the ONLY channel that commits the item password
        headlessly (the "put -w last to prompt" flavor does not commit): the value
        travels in argv — a documented, process-local OS boundary, same convention
        as the pre-existing hermes.keychain-db wrapper. Env and captured streams
        never carry secrets; unlock/create never use argv at all (pty channel).
- find-internet-password -a -s -g <file>  rc=0 + dump on STDOUT; the ``password: "…"``
                                          line is written to STDERR (verified on macOS 27);
                                          no match → rc 44; locked → rc 152. Lock state
                                          cannot be probed with a non-matching item (the
                                          item search never touches the secret partition),
                                          so attended-mode lock state is tracked by a
                                          process-level lease instead.
- dump-keychain <file>                    metadata ONLY — no passwords — works even while
                                          LOCKED (passwordless enumeration by design);
                                          missing file → rc 0 empty. One block per item,
                                          separated by ``keychain:`` header lines.
- lock-keychain / unlock-keychain <file>  unlock prompts on a TTY (pty-fed here); locked
                                          secret reads fail rc 152
- delete-internet-password -a -s <file>   rc=0 ("password has been deleted.")
- set-keychain-settings -ut <secs> <file> auto-lock timer (24h default mirrors the
                                          hermes.keychain-db wrapper pattern)

Unlock authority is process-level, not instance-level: ``enabled_backends()`` builds a
fresh backend per tool call, so attended unlocks are recorded in a generation-fenced
lease keyed by keychain file (``_LEASES``). Every backend instance sees the lease; the
keychain is physically re-locked when the lease expires (idle TTL) or the process exits
(``atexit`` relock registry).

Secrets never enter tool results, logs, or the session DB: only the fill path resolves,
and the browser layer types values straight into the page. All security invocations are
wrapped in a subprocess timeout so a consent prompt can never hang a tool call.
"""

from __future__ import annotations

import atexit
import os
import pathlib
from pathlib import Path
import pty
import re
import secrets
import select
import signal
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent.vault_backends.base import LoginBackend, UnlockRequired
from agent.vault_store import VaultError, VaultItemMeta, normalize_origin

_SEC = "/usr/bin/security"
_SEC_TIMEOUT = 30.0
# rc codes that mean "keychain locked / authorization refused" for secret reads:
# 36 = user interaction not allowed (background session), 152 = locked keychain.
_LOCKED_RCS = (36, 152)
_INET_MAX_PASSWORD = 4096

# Attended-mode unlock lease (process-level so fresh per-call backend instances all
# see an unlock; generation-fenced; physically re-locks on idle-TTL expiry or exit).
_LEASE_TTL = 30 * 60.0
_LEASES: Dict[str, Dict[str, Any]] = {}
_EXIT_RELOCKS: set = set()

_BLOCK_RE = re.compile(r"(?m)^keychain: ")
_ATTR = lambda name: re.compile(rf'(?m)^    "{name}"<blob>="([^"]*)"')  # noqa: E731
_SRVR_RE = _ATTR("srvr")
_ACCT_RE = _ATTR("acct")
_DESC_RE = _ATTR("desc")
_LABEL_RE = re.compile(r'(?m)^    0x00000007 <blob>="([^"]*)"')
_CDAT_RE = re.compile(r'(?m)^    "cdat"<timedate>=0x[0-9A-F]+  "(\d{14})Z')
_PASSWORD_RE = re.compile(r'(?m)^password: "((?:[^"\\]|\\.)*)"')
_PROMPT_RE = re.compile(r"[:\?]\s*$")


def _prompt_security(argv: Sequence[str], replies: Sequence[str],
                     timeout: float = _SEC_TIMEOUT, env: Optional[Dict[str, str]] = None,
                     cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run ``security`` feeding its TTY password prompts through a pty.

    The child gets a fresh pty as its controlling terminal; this side writes one reply
    line each time the drained output ends in a prompt (timeout fallback keeps a missing
    prompt from stalling). ``argv``/``env`` NEVER contain the reply lines. Returns a
    CompletedProcess: returncode 0 on clean exit, -9 if it had to be killed.
    """
    prev_sigchld = signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    pid, master = None, None
    out = b""
    written = 0
    last_feed = 0.0
    deadline = time.time() + timeout
    exited = False
    rc = None
    eof_sent = False
    try:
        pid, master = pty.fork()
        if pid == 0:  # child
            # Inherited SIG_IGN on SIGCHLD (uv-managed CPython / asyncio callers set it)
            # would break the child's own wait() inside `security`, hanging the binary
            # before its first prompt — reset the default disposition before exec.
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)
            try:
                os.execve(_SEC, [_SEC, *argv], dict(env or os.environ))
            except Exception:  # pragma: no cover
                os._exit(127)
        while time.time() < deadline:
            r, _, _ = select.select([master], [], [], 0.25)
            if r:
                try:
                    chunk = os.read(master, 4096)
                except OSError:
                    # Child closed its side (exiting or dead): stop reading; the
                    # post-loop reap will capture its real rc.
                    break
                if not chunk:
                    break
                out += chunk
            # Feed remaining replies ~blind: canonical-mode tty input queues whole
            # lines in order, so each prompt consumes the next queued reply
            # regardless of exact prompt timing. Feeding is paced only to avoid
            # bursting; the child's own tcsetattr drains depend on us reading.
            if written < len(replies) and time.time() - last_feed > 0.15:
                try:
                    os.write(master, (replies[written] + "\n").encode())
                except OSError:
                    exited = True
                    break
                written += 1
                last_feed = time.time()
            elif written >= len(replies) and not eof_sent and time.time() - last_feed > 0.5:
                # All replies fed: deliver EOF (Ctrl-D). It queues BEHIND the reply
                # lines, so each prompt consumes its line and the child's final
                # post-prompt read sees EOF — `security` lingers forever on that
                # read after create/unlock otherwise (verified empirically).
                eof_sent = True
                try:
                    os.write(master, b"\x04")
                except OSError:
                    pass
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                exited = True
                rc = os.waitstatus_to_exitcode(status)
                break
        if not exited:
            # EIO/EOF on the master usually means the child is done — reap it with
            # a short grace instead of assuming anything (its real rc is the truth).
            done, status = os.waitpid(pid, os.WNOHANG)
            grace = time.time() + 5.0
            while not done and time.time() < grace:
                time.sleep(0.05)
                done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                exited = True
                rc = os.waitstatus_to_exitcode(status)
            elif written >= len(replies):
                # Still alive with all replies fed: deliver EOF (Ctrl-D, not a close —
                # closing the master can SIGHUP the child) so prompt-blocked children
                # (create-keychain waits on stdin after the retype) finish cleanly.
                try:
                    os.write(master, b"\x04")
                except OSError:
                    pass
                done, status = os.waitpid(pid, os.WNOHANG)
                grace = time.time() + 5.0
                while not done and time.time() < grace:
                    time.sleep(0.05)
                    done, status = os.waitpid(pid, os.WNOHANG)
                if done:
                    exited = True
                    rc = os.waitstatus_to_exitcode(status)
    finally:
        if not exited and pid is not None:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        if master is not None:
            try:
                os.close(master)
            except OSError:
                pass
        try:
            signal.signal(signal.SIGCHLD, prev_sigchld)
        except ValueError:  # pragma: no cover — non-main thread
            pass
    return subprocess.CompletedProcess([_SEC, *argv], rc if rc is not None else -9, out, b"")


def _physical_lock(path: str) -> None:
    """Best-effort physical re-lock (attended-lease expiry / process exit)."""
    try:
        subprocess.run([_SEC, "lock-keychain", path],
                       capture_output=True, timeout=_SEC_TIMEOUT)
    except Exception:
        pass


def _purge_expired_leases() -> None:
    """Physically re-lock keychains whose attended lease expired (idle TTL)."""
    now = time.time()
    for path, lease in list(_LEASES.items()):
        if lease.get("unlocked") and now - float(lease.get("unlocked_at", 0)) > _LEASE_TTL:
            _physical_lock(path)
            lease["unlocked"] = False
            lease["generation"] = int(lease.get("generation", 0)) + 1
            _EXIT_RELOCKS.discard(path)


def _register_exit_relock(path: str) -> None:
    if not _EXIT_RELOCKS:
        atexit.register(_relock_all_on_exit)
    _EXIT_RELOCKS.add(path)


def _relock_all_on_exit() -> None:  # pragma: no cover (atexit hook)
    for path in list(_EXIT_RELOCKS):
        _physical_lock(path)


def _identifier_type(account: str) -> Optional[str]:
    if "@" in account:
        return "email"
    if account.isdigit():
        return "phone"
    return "username"


def _created_at(raw: Optional[str]) -> str:
    if not raw:
        return ""
    try:
        dt = datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return ""


def _unescape_dump(s: str) -> str:
    """``security`` renders quotes as ``\"`` inside the dump; undo just those two escapes."""
    return s.replace('\\"', '"').replace("\\\\", "\\")


def _value(match: Optional[re.Match]) -> str:
    return _unescape_dump(match.group(1)) if match else ""


def _origin_ok(origin: str) -> bool:
    try:
        normalize_origin(origin)
        return True
    except Exception:
        return False


def _origin_from_desc(desc: Optional[str]) -> Optional[str]:
    """The exact origin persisted at add time (scheme+host+port as saved), or None."""
    if not desc:
        return None
    m = re.match(r"^(https?://[^/]+)(?:/.*)?$", desc.strip())
    return m.group(1) if m else None


class MacOSKeychainLoginBackend(LoginBackend):
    """Login items in a dedicated macOS keychain file, read headlessly via ``security``.

    Two unlock modes, decided by whether a password sidecar exists:
    - password sidecar present → ``needs_unlock`` False; locked reads self-heal (unlock
      with the sidecar password and retry once). Unattended/headless mode, provisioned
      by ``hermes vault keychain init`` (rc 152 → auto-unlock → retry).
    - no sidecar → ``needs_unlock`` True; the surface prompts for the keychain master
      password (``browser_vault_unlock`` → :meth:`unlock`), recorded in a process-level
      generation-fenced lease so later fresh instances still see it. ``list_items``
      works while locked: the OS exposes item metadata without unlocking.
    """

    name = "keychain"
    display_name = "macOS Keychain"
    prefix = "kc:"

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}
        # Instance-level (not class-level): the mode is a property of the files present.
        self.needs_unlock = not self._pw_file().exists()
        _purge_expired_leases()

    # -- paths ---------------------------------------------------------------

    def _file(self) -> Path:
        explicit = str(self.cfg.get("file") or "").strip()
        if explicit:
            return Path(explicit).expanduser()
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()) / "vault" / "keychain.keychain-db"

    def _pw_file(self) -> Path:
        explicit = str(self.cfg.get("password_file") or "").strip()
        if explicit:
            return Path(explicit).expanduser()
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()) / "vault" / "keychain.pw"

    # -- plumbing ------------------------------------------------------------

    def _env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env.pop("DISPLAY", None)  # a consent dialog must never reach the GUI
        return env

    def _sec(self, *args: str, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run ``security`` with a hard timeout (the consent-prompt guard). No secrets
        in argv on this path — prompt ops go through :meth:`_sec_prompt`."""
        _purge_expired_leases()
        try:
            return subprocess.run(  # noqa: S603 — argv list, no shell
                [_SEC, *args], cwd=str(cwd) if cwd else str(self._file().parent),
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=_SEC_TIMEOUT, env=self._env())
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"keychain operation timed out after {_SEC_TIMEOUT:.0f}s") from exc
        except OSError as exc:
            raise RuntimeError(f"failed to invoke security: {exc}") from exc

    def _sec_prompt(self, argv: Sequence[str], replies: Sequence[str]) -> subprocess.CompletedProcess:
        """``security`` with password prompts fed through a pty — no secrets in argv/env."""
        _purge_expired_leases()
        return _prompt_security(argv, replies, env=self._env(), cwd=str(self._file().parent))

    def _unlock_with_password(self, password: str) -> bool:
        """Unlock feeding the password through the pty; the child never sees it in argv."""
        proc = self._sec_prompt(["unlock-keychain", str(self._file())], [password])
        return proc.returncode == 0

    def _auto_unlock(self) -> bool:
        """Unattended mode: unlock with the sidecar password (read into this process)."""
        pw_path = self._pw_file()
        try:
            pw = pw_path.read_text(encoding="utf-8").strip()
        except OSError:
            return False
        if not pw:
            return False
        ok = self._unlock_with_password(pw)
        if ok:
            _register_exit_relock(str(self._file()))
        return ok

    def _run_unlocked(self, fn, *args) -> subprocess.CompletedProcess:
        """Run one ``security`` call; on a locked rc, heal in sidecar mode or raise
        ``UnlockRequired`` for the surface, then retry ONCE."""
        proc = fn(*args)
        if proc.returncode not in _LOCKED_RCS:
            return proc
        if self.needs_unlock or not self._auto_unlock():
            # attended mode: clear the lease so the surface re-prompts authoritatively
            self._lease()["unlocked"] = False
            self._lease()["generation"] = int(self._lease()["generation"]) + 1
            raise UnlockRequired(self)
        return fn(*args)

    def _lease(self) -> Dict[str, Any]:
        return _LEASES.setdefault(str(self._file()),
                                  {"unlocked": False, "generation": 0, "unlocked_at": 0.0})

    def release(self) -> None:
        """Session release: physically re-lock the keychain and drop the attended lease."""
        lease = self._lease()
        if lease.get("unlocked") is True or not self.needs_unlock:
            try:
                self._sec("lock-keychain", str(self._file()))
            except Exception:
                pass
        lease["unlocked"] = False
        lease["generation"] = int(lease.get("generation", 0)) + 1
        _EXIT_RELOCKS.discard(str(self._file()))

    # -- LoginBackend contract ------------------------------------------------

    def is_unlocked(self) -> bool:
        _purge_expired_leases()
        if not self._file().exists():
            return False
        if not self.needs_unlock:
            # Sidecar mode is self-healing: locked reads are transparently unlocked.
            return True
        return self._lease().get("unlocked") is True

    def unlock(self, master_password: str) -> None:
        """Attended unlock (masked prompt on the surface). Records a generation-fenced
        lease every fresh instance in this process can see; the OS keeps the keychain
        open until the lease expires (idle TTL) or this process exits (atexit relock)."""
        if not self.needs_unlock:
            return
        proc = self._sec_prompt(["unlock-keychain", str(self._file())], [master_password])
        if proc.returncode != 0:
            raise RuntimeError("keychain unlock failed — wrong master password?")
        lease = self._lease()
        lease["unlocked"] = True
        lease["generation"] = int(lease.get("generation", 0)) + 1
        lease["unlocked_at"] = time.time()
        _register_exit_relock(str(self._file()))

    def _parse_handle(self, handle: str) -> Optional[Tuple[str, str]]:
        suffix = handle[len(self.prefix):] if handle.startswith(self.prefix) else ""
        server, _, account = suffix.partition("|")
        if not server:
            return None
        return server, account

    def list_items(self) -> List[VaultItemMeta]:
        file = self._file()
        if not file.exists():
            return []
        proc = self._sec("dump-keychain", str(file))
        if proc.returncode in _LOCKED_RCS:
            return []  # metadata is exposed while locked in practice; stay conservative
        if proc.returncode != 0:
            raise RuntimeError(f"keychain dump failed (rc={proc.returncode})")
        out: List[VaultItemMeta] = []
        for block in _BLOCK_RE.split(proc.stdout)[1:]:
            server = _value(_SRVR_RE.search(block))
            if not server:
                continue  # no server -> nothing to bind; not a fill target
            account = _value(_ACCT_RE.search(block))
            label = _value(_LABEL_RE.search(block)) or server
            origin = _origin_from_desc(_value(_DESC_RE.search(block)))
            # Exact-origin policy: an item is fillable ONLY on the origin persisted at
            # add time (scheme+host+port as saved). No origin on record -> not fillable.
            allowed = tuple([origin]) if origin and _origin_ok(origin) else tuple()
            out.append(VaultItemMeta(
                id=f"{self.prefix}{server}|{account}",
                kind="login",
                label=label,
                origin=origin,
                created_at=_created_at(_value(_CDAT_RE.search(block))),
                identifier_type=_identifier_type(account) if account else None,
                identifier=account or None,
                allowed_origins=allowed))
        return out

    def get_meta(self, handle: str) -> Optional[VaultItemMeta]:
        """Metadata for one handle. The authoritative origin comes from the item's
        persisted description (exact); ``allowed_origins`` is exactly that origin, or
        empty when the item predates origin recording (not fillable — never widened)."""
        parsed = self._parse_handle(handle)
        if parsed is None:
            return None
        server, account = parsed
        file = self._file()
        if not file.exists():
            return None
        try:
            proc = self._sec("find-internet-password", "-a", account, "-s", server,
                             "-g", str(file))
        except RuntimeError:
            return None
        if proc.returncode != 0:
            return None
        text = proc.stdout
        block_m = _BLOCK_RE.search(text)
        block = text[block_m.end():] if block_m else text
        if _value(_SRVR_RE.search(block)) != server:
            return None
        origin = _origin_from_desc(_value(_DESC_RE.search(block)))
        allowed = tuple([origin]) if origin and _origin_ok(origin) else tuple()
        return VaultItemMeta(
            id=handle,
            kind="login",
            label=_value(_LABEL_RE.search(block)) or server,
            origin=origin,
            created_at=_created_at(_value(_CDAT_RE.search(block))),
            identifier_type=_identifier_type(account) if account else None,
            identifier=account or None,
            allowed_origins=allowed)

    def resolve_password(self, handle: str) -> str:
        parsed = self._parse_handle(handle)
        if parsed is None:
            raise VaultError(f"malformed keychain handle {handle!r}")
        server, account = parsed
        file = self._file()
        if not file.exists():
            raise VaultError("keychain file missing — re-run `hermes vault keychain init`")
        proc = self._run_unlocked(self._sec, "find-internet-password", "-a", account,
                                  "-s", server, "-g", str(file))
        if proc.returncode == 44:
            raise VaultError("keychain item no longer exists")
        if proc.returncode != 0:
            raise RuntimeError(f"keychain lookup failed (rc={proc.returncode})")
        # The dump goes to stdout; `security` writes the password: line to STDERR (verified).
        match = _PASSWORD_RE.search(proc.stdout + proc.stderr)
        return _unescape_dump(match.group(1)) if match else ""

    # -- write path (hermes vault keychain add/rm) -----------------------------

    def add_item(self, server: str, account: str, password: str,
                 label: Optional[str] = None, origin: Optional[str] = None) -> str:
        """Insert or atomically replace the (server, account) item.

        ``-U`` updates in place (creates when missing), so a replace never deletes
        first: a failed write cannot lose the previous credential. ``-D`` persists
        the exact login origin (as saved) — the only origin fills will target.
        Returns the fill handle.
        """
        server = (server or "").strip().lower()
        account = (account or "").strip()
        if not server or not account:
            raise VaultError("keychain item needs a server and an account")
        if len(password) > _INET_MAX_PASSWORD:
            raise VaultError("keychain item password too long")
        file = self._file()
        if not file.exists():
            raise VaultError("no keychain file yet — run `hermes vault keychain init` first")
        # `security -U` matches on the FULL attribute tuple (label included): an
        # update that omits a previously-set attribute is a *different* item → rc 45.
        # Preserve the existing item's label/origin so in-place replaces stay atomic.
        if origin is None or label is None:
            target = f"{self.prefix}{server}|{account}"
            existing = next((m for m in self.list_items() if m.id == target), None)
            if existing is not None:
                origin = existing.origin
                label = label or existing.label
        argv = ["add-internet-password", "-a", account, "-s", server]
        if origin:
            argv += ["-D", origin]  # exact origin, exactly as saved (scheme+host+port)
        if label:
            argv += ["-l", label]
        # -w stays an argv value: empirically the ONLY channel `security` commits for
        # internet-password items (its pty prompt flavor does not commit headlessly).
        # Documented OS boundary — env/captured streams still never carry the secret.
        argv += ["-w", password, "-U", str(file)]
        proc = self._run_unlocked(self._sec, *argv)
        if proc.returncode != 0:
            raise VaultError(f"keychain add failed (rc={proc.returncode})")
        return f"{self.prefix}{server}|{account}"

    def resolve_otp(self, handle: str) -> str:
        raise NotImplementedError("keychain backend does not store TOTP secrets")

    def remove_item(self, handle: str) -> bool:
        parsed = self._parse_handle(handle)
        if parsed is None:
            return False
        server, account = parsed
        file = self._file()
        proc = self._run_unlocked(self._sec, "delete-internet-password", "-a", account,
                                  "-s", server, str(file))
        return proc.returncode == 0

    def status(self) -> Dict[str, object]:
        file, pw_path = self._file(), self._pw_file()
        return {
            "file": str(file), "password_file": str(pw_path),
            "file_exists": file.exists(), "password_file_exists": pw_path.exists(),
            "mode": "unattended" if pw_path.exists() else "prompt",
            "item_count": len(self.list_items()) if file.exists() else 0,
            "unlocked": self.is_unlocked() if file.exists() else False,
        }


def default_paths():
    """Module-level default paths (used by the CLI when configuring nothing explicitly)."""
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home())
    return home / "vault" / "keychain.keychain-db", home / "vault" / "keychain.pw"


def provision(file: Optional[Path] = None, password_file: Optional[Path] = None,
              force: bool = False) -> Tuple[Path, Path]:
    """Create a passworded keychain file + 0600 password sidecar (unattended mode).

    The keychain password is generated here, written only to the 0600 sidecar, and
    never printed. ``create-keychain`` prompts on the pty (fed from this process —
    the password never appears in argv/env) and runs with cwd = the file's parent so
    securityd's transient ``.fl*`` temp marker does not litter the working directory.
    """
    from hermes_cli.config import _secure_dir
    from utils import atomic_write_bytes

    kc_path = Path(file) if file is not None else default_paths()[0]
    pw_path = Path(password_file) if password_file is not None else default_paths()[1]
    kc_path = kc_path.expanduser().resolve()
    pw_path = pw_path.expanduser().resolve()
    if kc_path.exists() and not force:
        raise VaultError(
            f"keychain already exists at {kc_path} (use --force to re-create; items are lost)")
    kc_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _secure_dir(kc_path.parent)

    password = secrets.token_urlsafe(32)
    env = dict(os.environ)
    env.pop("DISPLAY", None)
    proc = _prompt_security(["create-keychain", str(kc_path)], [password, password],
                            env=env, cwd=str(kc_path.parent))
    if proc.returncode != 0:
        raise VaultError(f"keychain create failed (rc={proc.returncode})")
    atomic_write_bytes(pw_path, (password + "\n").encode("utf-8"), mode=0o600)
    # 24h auto-lock, mirroring the hermes.keychain-db wrapper pattern; best-effort.
    subprocess.run(  # noqa: S603 — argv list, no shell
        [_SEC, "set-keychain-settings", "-ut", "86400", str(kc_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=_SEC_TIMEOUT)
    return kc_path, pw_path