"""macOS Keychain login backend for the browser credential vault (``security`` CLI).

A dedicated, passworded keychain file (default ``<HERMES_HOME>/vault/keychain.keychain-db``)
holds internet-password items. The unlock password lives in a 0600 sidecar
(``<HERMES_HOME>/vault/keychain.pw``) when Hermes may unlock unattended, or is asked for
via the surface's masked prompt when it may not. Handles are ``kc:<server>|<account>``.

Why a dedicated file instead of the login keychain: the login keychain is tied to the
GUI security session and prompts — locked/headless reads fail with rc 36/152 (see the
hermes-secret-management skill). A dedicated file created with ``create-keychain -p``
unlocks with its own password, is headless-safe, profile-scoped under HERMES_HOME, and
never touches the keychain search list (verified live: ``security list-keychains`` is
unchanged after create).

``security`` CLI surface (verified on macOS 27):
- create-keychain -p <pw> <file>          headless rc=0; never adds to the search list
- add-internet-password -a <acct> -s <srvr> [-l label] -w <pw> <file>
                                          headless rc=0 WITHOUT ``-A``; the ``-A`` flag is
                                          INVALID here (rc 48). Duplicate item → rc 45.
- find-internet-password -a -s -g <file>  rc=0 + YAML-ish dump on STDOUT; the ``password: "…"``
                                          line is written to STDERR (verified on macOS 27); no
                                          match → rc 44; locked → rc 152 (empty output). Lock
                                          state cannot be probed with a non-matching item (the
                                          item search never touches the secret partition), so
                                          prompt-mode lock state is tracked per session instead.
- dump-keychain <file>                    metadata ONLY — no passwords — works even while
                                          LOCKED (passwordless enumeration by design);
                                          missing file → rc 0 empty. One block per item,
                                          separated by ``keychain:`` header lines.
- lock-keychain / unlock-keychain -p <pw> <file>; locked secret reads fail rc 152
- delete-internet-password -a -s <file>   rc=0 ("password has been deleted.")
- set-keychain-settings -ut <secs> <file> auto-lock timer (24h default mirrors the
                                          hermes.keychain-db wrapper pattern)

Secrets in argv: ``security`` has no env/stdin channel, so the keychain password and item
passwords travel in subprocess argv — the same convention every wrapper on this install
already uses (hermes.keychain-db scripts). The model never sees them: only the fill path
resolves, and values never enter tool results, logs, or the session DB. All security
invocations are wrapped in a subprocess timeout so a consent prompt can never hang a tool
call (the known failure mode for accidentally prompting operations).
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent.vault_backends.base import LoginBackend, UnlockRequired
from agent.vault_store import VaultItemMeta, normalize_origin

_SEC_TIMEOUT = 30.0
# rc codes that mean "keychain locked / authorization refused" for secret reads:
# 36 = user interaction not allowed (background session), 152 = locked keychain.
_LOCKED_RCS = (36, 152)

_BLOCK_RE = re.compile(r"(?m)^keychain: ")
_ATTR = lambda name: re.compile(rf'(?m)^    "{name}"<blob>="([^"]*)"')
_SRVR_RE = _ATTR("srvr")
_ACCT_RE = _ATTR("acct")
_LABEL_RE = re.compile(r'(?m)^    0x00000007 <blob>="([^"]*)"')
_CDAT_RE = re.compile(r'(?m)^    "cdat"<timedate>=0x[0-9A-F]+  "(\d{14})Z')
_PASSWORD_RE = re.compile(r'(?m)^password: "((?:[^"\\]|\\.)*)"')


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
    """``security`` renders quotes as ``\\"`` inside the dump; undo just those two escapes."""
    return s.replace('\\"', '"').replace("\\\\", "\\")


class MacOSKeychainLoginBackend(LoginBackend):
    """Login items in a dedicated macOS keychain file, read headlessly via ``security``.

    Two unlock modes, decided by whether a password sidecar exists:
    - password sidecar present → ``needs_unlock`` False; locked reads self-heal (unlock
      with the sidecar password and retry once). This is the unattended/headless mode
      provisioned by ``hermes vault keychain init`` (rc 152 → auto-unlock → retry).
    - no sidecar → ``needs_unlock`` True; the surface must prompt for the keychain master
      password (``browser_vault_unlock`` → ``unlock``) for the session, like 1Password.
      Note ``list_items`` still works while locked: macOS exposes item metadata (account,
      server, dates) without unlocking, only the password read is gated.
    """

    name = "keychain"
    display_name = "macOS Keychain"
    prefix = "kc:"

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}
        # Instance-level (not class-level): the mode is a property of the files present.
        self.needs_unlock = not self._pw_file().exists()
        # Prompt-mode lock state. macOS offers no lock probe via a non-matching item (item
        # search never touches the secret partition), so the session tracks it: set by a
        # successful unlock(), cleared when a secret read hits rc 152 (locked again — e.g. the
        # OS auto-lock timer after 24h).
        self._session_unlocked = False

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

    def _sec(self, *args: str, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run ``security`` with a hard timeout. The timeout is the consent-prompt guard:
        a hung auth dialog must surface as an error, never as a wedged tool call."""
        try:
            return subprocess.run(  # noqa: S603 — argv list, no shell
                ["/usr/bin/security", *args], cwd=str(cwd) if cwd else None,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=_SEC_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"keychain operation timed out after {_SEC_TIMEOUT:.0f}s") from exc
        except OSError as exc:
            raise RuntimeError(f"failed to invoke security: {exc}") from exc

    def _auto_unlock(self) -> bool:
        """Unlock with the sidecar password (attended unlocks use :meth:`unlock`)."""
        pw_path = self._pw_file()
        try:
            pw = pw_path.read_text(encoding="utf-8").strip()
        except OSError:
            return False
        if not pw:
            return False
        proc = self._sec("unlock-keychain", "-p", pw, str(self._file()))
        return proc.returncode == 0

    def _run_unlocked(self, fn, *args) -> subprocess.CompletedProcess:
        """Run one ``security`` call; when the keychain is locked (rc 36/152), heal in
        sidecar mode or raise ``UnlockRequired`` for the surface, then retry ONCE."""
        proc = fn(*args)
        if proc.returncode not in _LOCKED_RCS:
            return proc
        self._session_unlocked = False
        if self.needs_unlock or not self._auto_unlock():
            raise UnlockRequired(self)
        self._session_unlocked = True
        return fn(*args)

    # -- LoginBackend contract ------------------------------------------------

    def is_unlocked(self) -> bool:
        if not self._file().exists():
            return False
        if not self.needs_unlock:
            # Sidecar mode is self-healing: locked reads are transparently unlocked.
            return True
        return self._session_unlocked

    def unlock(self, master_password: str) -> None:
        proc = self._sec("unlock-keychain", "-p", master_password, str(self._file()))
        if proc.returncode != 0:
            raise RuntimeError("keychain unlock failed — wrong master password?")
        self._session_unlocked = True

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
                continue  # no server -> no origin to bind; not a fill target
            account = _value(_ACCT_RE.search(block))
            label = _value(_LABEL_RE.search(block)) or server
            origins = (f"https://{server}", f"http://{server}")
            allowed = tuple(o for o in origins if _origin_ok(o))
            if not allowed:
                continue
            dot_https = origins[0] if _origin_ok(origins[0]) else origins[1]
            out.append(VaultItemMeta(
                id=f"{self.prefix}{server}|{account}",
                kind="login",
                label=label,
                origin=dot_https,
                created_at=_created_at(_value(_CDAT_RE.search(block))),
                identifier_type=_identifier_type(account) if account else None,
                identifier=account or None,
                allowed_origins=allowed))
        return out

    def get_meta(self, handle: str) -> Optional[VaultItemMeta]:
        # The handle IS the item identity (server|account) — metadata is derivable with
        # no subprocess. Existence is only re-checked by resolve_password at fill time.
        parsed = self._parse_handle(handle)
        if parsed is None:
            return None
        server, account = parsed
        label = server
        origins = (f"https://{server}", f"http://{server}")
        allowed = tuple(o for o in origins if _origin_ok(o))
        if not allowed:
            return None
        return VaultItemMeta(
            id=handle, kind="login", label=label, origin=allowed[0], created_at="",
            identifier_type=_identifier_type(account) if account else None,
            identifier=account or None, allowed_origins=allowed)

    def resolve_password(self, handle: str) -> str:
        parsed = self._parse_handle(handle)
        if parsed is None:
            from agent.vault_store import VaultError
            raise VaultError(f"malformed keychain handle {handle!r}")
        server, account = parsed
        file = self._file()
        proc = self._run_unlocked(self._sec, "find-internet-password", "-a", account,
                                  "-s", server, "-g", str(file))
        if proc.returncode == 44:
            from agent.vault_store import VaultError
            raise VaultError("keychain item no longer exists")
        if proc.returncode != 0:
            raise RuntimeError(f"keychain lookup failed (rc={proc.returncode})")
        # The dump goes to stdout; `security` writes the password: line to STDERR (verified).
        match = _PASSWORD_RE.search(proc.stdout + proc.stderr)
        return _unescape_dump(match.group(1)) if match else ""

    # -- write path (hermes vault keychain add/rm) -----------------------------

    def add_item(self, server: str, account: str, password: str,
                 label: Optional[str] = None) -> str:
        """Insert or replace the (server, account) item. Returns the fill handle."""
        from agent.vault_store import VaultError

        server = (server or "").strip().lower()
        account = (account or "").strip()
        if not server or not account:
            raise VaultError("keychain item needs a server and an account")
        file = self._file()
        if not file.exists():
            raise VaultError("no keychain file yet — run `hermes vault keychain init` first")
        existing = self._run_unlocked(self._sec, "find-internet-password", "-a", account,
                                      "-s", server, "-g", str(file))
        argv = ["add-internet-password", "-a", account, "-s", server]
        if label:
            argv += ["-l", label]
        argv += ["-w", password, str(file)]  # security has no env/stdin channel (see module doc)
        if existing.returncode == 0:
            self._run_unlocked(self._sec, "delete-internet-password", "-a", account,
                               "-s", server, str(file))
        proc = self._run_unlocked(self._sec, *argv)
        if proc.returncode != 0:
            raise VaultError(f"keychain add failed (rc={proc.returncode})")
        return f"{self.prefix}{server}|{account}"

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


def _value(match: Optional[re.Match]) -> str:
    return _unescape_dump(match.group(1)) if match else ""


def _origin_ok(origin: str) -> bool:
    try:
        normalize_origin(origin)
        return True
    except Exception:
        return False


def default_paths():
    """Module-level default paths (used by the CLI when configuring nothing explicitly)."""
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home())
    return home / "vault" / "keychain.keychain-db", home / "vault" / "keychain.pw"


def provision(file: Optional[Path] = None, password_file: Optional[Path] = None,
              force: bool = False) -> Tuple[Path, Path]:
    """Create a passworded keychain file + 0600 password sidecar (unattended mode).

    The keychain password is generated here, written only to the 0600 sidecar, and
    never printed. ``create-keychain`` runs with cwd = the file's parent so securityd's
    transient ``.fl*`` temp marker does not litter the process working directory.
    """
    import secrets

    from hermes_cli.config import _secure_dir
    from utils import atomic_write_bytes
    from agent.vault_store import VaultError

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
    proc = subprocess.run(  # noqa: S603 — argv list, no shell
        ["/usr/bin/security", "create-keychain", "-p", password, str(kc_path)],
        cwd=str(kc_path.parent), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=_SEC_TIMEOUT)
    if proc.returncode != 0:
        raise VaultError(f"keychain create failed (rc={proc.returncode})")
    atomic_write_bytes(pw_path, (password + "\n").encode("utf-8"), mode=0o600)
    # 24h auto-lock, mirroring the hermes.keychain-db wrapper pattern; best-effort.
    subprocess.run(  # noqa: S603 — argv list, no shell
        ["/usr/bin/security", "set-keychain-settings", "-ut", "86400", str(kc_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=_SEC_TIMEOUT)
    return kc_path, pw_path