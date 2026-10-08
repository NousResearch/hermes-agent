"""Opt-in exact-host discovery with trusted in-process metadata projection.

dcli 6.2640.1 decrypts all logins internally, then outputs complete records
matching its URL substring filter. Never expose that stdout outside this backend.
Only browser_vault_fill consumes a selected password via an exact UUID read.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import secrets
import signal
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from agent.vault_backends.base import LoginBackend, UnlockRequired
from agent.vault_store import VaultError, VaultItemMeta, normalize_origin
from hermes_platform.resolver import locate_command
from hermes_constants import get_hermes_home

_TIMEOUT = 15.0
_MAX_OUTPUT = 256 * 1024
_MAX_MATCHES = 20
_HANDLE_TTL = 300.0
# Nonsecret, process-local, profile/config-bound selection metadata only.
_DISCOVERED = {}
_DISCOVERED_LOCK = threading.Lock()
# dcli's isUuid is CASE SENSITIVE and accepts only these version digits.
_UUID = re.compile(r"[0-9A-F]{8}-[0-9A-F]{4}-[0-5][0-9A-F]{3}-[0-9A-F]{4}-[0-9A-F]{12}")
_ENV_KEEP = ("PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "SystemRoot",
             "SYSTEMROOT", "TMPDIR", "TMP", "TEMP")


def _text(value, *, limit=1024):
    return isinstance(value, str) and 0 < len(value) <= limit and not any(ord(c) < 32 or ord(c) == 127 for c in value)


def _origin(value):
    try:
        if not _text(value, limit=4096):
            raise ValueError
        parsed = urlsplit(value)
        if (parsed.scheme not in ("https", "http") or not parsed.hostname
                or "@" in parsed.netloc or "\\" in value
                or any(c.isspace() for c in value)
                or not re.fullmatch(r"[A-Za-z0-9.-]+", parsed.hostname)):
            # Require ordinary ASCII DNS/IP host spelling. WHATWG browser URL
            # parsing differs from urllib for backslashes, escaped hosts, etc.
            raise ValueError
        return normalize_origin(value)
    except (ValueError, VaultError):
        raise VaultError("Dashlane login requires a valid HTTP(S) origin") from None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _host(value):
    if (not isinstance(value, str) or len(value) > 253 or value != value.lower()
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", value)
            or "." not in value
            or any(not part or len(part) > 63 or part.startswith("-") or part.endswith("-")
                   for part in value.split("."))):
        raise VaultError("Dashlane search requires a lowercase exact ASCII host, not a URL or wildcard")
    return value


_LISTING_REASONS = {
    "configuration": {"invalid_configuration"},
    "status_before": {"status_unavailable", "account_or_lock_state"},
    "search_process": {"process_failed"},
    "search_json": {"invalid_json"},
    "search_projection": {"invalid_response", "record_limit", "invalid_record",
                          "invalid_identity", "duplicate_identity", "invalid_origin",
                          "invalid_metadata", "selection_capacity"},
    "status_after": {"status_unavailable", "account_or_lock_state"},
}


class DashlaneListingError(VaultError):
    """Only fixed codes cross the tool boundary; never vendor text or fields."""

    def __init__(self, stage, reason):
        if stage not in _LISTING_REASONS or reason not in _LISTING_REASONS[stage]:
            raise ValueError("Invalid Dashlane diagnostic code")
        self.stage, self.reason = stage, reason
        Exception.__init__(self, "Dashlane listing failed: " + stage + "/" + reason)


class _ListingLocked(DashlaneListingError, UnlockRequired):
    def __init__(self, backend, stage):
        self.backend = backend
        DashlaneListingError.__init__(self, stage, "account_or_lock_state")


def listing_diagnostic(exc):
    # Revalidate attributes rather than serializing arbitrary exception data.
    if (isinstance(exc, DashlaneListingError) and type(exc.stage) is str
            and type(exc.reason) is str and exc.stage in _LISTING_REASONS
            and exc.reason in _LISTING_REASONS[exc.stage]):
        return {"stage": exc.stage, "reason": exc.reason}
    return {}


class DashlaneLoginBackend(LoginBackend):
    name = "dashlane"
    display_name = "Dashlane"
    prefix = "dl:"
    needs_unlock = True
    manual_unlock = True
    automatic_otp = False
    setup_hint = "Enroll/unlock with dcli sync in your own terminal; lock with dcli lock. Configure vault.dashlane account and search_hosts (or items) first."

    def __init__(self, cfg=None):
        self.cfg = {} if cfg is None else cfg
        self.unfillable_candidates = []
        self.listing_errors = []

    def _binary(self):
        explicit = self.cfg.get("binary_path", "")
        if not isinstance(explicit, str) or (explicit and not Path(explicit).is_absolute()):
            raise VaultError("Dashlane binary_path must be absolute")
        resolution = locate_command(explicit or "dcli")
        path = next((c.value for c in resolution.candidates if c.present), None)
        if path is None:
            raise VaultError("Dashlane CLI is unavailable")
        return path

    def _run(self, *args):
        # No inherited DASHLANE_* credentials, NODE_OPTIONS, debug flags, stdin,
        # clipboard, shell, stderr, or disk spool. A bounded reader works on Windows too.
        env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
        env["NO_COLOR"] = "1"
        output = bytearray()
        failed = threading.Event()
        stopped = threading.Event()
        proc = None
        reader = None
        deadline = time.monotonic() + _TIMEOUT
        try:
            proc = subprocess.Popen([self._binary(), *args], env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
                                    start_new_session=(os.name == "posix"))

            assert proc.stdout is not None
            stream = proc.stdout

            def read():
                try:
                    while not stopped.is_set() and (chunk := stream.read(4096)):
                        if len(output) + len(chunk) > _MAX_OUTPUT:
                            failed.set()
                            return
                        output.extend(chunk)
                except (OSError, ValueError):
                    failed.set()

            reader = threading.Thread(target=read, daemon=True)
            reader.start()
            reader.join(max(0, deadline - time.monotonic()))
            if reader.is_alive() or failed.is_set():
                raise VaultError("Dashlane CLI exceeded its time or output limit")
            if proc.wait(timeout=max(0, deadline - time.monotonic())) != 0:
                raise VaultError("Dashlane CLI failed; check enrollment and lock state in your terminal")
            return output.decode("utf-8", errors="strict")
        except (OSError, UnicodeError, subprocess.SubprocessError):
            raise VaultError("Dashlane CLI unavailable or invalid response") from None
        finally:
            stopped.set()
            if proc is not None:
                # A descendant can retain stdout after the CLI exits. Kill the
                # isolated POSIX group, not just its leader; never wait forever
                # or close a BufferedReader while another thread holds its lock.
                try:
                    if os.name == "posix":
                        os.killpg(proc.pid, signal.SIGKILL)
                    elif proc.poll() is None:
                        proc.kill()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                if proc.stdout is not None:
                    proc.stdout.close()  # unbuffered FileIO: no reader lock
                if reader is not None:
                    reader.join(1)
            output.clear()

    def _enrolled(self):
        if (not isinstance(self.cfg, dict)
                or set(self.cfg) - {"enabled", "binary_path", "account", "items", "search_hosts"}
                or self.cfg.get("enabled") is not True):
            raise VaultError("Dashlane requires explicit valid opt-in configuration")
        binary = self.cfg.get("binary_path", "")
        if not isinstance(binary, str) or (binary and (not _text(binary, limit=4096) or not Path(binary).is_absolute())):
            raise VaultError("Dashlane binary_path must be absolute")
        account = self.cfg.get("account")
        hosts = self.cfg.get("search_hosts", [])
        if not isinstance(hosts, list) or len(hosts) > 5:
            raise VaultError("Dashlane search_hosts must contain at most five exact hosts")
        for host in hosts:
            _host(host)
        if len(set(hosts)) != len(hosts):
            raise VaultError("Duplicate Dashlane search host")
        entries = self.cfg.get("items", [])
        if not _text(account) or not isinstance(entries, list) or len(entries) > 100:
            raise VaultError("Configure a single Dashlane account and at most 100 enrolled items")
        metas = {}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) - {"id", "label", "origin", "identifier", "identifier_type", "source_host"}:
                raise VaultError("Invalid Dashlane enrollment metadata")
            item_id = entry.get("id")
            identifier = entry.get("identifier")
            field = entry.get("identifier_type", "email")
            label = entry.get("label")
            if (not isinstance(item_id, str) or not _UUID.fullmatch(item_id) or item_id in metas
                    or not _text(identifier) or not _text(label) or field not in ("email", "username")):
                raise VaultError("Invalid or ambiguous Dashlane enrollment metadata")
            if "source_host" in entry:
                # Explicit enrollment of an exact bare-host source, never URL repair.
                _host(entry["source_host"])
            origin = _origin(entry.get("origin"))
            if origin != entry["origin"]:
                raise VaultError("Dashlane enrollment must name an exact normalized origin")
            metas[item_id] = VaultItemMeta(id=self.prefix + item_id, kind="login", label=str(label),
                                          origin=origin, created_at="", identifier=identifier,
                                          identifier_type=field, allowed_origins=(origin,))
        return account, metas

    def _require_unlocked(self):
        account, _ = self._enrolled()
        # Strict whole-response match rejects missing, duplicated or ambiguous status.
        status = self._run("status").splitlines()
        if status != ["Logged in: Yes", f"Login: {account}", "Locked: No"]:
            raise UnlockRequired(self)

    def is_unlocked(self):
        try:
            self._require_unlocked()
            return True
        except (VaultError, UnlockRequired):
            return False

    def _listing_configuration(self, host=None):
        try:
            enrolled = self._enrolled()
            if host is not None:
                _host(host)
                if host not in self.cfg.get("search_hosts", []):
                    raise VaultError("Host not enabled")
            return enrolled
        except Exception:
            raise DashlaneListingError("configuration", "invalid_configuration") from None

    def _listing_status(self, stage):
        try:
            self._require_unlocked()
        except UnlockRequired:
            raise _ListingLocked(self, stage) from None
        except Exception:
            raise DashlaneListingError(stage, "status_unavailable") from None

    def list_items(self):
        self.unfillable_candidates = []
        self.listing_errors = []
        _, metas = self._listing_configuration()
        self._listing_status("status_before")
        found = list(metas.values())
        for host in self.cfg.get("search_hosts", []):
            try:
                found.extend(self.search_items(host))
            except DashlaneListingError as exc:
                # Isolate malformed host results, but never suppress account drift.
                if exc.stage != "search_projection":
                    raise
                self.listing_errors.append(listing_diagnostic(exc))
        if self.cfg.get("search_hosts"):
            self._listing_status("status_after")
        return found

    def _scope(self):
        self._enrolled()
        return (str(get_hermes_home().resolve()),
                hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest())

    def search_items(self, host):
        """Exact-host, bounded metadata API. Config is the explicit trust opt-in.

        Multiple results are choices, never an automatic first-match selection.
        No raw records or secrets are retained in the selection cache.
        """
        self._listing_configuration(host)
        self._listing_status("status_before")
        try:
            raw = self._run("password", "--output", "json", "url=" + host)
        except Exception:
            raise DashlaneListingError("search_process", "process_failed") from None
        try:
            records = json.loads(raw, object_pairs_hook=_unique_object)
        except (ValueError, TypeError, RecursionError):
            raise DashlaneListingError("search_json", "invalid_json") from None
        finally:
            raw = None
        self._listing_status("status_after")
        if not isinstance(records, list):
            raise DashlaneListingError("search_projection", "invalid_response")
        if len(records) > _MAX_MATCHES:
            raise DashlaneListingError("search_projection", "record_limit")
        projected, candidates, seen = [], [], set()
        for record in records:
            if not isinstance(record, dict):
                raise DashlaneListingError("search_projection", "invalid_record")
            try:
                origin = _origin(record.get("url"))
            except VaultError:
                # Recognize only an exact bare hostname as non-authorizing metadata.
                # No scheme is supplied or inferred, and no fill handle is minted.
                try:
                    saved_host = _host(record.get("url"))
                except VaultError:
                    raise DashlaneListingError("search_projection", "invalid_origin") from None
                if saved_host != host:
                    continue
                origin = None
            if origin is not None and urlsplit(origin).hostname != host:
                continue
            raw_id = record.get("id")
            if (not isinstance(raw_id, str) or len(raw_id) != 38
                    or raw_id[0] != "{" or raw_id[-1] != "}"
                    or not _UUID.fullmatch(raw_id[1:-1])):
                raise DashlaneListingError("search_projection", "invalid_identity")
            if raw_id in seen:
                raise DashlaneListingError("search_projection", "duplicate_identity")
            seen.add(raw_id)
            field = "email" if record.get("email") else "login"
            identifier, label = record.get(field), record.get("title") or host
            if not _text(identifier) or not _text(label):
                raise DashlaneListingError("search_projection", "invalid_metadata")
            if origin is None:
                candidates.append({"backend": self.name, "source_id": raw_id[1:-1],
                                   "website_host": host, "identifier": identifier,
                                   "identifier_type": "email" if field == "email" else "username",
                                   "available": False, "fillable": False,
                                   "stage": "search_projection", "reason": "invalid_origin",
                                   "status": "explicit_origin_binding_required"})
                continue
            handle = self.prefix + "search-" + secrets.token_hex(24)
            meta = VaultItemMeta(id=handle, kind="login", label=label, origin=origin,
                                 created_at="", identifier=identifier,
                                 identifier_type="email" if field == "email" else "username",
                                 allowed_origins=(origin,))
            projected.append((raw_id[1:-1], meta))
        # All validation succeeds before any handles become usable.
        scope, now = self._scope(), time.monotonic()
        with _DISCOVERED_LOCK:
            for key, (old_scope, expires, _, _) in list(_DISCOVERED.items()):
                if expires <= now:
                    del _DISCOVERED[key]
            if len(_DISCOVERED) + len(projected) > 1000:
                raise DashlaneListingError("search_projection", "selection_capacity")
            for item_id, meta in projected:
                _DISCOVERED[meta.id] = (scope, now + _HANDLE_TTL, item_id, meta)
        self.unfillable_candidates = (self.unfillable_candidates + candidates)[-100:]
        return [meta for _, meta in projected]

    def _selection(self, handle):
        scope = self._scope()
        with _DISCOVERED_LOCK:
            entry = _DISCOVERED.get(handle)
            if entry and entry[0] == scope and entry[1] > time.monotonic():
                return entry[2], entry[3]
        return None

    def get_meta(self, handle):
        _, metas = self._enrolled()
        if not isinstance(handle, str) or not handle.startswith(self.prefix):
            return None
        if handle.startswith(self.prefix + "search-"):
            selected = self._selection(handle)
            return selected[1] if selected else None
        return metas.get(handle[len(self.prefix):])

    def resolve_password(self, handle):
        scope = self._scope()
        meta = self.get_meta(handle)
        if meta is None:
            raise VaultError("Unknown Dashlane login handle")
        self._require_unlocked()
        if handle.startswith(self.prefix + "search-"):
            selected = self._selection(handle)
            if selected is None:
                raise VaultError("Dashlane selection expired; list again")
            item_id = selected[0]
        else:
            item_id = handle[len(self.prefix):]
        try:
            item = json.loads(self._run("read", "dl://" + item_id), object_pairs_hook=_unique_object)
        except (ValueError, RecursionError):
            raise VaultError("Invalid Dashlane item response") from None
        self._require_unlocked()
        # Do not trust metadata from a stale list, substring match, title lookup,
        # account switch, or edited vault entry. One record supplies all bindings.
        if not isinstance(item, dict) or item.get("id") != "{" + item_id + "}":
            raise VaultError("Dashlane item identity changed")
        field = "email" if meta.identifier_type == "email" else "login"
        source_host = next((entry.get("source_host") for entry in self.cfg.get("items", [])
                            if entry["id"] == item_id), None) if not handle.startswith(self.prefix + "search-") else None
        source_matches = (item.get("url") == source_host if source_host is not None
                          else _origin(item.get("url")) == meta.origin)
        if item.get(field) != meta.identifier or not source_matches or self._scope() != scope:
            raise VaultError("Dashlane login account or origin changed; review enrollment")
        password = item.get("password")
        if not isinstance(password, str) or not password:
            raise VaultError("Dashlane login has no password")
        # The selected capability must still be live after bounded CLI reads.
        # Expiry or a scope change during those reads cannot authorize a fill.
        if handle.startswith(self.prefix + "search-"):
            current = self._selection(handle)
            if current is None or current != (item_id, meta):
                raise VaultError("Dashlane selection expired or changed; list again")
        return password
