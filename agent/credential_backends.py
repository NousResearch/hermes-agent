"""Platform backends for the credential store: Bitwarden Secrets Manager and 1Password.

`/creds set` writes a named secret to a platform when that integration is set up
and reachable, and keeps only an *index* entry (backend + locator) in the local
store — the value itself lives in the platform alone. When no platform is
available the local Fernet store is used exactly as before, so `/creds` never
stops working. Reads route back through the same backend
(``CredentialStore.get`` → :func:`read_ref`), so ``/creds show``, ``/creds
inject`` and ``execute_code(credentials=)`` behave identically for
platform-backed entries.

Vendor CLI syntax this module relies on (verified against vendor docs):

  bws secret create <KEY> <VALUE> <PROJECT_ID> [--note N] --output json
  bws secret get <ID> --output json | edit <ID> --value V | delete <ID>
  op item create --category login --title T --vault V password=VALUE --format json
  op item edit <ID> password=VALUE --vault V   |   op item delete <ID> --vault V
  op read op://<vault>/<item-id>/password

Hygiene: tokens ride the child's environment only (``run_secret_cli`` builds a
minimal allowlisted env — never argv, never our process). Values *are* CLI
arguments: neither vendor CLI offers a stdin form for them, so the child is kept
short-lived and callers register the value with the redactor before any output
path can echo it.
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LOCAL = "local"
BACKEND_NAMES = ("bitwarden", "onepassword")
_CLI_TIMEOUT_SECONDS = 60.0
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class BackendError(RuntimeError):
    """A platform backend could not complete an operation.

    Messages are user-facing and must never contain a secret value.
    """


class AmbiguousBackendError(BackendError):
    """Several platforms are available under ``auto`` — the caller must choose."""

    def __init__(self, names: list[str]):
        self.names = list(names)
        super().__init__("more than one secret platform is available: " + ", ".join(self.names))


@dataclass(frozen=True)
class BackendStatus:
    name: str
    display_name: str
    available: bool
    reason: str = ""


def _secrets_cfg() -> dict:
    """``secrets:`` from config.yaml — the section `hermes secrets` writes."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly().get("secrets")
    except Exception as exc:  # config unreadable must not break /creds
        logger.debug("could not read secrets config: %s", exc, exc_info=True)
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _section(name: str) -> dict:
    section = _secrets_cfg().get(name)
    return section if isinstance(section, dict) else {}


def _token_value(env_name: str) -> str:
    """The token behind *env_name*: process env first (env_loader already ran), else .env."""
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    try:
        from hermes_cli.config import load_env

        return str(load_env().get(env_name, "") or "").strip()
    except Exception as exc:
        logger.debug("could not read .env for %s: %s", env_name, exc, exc_info=True)
        return ""


def _run_json(argv: list[str], *, extra_env: dict[str, str], allow_env: tuple[str, ...],
              label: str) -> dict:
    """Run a vendor CLI that answers JSON; non-zero exit or bad JSON → BackendError."""
    from agent.secret_sources.base import run_secret_cli

    proc = run_secret_cli(argv, allow_env=allow_env, extra_env=extra_env,
                          timeout=_CLI_TIMEOUT_SECONDS)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:300]
        raise BackendError(f"{label} failed (exit {proc.returncode}): {detail or 'no output'}")
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise BackendError(f"{label} returned non-JSON output: {exc}") from exc
    if not isinstance(payload, dict):
        raise BackendError(f"{label} returned unexpected shape: {type(payload).__name__}")
    return payload


def _run_ok(argv: list[str], *, extra_env: dict[str, str], allow_env: tuple[str, ...],
            label: str) -> None:
    """Run a vendor CLI that only needs success/failure → BackendError on non-zero exit."""
    from agent.secret_sources.base import run_secret_cli

    proc = run_secret_cli(argv, allow_env=allow_env, extra_env=extra_env,
                          timeout=_CLI_TIMEOUT_SECONDS)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:300]
        raise BackendError(f"{label} failed (exit {proc.returncode}): {detail or 'no output'}")


class CredentialBackend(ABC):
    """One secret platform: availability probe plus create/read/update/delete."""

    name: str            # config + CLI key: bitwarden | onepassword
    display_name: str    # user-facing

    @abstractmethod
    def status(self) -> BackendStatus:
        """Is this platform usable right now, and if not, why not?"""

    @abstractmethod
    def create(self, name: str, value: str, *, note: str = "") -> str:
        """Store *value* under *name*; return the backend-local locator to record."""

    @abstractmethod
    def read(self, ref: str) -> str:
        """Plaintext value for a stored locator."""

    @abstractmethod
    def update(self, ref: str, value: str) -> None:
        """Replace the value behind an existing locator."""

    @abstractmethod
    def delete(self, ref: str) -> None:
        """Remove the secret behind a locator."""


class BitwardenBackend(CredentialBackend):
    """Bitwarden Secrets Manager (``bws``) — the integration `hermes secrets bitwarden` sets up."""

    name = "bitwarden"
    display_name = "Bitwarden Secrets Manager"

    def _cfg(self) -> dict:
        return _section("bitwarden")

    def _token_env(self) -> str:
        return str(self._cfg().get("access_token_env") or "BWS_ACCESS_TOKEN")

    def _project_id(self) -> str:
        return str(self._cfg().get("project_id") or "").strip()

    def _binary(self, *, install: bool):
        from agent.secret_sources.bitwarden import find_bws

        return find_bws(install_if_missing=install)

    def status(self) -> BackendStatus:
        cfg = self._cfg()
        if not cfg.get("enabled"):
            return BackendStatus(self.name, self.display_name, False,
                                 "disabled — run `hermes secrets bitwarden setup`")
        if not self._project_id():
            return BackendStatus(self.name, self.display_name, False,
                                 "secrets.bitwarden.project_id is empty — re-run setup")
        if not _token_value(self._token_env()):
            return BackendStatus(self.name, self.display_name, False,
                                 f"{self._token_env()} is not set — run `hermes secrets bitwarden token`")
        if self._binary(install=False) is None:
            return BackendStatus(self.name, self.display_name, False,
                                 "bws binary not found — run `hermes secrets bitwarden setup`")
        return BackendStatus(self.name, self.display_name, True, f"project {self._project_id()}")

    def _require_ready(self) -> tuple[str, dict[str, str], tuple[str, ...]]:
        """(bws path, extra child env, allow_env) or BackendError carrying the status reason."""
        status = self.status()
        if not status.available:
            raise BackendError(f"{self.display_name} is unavailable: {status.reason}")
        bws = self._binary(install=bool(self._cfg().get("auto_install", True)))
        if bws is None:
            raise BackendError(f"{self.display_name} is unavailable: bws binary not found")
        extra = {"BWS_ACCESS_TOKEN": _token_value(self._token_env())}
        server = str(self._cfg().get("server_url") or os.environ.get("BWS_SERVER_URL", "") or "").strip()
        allow: tuple[str, ...] = ("BWS_SERVER_URL",)
        if server:
            extra["BWS_SERVER_URL"] = server
        return str(bws), extra, allow

    def create(self, name: str, value: str, *, note: str = "") -> str:
        from agent.secret_sources.base import is_valid_env_name

        # `_run_bws_list` (and therefore every read path) only surfaces keys that
        # are legal env-var names, so a hyphenated name would save but never read.
        if not is_valid_env_name(name):
            raise BackendError(
                "Bitwarden secret keys are environment-variable names — use letters, digits and "
                "underscores only (no hyphens), or store this one with --backend local."
            )
        bws, extra, allow = self._require_ready()
        argv = [bws, "secret", "create", name]
        if note:
            argv += ["--note", note]
        argv += ["--output", "json"]
        if value.startswith("-"):
            # clap would read a leading '-' value as a flag; `--` ends flag parsing
            # so VALUE and PROJECT_ID stay positional.
            argv.append("--")
        argv += [value, self._project_id()]
        payload = _run_json(argv, extra_env=extra, allow_env=allow, label="bws secret create")
        ref = str(payload.get("id") or "")
        if not ref:
            raise BackendError("bws secret create returned no id")
        return ref

    def read(self, ref: str) -> str:
        bws, extra, allow = self._require_ready()
        payload = _run_json([bws, "secret", "get", ref, "--output", "json"],
                            extra_env=extra, allow_env=allow, label="bws secret get")
        value = payload.get("value")
        if not isinstance(value, str) or value == "":
            raise BackendError("bws secret get returned no value")
        return value

    def update(self, ref: str, value: str) -> None:
        bws, extra, allow = self._require_ready()
        _run_json([bws, "secret", "edit", ref, "--value", value, "--output", "json"],
                  extra_env=extra, allow_env=allow, label="bws secret edit")

    def delete(self, ref: str) -> None:
        bws, extra, allow = self._require_ready()
        _run_ok([bws, "secret", "delete", ref], extra_env=extra, allow_env=allow,
                label="bws secret delete")


class OnePasswordBackend(CredentialBackend):
    """1Password (``op``) — the integration `hermes secrets onepassword` sets up."""

    name = "onepassword"
    display_name = "1Password"

    def _cfg(self) -> dict:
        return _section("onepassword")

    def _token_env(self) -> str:
        return str(self._cfg().get("service_account_token_env") or "OP_SERVICE_ACCOUNT_TOKEN")

    def _vault(self) -> str:
        """Configured vault, else the vault named by the first ``op://`` mapping.

        Hermes's 1Password setup maps env vars to references, so an existing
        ``op://<vault>/...`` line already names the vault these items belong in.
        """
        vault = str(self._cfg().get("vault") or "").strip()
        if vault:
            return vault
        env_map = self._cfg().get("env")
        if isinstance(env_map, dict):
            for ref in env_map.values():
                parts = str(ref).split("/")
                if len(parts) >= 4 and parts[0] == "op:" and parts[2].strip():
                    return parts[2].strip()
        return ""

    def _binary(self):
        from agent.secret_sources.onepassword import find_op

        return find_op(str(self._cfg().get("binary_path") or ""))

    def status(self) -> BackendStatus:
        if not self._cfg().get("enabled"):
            return BackendStatus(self.name, self.display_name, False,
                                 "disabled — run `hermes secrets onepassword setup`")
        vault = self._vault()
        if not vault:
            return BackendStatus(self.name, self.display_name, False,
                                 "no vault: set secrets.onepassword.vault (or add an op:// mapping)")
        if self._binary() is None:
            return BackendStatus(self.name, self.display_name, False,
                                 "op CLI not found — install the 1Password CLI or set "
                                 "secrets.onepassword.binary_path")
        token_note = "" if _token_value(self._token_env()) else \
            f" (no {self._token_env()} in env — an interactive `op signin` session is required)"
        return BackendStatus(self.name, self.display_name, True, f"vault {vault}{token_note}")

    def _require_ready(self) -> tuple[str, str, list[str], dict[str, str], tuple[str, ...]]:
        """(op path, vault, base argv, extra child env, allow_env) or BackendError."""
        status = self.status()
        if not status.available:
            raise BackendError(f"{self.display_name} is unavailable: {status.reason}")
        op = self._binary()
        if op is None:
            raise BackendError(f"{self.display_name} is unavailable: op CLI not found")
        cfg = self._cfg()
        argv = [str(op)]
        account = str(cfg.get("account") or "").strip()
        if account:
            argv += ["--account", account]
        token_env = self._token_env()
        token = _token_value(token_env)
        extra = {token_env: token} if token else {}
        return str(op), self._vault(), argv, extra, (token_env,)

    @staticmethod
    def _item_id(ref: str) -> str:
        parts = ref.split("/")
        if len(parts) < 5 or parts[0] != "op:":
            raise BackendError(f"not a 1Password reference: {ref}")
        return parts[3]

    def create(self, name: str, value: str, *, note: str = "") -> str:
        _op, vault, base, extra, allow = self._require_ready()
        argv = base + ["item", "create", "--category", "login", "--title", name,
                       "--vault", vault, "--format", "json", f"password={value}"]
        payload = _run_json(argv, extra_env=extra, allow_env=allow, label="op item create")
        item_id = str(payload.get("id") or "")
        if not item_id:
            raise BackendError("op item create returned no item id")
        return f"op://{vault}/{item_id}/password"

    def read(self, ref: str) -> str:
        from agent.secret_sources.onepassword import fetch_onepassword_secrets

        cfg = self._cfg()
        # use_cache=False: a credential updated a second ago must not read stale
        # for the cache TTL (one `op read` per read is the honest cost).
        secrets, warnings = fetch_onepassword_secrets(
            references={"value": ref}, account=str(cfg.get("account") or "").strip(),
            token_env=self._token_env(), binary_path=str(cfg.get("binary_path") or ""),
            use_cache=False)
        if "value" in secrets:
            return secrets["value"]
        raise BackendError("; ".join(warnings) or f"1Password returned no value for {ref}")

    def update(self, ref: str, value: str) -> None:
        _op, vault, base, extra, allow = self._require_ready()
        # Service accounts must name the vault on item edits.
        argv = base + ["item", "edit", self._item_id(ref), "--vault", vault,
                       f"password={value}"]
        _run_ok(argv, extra_env=extra, allow_env=allow, label="op item edit")

    def delete(self, ref: str) -> None:
        _op, vault, base, extra, allow = self._require_ready()
        argv = base + ["item", "delete", self._item_id(ref), "--vault", vault]
        _run_ok(argv, extra_env=extra, allow_env=allow, label="op item delete")


def backend_classes() -> tuple:
    """Backend classes in ``auto``-prompt order. Tests swap this out."""
    return (BitwardenBackend, OnePasswordBackend)


def all_backends() -> list[CredentialBackend]:
    return [cls() for cls in backend_classes()]


def statuses() -> list[BackendStatus]:
    return [backend.status() for backend in all_backends()]


def available_backends() -> list[CredentialBackend]:
    return [backend for backend in all_backends() if backend.status().available]


def get_backend(name: str) -> CredentialBackend:
    for backend in all_backends():
        if backend.name == name:
            return backend
    raise BackendError(f"unknown credential backend {name!r} "
                       f"(choose from: {', '.join(BACKEND_NAMES)}, {LOCAL})")


def read_ref(backend_name: str, ref: str) -> str:
    """Resolve a platform-backed credential from its index entry."""
    backend = get_backend(backend_name)
    status = backend.status()
    if not status.available:
        raise BackendError(f"{backend.display_name} is unavailable: {status.reason}")
    return backend.read(ref)


def _resolve_target(requested: str) -> str:
    """Turn ``--backend`` (or ``auto``) into the backend name to write to."""
    if requested in (None, "", "auto"):
        available = available_backends()
        if len(available) > 1:
            raise AmbiguousBackendError([backend.name for backend in available])
        return available[0].name if available else LOCAL
    if requested == LOCAL:
        return LOCAL
    backend = get_backend(requested)
    status = backend.status()
    if not status.available:
        raise BackendError(f"{backend.display_name} is unavailable: {status.reason}")
    return backend.name


def _write_remote(backend: CredentialBackend, store, name: str, value: str, *,
                  overwrite: bool, note: str) -> tuple[str, list[str]]:
    """Create or update the platform secret; returns (locator, warnings)."""
    existing = store.entry(name)
    warnings: list[str] = []
    if existing and existing.get("backend") == backend.name and existing.get("ref"):
        backend.update(existing["ref"], value)
        return existing["ref"], warnings
    ref = backend.create(name, value, note=note)
    if existing:
        warnings.extend(_drop_previous(store, existing, keep=backend.name))
    return ref, warnings


def _drop_previous(store, entry: dict, *, keep: str) -> list[str]:
    """Remove the credential's previous home once the new copy exists (never before)."""
    previous = entry.get("backend", LOCAL)
    if previous == LOCAL or previous == keep:
        return []  # a local value disappears when the index entry is replaced
    try:
        get_backend(previous).delete(entry["ref"])
    except BackendError as exc:
        logger.warning("could not delete old %s copy: %s", previous, exc, exc_info=True)
        return [f"the old copy in {previous} could not be deleted: {exc}"]
    return []


def save_credential(name: str, value: str, *, backend: str = "auto",
                    overwrite: bool = False, note: str = "", store=None) -> dict:
    """Write a credential to a platform when available, else to the local store.

    Returns ``{"success", "backend", "warnings", ...}``. Raises
    :class:`AmbiguousBackendError` when ``auto`` matches several platforms (the
    caller prompts) and :class:`BackendError` when an explicitly requested
    platform cannot be used — only an ``auto`` write falls back to local.
    """
    from agent.credential_store import get_store

    store = store if store is not None else get_store()
    existing = store.entry(name)
    if existing and not overwrite:
        return {"success": False,
                "error": f"Credential '{name}' already exists. Update it with "
                         f"`/creds set {name} <value> --force`."}

    target = _resolve_target(backend)
    if target == LOCAL:
        result = store.set(name, value, overwrite=overwrite)
        if result.get("success"):
            result.update({"backend": LOCAL, "warnings": []})
        return result

    platform = get_backend(target)
    try:
        ref, warnings = _write_remote(platform, store, name, value,
                                      overwrite=overwrite, note=note)
    except BackendError as exc:
        if backend not in (None, "", "auto"):
            raise  # the caller named this platform: failing quietly elsewhere would lie
        logger.warning("%s write failed, storing locally instead: %s",
                       platform.display_name, exc, exc_info=True)
        result = store.set(name, value, overwrite=overwrite)
        if result.get("success"):
            result.update({"backend": LOCAL, "warnings": [
                f"{platform.display_name} write failed ({exc}); stored in the local store instead"]})
        return result
    store.set_remote(name, platform.name, ref, overwrite=True)
    return {"success": True, "backend": platform.name, "warnings": warnings}


def remove_credential(name: str, *, store=None) -> dict:
    """Delete a credential — from its platform first, then the local index.

    A failed platform delete keeps the index entry so the user can retry rather
    than stranding an orphaned secret they think Hermes removed.
    """
    from agent.credential_store import get_store

    store = store if store is not None else get_store()
    entry = store.entry(name)
    if entry is None:
        return {"success": False, "error": f"Credential '{name}' not found."}
    backend_name = str(entry.get("backend") or LOCAL)
    if backend_name != LOCAL:
        backend = get_backend(backend_name)
        try:
            backend.delete(entry["ref"])
        except BackendError as exc:
            return {"success": False, "error":
                    f"Could not delete '{name}' from {backend.display_name}: {exc}\n"
                    f"The entry was kept so you can retry."}
    store.delete(name)
    return {"success": True, "name": name, "backend": backend_name}


def format_statuses(store=None) -> str:
    """`/creds status` body: every platform's availability plus the local store."""
    from agent.credential_store import get_store

    store = store if store is not None else get_store()
    lines = ["Credential backends:", ""]
    for status in statuses():
        mark = "available  " if status.available else "unavailable"
        lines.append(f"  {status.display_name:<26} {mark}  {status.reason}")
    local_count = len(store.list())
    lines.append(f"  {'Local encrypted store':<26} available  "
                 f"{store.base_dir} ({local_count} credential(s))")
    lines += ["", "Set a target explicitly with /creds set --backend <name>, or let /creds prompt."]
    return "\n".join(lines)