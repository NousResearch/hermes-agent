"""Single-use OAuth grant hygiene: strip cloned grants from profiles, heal forked grants.

Split out of ``hermes_cli/auth.py`` and re-exported there; origin helpers are imported lazily
inside each function so ``hermes_cli.auth.<name>`` patches still intercept (and no import cycle).
"""

from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from hermes_cli.auth_constants import _decode_jwt_claims

# Log-record parity with the origin module (caplog tests pin "hermes_cli.auth").
logger = logging.getLogger("hermes_cli.auth")

# Pool providers whose OAuth refresh tokens are SINGLE-USE: redeeming rotates the pair and
# revokes the old one, so a grant forked into two auth.json files is ONE credential with two
# owners — the first to refresh strands the other with ``invalid_grant`` /
# ``refresh_token_reused``.
# Profiles must never receive a copy: ONE grant lives at the global root and named profiles read
# it through the ``read_credential_pool`` root fallback.
SINGLE_USE_REFRESH_POOL_PROVIDERS = frozenset({"anthropic", "openai-codex", "xai-oauth"})

# Singleton credential files holding the same single-use grants outside ``auth.json``. Copying one
# into a profile re-seeds a forked pool row on the profile's next ``load_pool()``.
SINGLE_USE_OAUTH_SINGLETON_FILES = (".anthropic_oauth.json",)

# Providers whose device-code grants live under ``providers.<id>`` (not only the pool).
_DEVICE_CODE_BLOCK_PROVIDERS = ("openai-codex", "xai-oauth")


def _is_oauth_pool_payload(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    # Legacy rows predating ``auth_type``: an Anthropic OAuth access token or any row carrying a
    # refresh token is an OAuth grant.
    return (
        str(entry.get("auth_type") or "").strip().lower() == "oauth"
        or bool(str(entry.get("refresh_token") or "").strip())
        or str(entry.get("access_token") or "").startswith("sk-ant-oat"))


def _is_pkce_row(row: Dict[str, Any]) -> bool:
    return str(row.get("source") or "").endswith("hermes_pkce")


def strip_cloned_single_use_oauth_grants(profile_dir: Path) -> Dict[str, Any]:
    """Remove forked single-use OAuth grants from a freshly cloned profile.

    Called after any path that copies credential files between profiles (``hermes profile create
    --clone-all``, the dashboard/TUI ``mirror_credentials`` flow). API-key pool rows are kept — a
    static key is safe to duplicate. Returns ``{"pool": [...provider ids], "providers": [...],
    "files": [...]}`` of what was stripped. Never raises: a clone must not fail because hygiene
    could not run — the caller logs the summary.
    """
    from hermes_cli.auth import _save_auth_store
    stripped: Dict[str, Any] = {"pool": [], "providers": [], "files": []}
    profile_dir = Path(profile_dir)
    for name in SINGLE_USE_OAUTH_SINGLETON_FILES:
        target = profile_dir / name
        try:
            if target.is_file() or target.is_symlink():
                target.unlink()
                stripped["files"].append(name)
        except OSError:
            logger.debug("Could not remove cloned %s from %s", name, profile_dir, exc_info=True)
    auth_path = profile_dir / "auth.json"
    if not auth_path.is_file():
        return stripped
    try:
        store = json.loads(auth_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        store = None
    if not isinstance(store, dict):
        return stripped
    changed = False
    pool = store.get("credential_pool")
    if isinstance(pool, dict):
        for provider_id in list(pool):
            entries = pool.get(provider_id)
            if (provider_id not in SINGLE_USE_REFRESH_POOL_PROVIDERS
                    or not isinstance(entries, list)):
                continue
            kept = [e for e in entries if not _is_oauth_pool_payload(e)]
            if len(kept) != len(entries):
                changed = True
                stripped["pool"].append(provider_id)
                if kept:
                    pool[provider_id] = kept
                else:
                    # No local rows → read_credential_pool falls back to the root slice.
                    del pool[provider_id]
    providers = store.get("providers")
    if isinstance(providers, dict):
        # _load_provider_state has the same root fallback, so dropping the copy keeps the
        # profile working while removing the fork.
        for provider_id in _DEVICE_CODE_BLOCK_PROVIDERS:
            block = providers.get(provider_id)
            if isinstance(block, dict) and block:
                del providers[provider_id]
                stripped["providers"].append(provider_id)
                changed = True
    if not changed:
        return stripped
    try:
        _save_auth_store(store, target_path=auth_path)
    except Exception:
        logger.debug(
            "Failed to strip cloned single-use OAuth grants from %s", auth_path, exc_info=True)
    return stripped


_OAUTH_TOKEN_FIELDS = (
    "access_token", "refresh_token", "expires_at", "expires_at_ms", "last_refresh")

_oauth_heal_notices: List[str] = []

# provider -> (profile auth.json path, auth.json mtime_ns, singleton mtime_ns) of the last store
# verified fork-free; lets load_pool() skip the locked scan.
_oauth_heal_clean_marks: Dict[str, Tuple[str, Optional[int], Optional[int]]] = {}


def consume_oauth_heal_notices() -> List[str]:
    """Return (and clear) human-readable notes about heals run in this process.

    ``hermes auth list`` / ``hermes auth status`` print them so the user sees the consolidation.
    """
    from hermes_cli.auth import _oauth_heal_notices
    notes = list(_oauth_heal_notices)
    _oauth_heal_notices.clear()
    return notes


def _oauth_identity(entry: Dict[str, Any]) -> Optional[str]:
    """Stable account identity for an OAuth row when the token carries one.

    Codex / xAI access tokens are JWTs with ``sub`` / ``email`` / ``chatgpt_account_id`` claims;
    Anthropic ``sk-ant-oat`` tokens carry none (None → lineage rests on id / token material).
    """
    from hermes_cli.auth import _nonempty_str
    if not isinstance(entry, dict):
        return None
    for token in (entry.get("access_token"), entry.get("id_token")):
        claims = _decode_jwt_claims(token)
        if not claims:
            continue
        nested = claims.get("https://api.openai.com/auth")
        account = nested.get("chatgpt_account_id") if isinstance(nested, dict) else None
        for value in (account, claims.get("sub"), claims.get("email")):
            if _nonempty_str(value):
                return value.strip()
    return None


def _oauth_freshness(entry: Dict[str, Any]) -> float:
    """Best-effort 'how recently was this pair issued' score (epoch seconds).

    A rotation always issues a later-expiring access token, so ``expires_at`` ordering identifies
    the live copy; ``last_refresh`` and the JWT ``exp`` claim are fallbacks.
    """
    from agent.credential_pool import _parse_absolute_timestamp
    stamps = [entry.get(k) for k in ("expires_at_ms", "expires_at", "last_refresh")]
    best = max((ts for ts in map(_parse_absolute_timestamp, stamps) if ts), default=0.0)
    if best == 0.0:
        exp = _decode_jwt_claims(entry.get("access_token")).get("exp")
        best = _parse_absolute_timestamp(exp) or 0.0
    return best


def _find_root_counterpart(
    profile_row: Dict[str, Any], root_rows: List[Dict[str, Any]]) -> Optional[int]:
    """Index of the root OAuth row that shares a grant lineage with *profile_row*.

    Fallback per the one-grant-at-root rule: same provider + same OAuth client — every Anthropic
    ``hermes_pkce`` grant uses one client id and carries no claims, so two Anthropic OAuth rows
    with no contrary identity are one lineage.
    """
    from hermes_cli.auth import _nonempty_str
    candidates = [i for i, r in enumerate(root_rows) if _is_oauth_pool_payload(r)]
    if not candidates:
        return None
    pid = profile_row.get("id")
    for i in candidates:
        if pid and root_rows[i].get("id") == pid:
            return i
    p_ident = _oauth_identity(profile_row)
    for i in candidates:
        r_ident = _oauth_identity(root_rows[i])
        if p_ident and r_ident and p_ident == r_ident:
            return i
    for key in ("refresh_token", "access_token"):
        p_val = profile_row.get(key)
        if not _nonempty_str(p_val):
            continue
        for i in candidates:
            if root_rows[i].get(key) == p_val:
                return i
    # Fallback: same provider + same client. Only a contradicting identity (both sides carry
    # claims and they differ from every root row) blocks it.
    if p_ident:
        for i in candidates:
            if not _oauth_identity(root_rows[i]):
                return i
        return None
    return candidates[0]


def _adopt_oauth_material(target: Dict[str, Any], winner: Dict[str, Any]) -> Dict[str, Any]:
    """Return *target* carrying *winner*'s token pair, status markers cleared."""
    from hermes_cli.auth import _POOL_STATUS_FIELDS
    merged = dict(target)
    for key in _OAUTH_TOKEN_FIELDS:
        if winner.get(key) is not None:
            merged[key] = winner[key]
        else:
            merged.pop(key, None)
    merged.update(dict.fromkeys(_POOL_STATUS_FIELDS))
    return merged


def _singleton_as_row(path: Path) -> Optional[Dict[str, Any]]:
    """Read a ``.anthropic_oauth.json`` as a pool-row-shaped dict, or None."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not str(data.get("accessToken") or "").strip():
        return None
    return {
        "access_token": data.get("accessToken"),
        "refresh_token": data.get("refreshToken"),
        "expires_at_ms": data.get("expiresAt")}


def heal_forked_single_use_oauth_grants(provider_id: str) -> Optional[Dict[str, Any]]:
    """One-time heal for installs that ALREADY forked a single-use grant into a profile.

    Forked copies are one credential with several owners: whichever profile rotated last holds the
    only live refresh token and every other copy (root included) is spent. Runs at profile
    ``load_pool()`` time for ``SINGLE_USE_REFRESH_POOL_PROVIDERS``: finds profile rows sharing
    LINEAGE with a root row (same pool id, or same account identity / token material), keeps the
    freshest rotation, writes it into ROOT when root's is older, and strips the profile's copy so
    the profile borrows root from then on. Idempotent; never touches API-key rows; never deletes a
    row with no root counterpart (an independent ``hermes -p <p> auth add`` grant, or the only
    surviving copy); reads only the two auth.json files the root fallback already reads. Returns
    ``{"adopted", "stripped_ids", "files", "providers_block"}`` when something healed, else None.
    Never raises.
    """
    if provider_id not in SINGLE_USE_REFRESH_POOL_PROVIDERS:
        return None
    try:
        return _heal_forked_single_use_oauth_grants(provider_id)
    except Exception:
        logger.debug("%s: forked-OAuth heal skipped", provider_id, exc_info=True)
        return None


def _heal_forked_provider_block(
    profile_store: Dict[str, Any], root_store: Dict[str, Any], provider_id: str) -> Optional[bool]:
    """Consolidate a forked ``providers.<id>`` device-code block into root.

    Returns None when nothing matched, False when the profile copy was dropped (root already
    newest), True when the profile copy was fresher and was adopted into root.
    """
    p_providers, r_providers = profile_store.get("providers"), root_store.get("providers")
    if not (isinstance(p_providers, dict) and isinstance(r_providers, dict)):
        return None
    p_block, r_block = p_providers.get(provider_id), r_providers.get(provider_id)
    if not (isinstance(p_block, dict) and p_block and isinstance(r_block, dict) and r_block):
        return None

    def _flat(block: Dict[str, Any]) -> Dict[str, Any]:
        tokens = block.get("tokens") if isinstance(block.get("tokens"), dict) else {}
        return {**tokens, "last_refresh": block.get("last_refresh")}

    p_flat, r_flat = _flat(p_block), _flat(r_block)
    p_ident, r_ident = _oauth_identity(p_flat), _oauth_identity(r_flat)
    if p_ident and r_ident and p_ident != r_ident:
        return None
    adopted = _oauth_freshness(p_flat) > _oauth_freshness(r_flat)
    if adopted:
        r_providers[provider_id] = dict(p_block)
    del p_providers[provider_id]
    return adopted


def _mtime_ns(p: Optional[Path]) -> Optional[int]:
    try:
        return p.stat().st_mtime_ns if p is not None else None
    except OSError:
        return None


def _pool_rows(store: Dict[str, Any], provider_id: str) -> Tuple[Any, List[Any]]:
    """``(store["credential_pool"], its provider_id rows-or-[])`` — pool may be None/non-dict."""
    pool = store.get("credential_pool")
    rows = pool.get(provider_id) if isinstance(pool, dict) else None
    return pool, rows if isinstance(rows, list) else []


def _adopt_if_fresher(
    target: Dict[str, Any], candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """*target* carrying *candidate*'s pair when the candidate rotated later, else None."""
    fresher = _oauth_freshness(candidate) > _oauth_freshness(target)
    return _adopt_oauth_material(target, candidate) if fresher else None


class _HealPass:
    """One heal pass over a profile store vs the root store for *provider_id*."""

    def __init__(self, profile_store, root_store, provider_id, root_singleton: Optional[Path]):
        self.profile_store, self.root_store = profile_store, root_store
        self.provider_id = provider_id
        self.summary: Dict[str, Any] = {
            "adopted": False, "stripped_ids": [], "files": [], "providers_block": False}
        self.profile_changed = self.root_changed = False
        self.p_pool, self.p_rows = _pool_rows(profile_store, provider_id)
        self.r_pool, self.r_rows = _pool_rows(root_store, provider_id)
        self.r_oauth = [r for r in self.r_rows if _is_oauth_pool_payload(r)]
        self.root_singleton = root_singleton
        self.root_singleton_row = (
            _singleton_as_row(root_singleton)
            if root_singleton is not None and root_singleton.exists() else None)

    def _adopt_root_row(self, idx: int, row: Dict[str, Any]) -> None:
        merged = _adopt_if_fresher(self.r_rows[idx], row)
        if merged is not None:
            self.r_rows[idx] = merged
            self.root_changed = self.summary["adopted"] = True

    def _adopt_root_singleton(self, row: Dict[str, Any]) -> None:
        merged = _adopt_if_fresher(self.root_singleton_row, row)
        if merged is not None:
            self.root_singleton_row = merged
            self.summary["adopted"] = True

    def heal_pool_rows(self) -> None:
        kept_rows: List[Any] = []
        for row in self.p_rows:
            if not _is_oauth_pool_payload(row):
                kept_rows.append(row)  # API keys are safe to duplicate
                continue
            match_idx = _find_root_counterpart(row, self.r_rows)
            if match_idx is not None:
                self._adopt_root_row(match_idx, row)
            # No root pool counterpart. Root's grant may live only in its .anthropic_oauth.json
            # (the ``hermes auth`` PKCE shape); a profile hermes_pkce-family row is its copy.
            elif _is_pkce_row(row) and self.root_singleton_row is not None and not self.r_oauth:
                self._adopt_root_singleton(row)
            else:
                # Root holds no copy of this lineage (independent account, or root never had the
                # grant): the profile's row may be the only surviving copy — leave it alone.
                kept_rows.append(row)
                continue
            self.summary["stripped_ids"].append(row.get("id"))
            self.profile_changed = True
        if self.profile_changed and isinstance(self.p_pool, dict):
            if kept_rows:
                self.p_pool[self.provider_id] = kept_rows
            else:
                self.p_pool.pop(self.provider_id, None)

    def heal_provider_block(self) -> None:
        if self.provider_id not in _DEVICE_CODE_BLOCK_PROVIDERS:
            return
        block_result = _heal_forked_provider_block(
            self.profile_store, self.root_store, self.provider_id)
        if block_result is not None:
            self.profile_changed = self.summary["providers_block"] = True
            if block_result:
                self.root_changed = self.summary["adopted"] = True

    def heal_profile_singleton(self, profile_singleton: Optional[Path]) -> None:
        if profile_singleton is None or not profile_singleton.exists():
            return
        from hermes_cli.auth import _is_same_auth_store
        if self.root_singleton is not None and _is_same_auth_store(profile_singleton, self.root_singleton):
            return  # an aliased singleton pair is one shared grant, not a fork: never self-compare/unlink
        # See #101356.
        p_single = _singleton_as_row(profile_singleton)
        root_has_grant = bool(self.r_oauth) or self.root_singleton_row is not None
        # Otherwise root has NO grant for this provider (or the file is not a grant): the
        # profile's singleton may be the only surviving copy — never delete it.
        if p_single is None or not root_has_grant:
            return
        if self.root_singleton_row is not None:
            self._adopt_root_singleton(p_single)
        else:
            # Root only has pool rows: fold the singleton's pair into the freshest-matching
            # root pkce row, if any.
            idx = next(
                (i for i, r in enumerate(self.r_rows)
                 if _is_oauth_pool_payload(r) and _is_pkce_row(r)), None)
            if idx is not None:
                self._adopt_root_row(idx, p_single)
        try:
            profile_singleton.unlink()
            self.summary["files"].append(profile_singleton.name)
        except OSError:
            logger.debug("could not remove %s", profile_singleton, exc_info=True)

    def sync_root_singleton_with_pkce_row(self) -> None:
        """Keep root's singleton and its ``hermes_pkce``-seeded pool row in step.

        Root's next load_pool() re-seeds that row FROM the singleton file, so a stale file would
        resurrect the spent pair (and a stale row would be overwritten by a fresh file).
        """
        if not (
            self.summary["adopted"] and self.root_singleton is not None
            and self.root_singleton_row is not None):
            return
        pkce_idx = next(
            (i for i, r in enumerate(self.r_rows)
             if _is_oauth_pool_payload(r) and r.get("source") == "hermes_pkce"), None)
        if pkce_idx is None:
            return
        pkce_row = self.r_rows[pkce_idx]
        row_fresh = _oauth_freshness(pkce_row)
        single_fresh = _oauth_freshness(self.root_singleton_row)
        if row_fresh > single_fresh:
            self.root_singleton_row = _adopt_oauth_material(self.root_singleton_row, pkce_row)
        elif single_fresh > row_fresh:
            self.r_rows[pkce_idx] = _adopt_oauth_material(pkce_row, self.root_singleton_row)
            self.root_changed = True

    @property
    def dirty(self) -> bool:
        return bool(self.profile_changed or self.root_changed or self.summary["adopted"])

    def notice(self, profile_name: str) -> str:
        summary = self.summary
        log_bits = [bit for bit, present in (
            (f"pool rows {summary['stripped_ids']}", summary["stripped_ids"]),
            (f"providers.{self.provider_id} block", summary["providers_block"]),
            (", ".join(summary["files"]), summary["files"])) if present]
        verdict = (
            "profile copy was the live pair; root updated"
            if summary["adopted"] else "root copy already newest; profile copy dropped")
        return (
            f"profile {profile_name}: consolidated forked {self.provider_id} OAuth grant "
            f"({'; '.join(log_bits) or 'no-op'}) into the root grant — {verdict}; "
            f"this profile now borrows the root grant (#100339)")


def _heal_forked_single_use_oauth_grants(provider_id: str) -> Optional[Dict[str, Any]]:
    from hermes_cli.auth import (
        _auth_file_path, _auth_store_lock, _global_auth_file_path, _load_auth_store,
        _is_same_auth_store, _oauth_heal_clean_marks, _oauth_heal_notices, _same_path,
        _save_auth_store)
    root_path = _global_auth_file_path()
    if root_path is None:
        return None  # classic mode: nothing to consolidate into
    if os.environ.get("PYTEST_CURRENT_TEST"):
        # Same seat belt as the write-through paths: never touch the real user's
        # ~/.hermes/auth.json from a test that forgot to isolate HOME.
        real_home_env = os.environ.get("HOME", "")
        if real_home_env and _same_path(root_path, Path(real_home_env) / ".hermes" / "auth.json"):
            return None
    profile_path = _auth_file_path()
    profile_home = profile_path.parent
    is_anthropic = provider_id == "anthropic"
    profile_singleton = profile_home / ".anthropic_oauth.json" if is_anthropic else None
    root_singleton = root_path.parent / ".anthropic_oauth.json" if is_anthropic else None

    # Hot-path short-circuit: load_pool() runs per model call. Once this profile's store was
    # verified clean for *provider_id*, skip the locked read-modify-write until the profile's own
    # files change (mtime key).
    fingerprint = (str(profile_path), _mtime_ns(profile_path), _mtime_ns(profile_singleton))
    if _oauth_heal_clean_marks.get(provider_id) == fingerprint:
        return None
    if fingerprint[1] is None and fingerprint[2] is None:
        _oauth_heal_clean_marks[provider_id] = fingerprint
        return None
    if _is_same_auth_store(profile_path, root_path):
        # The profile's auth.json IS the root store (symlink/hardlink alias — a deliberate way to
        # share one grant). Both "sides" would read the same file, every OAuth row would match
        # itself, and the strip would write through the alias and delete the shared credential.
        # Nothing to consolidate; the mtime mark keeps this off the per-call hot path.
        _oauth_heal_clean_marks[provider_id] = fingerprint
        # See #101356.
        logger.debug("%s: forked-OAuth heal skipped, %s is the root store", provider_id, profile_path)
        return None

    # Lock order: active (profile) store first, then the root source store — the same order
    # ``_provider_state_transaction`` uses.
    with _auth_store_lock():
        profile_store = (
            _load_auth_store(profile_path) if profile_path.exists() else {"providers": {}})
        with _auth_store_lock(target_path=root_path):
            root_store = _load_auth_store(root_path) if root_path.exists() else {"providers": {}}
            run = _HealPass(profile_store, root_store, provider_id, root_singleton)
            run.heal_pool_rows()
            run.heal_provider_block()
            run.heal_profile_singleton(profile_singleton)
            if not run.dirty:
                _oauth_heal_clean_marks[provider_id] = fingerprint
                return None
            run.sync_root_singleton_with_pkce_row()
            summary = run.summary
            if run.root_changed:
                if isinstance(run.r_pool, dict):
                    run.r_pool[provider_id] = run.r_rows
                else:
                    root_store["credential_pool"] = {provider_id: run.r_rows}
                _save_auth_store(root_store, target_path=root_path)
            singleton_row = run.root_singleton_row
            if summary["adopted"] and root_singleton is not None and singleton_row is not None:
                from agent.anthropic_credentials import _write_hermes_oauth_credentials
                _write_hermes_oauth_credentials(
                    singleton_row.get("access_token") or "", singleton_row.get("refresh_token"),
                    singleton_row.get("expires_at_ms"), target=root_singleton)
            if run.profile_changed and profile_path.exists():
                _save_auth_store(profile_store, target_path=profile_path)
    message = run.notice(profile_home.name)
    logger.info(message)
    _oauth_heal_notices.append(message)
    return summary
