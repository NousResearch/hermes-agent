"""What ``model.options`` knows about subscription usage without waiting on the network.

Usage windows come from a provider API (:func:`agent.account_usage.fetch_account_usage`), so a
picker reads this cache and asks for a background refresh instead of fetching inline. Every fetch
lands here, including the one ``session.usage`` runs after each turn, so a picker is at most one
turn or one refresh interval behind.

Slots are keyed per profile home AND per account: one process may serve many profiles, and a
provider may have a pool of credentials whose usage must not contaminate each other. A fetcher
that can identify its account (a decoded Codex JWT principal) dedupes several credentials of one
account onto a single slot; fetchers that cannot get one slot per credential, never a
provider-wide singleton. The legacy provider slot (``identity_id=None``) remains the single-account
gauge: a per-turn or ``/usage`` fetch still lands there, so non-pooled chips are unchanged.
"""

from __future__ import annotations

import threading
import time
from contextvars import copy_context
from typing import TYPE_CHECKING, Any, Iterable, Optional

if TYPE_CHECKING:
    from agent.account_usage import AccountUsageSnapshot

# A picker opened again within this long reuses what it has; usage windows move in minutes, not seconds.
REFRESH_AFTER_S = 120.0
# A snapshot older than this renders as state ``unknown`` instead of a confident gauge: a benched
# credential's windows stop updating, and reporting stale numbers as live would mislabel an
# account that may have recovered (or worsened) in the meantime.
SNAPSHOT_STALE_AFTER_S = 15 * 60.0

_lock = threading.Lock()
# slot key (home, provider, identity_id, base_url) → snapshot
_snapshots: dict[tuple[str, str, str, str], AccountUsageSnapshot] = {}
# slot key → monotonic time the stored snapshot's fetch STARTED (late-fetch guard)
_started: dict[tuple[str, str, str, str], float] = {}
# slot key → monotonic time of the last refresh try (success or failure)
_last_try: dict[tuple[str, str, str, str], float] = {}
_inflight: set[tuple[str, str, str, str]] = set()


def _key(provider: str, identity_id: Optional[str] = None, base_url: Optional[str] = None) -> tuple[str, str, str, str]:
    from hermes_constants import hermes_home_key

    return (
        hermes_home_key(),
        str(provider or "").strip().lower(),
        str(identity_id or "").strip(),
        str(base_url or "").strip().rstrip("/"),
    )


def has_account_usage(provider: str) -> bool:
    """Whether *provider* can report usage windows at all (a built-in fetcher or a plugin hook)."""
    from agent.account_usage import _USAGE_FETCHERS
    from providers import get_provider_profile
    from providers.base import ProviderProfile

    slug = str(provider or "").strip().lower()
    if slug in _USAGE_FETCHERS:
        return True
    profile = get_provider_profile(slug) if slug else None
    return profile is not None and type(profile).fetch_account_usage is not ProviderProfile.fetch_account_usage


def _identity_id_for(provider: str, entry: Any) -> str:
    """Stable non-secret account id of one pool entry, for slot keys and wire ``accounts[].id``.

    A trusted decoded identity (Codex JWT principal) wins: several credentials of one account
    share a slot, so the account is counted once. Without one, the entry's token fingerprint
    stands in — same token ⇒ same account, and a replaced token never reuses the old quota.
    A metadata-only row (no token) falls back to the entry id: nothing better is knowable.
    """
    token = str(getattr(entry, "runtime_api_key", "") or "").strip()
    if not token:
        return f"entry:{getattr(entry, 'id', '')}"
    if provider == "openai-codex":
        from agent.credential_pool import _codex_principal_identity

        principal = _codex_principal_identity(token)
        if principal:
            return f"codex:{principal[0]}:{principal[1]}"
    import hashlib

    return f"fp:{hashlib.sha256(token.encode('utf-8')).hexdigest()[:16]}"


def remember_account_usage(
    provider: Optional[str], snapshot: Optional[AccountUsageSnapshot],
    *, identity_id: Optional[str] = None, base_url: Optional[str] = None,
    started_monotonic: Optional[float] = None,
) -> None:
    """Land one fetch result.

    ``identity_id`` given (per-credential fetch): the snapshot is stored under that account's
    slot only — one pool entry's usage must never pose as the provider-wide gauge. The snapshot's
    own trusted ``identity`` (decoded from the very token it was fetched with) wins over the
    caller's id, so a replaced credential cannot repopulate its old account's slot.

    ``identity_id`` absent (per-turn / ``/usage`` fetch): stored in the provider's legacy slot —
    the single-account gauge — and additionally in the account's identity slot when the fetcher
    identified it, so a pooled picker sees per-turn data without waiting on its own refresh.

    A snapshot without windows is never stored: a failed or empty fetch must not erase a fresher
    snapshot. A fetch that STARTED before the freshest stored one cannot replace it either
    (``started_monotonic``), however late its response arrives.
    """
    if not provider or snapshot is None or not snapshot.windows:
        return
    started = time.monotonic() if started_monotonic is None else started_monotonic
    with _lock:
        targets = []
        snapshot_identity = getattr(snapshot, "identity", None)
        if identity_id is None:
            targets.append(_key(provider, None, base_url))
            if snapshot_identity:
                targets.append(_key(provider, snapshot_identity, None))
        else:
            targets.append(_key(provider, snapshot_identity or identity_id, None))
        for key in targets:
            if key in _started and _started[key] > started:
                continue  # a fresher fetch already landed; this late result must not replace it
            _snapshots[key] = snapshot
            _started[key] = started


def cached_account_usage(
    provider: str, *, identity_id: Optional[str] = None, base_url: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    with _lock:
        return _snapshots.get(_key(provider, identity_id, base_url))


def snapshot_is_stale(snapshot: Optional[AccountUsageSnapshot]) -> bool:
    """True when the snapshot is too old to render as a live gauge (the picker shows state
    ``unknown`` instead of trusting its percentages)."""
    from agent.account_usage import AccountUsageSnapshot as _Snapshot

    if not isinstance(snapshot, _Snapshot):
        return True
    try:
        age = time.time() - snapshot.fetched_at.timestamp()
    except (AttributeError, TypeError, ValueError, OSError, OverflowError):
        return True
    return age < 0 or age > SNAPSHOT_STALE_AFTER_S


def _claim_refresh(key: tuple[str, str, str, str], now: float) -> bool:
    """Take the refresh right for *key* unless one is inflight or tried too recently.

    A failed or empty fetch still counts as a try, so a dead credential (or a provider with no
    usage API) isn't re-asked on every picker open."""
    with _lock:
        if key in _inflight or now - _last_try.get(key, float("-inf")) < REFRESH_AFTER_S:
            return False
        _inflight.add(key)
        _last_try[key] = now
        return True


def _release_refresh(key: tuple[str, str, str, str]) -> None:
    with _lock:
        _inflight.discard(key)


def refresh_account_usage_async(providers: Iterable[str]) -> list[threading.Thread]:
    """Background-refresh the provider-wide (legacy) slot for each provider not tried within
    ``REFRESH_AFTER_S``. Returns the started threads (daemon; callers never join them)."""
    started: list[threading.Thread] = []
    now = time.monotonic()
    for provider in dict.fromkeys(providers):
        key = _key(provider)
        if not _claim_refresh(key, now):
            continue
        thread = threading.Thread(
            target=copy_context().run, args=(_refresh_legacy, provider, key),
            name="hermes-account-usage-refresh", daemon=True)
        thread.start()
        started.append(thread)
    return started


def _refresh_legacy(provider: str, key: tuple[str, str, str, str]) -> None:
    from agent.account_usage import fetch_account_usage

    try:
        fetch_account_usage(provider)  # remembers its own result
    finally:
        _release_refresh(key)


def refresh_account_usage_entries_async(requests: Iterable[dict]) -> list[threading.Thread]:
    """Background-refresh per-credential usage slots. *requests* entries are
    ``{"provider", "identity_id", "base_url", "api_key"}`` — the live credentials to read.

    One worker thread fetches all due entries serially (no thread storm for a 10-credential
    pool), read-only: a 401 or any other failure is reported, never repaired by rotating or
    refreshing the credential. Throttled per account slot, failures included."""
    due: list[dict] = []
    now = time.monotonic()
    for request in requests:
        provider = str(request.get("provider") or "").strip().lower()
        identity_id = str(request.get("identity_id") or "").strip()
        if not provider or not identity_id or not str(request.get("api_key") or "").strip():
            continue
        if not _claim_refresh(_key(provider, identity_id), now):
            continue
        due.append(request)
    if not due:
        return []
    thread = threading.Thread(
        target=copy_context().run, args=(_refresh_entries, list(due)),
        name="hermes-account-usage-refresh", daemon=True)
    thread.start()
    return [thread]


def _refresh_entries(requests: list[dict]) -> None:
    from agent.account_usage import fetch_account_usage

    for request in requests:
        provider = request["provider"]
        key = _key(provider, request["identity_id"])
        try:
            fetch_account_usage(
                provider, base_url=request.get("base_url"), api_key=request.get("api_key"),
                read_only=True, identity_id=request["identity_id"],
            )
        finally:
            _release_refresh(key)
