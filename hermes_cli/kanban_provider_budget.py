"""Per-provider concurrency budget for the Kanban dispatcher (#123654).

Counts running workers per RESOLVED provider — never per profile identity — and
defers (never kills) any spawn that would exceed its provider's budget. The
budget key is the "requested route at admission": the task's
``model_override``/``provider_override`` first, else the assignee profile's
configured model route, resolved through the same pure stage-1 function the
CLI uses (``hermes_cli.model_route.resolve_requested_route``) plus a
credential-free custom-endpoint keying step.

Nothing here runs stage-2 runtime resolution: no ``resolve_runtime_provider``,
no credential pools, no network I/O. An ``auto`` route buckets as ``auto``
(D6, O1); a custom endpoint keys by its normalized base URL (O2); an
unresolvable route keys as ``unknown``. Every dispatcher claim records its key
on ``task_runs.provider_key`` even when the budget is disabled (O3), so counts
survive restarts and later profile edits.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from hermes_cli.model_route import resolve_requested_route
from utils import base_url_host_matches, base_url_hostname

logger = logging.getLogger(__name__)

# Hosts whose URLs never take the auto + local-base_url bypass (mirrors
# runtime_provider._LOCAL_BYPASS_CLOUD_HOSTS; D2 step 2). Imported lazily by
# runtime_provider itself so the two lists can never drift.
_LOCAL_BYPASS_CLOUD_HOSTS = ("openrouter.ai", "anthropic.com", "openai.com")


# ---------------------------------------------------------------------------
# Config parsing (D7)
# ---------------------------------------------------------------------------


@dataclass
class ProviderBudgets:
    """Parsed ``kanban.provider_concurrency`` mapping.

    ``caps`` maps budget keys (canonical provider ids, ``custom:<url>``,
    pseudo-keys) to positive caps; a key present with value ``None`` is
    EXPLICITLY unbudgeted (``key: null`` in YAML). ``default`` is the per-key
    fallback cap applied separately to every unlisted key (NOT a shared pool),
    or ``None`` when unset (unlisted keys are unbudgeted). ``sources`` keeps
    each cap's original config key (log-safe form) so the duplicate-after-
    normalization warning can name both (D7).
    """

    caps: Dict[str, Optional[int]] = field(default_factory=dict)
    default: Optional[int] = None
    sources: Dict[str, str] = field(default_factory=dict)

    def cap_for(self, key: str) -> Optional[int]:
        """Effective cap for ``key``: explicit entry wins, else ``default``.

        An explicit ``key: null`` entry means "no budget for this key" and
        overrides ``default``.
        """
        if key in self.caps:
            return self.caps[key]
        return self.default

    def __bool__(self) -> bool:
        return bool(self.caps) or self.default is not None


_warned_bad_entries: set[Tuple[str, str]] = set()


def normalize_endpoint_key(url: str) -> str:
    """Canonical endpoint key form: lowercase scheme/host, no userinfo, no
    query/fragment, no default port, no trailing slash.

    ``https://User:Pass@LLM.Example.INTERNAL:443/v1/?x=1#f`` ->
    ``https://llm.example.internal/v1``. Secrets embedded in the URL never
    reach keys, logs, or diagnostics (D2, MoA I13). Unparseable input comes
    back as ``<invalid url>``.
    """
    raw = (url or "").strip()
    if not raw:
        return "<invalid url>"
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(raw if "://" in raw else f"https://{raw}")
        scheme = (parts.scheme or "https").lower()
        host = (parts.hostname or "").lower().rstrip(".")
        if not host:
            return "<invalid url>"
        port = parts.port
        default = {"https": 443, "http": 80}.get(scheme)
        if port is None or port == default:
            netloc = host
        else:
            netloc = f"{host}:{port}"
        path = (parts.path or "/").rstrip("/")
        return urlunsplit((scheme, netloc, path, "", ""))
    except Exception:
        return "<invalid url>"


def log_safe_key(raw: str) -> str:
    """Log/diagnostic-safe form of a provider-concurrency key (D7/D9).

    Plain provider/bucket names pass through; anything URL-shaped is reduced
    to ``custom:<normalized url>`` (userinfo, query and fragment stripped), so
    a secret embedded in a config key can never reach a log line. Unparseable
    URLs become ``custom:<invalid url>``.
    """
    s = str(raw or "").strip()
    if "://" not in s:
        return s
    url = s.split(":", 1)[1] if s.lower().startswith("custom:") else s
    norm = normalize_endpoint_key(url)
    return f"custom:{norm}" if s.lower().startswith("custom:") else norm


def _warn_bad_entry_once(raw_key: str, raw_value: Any, reason: str) -> None:
    marker = (str(raw_key), str(raw_value))
    if marker in _warned_bad_entries:
        return
    _warned_bad_entries.add(marker)
    logger.warning(
        "kanban provider_concurrency: dropping entry %s=%r (%s)",
        log_safe_key(str(raw_key)), raw_value, reason
    )


def parse_provider_concurrency(raw: Any) -> Optional[ProviderBudgets]:
    """Parse ``kanban.provider_concurrency``; None when absent/disabled.

    Not a mapping or empty -> None (budget off, no per-tick work). Keys are
    stripped + lowercased; reserved bucket keys (``default``, ``auto``,
    ``moa``, ``unknown``, bare ``custom``) are handled BEFORE any
    canonicalization; ``custom:<url>`` keys are normalized; anything else goes
    through ``auth.resolve_provider`` (pure for non-auto names). Values must be
    a positive int or ``null`` (explicitly no budget). Invalid entries drop
    with one WARNING per distinct (key, value) per process. Two keys that
    normalize to one keep the SMALLER cap (warning names both).
    """
    if not isinstance(raw, dict) or not raw:
        return None
    budgets = ProviderBudgets()
    for raw_key, raw_value in raw.items():
        key = str(raw_key or "").strip().lower()
        if not key:
            continue
        # Value validation first: `null` is a valid "no budget" entry.
        if raw_value is None:
            value: Optional[int] = None
        elif isinstance(raw_value, bool):
            _warn_bad_entry_once(raw_key, raw_value, "boolean is not a cap")
            continue
        elif isinstance(raw_value, int) and raw_value > 0:
            value = int(raw_value)
        else:
            _warn_bad_entry_once(raw_key, raw_value, "must be a positive int or null")
            continue

        if key == "default":
            budgets.default = value
            continue
        if key in ("auto", "moa", "unknown", "custom"):
            # Pseudo-buckets, never canonicalized (B5).
            budgets.caps[key] = value
            continue
        if "://" in key or key.startswith("custom:"):
            if "://" not in key:
                _warn_bad_entry_once(raw_key, raw_value, "custom keys are keyed by URL (O2)")
                continue
            url = key.split(":", 1)[1] if key.startswith("custom:") else key
            norm = normalize_endpoint_key(url)
            if norm == "<invalid url>":
                _warn_bad_entry_once(raw_key, raw_value, "unparseable URL")
                continue
            _merge_cap(budgets, f"custom:{norm}", value, raw_key)
            continue
        try:
            from hermes_cli.auth import resolve_provider

            canonical = resolve_provider(key)
            if canonical in ("", "auto"):
                # resolve_provider("auto") would run credential detection; treat
                # as the auto bucket instead (defensive; reserved keys above
                # already captured it).
                _warn_bad_entry_once(raw_key, raw_value, "cannot canonicalize")
                continue
        except Exception:
            # Unknown to the dispatcher's registry (e.g. a plugin provider not
            # loaded here): keep verbatim so it can still match a verbatim key.
            canonical = key
        _merge_cap(budgets, canonical, value, raw_key)
    return budgets if budgets else None


def _merge_cap(budgets: ProviderBudgets, key: str, value: Optional[int], raw_key: str) -> None:
    shown = log_safe_key(str(raw_key))
    if key in budgets.caps:
        existing = budgets.caps[key]
        if value is None or existing is None:
            winner: Optional[int] = None if (value is None or existing is None) else min(existing, value)
        else:
            winner = min(existing, value)
        if winner != existing:
            # Names BOTH original keys (D7) in their log-safe forms.
            logger.warning(
                "kanban provider_concurrency: keys %r and %r normalize to %r; "
                "keeping the smaller cap %r", budgets.sources.get(key, key), shown, key, winner,
            )
        budgets.caps[key] = winner
    else:
        budgets.caps[key] = value
        budgets.sources[key] = shown


_parsed_budget_cache: Dict[str, Optional[ProviderBudgets]] = {}


def parse_provider_concurrency_cached(raw: Any) -> Optional[ProviderBudgets]:
    """``parse_provider_concurrency`` memoized per distinct mapping (D7/MoA B2).

    The tick carries the RAW config mapping and parses it per tick; identical
    content (the gateway and daemon re-read the same config every tick) hits
    the cache, and any config edit produces a new distinct mapping and
    re-parses. Keyed on a stable repr of the mapping, so a mutated-in-place
    dict with changed content re-parses. Bounded: past 64 distinct mappings the
    cache resets (operators do not edit config that often).
    """
    if not isinstance(raw, dict) or not raw:
        return None
    cache_key = repr(sorted((str(k), repr(v)) for k, v in raw.items()))
    if cache_key not in _parsed_budget_cache:
        if len(_parsed_budget_cache) >= 64:
            _parsed_budget_cache.clear()
        _parsed_budget_cache[cache_key] = parse_provider_concurrency(raw)
    return _parsed_budget_cache[cache_key]


# ---------------------------------------------------------------------------
# Route -> budget key (D2)
# ---------------------------------------------------------------------------

_key_resolution_failures: set[Tuple[str, str, str]] = set()


def local_endpoint_bypass_applies(cfg_base_url: str, cfg_provider: str) -> bool:
    """Whether provider ``auto``/unset with ``model.base_url`` at a
    non-cloud local endpoint routes custom (mirrors
    ``runtime_provider._local_endpoint_bypass``'s condition; MoA I10).

    Pure: takes the config values as data so the dispatcher never runs
    ``_get_model_config()`` (which may HTTP auto-detect a model). ONE
    condition shared by the runtime's bypass and the budget's key step
    (D2), so the two cannot drift.
    """
    cfg_provider_norm = (cfg_provider or "").strip().lower()
    bu = (cfg_base_url or "").strip()
    if not bu:
        return False
    if any(base_url_host_matches(bu, host) for host in _LOCAL_BYPASS_CLOUD_HOSTS):
        return False
    return cfg_provider_norm in ("", "auto")


def _resolves_to_custom(name: str) -> bool:
    try:
        from hermes_cli.auth import resolve_provider

        return resolve_provider(name) == "custom"
    except Exception:
        return False


def route_key(
    route, model_config: Dict[str, Any], user_providers: Optional[dict] = None,
    custom_providers: Optional[list] = None,
) -> str:
    """Budget key for a resolved ``RequestedRoute`` + its profile's model config.

    Steps (D2): moa -> ``moa``; auto with a local custom endpoint -> custom
    URL key; ``expand_direct_api_alias``; custom-class (bare ``custom``, a
    named ``providers:``/``custom_providers:`` entry, or a custom-resolving
    alias) -> ``custom:<normalized url>`` (``custom`` when no URL resolves);
    anything else -> the canonical registry id; unknown -> ``unknown``.
    """
    try:
        requested = (route.requested_provider or "").strip()
        if requested == "moa":
            return "moa"
        explicit_base_url = (route.explicit_base_url or "").strip()

        # Step 2: auto/unset with no explicit URL and a local config endpoint
        # is custom-class (the same pure predicate the runtime's
        # _local_endpoint_bypass uses — D2 says one condition, used by both).
        if requested in ("", "auto") and not explicit_base_url:
            cfg_base_url = str(model_config.get("base_url") or "").strip()
            cfg_provider = str(model_config.get("provider") or "").strip()
            if local_endpoint_bypass_applies(cfg_base_url, cfg_provider):
                return f"custom:{normalize_endpoint_key(cfg_base_url)}"
            if requested == "auto":
                return "auto"
        elif requested == "auto":
            return "auto"

        # Step 3: the same normalization the runtime applies first.
        from hermes_cli.runtime_provider_custom import expand_direct_api_alias

        requested, explicit_base_url = expand_direct_api_alias(requested, explicit_base_url)
        requested = (requested or "").strip()

        # Step 4: custom-class routes key by normalized base URL.
        if requested == "custom" or _named_custom_entry(requested, user_providers, custom_providers) \
                or (requested and _resolves_to_custom(requested)):
            entry = _named_custom_entry(requested, user_providers, custom_providers) or {}
            base_url = explicit_base_url or str(entry.get("base_url") or "").strip() \
                or str(model_config.get("base_url") or "").strip()
            if base_url:
                return f"custom:{normalize_endpoint_key(base_url)}"
            return "custom"

        # Step 5: canonical registry id for non-auto names.
        if requested:
            from hermes_cli.auth import resolve_provider

            try:
                return resolve_provider(requested)
            except Exception:
                return "unknown"
        return "auto"
    except Exception as exc:
        return "unknown"


def _named_custom_entry(name: str, user_providers: Optional[dict], custom_providers: Optional[list]) -> Optional[dict]:
    """A ``providers:``/``custom_providers:`` entry matching ``name`` (pure config read)."""
    if not name:
        return None
    norm = name.strip().lower()
    for key, entry in (user_providers or {}).items():
        if isinstance(entry, dict) and str(key).strip().lower() == norm:
            return entry
    for entry in (custom_providers or []):
        if isinstance(entry, dict) and str(entry.get("name") or "").strip().lower() == norm:
            return entry
    return None


class RouteKeyResolver:
    """Per-tick resolver caching key resolution by (assignee, model, provider).

    Inputs for one task (D2): model = ``task.model_override``, provider =
    ``task.provider_override`` when a model override is set (mirrors
    ``_worker_argv``), else None; all config reads happen inside the caller's
    profile scope via ``inputs_for_profile`` — and, when ``scope`` is given,
    the WHOLE resolution (stage-1 route + key step, including their transitive
    reads: ``model_switch`` aliases, ``OPENAI_BASE_URL`` via
    ``expand_direct_api_alias``) runs inside it too, so a launch profile's
    config can never leak into an assignee's key (#123654 R1 Q-F1).

    Failure semantics (O3/T15c): a row whose route cannot be resolved at all
    returns ``None`` — the claim records NULL and the provider gate is skipped
    (never blocks the spawn) — while a route that RESOLVES but names no known
    provider returns the ``"unknown"`` bucket (D2 step 5), which IS subject
    to ``unknown``/``default`` caps. Exceptions log once per (assignee, model,
    provider) per process with the exception type only, never text or a raw
    URL.
    """

    def __init__(self, *, profile_exists: Optional[Callable[[str], bool]] = None,
                 profile_inputs: Optional[Callable[[str], Optional[dict]]] = None,
                 scope: Optional[Callable[[str], Any]] = None):
        self._profile_exists = profile_exists
        self._profile_inputs = profile_inputs
        self._scope = scope
        self._cache: Dict[Tuple[str, str, str], Optional[str]] = {}
        self._inputs_cache: Dict[str, Optional[dict]] = {}

    def inputs_for_profile(self, assignee: str) -> Optional[dict]:
        """``{model_config, user_providers, custom_providers, env_provider}``
        for ``assignee``'s profile home, read inside the profile scope; None
        when the assignee is not a Hermes profile or its config is unreadable."""
        if not assignee:
            return None
        if assignee in self._inputs_cache:
            return self._inputs_cache[assignee]
        inputs = None
        try:
            if self._profile_exists is not None and not self._profile_exists(assignee):
                inputs = None
            else:
                inputs = (self._profile_inputs or _default_profile_inputs)(assignee)
        except Exception as exc:
            _log_key_failure_once(assignee, "", "", exc)
            inputs = None
        self._inputs_cache[assignee] = inputs
        return inputs

    def _resolve_scoped(self, assignee: str, model: str, provider: str) -> Optional[str]:
        inputs = self.inputs_for_profile(assignee)
        if inputs is None:
            return "unknown"
        route = resolve_requested_route(
            model=model or None,
            provider=provider or None,
            model_config=inputs["model_config"],
            user_providers=inputs.get("user_providers"),
            custom_providers=inputs.get("custom_providers"),
            env_provider=inputs.get("env_provider"),
        )
        return route_key(
            route,
            inputs["model_config"],
            user_providers=inputs.get("user_providers"),
            custom_providers=inputs.get("custom_providers"),
        )

    def resolve(self, assignee: str, model_override: Optional[str], provider_override: Optional[str]) -> Optional[str]:
        """Budget key for one task row; ``None`` on a resolution failure."""
        model = (model_override or "").strip()
        provider = (provider_override or "").strip() if model else ""
        cache_key = (assignee, model, provider)
        if cache_key in self._cache:
            return self._cache[cache_key]
        key: Optional[str] = None
        try:
            if self._scope is not None:
                with self._scope(assignee):
                    key = self._resolve_scoped(assignee, model, provider)
            else:
                key = self._resolve_scoped(assignee, model, provider)
        except Exception as exc:
            _log_key_failure_once(assignee, model, provider, exc)
            key = None
        self._cache[cache_key] = key
        return key


def _log_key_failure_once(assignee: str, model: str, provider: str, exc: Exception) -> None:
    marker = (assignee, model, provider)
    if marker in _key_resolution_failures:
        return
    _key_resolution_failures.add(marker)
    logger.warning(
        "kanban provider budget: could not resolve provider key for assignee=%r "
        "model_override=%r provider_override=%r (%s); recording NULL and leaving "
        "the row unbudgeted this tick",
        assignee, log_safe_key(model), log_safe_key(provider), type(exc).__name__,
    )


def _default_profile_inputs(assignee: str) -> Optional[dict]:
    """Snapshot the assignee profile's route-relevant config inside its scope.

    Runs under the caller's ``_worker_profile_scope`` binding (the dispatcher
    enters it before calling); reads are profile-scoped so a multiplexed
    dispatcher never resolves a route from the launch profile's config.
    """
    from hermes_cli.config import load_config_readonly

    cfg = load_config_readonly() or {}
    model_config = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    return {
        "model_config": model_config,
        "user_providers": cfg.get("providers") if isinstance(cfg.get("providers"), dict) else None,
        "custom_providers": cfg.get("custom_providers") if isinstance(cfg.get("custom_providers"), list) else None,
        "env_provider": _scoped_env_provider(),
    }


def _scoped_env_provider() -> str:
    from agent.secret_scope import get_secret_str

    return (get_secret_str("HERMES_INFERENCE_PROVIDER", "") or "").strip()


# ---------------------------------------------------------------------------
# Counting (D4) and diagnostics snapshot (D9)
# ---------------------------------------------------------------------------


def count_running_by_provider(
    conn,
    resolver: RouteKeyResolver,
    *,
    profile_exists: Optional[Callable[[str], bool]] = None,
    inferred: Optional[Dict[str, int]] = None,
) -> Counter:
    """Counter of running rows per budget key on one board.

    ``task_runs.provider_key`` is authoritative. A NULL key (pre-upgrade run
    or a resolution error at claim) whose assignee is a Hermes profile is
    re-derived from the task's current overrides + the profile's current
    config and counted as inferred (``inferred[key] += 1``) so diagnostics can
    show it; a re-derivation that FAILS (resolver returns None) is not
    counted anywhere — its bucket is unknowable this tick. Rows whose assignee
    is not a Hermes profile (control-plane lanes) are excluded — the
    dispatcher did not spawn them and cannot know their provider.
    """
    counts: Counter = Counter()
    rows = conn.execute(
        "SELECT t.id, t.assignee, t.model_override, t.provider_override, r.provider_key "
        "FROM tasks t LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running'"
    ).fetchall()
    for row in rows:
        key = row["provider_key"]
        if key:
            counts[key] += 1
            continue
        assignee = row["assignee"]
        if not assignee:
            continue
        if profile_exists is not None and not profile_exists(assignee):
            continue
        derived = resolver.resolve(assignee, row["model_override"], row["provider_override"])
        if derived is None:
            continue
        counts[derived] += 1
        if inferred is not None:
            inferred[derived] = (inferred.get(derived, 0) or 0) + 1
    return counts


_warned_count_boards: set[str] = set()


def warn_sibling_count_failure_once(slug: str) -> None:
    """One WARNING per sibling board per process when its count fails (MoA I7)."""
    if slug in _warned_count_boards:
        return
    _warned_count_boards.add(slug)
    logger.warning(
        "kanban provider budget: could not count running workers on sibling "
        "board %s; its workers are not counted against any provider budget "
        "this tick", slug,
    )


def host_wide_running_counts(
    conn, *,
    board: Optional[str],
    resolver: RouteKeyResolver,
    profile_exists: Optional[Callable[[str], bool]] = None,
    iter_other_boards: Optional[Callable[..., Any]] = None,
    warn: bool = True,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """D4 host-wide counting: this board + every sibling board.

    The ONE counting path shared by the dispatcher gate and the diagnostics
    snapshot (D9), so the two can never disagree. Provider quotas are
    account-wide, so per-board counting would multiply the budget by the
    number of boards. Sibling boards come from the injected
    ``iter_other_boards(board)`` iterator (fail-open per board, same posture
    as the host cap): a sibling whose count raises logs one WARNING per board
    per process and is skipped — one corrupt board cannot halt every
    provider. Returns ``(counts, inferred)``.
    """
    inferred: Dict[str, int] = {}
    counts: Dict[str, int] = dict(
        count_running_by_provider(conn, resolver, profile_exists=profile_exists, inferred=inferred))
    if iter_other_boards is not None:
        for slug, other in iter_other_boards(board):
            try:
                sib_inferred: Dict[str, int] = {}
                sib = count_running_by_provider(
                    other, resolver, profile_exists=profile_exists, inferred=sib_inferred)
                for key, n in sib.items():
                    counts[key] = counts.get(key, 0) + n
                for key, n in sib_inferred.items():
                    inferred[key] = inferred.get(key, 0) + n
            except Exception:
                if warn:
                    warn_sibling_count_failure_once(str(slug))
    return counts, inferred


def provider_budget_snapshot(
    conn, board: Optional[str], budgets: ProviderBudgets, *,
    resolver: RouteKeyResolver, profile_exists: Optional[Callable[[str], bool]] = None,
    iter_other_boards: Optional[Callable[..., Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Read-only diagnostics snapshot: per-key ``{running, inferred, cap, waiting}``.

    Running counts are HOST-WIDE — the same ``host_wide_running_counts`` path
    the dispatcher's gate uses (D9), so diagnostics can never disagree with
    enforcement; ``waiting`` counts ready/review rows on THIS board whose
    resolved key is at cap. Unlisted keys with no running/waiting rows do not
    appear.
    """
    counts, inferred = host_wide_running_counts(
        conn, board=board, resolver=resolver, profile_exists=profile_exists,
        iter_other_boards=iter_other_boards)
    waiting: Counter = Counter()
    for lane in ("ready", "review"):
        for row in conn.execute(
            "SELECT id, assignee, model_override, provider_override FROM tasks "
            f"WHERE status = '{lane}' AND claim_lock IS NULL"
        ).fetchall():
            if not row["assignee"]:
                continue
            key = resolver.resolve(row["assignee"], row["model_override"], row["provider_override"])
            if key is None:
                continue
            cap = budgets.cap_for(key)
            if cap is not None and counts.get(key, 0) >= cap:
                waiting[key] += 1
    snap: Dict[str, Dict[str, Any]] = {}
    for key in set(counts) | set(waiting) | set(budgets.caps):
        cap = budgets.cap_for(key)
        snap[key] = {
            "running": int(counts.get(key, 0)),
            "inferred": int(inferred.get(key, 0)),
            "cap": cap,
            "waiting": int(waiting.get(key, 0)),
        }
    return snap


def describe_budget_line(snap: Dict[str, Dict[str, Any]], budgets_default: Optional[int]) -> str:
    """The ``kanban.provider_concurrency:`` diagnostics line (D9)."""
    if not snap:
        return "off" if budgets_default is None and not snap else "all idle"
    parts = []
    for key in sorted(snap):
        data = snap[key]
        cap = data["cap"]
        running = data["running"]
        inferred = data["inferred"]
        waiting = data["waiting"]
        piece = f"{key} {running}"
        if cap is not None:
            piece += f"/{cap}"
        if inferred:
            piece += f" ({inferred} inferred"
            if waiting:
                piece += f"; at cap, {waiting} waiting"
            piece += ")"
        elif waiting:
            piece += f" (at cap, {waiting} waiting)"
        parts.append(piece)
    return ", ".join(parts)
