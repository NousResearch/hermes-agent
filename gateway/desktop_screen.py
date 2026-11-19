"""Desktop intake screen: automatic hold for block-tier submits on the
Desktop App Chat (W1 build queue, owner order 16 Nov 2026; CBA
section 4: placement class BLOCKING, purpose id ``desktop-screen``).

The W1 twin of the Phase 2 gateway guardrail
(``gateway/guardrail.py``, owner decision 14 Nov 2026): the same
five-question guardrail battery through the local RNR Judgement
Engine (``/v1/screen``), held BEFORE the submit is durably admitted -
no session turn runs for a held message. Wired at the SINGLE shared
choke point every surface funnels through: ``SessionAuthority._submit``
(r4, 17 Nov 2026 - the owner's live test proved the Desktop app's own
local backend ``hermes serve`` submits straight to the session
authority, so per-surface hooks at the gateway WS control plane never
saw desktop traffic). Machine producers (kanban cards, hosted rooms,
a2a, automation envelopes, cron) either use their own admission paths
or carry the server-side ``_kanban_card`` marker - none are screened.

Trust rule (r5, 17 Nov): payload FIELDS are never a trust boundary.
A submit is not exempted because it LOOKS like a CLI turn
(``cli_turn_v1`` is committed from ordinary canonical params -
``api_text`` / ``turn_author`` / ``model_override`` / ``clarify_batch``
- that the desktop app's own backend sends in normal flows). The
boundary is the session's frozen launch source, decided by the DB
row.

Scope: sessions launched from the desktop surfaces (frozen launch
source ``gui`` or ``desktop`` - the label a Desktop chat migrated
from main keeps). Configurable via ``HERMES_DESKTOP_SCREEN_SOURCES``
(comma-separated launch-source labels; use ``all`` for every
interactive source). TUI/CLI/ACP viewers stay unscreened by default
and can be added later by env alone.

Verdicts (identical semantics to Phase 2):

- ``allow``      -> proceeds (outcome ``accepted`` reported)
- ``review``     -> proceeds, logged (outcome ``escalated``)
- ``block``      -> HELD: a RuntimeStoreError refusal (reason
                    ``desktop_screen_hold``, visible hold notice as the
                    wire message) with outcome ``escalated``.
                    The sender IS the local owner, so the refused
                    submit's error toast is the alert; every hold is
                    also logged for the owner dashboard.
- ``chain_down`` -> proceeds (the engine already applied its
                    fail_mode rule; hard-three held inside the
                    engine), logged
- unreachable / malformed response -> proceeds + log. We cannot hold
  what we cannot judge, and a dead or lying engine must not lock the
  owner out of his own desktop (documented, deliberate fail-open -
  the same rule Phase 2 set for the household).

No-raise guarantee: the public entry point wraps every step; any
error degrades to ``proceed`` with a log line. All engine response
parsing is type-defensive; all numeric env values fall back to
defaults.

Event-loop safety (the Phase 2 review B4 rule): ALL HTTP work -
screen and outcome - runs off the event loop via ``asyncio.to_thread``;
the screen call additionally has an overall deadline (per-socket
timeout + 1s) so a drip-feeding peer cannot stretch it, and outcome
reports are fire-and-forget tasks that can never block or delay an
intake decision.

The judge advises, it never replaces owner gates: holds are audited
(log + outcomes), and only review-tier and chain-down passes plus
block-tier holds reach the logs the owner dashboard reads.

Configuration is environment-driven (the gateway's existing override
pattern) so the core change is inert until the operator opts in:

- HERMES_DESKTOP_SCREEN              "" (off) | "1"/"true" (on)
- HERMES_DESKTOP_SCREEN_ENDPOINT     default http://127.0.0.1:8201/v1/screen
- HERMES_DESKTOP_SCREEN_TIMEOUT_S    default 4.0 (hard cap: timeout + 1s)
- HERMES_DESKTOP_SCREEN_CALLER       default hermes-desktop
- HERMES_DESKTOP_SCREEN_PURPOSE      default desktop-screen
- HERMES_DESKTOP_SCREEN_SOURCES      default gui,desktop ("all" = every
                                      interactive launch source)
- HERMES_DESKTOP_SCREEN_HOLD_NOTICE  default visible hold text
- HERMES_DESKTOP_SCREEN_LOG          default ~/.hermes/logs/desktop-screen.log

Deployment coupling: the purpose header needs the engine-side
``/v1/screen`` purpose override (RNR-Judgement-Engine branch
``hermes/judgement-engine-integrator/w1-screen-purpose-override``);
without it the screen still works, it just attributes under
``gateway-screen`` in /metrics.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("hermes.gateway.desktop_screen")

from hermes_state_runtime import RuntimeStoreError

MAX_TEXT_CHARS = 32000  # per-chunk chars (engine rejects > 64*1024 chars)
_SEAM_OVERLAP = 1024    # carry-over; payloads up to ~1 KB cannot hide in a seam
_MAX_SCREEN_CHARS = 4_000_000  # above this the submit is HELD (never truncated)
_SCREEN_BUDGET_S = 6.0  # hard cap for one submit's total screening

_DEFAULT_ENDPOINT = "http://127.0.0.1:8201/v1/screen"
_DEFAULT_HOLD_NOTICE = ("Held by the desktop intake screen (block tier): "
                        "your message was not delivered. It matched the "
                        "injection/abuse guardrail battery; see "
                        "~/.hermes/logs/desktop-screen.log and retry "
                        "rephrased if this is a false positive.")
_DEFAULT_SOURCES = "gui,desktop"
_INTERACTIVE_SOURCES = frozenset({"cli", "tui", "gui", "desktop", "acp"})

# Fire-and-forget outcome tasks (kept referenced so CPython cannot GC
# them mid-run; discarded on completion).
_OUTCOME_TASKS: set = set()

# Review 1c-r9-F1: the 'disabled' skip record is emitted once per
# process (a default install must not append a line per CLI turn).
_DISABLED_LOGGED = False


def _log_path() -> Path:
    # Resolved per call, not at import: env overrides and test patches
    # made after import must be honoured (the guardrail rule).
    return Path(os.environ.get(
        "HERMES_DESKTOP_SCREEN_LOG",
        os.path.expanduser("~/.hermes/logs/desktop-screen.log")))


def _flag(name: str) -> str:
    return os.environ.get(name, "").strip()


def desktop_screen_enabled() -> bool:
    return _flag("HERMES_DESKTOP_SCREEN").lower() in ("1", "true", "yes", "on")


def _float_env(name: str, default: float) -> float:
    try:
        value = float(_flag(name))
        if not math.isfinite(value) or value <= 0:
            return default
        return value
    except (TypeError, ValueError):
        return default


def _effective_endpoint() -> str:
    return _flag("HERMES_DESKTOP_SCREEN_ENDPOINT") or _DEFAULT_ENDPOINT


def _outcome_endpoint() -> str:
    endpoint = _effective_endpoint()
    if "/v1/screen" in endpoint:
        return endpoint.replace("/v1/screen", "/v1/outcome")
    return endpoint.rsplit("/", 1)[0] + "/outcome"


def _caller() -> str:
    return _flag("HERMES_DESKTOP_SCREEN_CALLER") or "hermes-desktop"


def _purpose() -> str:
    return _flag("HERMES_DESKTOP_SCREEN_PURPOSE") or "desktop-screen"


def _scoped_sources() -> frozenset:
    raw = _flag("HERMES_DESKTOP_SCREEN_SOURCES") or _DEFAULT_SOURCES
    if raw.lower() == "all":
        return _INTERACTIVE_SOURCES
    labels = {part.strip() for part in raw.split(",") if part.strip()}
    return frozenset(labels & _INTERACTIVE_SOURCES) or frozenset({"gui", "desktop"})


def _hold_notice() -> str:
    return _flag("HERMES_DESKTOP_SCREEN_HOLD_NOTICE") or _DEFAULT_HOLD_NOTICE


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    headers = dict(headers, **{"Content-Type": "application/json"})
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _log(record: dict) -> None:
    record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        line = json.dumps(record, ensure_ascii=True)
    except (TypeError, ValueError):
        # A non-serialisable value (e.g. bytes text from a fuzzed
        # payload) must never lose the log line: fall back to a safe
        # rendering, never raise.
        try:
            line = json.dumps(
                {k: repr(v) for k, v in record.items()}, ensure_ascii=True)
        except Exception:
            return
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        logger.warning("desktop_screen: log write failed", exc_info=True)


def _report_outcome_sync(result: dict, outcome: str, note: str,
                         question_id) -> None:
    request_id = result.get("request_id")
    if not request_id:
        return
    _post(_outcome_endpoint(), {
        "request_id": request_id,
        "question_id": question_id,
        "outcome": outcome,
        "note": note[:200],
    }, {}, 3.0)


def _spawn_outcome_report(result: dict, outcome: str, note: str,
                          question_id) -> None:
    """Fire-and-forget, off-loop, deadline-capped outcome report.

    Never awaited by the intake path (the Phase 2 review B4 rule): a
    slow or dead outcome endpoint cannot delay or block a gate
    decision.
    """
    try:
        task = asyncio.get_running_loop().create_task(asyncio.wait_for(
            asyncio.to_thread(_report_outcome_sync, result, outcome,
                              note, question_id),
            timeout=4.0))
    except RuntimeError:
        return
    _OUTCOME_TASKS.add(task)

    def _done(finished):
        _OUTCOME_TASKS.discard(finished)
        if not finished.cancelled() and finished.exception() is not None:
            logger.debug("desktop_screen: outcome report failed",
                         exc_info=finished.exception())

    task.add_done_callback(_done)


async def drain_outcome_tasks(timeout: float = 5.0) -> None:
    """Best-effort wait for pending outcome reports (tests, clean
    shutdown, the sync chat bridge). Never raises, and never touches
    tasks bound to OTHER loops: ``_OUTCOME_TASKS`` is module-global
    while gates may run on several threads' ephemeral loops at once -
    gathering a foreign task raises "future belongs to a different
    loop", which in the chat bridge used to escape and discard an
    already-computed hold verdict (review 1c-r2-1). Only this loop's
    own tasks are awaited; done/foreign leftovers are pruned from the
    set so it cannot grow without bound."""
    if not _OUTCOME_TASKS:
        return
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        return
    mine = [t for t in list(_OUTCOME_TASKS) if t.get_loop() is running]
    if mine:
        try:
            await asyncio.wait_for(
                asyncio.gather(*mine, return_exceptions=True),
                timeout=timeout)
        except Exception:
            # A stuck/slow report is cancelled below or logged by the
            # task callback; the drain itself must never raise.
            for stuck in mine:
                stuck.cancel()
    for finished in list(_OUTCOME_TASKS):
        if finished.done():
            _OUTCOME_TASKS.discard(finished)


def _parts_texts(value) -> list:
    """Text pieces of a submit-shaped value that may be a plain string
    or a multimodal parts list (dict 'text' fields and bare strings,
    deduplicated)."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        parts: list = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                if item["text"].strip() and item["text"] not in parts:
                    parts.append(item["text"])
            elif isinstance(item, str):
                if item.strip() and item not in parts:
                    parts.append(item)
        return parts
    return []


def submit_texts(params) -> list:
    """Every model-visible text of one submit-shaped params/payload
    dict, in deterministic order, deduplicated. Reads BOTH the raw
    submit shape (top-level ``api_text`` / ``text``) AND the committed
    admission shape, where ``admit_cli_turn`` nests ``api_text`` under
    ``cli_turn_v1`` (review r5-2: the authority gate receives the
    COMMITTED payload, so a resolver that only reads top-level
    ``api_text`` is dead code at the only real call site - the hostile
    model-facing text rode in unscreened behind a benign ``text``).
    A multimodal parts list yields its text parts (cycle 1d: the API
    turn's ``user_message`` can be one). Identical values are screened
    once. Empty list when there is nothing to screen."""
    texts: list = []
    candidates = []
    if isinstance(params, dict):
        candidates.append(params.get("api_text"))
        cli = params.get("cli_turn_v1")
        if isinstance(cli, dict):
            candidates.append(cli.get("api_text"))
        candidates.append(params.get("text"))
    for value in candidates:
        for part in _parts_texts(value):
            if part not in texts:
                texts.append(part)
    return texts


def segments(text) -> list:
    """Split one model-visible text into judgeable chunks covering
    EVERY character, with a small overlap so a payload cannot hide in
    a chunk seam (reviews r6-1 truncation, r7-3 seam split, r7-1
    silent tail drop: no slice cap - oversized text is HELD by the
    caller, never truncated)."""
    value = (text or "").strip() if isinstance(text, str) else ""
    if not value:
        return []
    step = MAX_TEXT_CHARS - _SEAM_OVERLAP
    return [value[i:i + MAX_TEXT_CHARS]
            for i in range(0, len(value), step)]


async def authority_gate(authority, subject, session_id, payload, *,
                         kanban_card=False) -> None:
    """The SINGLE shared intake gate, called from
    ``SessionAuthority._submit`` before durable admission - the choke
    point EVERY viewer submit funnels through (the desktop app's own
    local backend ``hermes serve``, the gateway WS control plane, TUI,
    queued edits). The per-surface hooks this replaces missed the
    serve path entirely (the owner's 17 Nov live test: attack submits
    admitted unscreened through ``hermes serve``).

    Screens only what the desktop screen is FOR - an interactive
    viewer's own text on a desktop-launched session:

    - inert unless ``HERMES_DESKTOP_SCREEN`` is on;
    - machine producers structurally out: a kanban card marker (the
      kanban dispatcher's trusted server-side marker) skips the gate;
    - scope: the session's frozen launch source must be a screened
      desktop surface (review F4: the DB row decides, cache is
      write-through); classic-CLI and oneshot fleet sessions are
      different launch sources and stay out by scope - payload
      branding is never trusted (review r5-1: ``cli_turn_v1`` is
      committed from ordinary canonical params the desktop backend
      sends in normal flows);
    - screens EVERY model-visible text (``api_text`` AND ``text``,
      review N1), holding on the first block verdict.

    No-raise on its own bookkeeping; a block verdict raises the
    control-plane refusal (a RuntimeStoreError with reason
    ``desktop_screen_hold`` and the visible hold notice as its wire
    message) so ANY client surface - desktop app, TUI, web server -
    shows the hold.
    """
    try:
        if not desktop_screen_enabled() or kanban_card:
            return
        if not isinstance(payload, dict):
            return
        texts = submit_texts(payload)
        if not texts:
            return
        if not session_in_scope(authority, session_id):
            return
        who = str(subject or "unknown")
        # One screening budget for the WHOLE submit - every text of it
        # shares the same blocking window (review r7-5).
        deadline = time.monotonic() + _SCREEN_BUDGET_S
        for text in texts:
            held, reasons = await screen(who, session_id, text,
                                         deadline=deadline)
            if held:
                raise hold_error(reasons)
    except RuntimeStoreError as exc:
        if getattr(exc, "reason", "") == "desktop_screen_hold":
            raise  # the hold IS the verdict - never degrade it to proceed
        logger.warning("desktop_screen: authority gate degraded to "
                       "proceed", exc_info=True)
    except Exception:
        logger.warning("desktop_screen: authority gate degraded to "
                       "proceed", exc_info=True)


def _state_db_path() -> Path:
    """The default profile state DB at CALL time (runtime HERMES_HOME
    redirects honoured). Isolated for tests and for the serve path's
    read-only scope lookup."""
    from hermes_state import _default_db_path
    return _default_db_path()


def _source_from_store(db, session_id: str) -> str | None:
    """The session's frozen launch source, resolved WITHOUT a direct
    store handle when the store is a gateway client proxy (review
    1c-r4: in the Desktop serve path ``agent._session_db`` is a
    ``GatewaySessionStore`` client whose ``_read_ctx`` raises
    ``GatewayClientError`` - the gate then skipped itself silently and
    the 18 Nov live test went unscreened). The DB ROW is authoritative
    and read FIRST (module F4 rule); the client's adopted snapshot is a
    fallback only (review 1c-r5-F2: a stale/poisoned snapshot must not
    win over the row). Never raises; None means 'cannot tell'."""
    source = _source_from_db(session_id)
    if source:
        return source
    snapshots = None
    try:
        snapshots = getattr(db, "snapshots", None)
    except Exception:
        snapshots = None
    if isinstance(snapshots, dict):
        snapshot = snapshots.get(session_id)
        if isinstance(snapshot, dict):
            found_source = snapshot.get("source")
            found_created = snapshot.get("created_source")
            resolved = _resolve_label(found_source, found_created)
            if resolved:
                # Review 1c-r6: the snapshot pair goes through the SAME
                # label rule, so a re-bound local+desktop snapshot
                # rescues and a machine transport never slips in.
                return resolved
            value = snapshot.get("launch_source")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _resolve_label(source, created) -> str | None:
    """Scope label from (source, created_source): the current transport
    wins when it is a screened surface; a ``local`` re-bind keeps the
    created desktop surface (review 1c-r5-F1); machine transports stay
    out. Never raises."""
    source = (source or "").strip() if isinstance(source, str) else ""
    created = (created or "").strip() if isinstance(created, str) else ""
    if source in _scoped_sources():
        return source
    if source == "local" and created in _scoped_sources():
        return created
    return source or created or None


_SQLITE_LOOKUP_TIMEOUT_S = 0.25  # bound the locked-DB worst case (query ~1.4 ms warm)


def _source_from_db(session_id: str) -> str | None:
    """READ-ONLY state-DB lookup of the session's launch label.

    ``source`` is the current transport; ``created_source`` records the
    surface the session was created on. A session re-bound to the
    ``local`` transport keeps ``created_source='desktop'`` (review
    1c-r5-F1: 47 live archived desktop rows) - those are still desktop
    viewer sessions and must screen, while machine transports
    (oneshot/subagent/cli) stay out."""
    try:
        import sqlite3
        path = _state_db_path()
        if not path.exists():
            return None
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True,
                               timeout=_SQLITE_LOOKUP_TIMEOUT_S)
        try:
            try:
                row = conn.execute(
                    "SELECT source, created_source FROM sessions WHERE id=?",
                    (session_id,)).fetchone()
            except sqlite3.OperationalError:
                row = conn.execute(
                    "SELECT source FROM sessions WHERE id=?",
                    (session_id,)).fetchone()
                row = (row[0], None) if row else None
        finally:
            conn.close()
        if row is None:
            return None
        return _resolve_label(row[0], row[1])
    except Exception:
        logger.debug("desktop_screen: read-only source lookup failed",
                     exc_info=True)
        return None


def session_in_scope(authority, session_id: str) -> bool:
    """True when the session's frozen launch source is a screened
    desktop surface. Never raises (an unreadable source means not in
    scope, except in the Desktop backend - see below). Resolves the DB
    row FIRST and uses ``launch_sources`` only as a write-through cache
    (review F4: a stale or poisoned cache entry must fail in the
    SCREENED direction, not silently skip the gate). Three resolution
    paths (review 1c-r4): a direct store handle (``_read_ctx``), a
    gateway client proxy (adopted snapshot / read-only state-DB read),
    and - only when neither can tell - the Desktop backend process
    itself, which exists to serve its desktop viewer and must not
    silently disable the gate on the surface it protects."""
    try:
        session_id = str(session_id or "")
        if not session_id:
            return False
        sources = getattr(authority, "launch_sources", None)
        db = getattr(authority, "db", None)
        source = None
        if db is not None:
            try:
                read_ctx = getattr(db, "_read_ctx", None)
            except Exception:
                # A client proxy whose attribute access itself raises
                # must fall through to the resolver, not abort the
                # scope check (review 1c-r4 test finding).
                read_ctx = None
            if callable(read_ctx):
                try:
                    with read_ctx() as conn:
                        try:
                            found = conn.execute(
                                "SELECT source, created_source FROM sessions "
                                "WHERE id=?", (session_id,)).fetchone()
                        except Exception:
                            found = conn.execute(
                                "SELECT source FROM sessions WHERE id=?",
                                (session_id,)).fetchone()
                            found = (found[0], None) if found else None
                    if found is not None:
                        created = found[1] if len(found) > 1 else None
                        source = _resolve_label(found[0], created)
                except Exception:
                    source = None
            if not (isinstance(source, str) and source.strip()):
                source = _source_from_store(db, session_id)
        if not (isinstance(source, str) and source.strip()):
            if _flag("HERMES_DESKTOP").lower() in ("1", "true", "yes", "on"):
                logger.info("desktop_screen: session source unresolvable "
                            "in the Desktop backend; screening (process scope)")
                return True
            return False
        source = source.strip()
        if isinstance(sources, dict):
            sources[session_id] = source
        return source in _scoped_sources()
    except Exception:
        logger.warning("desktop_screen: source lookup failed; not screening",
                       exc_info=True)
        return False


async def screen(subject: str, session_id: str, text: str,
                 deadline=None) -> tuple:
    """Screen one desktop submit; returns ``(held, reasons)``. ``held``
    is True only for a block-tier verdict (or a coverage hold).
    ``deadline`` (monotonic) lets one submit share ONE screening
    budget across its several texts (review r7-5). NEVER raises:
    every failure degrades to 'proceed', logged - except the
    coverage holds, which are verdicts."""
    try:
        return await _screen_inner(subject, session_id, text,
                                   deadline=deadline)
    except Exception:
        logger.exception("desktop_screen: unexpected error; proceeding")
        return False, []


async def _screen_inner(subject: str, session_id: str, text: str,
                        deadline=None) -> tuple:
    if not desktop_screen_enabled():
        return False, []
    text = (text or "").strip()
    if not text:
        return False, []
    base = {"subject": str(subject or "unknown"),
            "session": str(session_id or ""), "preview": text[:120],
            "chars": len(text)}
    if len(text) > _MAX_SCREEN_CHARS:
        # Never truncate silently (review r7-1): text too large to
        # judge in the blocking window is HELD, not admitted screened.
        _log(dict(base, tier="oversized_unscreened"))
        return True, ["oversized_unscreened"]
    endpoint = _effective_endpoint()
    timeout = _float_env("HERMES_DESKTOP_SCREEN_TIMEOUT_S", 4.0)
    headers = {"X-Judge-Caller": _caller(), "X-Judge-Purpose": _purpose()}

    def call(chunk: str, spent: float) -> dict:
        # Per-chunk socket budget is what remains of the submit's
        # overall screening budget (review r6-1: coverage is the
        # invariant - every model-visible character is judged within
        # the blocking window, or the submit is held).
        budget = max(0.5, min(timeout, _SCREEN_BUDGET_S - spent))
        return _post(endpoint, {"text": chunk}, headers, budget)

    chunks = segments(text)
    if deadline is None:
        deadline = time.monotonic() + _SCREEN_BUDGET_S
    judged = 0
    result = None
    tail_reasons: list = []
    for index, chunk in enumerate(chunks):
        if not chunk.strip():
            # Review r8-HIGH: a whitespace-only chunk carries no
            # payload, and sending it earns a deterministic engine
            # 400 that used to fail the whole tail open. Skip it.
            _log(dict(base, tier="blank_chunk_skipped", chunk=index))
            continue
        spent = _SCREEN_BUDGET_S - (deadline - time.monotonic())
        if time.monotonic() >= deadline:
            # Coverage invariant: a tail we could not judge in budget
            # is HELD (fail-closed for coverage - unlike an
            # unreachable engine, which stays fail-open per Phase 2).
            _log(dict(base, tier="screening_incomplete_budget",
                      chunk=index, judged=judged, chunks=len(chunks)))
            return True, ["screening_incomplete_budget"]
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(call, chunk, spent),
                timeout=max(0.5, min(timeout, _SCREEN_BUDGET_S - spent)) + 1.0)
            if not isinstance(result, dict):
                raise ValueError(f"non-dict engine response: {type(result)!r}")
        except Exception as err:
            # Review r8-HIGH (defence in depth): a deterministic client
            # rejection (4xx) means the engine CANNOT judge this chunk -
            # that is a coverage hole, so HOLD. A transport failure at
            # the budget boundary is coverage exhaustion, also HOLD.
            # Only a genuine mid-budget engine failure stays fail-open
            # (Phase 2: a dead engine must not lock the owner out).
            rejected = (isinstance(err, urllib.error.HTTPError)
                        and 400 <= int(getattr(err, "code", 0) or 0) < 500)
            transient = (isinstance(err, urllib.error.HTTPError)
                         and int(getattr(err, "code", 0) or 0) in (408, 429))
            if transient:
                # 408/429 are TRANSIENT (proxy timeout, rate limit): the
                # engine is alive but busy. A rate-limited engine must
                # not lock the owner out, so this joins the documented
                # fail-open class (Phase 2), logged distinctly.
                _log(dict(base, tier="engine_rate_limited_or_timeout",
                          chunk=index, judged=judged, chunks=len(chunks),
                          error=str(err)[:120]))
                return False, []
            if rejected:
                _log(dict(base, tier="engine_rejected_chunk",
                          chunk=index, judged=judged, chunks=len(chunks),
                          error=str(err)[:120]))
                return True, ["engine_rejected_chunk"]
            if time.monotonic() >= deadline - 0.5:
                _log(dict(base, tier="screening_incomplete_budget",
                          chunk=index, judged=judged, chunks=len(chunks),
                          error=str(err)[:120]))
                return True, ["screening_incomplete_budget"]
            _log(dict(base, tier="engine_unreachable_or_malformed",
                      chunk=index, judged=judged, chunks=len(chunks),
                      error=str(err)[:160]))
            return False, []
        judged += 1
        tier = result.get("tier")
        tier = tier if isinstance(tier, str) else None
        raw_reasons = result.get("reasons")
        if not isinstance(raw_reasons, list):
            raw_reasons = []
        reasons = [str(r) for r in raw_reasons
                   if isinstance(r, str) and r.strip()][:8]
        served_by = result.get("served_by")
        served_by = (served_by.get("provider")
                     if isinstance(served_by, dict) else None)
        scores = (result.get("scores")
                  if isinstance(result.get("scores"), dict) else {})
        _log(dict(base, tier=tier, reasons=reasons, chunk=index,
                  judged=judged, chunks=len(chunks),
                  served_by=served_by, scores=scores))
        question_id = ("q_" + reasons[0]) if reasons else None
        if tier == "block":
            _spawn_outcome_report(
                result, "escalated",
                "block-tier held automatically (desktop screen)", question_id)
            return True, reasons
        if tier in ("review", "chain_down"):
            _spawn_outcome_report(
                result, "escalated",
                f"{tier}-tier (desktop screen advisory pass)", question_id)
            tail_reasons = reasons
        else:
            # Review r7-4: report the outcome for EVERY judged chunk,
            # not just the last one (the usefulness join keys are
            # per-judgement).
            _spawn_outcome_report(
                result, "accepted", "allow-tier (desktop screen)", None)
    return False, tail_reasons


def hold_error(reasons) -> Exception:
    """The control-plane refusal for a held submit: a RuntimeStoreError
    subclass whose wire message is the owner-visible hold notice (the
    dispatcher's error frame carries ``wire_message``; the frame's data
    block is the fixed refusal shape and does NOT carry the tier
    reasons - the reasons live in the screen log), so the desktop
    client shows why the message was not delivered."""
    from hermes_state_runtime import RuntimeStoreError

    class _Hold(RuntimeStoreError):
        wire_code = 4001

        def __init__(self, tier_reasons):
            self.reasons = [str(r) for r in (tier_reasons or [])][:8]
            self.wire_message = _hold_notice()
            super().__init__("desktop_screen_hold")

    return _Hold(reasons)


# ---------------------------------------------------------------------------
# Desktop serve chat gate (cycle 1c, owner-approved 17 Nov 2026)
#
# The Desktop app's isolated ``hermes serve`` backend runs viewer turns
# through ``hermes_cli/cli_chat_turn_mixin.chat`` - a path that never
# reaches the gateway admission gate (``SessionAuthority._submit``),
# which is why the 17 Nov live test passed an injection through
# unscreened. ``chat_gate`` screens the SAME model-visible texts with
# the SAME engine gate for desktop/gui-sourced sessions, before the
# user message is staged or any model call is made.


class _ChatAuthority:
    """Minimal authority adapter for the serve chat path: the gate only
    needs the session store (``session_in_scope`` reads the frozen
    launch source from it)."""

    def __init__(self, db):
        self.db = db
        self.launch_sources = {}


def _message_texts(message) -> list:
    """Model-visible text pieces of one chat() message (viewer strings
    and multimodal text parts, deduplicated). ``TimelineNotification``
    rows are machine-produced UI metadata (never viewer input - and a
    str SUBCLASS in real code, so they are excluded BEFORE the str
    check) and images are A6-deferred, so neither is screened."""
    try:
        from tools.process_registry_notifications import TimelineNotification
        if isinstance(message, TimelineNotification):
            return []
    except Exception:
        pass
    return _parts_texts(message)


def _screen_texts_sync(subject: str, session_id: str, texts: list) -> tuple:
    """Sync bridge for the non-async serve chat thread. Shares ONE
    screening budget across the message's texts and reuses the SAME
    async ``screen`` so chunking, budget, tiers, outcome reporting and
    logging are identical to the gateway gate. Returns
    ``(held, reasons)``. When the caller is inside a running event
    loop, the screen runs on a dedicated thread with its own loop
    (the chat path may be invoked from async serve code)."""
    if not texts:
        return False, []

    def _run() -> tuple:
        deadline = time.monotonic() + _SCREEN_BUDGET_S

        async def _screen_all() -> tuple:
            held, reasons = False, []
            for text in texts:
                held, reasons = await screen(
                    subject, session_id, text, deadline=deadline)
                if held:
                    break
            # Outcome reports are fire-and-forget tasks bound to THIS
            # loop; asyncio.run cancels pending tasks on exit, which
            # would silently lose desktop-chat hold/allow telemetry.
            # The drain is per-loop (review 1c-r2-1) and must never be
            # able to discard the verdict already computed.
            try:
                await drain_outcome_tasks(timeout=2.0)
            except Exception:
                pass
            return held, reasons

        return asyncio.run(_screen_all())

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result()


def _resolve_scope(stores: list, session_id: str):
    """Chat-path scope: row first from ANY usable store (review 1c-r9-F3:
    ``or`` picked the first truthy store, not the one that can answer),
    then the read-only state DB, then each store's adopted snapshot.
    A resolved label decides immediately (row authoritative). Returns
    True/False when resolved, None when nothing can tell (the caller
    then applies the Desktop-backend process scope)."""
    for store in stores:
        try:
            read_ctx = getattr(store, "_read_ctx", None)
        except Exception:
            read_ctx = None
        if callable(read_ctx):
            try:
                with read_ctx() as conn:
                    try:
                        found = conn.execute(
                            "SELECT source, created_source FROM sessions "
                            "WHERE id=?", (session_id,)).fetchone()
                    except Exception:
                        found = conn.execute(
                            "SELECT source FROM sessions WHERE id=?",
                            (session_id,)).fetchone()
                        found = (found[0], None) if found else None
                if found is not None:
                    label = _resolve_label(found[0],
                                           found[1] if len(found) > 1 else None)
                    if label:
                        return label in _scoped_sources()
            except Exception:
                pass
    label = _source_from_db(session_id)
    if label:
        return label in _scoped_sources()
    for store in stores:
        label = _source_from_store(store, session_id)
        if label:
            return label in _scoped_sources()
    return None


def chat_gate(chat, agent, message) -> str | None:
    """Screen one Desktop serve chat turn; returns the hold notice text
    when the submit is held (the caller shows it instead of running
    the turn), None when the turn may proceed. NEVER raises: every
    failure degrades to 'proceed', logged. Every skip writes a
    ``skipped_<reason>`` record to the screen log (review 1c-r8: a
    silent skip hid a live gate miss twice)."""
    reason = None
    try:
        if not desktop_screen_enabled():
            reason = "disabled"
            return None
        session_id = str(getattr(chat, "session_id", "")
                         or getattr(agent, "session_id", "") or "")
        if not session_id:
            reason = "no_session"
            return None
        # The CLI chat object holds the store itself (self._session_db);
        # the agent's copy can be None in the serve flow (review 1c-r8).
        stores = []
        for candidate in (getattr(agent, "_session_db", None),
                          getattr(chat, "_session_db", None)):
            if candidate is not None and all(candidate is not s
                                             for s in stores):
                stores.append(candidate)
        if not stores:
            # Review 1c-r9-F6: no store at all in the Desktop backend
            # still screens under process scope (the secure direction);
            # every other process skips, logged.
            if _flag("HERMES_DESKTOP").lower() in ("1", "true", "yes", "on"):
                texts = _message_texts(message)
                if not texts:
                    reason = "no_text"
                    return None
                reason = "process_scope_no_store"
                held, reasons = _screen_texts_sync(
                    "desktop-chat", session_id, texts)
                if not held:
                    return None
                shown = "; ".join(reasons[:3]) if reasons else "block tier"
                return f"{_hold_notice()} [{shown}]"
            reason = "no_store"
            return None
        scope = _resolve_scope(stores, session_id)
        if scope is None:
            # Cannot tell: the Desktop backend screens under process
            # scope (r4 behaviour, restored in r9); elsewhere skip.
            if _flag("HERMES_DESKTOP").lower() not in ("1", "true", "yes", "on"):
                reason = "out_of_scope"
                return None
        elif not scope:
            reason = "out_of_scope"
            return None
        texts = _message_texts(message)
        if not texts:
            reason = "no_text"
            return None
        held, reasons = _screen_texts_sync("desktop-chat", session_id, texts)
        if not held:
            return None
        shown = "; ".join(reasons[:3]) if reasons else "block tier"
        return f"{_hold_notice()} [{shown}]"
    except Exception:
        logger.warning("desktop_screen: chat gate failed; proceeding",
                       exc_info=True)
        reason = "error"
        return None
    finally:
        if reason is not None:
            try:
                # Review 1c-r9-F1/F2: skip records carry the REASON ONLY
                # - never message text (a default install must stay
                # inert, and machine notifications must never be
                # persisted). Review F5: the error path is logged too.
                global _DISABLED_LOGGED
                emit = True
                if reason == "disabled":
                    emit = not _DISABLED_LOGGED
                    _DISABLED_LOGGED = True
                if emit:
                    session = ""
                    try:
                        session = str(getattr(chat, "session_id", "") or "")
                    except Exception:
                        pass
                    _log({"subject": "desktop-chat",
                          "session": session,
                          "tier": f"skipped_{reason}"})
            except Exception:
                pass


# ---------------------------------------------------------------------------
# API-turn admission gate (cycle 1d, owner-approved 19 Nov 2026)
#
# The Desktop app's turns enter through the serve's HTTP api_server ->
# gateway/session_api_turn.py, which admits via admit_session_input
# DIRECTLY (principal 'api') - bypassing SessionAuthority._submit (seam
# 1) and cli_chatTurnMixin.chat (seam 2). Proven live 19 Nov: the
# owner's desktop test session produced no gate activity on either
# gated seam. This is seam 3.


def api_turn_gate(authority, session_id, payload) -> None:
    """Screen one API-turn admission BEFORE ``admit_session_input``
    writes it. Same engine, chunking, budget, tiers, logging and
    outcome reporting as the other seams; raises the control-plane
    hold (RuntimeStoreError reason ``desktop_screen_hold``) on a block
    verdict so the HTTP surface returns the admission refusal.
    NEVER raises otherwise; every skip writes a ``skipped_<reason>``
    record (reason only, never text) with subject ``desktop-api``."""
    reason = None
    try:
        if not desktop_screen_enabled():
            reason = "disabled"
            return
        texts = submit_texts(payload)
        if not texts:
            reason = "no_text"
            return
        if not session_in_scope(authority, session_id):
            reason = "out_of_scope"
            return
        held, reasons = _screen_texts_sync("desktop-api", session_id, texts)
        if not held:
            return
        raise hold_error(reasons)
    except RuntimeStoreError:
        raise
    except Exception:
        logger.warning("desktop_screen: api turn gate failed; proceeding",
                       exc_info=True)
        reason = "error"
        return
    finally:
        if reason is not None:
            try:
                global _DISABLED_LOGGED
                emit = True
                if reason == "disabled":
                    emit = not _DISABLED_LOGGED
                    _DISABLED_LOGGED = True
                if emit:
                    session = ""
                    try:
                        session = str(session_id or "")
                    except Exception:
                        pass
                    _log({"subject": "desktop-api",
                          "session": session,
                          "tier": f"skipped_{reason}"})
            except Exception:
                pass
