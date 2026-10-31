"""``hermes wall`` — broadcast one message to every registered agent.

``wall`` has unix ``wall(1)`` semantics and nothing narrower: ONE message, EVERY
registered agent, no exceptions. It exists because an operator about to take a
shared service down needs one command whose *exit status* answers "did the whole
fleet get the notice?", not "did the HTTP call return 200?".

Design boundaries:

- **No delivery machinery of its own.** Each target is delivered through the
  existing peer transport (``hermes peer dm <peer>/<profile> --wait N --json``),
  which owns idempotency-key derivation, retry/replay and admission. ``wall``
  adds enumeration, settle verification and reporting — nothing else.
- **Enumerate, never assume.** The roster comes from the gateway's own registry
  chokepoint, :func:`hermes_cli.profiles.profiles_to_serve` (``multiplex=True``):
  default plus every live named profile. A profile directory that exists but
  that the gateway does not serve is reported as ``unroutable``, never skipped
  silently.
- **Verify the settle.** A send is not a delivery. A target counts as reached
  only when the transport reports ``result=delivered`` or the delivery ledger
  (``tools.bot_delivery_queue``) holds a terminal ``delivered`` row for it. An
  unanswered receipt is retried once, then reported as NOT reached — which is
  what makes the process exit status worth reading.
- **Chunk, never truncate.** The peer transport accepts an unbounded body
  (``gateway/platforms/api_server.py`` type-checks the string and nothing more),
  so what clips a long body is the *producer*, and a clipped body ends in a
  truncation marker. ``wall`` refuses a body that already carries a marker and
  splits anything over :data:`CHUNK_MAX_CHARS` into ordinal-labelled parts, so a
  recipient never gets half a sentence presented as a whole one.

Exit codes: ``0`` every registered agent reached; ``1`` at least one agent was
not reached; ``2`` the broadcast was refused before anything was sent (empty
body, a body that is already clipped, or a body too large to chunk).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

#: Largest body delivered in one part. The peer transport itself imposes no cap;
#: this is the *producer* budget — a long body pushed through a shell argv or a
#: tool argument is what comes back clipped. Keep it comfortably above a normal
#: notice and comfortably below the point where a caller starts clipping.
CHUNK_MAX_CHARS = 1000

#: A broadcast needing more parts than this is not a notice, it is a document —
#: refuse rather than turn one operator action into dozens of agent turns.
MAX_CHUNKS = 5

#: Substrings that mean "this text was clipped by something upstream". A body
#: carrying one is refused outright: forwarding it would broadcast an
#: announcement that is silently missing its middle.
TRUNCATION_MARKERS = ("[truncated]", "[output truncated]", "[clipped]")

#: Wait before the single retry of an unsettled target.
RETRY_DELAY_S = 10.0

#: Targets delivered at once. A wall is simultaneous by nature (that is the
#: point of ``wall(1)``), and the gateway's api_server admits concurrent runs.
DEFAULT_MAX_WORKERS = 8

# Terminal delivery states (mirrors tools.bot_delivery_queue).
STATE_DELIVERED = "delivered"
STATE_QUEUED = "queued"
STATE_FAILED = "failed"
STATE_UNKNOWN = "unknown"
STATE_REFUSED = "refused"
STATE_UNREACHABLE = "unreachable"
STATE_UNROUTABLE = "unroutable"

#: States that count as "this agent got the message".
REACHED_STATES = frozenset({STATE_DELIVERED})

#: The transport's own result enum -> wall state. ``receipt`` means the peer
#: admitted the message but has not run the target's turn yet: unsettled.
_RESULT_TO_STATE = {
    "delivered": STATE_DELIVERED,
    "receipt": STATE_QUEUED,
    "failed": STATE_FAILED,
    "unknown": STATE_UNKNOWN,
    "refused": STATE_REFUSED,
}

#: A ledger row's terminal status -> wall state (tools.bot_delivery_queue).
_LEDGER_TO_STATE = {
    "delivered": STATE_DELIVERED,
    "failed": STATE_FAILED,
    "expired": STATE_FAILED,
    "cancelled": STATE_FAILED,
    "ambiguous": STATE_UNKNOWN,
}


class WallRefused(Exception):
    """The broadcast was refused before anything was sent."""


class WallUnavailable(Exception):
    """The host cannot broadcast at all (no peer, no registry)."""


@dataclass(frozen=True)
class Agent:
    """One registered agent: its profile name, its home, and its peer target."""

    name: str
    home: Path
    target: str


@dataclass
class Outcome:
    """What happened to one agent."""

    target: str
    state: str
    attempts: int = 0
    delivery_id: str = ""
    detail: str = ""

    @property
    def reached(self) -> bool:
        return self.state in REACHED_STATES


# --------------------------------------------------------------------------- #
# body planning
# --------------------------------------------------------------------------- #

def truncation_marker(message: str) -> Optional[str]:
    """The truncation marker *message* carries, or ``None`` if it is intact."""
    lowered = (message or "").lower()
    for marker in TRUNCATION_MARKERS:
        if marker in lowered:
            return marker
    return None


#: Clause endings a part may stop at, in preference order. A part that stops
#: mid-clause cannot be told apart from a transport clip, so a split prefers
#: sentence punctuation over a paragraph break, and a hand-wrapped newline (which
#: lands mid-sentence and renders as a space) last of all.
_SENTENCE_ENDINGS = (". ", "; ", ": ", "! ", "? ")
_PARAGRAPH_ENDINGS = ("\n\n",)
_LINE_ENDINGS = ("\n",)

#: Do not back up further than this fraction of the window hunting a clause end.
_CLAUSE_FLOOR_DIVISOR = 3

#: Budget held back from every part: its ordinal prefix, plus the explicit
#: continuation marker a non-final part carries.
_ORDINAL_RESERVE = len("(99/99) ")
_CONTINUATION_TEMPLATE = " [continues in {n}/{total}]"


def _cut_point(text: str, limit: int) -> int:
    """Where to cut *text* so the first part is at most *limit* chars.

    The last sentence boundary inside the window wins; then a paragraph break,
    then a bare line break, then a whitespace break. A part that ends mid-clause
    reads as a clipped message even when the split was deliberate, which is the
    ambiguity the ordinal alone does not resolve.
    """
    window = text[:limit]
    floor = max(1, limit // _CLAUSE_FLOOR_DIVISOR)
    for endings in (_SENTENCE_ENDINGS, _PARAGRAPH_ENDINGS, _LINE_ENDINGS):
        best = 0
        for ending in endings:
            idx = window.rfind(ending)
            if idx != -1:
                end = idx + len(ending)
                if end >= floor and end > best:
                    best = end
        if best:
            return best
    idx = window.rfind(" ")
    return idx if idx > 0 else 0


def _split_clause_aware(text: str, room: int) -> list[str]:
    """Split *text* into parts of at most *room* chars, at clause boundaries."""
    parts: list[str] = []
    rest = text
    while len(rest) > room:
        cut = _cut_point(rest, room)
        if cut <= 0:
            break
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        parts.append(rest)
    return parts


def plan_chunks(
    message: str,
    *,
    max_chunk_chars: int = CHUNK_MAX_CHARS,
    max_chunks: int = MAX_CHUNKS,
) -> list[str]:
    """Split *message* into ordered parts, or raise :class:`WallRefused`.

    A body that fits is returned unchanged (the common case: announcements are
    short). A longer body is split at CLAUSE boundaries — falling back to a
    whitespace break only when the window holds no clause end — and every part
    is prefixed with its ordinal, ``(2/3) ...``, so the recipient can see where
    in the message it sits. Every non-final part also carries an explicit
    continuation marker, ``[continues in 3/3]``, because the parts arrive as
    separate messages in separate turns: while a recipient holds only part one,
    a mid-clause ending is otherwise indistinguishable from a transport clip.
    Splitting normalises runs of whitespace to a single space, so a chunked body
    is word-for-word identical to what was passed but is not byte-for-byte
    identical when it contained newlines; a body that fits is never touched.
    """
    body = (message or "").strip()
    if not body:
        raise WallRefused("empty message: nothing to broadcast")
    marker = truncation_marker(body)
    if marker:
        raise WallRefused(
            f"message carries a truncation marker ({marker!r}) — it was clipped "
            f"before it reached wall, and broadcasting it would deliver a notice "
            f"that is silently missing its middle. Re-build the body from the "
            f"original source and pass it intact (or pipe it on stdin).")
    if len(body) <= max_chunk_chars:
        return [body]

    room = max_chunk_chars - _ORDINAL_RESERVE - len(
        _CONTINUATION_TEMPLATE.format(n=99, total=99))
    if room < 32:
        raise WallRefused(
            f"chunk budget too small: max_chunk_chars={max_chunk_chars} leaves "
            f"{room} chars for the text of a part once its ordinal and the "
            f"continuation marker are reserved")
    parts = _split_clause_aware(body, room)
    oversized = [p for p in parts if len(p) > room]
    if oversized:
        raise WallRefused(
            f"a single token is {len(oversized[0])} chars, over the {room}-char "
            f"chunk budget; shorten it or raise wall.chunk_max_chars")
    if len(parts) > max_chunks:
        raise WallRefused(
            f"message is {len(body)} chars => {len(parts)} parts at "
            f"{max_chunk_chars} chars/part, over the {max_chunks}-part limit for one "
            f"broadcast; shorten it, raise wall.max_chunks, or use 'hermes send'")
    if len(parts) == 1:
        return [body]
    total = len(parts)
    rendered: list[str] = []
    for index, part in enumerate(parts, start=1):
        prefix = f"({index}/{total}) "
        if index < total:
            part = f"{part}{_CONTINUATION_TEMPLATE.format(n=index + 1, total=total)}"
        rendered.append(f"{prefix}{part}")
    return rendered


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

def _default_home() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home())


def peer_name() -> str:
    """The registered peer that fronts this host's own agents.

    One peer (the local gateway) is the normal case; more than one is ambiguous
    — ``wall`` would be guessing which fleet "every registered agent" means — so
    that is refused rather than resolved arbitrarily.
    """
    from hermes_cli.config import load_config

    peers = (load_config() or {}).get("bot_peers") or {}
    if not isinstance(peers, dict) or not peers:
        raise WallUnavailable(
            "no bot_peers registered (hermes peer add <name> --url <url>); wall has "
            "no transport to broadcast over")
    if len(peers) > 1:
        raise WallUnavailable(
            f"{len(peers)} peers registered ({', '.join(sorted(peers))}); wall needs "
            f"exactly one to mean 'every registered agent'")
    return next(iter(peers))


def registry() -> tuple[list[Agent], list[str]]:
    """``(served agents, registered-but-not-served profile names)``.

    The served list is the gateway's own answer (:func:`profiles_to_serve`), so
    ``wall`` and the gateway can never disagree about who is part of the fleet.
    """
    from hermes_cli.profiles import profiles_to_serve

    peer = peer_name()
    served = list(profiles_to_serve(multiplex=True))
    agents = [Agent(name=name, home=Path(home), target=f"{peer}/{name}")
              for name, home in served]
    served_names = {agent.name for agent in agents}

    unserved: list[str] = []
    profiles_root = _default_home() / "profiles"
    if profiles_root.is_dir():
        for entry in sorted(os.listdir(profiles_root)):
            if entry.startswith(".") or entry in served_names:
                continue
            if (profiles_root / entry).is_dir():
                unserved.append(entry)
    return agents, unserved


# --------------------------------------------------------------------------- #
# delivery
# --------------------------------------------------------------------------- #

def _peer_command() -> list[str]:
    """The ``hermes`` entry point used to reach the peer transport."""
    exe = shutil.which("hermes")
    if exe:
        return [exe]
    return [sys.executable, "-m", "hermes_cli.main"]


def _last_json_object(text: str) -> Optional[dict]:
    """The last JSON object printed on *text* (the peer verb's ``--json`` line)."""
    for line in reversed((text or "").splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _run_peer_dm(target: str, chunk: str, wait_seconds: Optional[float],
                 timeout: Optional[float]) -> tuple[int, Optional[dict], str]:
    """Deliver one chunk via ``hermes peer dm`` and return ``(rc, envelope, stderr)``."""
    cmd = _peer_command() + ["peer", "dm", target, chunk, "--json"]
    if wait_seconds is not None:
        cmd += ["--wait", str(wait_seconds)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 1, None, f"peer dm exceeded its {timeout}s local budget"
    except OSError as exc:
        return 1, None, f"could not run the peer transport: {exc}"
    return proc.returncode, _last_json_object(proc.stdout), (proc.stderr or "").strip()


def _state_from_attempt(rc: int, envelope: Optional[dict], stderr: str) -> tuple[str, str]:
    """Map one peer-transport attempt onto ``(state, detail)``."""
    if envelope is None:
        # No envelope: the transport never completed a request. A 404 from the
        # multiplexed gateway means the profile is not served, which is a
        # different (and actionable) failure from an unreachable gateway.
        detail = stderr.splitlines()[-1] if stderr else f"peer dm exited {rc}"
        if "HTTP 404" in detail:
            return STATE_UNROUTABLE, detail
        if rc == 2:
            return STATE_REFUSED, detail
        return STATE_UNREACHABLE, detail

    result = str(envelope.get("result") or "")
    if result:
        state = _RESULT_TO_STATE.get(result, STATE_UNKNOWN)
        detail = str(envelope.get("detail") or envelope.get("reason")
                     or envelope.get("error") or result)
        return state, detail
    # Legacy peer: no delivery envelope, just a reply.
    if rc == 0 and "reply" in envelope:
        return STATE_DELIVERED, "reply received (peer predates the delivery envelope)"
    if rc == 2:
        return STATE_REFUSED, str(envelope.get("error") or "peer refused the request")
    return STATE_UNKNOWN, "peer returned no delivery result"


def ledger_state(home: Path, delivery_id: str) -> str:
    """The wall state the delivery ledger holds for *delivery_id*, else ``''``.

    The ledger is the settle authority: an unknown or unreadable row is *not*
    evidence of delivery, so it can only ever confirm a reach, never invent one.
    """
    if not delivery_id:
        return ""
    try:
        from tools.bot_delivery_queue import read_record

        record = read_record(home, delivery_id) or {}
    except Exception:  # a ledger we cannot read is not a delivery we can claim
        return ""
    return _LEDGER_TO_STATE.get(str(record.get("status") or ""), "")


def deliver_agent(
    agent: Agent,
    chunks: Sequence[str],
    *,
    wait_seconds: Optional[float] = None,
    runner: Callable[[str, str, Optional[float], Optional[float]],
                     tuple[int, Optional[dict], str]] = _run_peer_dm,
    retry_delay: float = RETRY_DELAY_S,
    sleep: Callable[[float], None] = time.sleep,
    timeout: Optional[float] = None,
) -> Outcome:
    """Deliver every chunk of the broadcast to one agent, verifying the settle."""
    attempts = 0
    delivery_id = ""
    detail = ""
    state = STATE_UNKNOWN
    for index, chunk in enumerate(chunks, start=1):
        state, chunk_detail, chunk_attempts, chunk_id = _deliver_chunk(
            agent, chunk, wait_seconds=wait_seconds, runner=runner,
            retry_delay=retry_delay, sleep=sleep, timeout=timeout)
        attempts += chunk_attempts
        detail = chunk_detail
        delivery_id = chunk_id or delivery_id
        if state not in REACHED_STATES:
            if len(chunks) > 1:
                detail = f"part {index}/{len(chunks)} not reached: {chunk_detail}"
            return Outcome(target=agent.target, state=state, attempts=attempts,
                           delivery_id=delivery_id, detail=detail)
    return Outcome(target=agent.target, state=state, attempts=attempts,
                   delivery_id=delivery_id, detail=detail)


def _deliver_chunk(
    agent: Agent,
    chunk: str,
    *,
    wait_seconds: Optional[float],
    runner,
    retry_delay: float,
    sleep,
    timeout: Optional[float],
) -> tuple[str, str, int, str]:
    """One chunk: send, verify the settle, retry once if it is unsettled."""
    attempts = 0
    last_detail = ""
    delivery_id = ""
    state = STATE_UNKNOWN
    for attempt in range(1, 3):
        attempts = attempt
        rc, envelope, stderr = runner(agent.target, chunk, wait_seconds, timeout)
        state, detail = _state_from_attempt(rc, envelope, stderr)
        delivery_id = str((envelope or {}).get("delivery_id") or "") or delivery_id
        settled = ledger_state(agent.home, delivery_id)
        if settled:
            return settled, detail, attempts, delivery_id
        if state in REACHED_STATES:
            return state, detail, attempts, delivery_id
        last_detail = detail
        if attempt == 1:
            # "Unsettled" covers an unanswered receipt and a request that never
            # completed; both are worth exactly one more try. A refusal is a
            # defect in the request, so retrying it would only repeat the flaw.
            if state in {STATE_REFUSED, STATE_UNROUTABLE, STATE_FAILED}:
                return state, detail, attempts, delivery_id
            sleep(retry_delay)
    return state, last_detail, attempts, delivery_id


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def _emit_roster(outcome: Outcome, out) -> None:
    fields = [f"target={outcome.target}", f"state={outcome.state}",
              f"attempts={outcome.attempts}"]
    if outcome.detail:
        fields.append(f"detail={json.dumps(outcome.detail)}")
    print("wall " + " ".join(fields), file=out)


def run_wall(
    message: str,
    *,
    wait_seconds: Optional[float] = None,
    reason: str = "",
    out=None,
    err=None,
    runner=_run_peer_dm,
    sleep: Callable[[float], None] = time.sleep,
    registry_fn: Callable[[], tuple[list[Agent], list[str]]] = registry,
    max_workers: int = DEFAULT_MAX_WORKERS,
    retry_delay: float = RETRY_DELAY_S,
    timeout: Optional[float] = None,
    as_json: bool = False,
    chunk_max_chars: int = CHUNK_MAX_CHARS,
    max_chunks: int = MAX_CHUNKS,
) -> int:
    """Broadcast *message* to every registered agent; return the exit code."""
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    try:
        chunks = plan_chunks(message, max_chunk_chars=chunk_max_chars,
                             max_chunks=max_chunks)
    except WallRefused as exc:
        print(f"wall: refused — {exc}", file=err)
        return 2
    try:
        agents, unserved = registry_fn()
    except WallUnavailable as exc:
        print(f"wall: {exc}", file=err)
        return 2
    if not agents:
        print("wall: no registered agents (nothing to broadcast to)", file=err)
        return 2

    target_count = len(agents)
    if not as_json:
        header = f"wall: {target_count} registered agents"
        if len(chunks) > 1:
            header += f", {len(chunks)} parts ({len(message.strip())} chars)"
        if reason:
            header += f" — reason: {reason}"
        print(header, file=out)

    outcomes: list[Outcome] = []
    workers = max(1, min(max_workers, target_count))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(deliver_agent, agent, chunks, wait_seconds=wait_seconds,
                               runner=runner, retry_delay=retry_delay, sleep=sleep,
                               timeout=timeout)
                   for agent in agents]
        for future in futures:
            outcomes.append(future.result())

    reached = [o for o in outcomes if o.reached]
    unreachable = [o for o in outcomes
                   if o.state in {STATE_UNREACHABLE, STATE_UNROUTABLE, STATE_QUEUED}]
    exit_code = 0 if len(reached) == target_count and not unserved else 1

    if as_json:
        print(json.dumps({
            "object": "hermes.wall.result",
            "reason": reason or None,
            "agents_total": target_count,
            "chunks": len(chunks),
            "reached": len(reached),
            "unreachable": len(unreachable),
            "unserved_profiles": unserved,
            "exit_code": exit_code,
            "roster": [
                {"target": o.target, "state": o.state, "attempts": o.attempts,
                 "delivery_id": o.delivery_id or None, "detail": o.detail or None}
                for o in outcomes
            ],
        }), file=out)
        return exit_code

    for outcome in outcomes:
        _emit_roster(outcome, out)
    for name in unserved:
        print(f"wall target={name} state={STATE_UNROUTABLE} attempts=0 "
              f"detail=\"registered profile not served by the gateway, no exception\"",
              file=out)
    summary = (f"wall: reached={len(reached)}/{target_count} "
               f"unreachable={len(unreachable)}")
    if unserved:
        summary += f" unroutable={len(unserved)}"
    summary += f" exit={exit_code}"
    print(summary, file=out)
    return exit_code


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _wall_config() -> dict:
    from hermes_cli.config import load_config

    section = (load_config() or {}).get("wall")
    return section if isinstance(section, dict) else {}


def cmd_wall(args) -> int:
    message = (getattr(args, "message", None) or "").strip()
    if not message and not sys.stdin.isatty():
        message = sys.stdin.read().strip()
    if not message:
        print("Message required (argument or stdin).", file=sys.stderr)
        return 2
    config = _wall_config()
    wait_seconds = getattr(args, "wait_seconds", None)
    if wait_seconds is not None and wait_seconds < 0:
        print("--wait must be >= 0 seconds.", file=sys.stderr)
        return 2
    return run_wall(
        message,
        wait_seconds=wait_seconds,
        reason=(getattr(args, "reason", "") or "").strip(),
        as_json=bool(getattr(args, "json", False)),
        retry_delay=float(config.get("retry_delay_seconds", RETRY_DELAY_S)),
        timeout=config.get("peer_timeout_seconds") or None,
        max_workers=int(config.get("max_workers", DEFAULT_MAX_WORKERS)),
        chunk_max_chars=int(config.get("chunk_max_chars", CHUNK_MAX_CHARS)),
        max_chunks=int(config.get("max_chunks", MAX_CHUNKS)),
    )


def build_wall_parser(subparsers) -> None:
    """Attach the ``wall`` subcommand to *subparsers*."""
    parser = subparsers.add_parser(
        "wall", help="Broadcast one message to every registered agent",
        description=(
            "Send ONE message to EVERY registered agent (unix wall(1), fleet-wide). "
            "The roster is the gateway's own profile registry — default plus every "
            "live named profile — delivered through the existing peer transport, and "
            "each target is reported with the state that was actually verified "
            "(delivered / queued / unreachable / unroutable), never a bare send.\n\n"
            "Exit 0 only when every registered agent is confirmed reached; 1 when "
            "any agent is not; 2 when the broadcast is refused before sending."),
        epilog=(
            "Examples:\n"
            "  hermes wall \"gateway restarting in 5 minutes\"\n"
            '  hermes wall "switching model pins, expect a slow turn" --reason deploy\n'
            "  hermes wall --wait 600 < notice.txt\n"
            "  hermes wall \"short note\" --json\n\n"
            "A body containing a truncation marker is refused (it was already "
            "clipped) and a body over wall.chunk_max_chars is split into "
            "ordinal-labelled parts; a body needing more than wall.max_chunks "
            "parts is refused. --reason labels the broadcast for the operator's "
            "record; the message body itself is delivered byte for byte."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("message", nargs="?", default=None,
                        help="Message text (or pipe it on stdin)")
    parser.add_argument("--wait", dest="wait_seconds", type=float, default=None,
                        help="Seconds to wait for each target's turn to settle "
                             "(default: the peer transport's own default)")
    parser.add_argument("--reason", default="",
                        help="Optional label for why the fleet is being broadcast to")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Emit one JSON result instead of roster lines")
    parser.set_defaults(func=cmd_wall)
