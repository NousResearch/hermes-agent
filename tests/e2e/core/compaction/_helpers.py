"""Lane-private harness for the compaction property / liveness suites (class C4).

Everything real except the LLM: a real ``AIAgent`` with a real ``SessionDB`` on disk, real tools
(``read_file`` on seeded files), real compaction, all talking to the recording loopback provider in
``tests/fakes/fake_llm_provider.py``. The provider is shared per module; each scenario installs its own
main-turn and summarizer responders on the ``Dispatch`` object.

The invariants live here so every test file asserts the SAME properties:

* ``assert_wire_request_ok``      — every request the model sees: system first, first user (job/task
  anchor) present, tool_call/tool_result pairs matched, no user->user.
* ``assert_persisted_prefix``     — what state.db holds at request time is exactly the history prefix
  the request carries; the rest is only the in-flight turn (persisted == sent).
* ``assert_db_recoverable``       — no user message is lost: each is live (``active=1``) or archived
  as summarized history (``compacted=1``); a superseded duplicate (``active=0, compacted=0``) always
  has an identical live/archived copy.
* ``assert_db_matches_history``   — the live rows equal the in-memory history the host keeps.
"""

from __future__ import annotations

import base64
import json
import random
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from tests.fakes.fake_llm_provider import MODEL_ID, Error, FakeLLMServer, Hang, Text, ToolCall

# Every compaction knob is tiny so a handful of turns crosses the trigger. The window must stay at the
# 64K agent floor; ``threshold_tokens`` (an absolute cap) pulls the trigger far below the ratio floor.
CONTEXT_LENGTH = 64_000
THRESHOLD_TOKENS = 12_000
# Seconds the "slow" summarizer hangs; the summary timeout is pinned far below it.
SLOW_HANG_SECONDS = 60.0
SLOW_SUMMARY_TIMEOUT_SECONDS = 3.0
# Seeded session shape shared by the automatic suites.
SEEDS = (11, 23, 37)
N_TURNS = 10
# Legitimate summarizer traffic per turn: max_attempts (3) x (primary + one main-model retry) plus the
# stall fallback route. A runaway compaction loop (the 877-call class) blows far past this.
MAX_SUMMARY_CALLS_PER_TURN = 8
# Upper bound on one turn's wall time. The slowest legitimate turn is a slow-summarizer turn that waits
# one pinned timeout per attempt; 10x headroom over that keeps the bound load-proof.
TURN_WALL_BOUND_SECONDS = 120.0

MARKER_RE = re.compile(r"\bU\d+x\d+x[0-9a-f]{6}\b")
GOOD_SUMMARY_TOKEN = "SUMMARY-OK"
# Bodies of the bad summaries; none may ever reach the model or state.db as context.
REFUSAL_TEXT = "I'm sorry, but I can't help with summarizing this conversation."
TRUNCATED_TEXT = "## Goal\nThe user is work"

_PNG_1PX = base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154"
    "78da63f8cfc0f01f0005000201ffa3a8b2b10000000049454e44ae426082")).decode()


def estimate_prompt_tokens(body: dict[str, Any]) -> int:
    """What the fake provider reports as ``usage.prompt_tokens``: chars/4 of the whole request."""
    return len(json.dumps(body.get("messages", []))) // 4 + len(json.dumps(body.get("tools", []))) // 4


# ---------------------------------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------------------------------


class Dispatch:
    """Per-scenario responders behind one module-scoped provider."""

    def __init__(self) -> None:
        self.main: Callable[[dict[str, Any]], Any] = lambda _rec: Text("ok")
        self.aux: Callable[[dict[str, Any]], Any] = lambda _rec: Text(good_summary(0))


def start_provider() -> tuple[FakeLLMServer, Dispatch]:
    dispatch = Dispatch()
    server = FakeLLMServer(
        lambda rec: dispatch.main(rec), aux=lambda rec: dispatch.aux(rec),
        prompt_tokens_fn=estimate_prompt_tokens)
    server.start()
    return server, dispatch


def good_summary(n: int) -> str:
    return (
        f"## Goal\nKeep helping with the seeded task ({GOOD_SUMMARY_TOKEN}-{n}).\n"
        "## Progress\n### Done\n- Read several files and answered.\n"
        "## Next Steps\n- Continue with the latest request.\n")


def summary_responder(mode: str, calls: list) -> Callable[[dict[str, Any]], Any]:
    """Summarizer behaviours from the C4 harness spec."""

    def respond(rec: dict[str, Any]) -> Any:
        calls.append(rec)
        n = len(calls)
        if mode == "good":
            return Text(good_summary(n), chunk_chars=64)
        if mode == "empty":
            return Text("")
        if mode == "refusal":
            return Text(REFUSAL_TEXT)
        if mode == "http500":
            return Error(500, "summarizer upstream exploded")
        if mode == "truncated":
            return Text(TRUNCATED_TEXT, finish_reason="length")
        if mode == "slow":
            return Hang(SLOW_HANG_SECONDS)
        raise AssertionError(f"unknown summarizer mode {mode!r}")

    return respond


def compaction_config(*, extra: str = "", slow: bool = False, threshold_tokens: int = THRESHOLD_TOKENS) -> str:
    cfg = (
        "compression:\n"
        f"  threshold_tokens: {threshold_tokens}\n"
        "  protect_last_n: 4\n"
        + extra
        + "auxiliary:\n"
        "  title_generation:\n"
        "    enabled: false\n"
    )
    if slow:
        cfg += f"  compression:\n    timeout: {SLOW_SUMMARY_TIMEOUT_SECONDS}\n"
    return cfg


def write_home(hermes_home: Path, base_url: str, extra_config: str) -> None:
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  provider: custom\n"
        f"  base_url: {base_url}\n"
        f"  default: {MODEL_ID}\n"
        f"  context_length: {CONTEXT_LENGTH}\n"
        "agent:\n"
        "  api_max_retries: 1\n"
        "updates:\n"
        "  check: false\n"  # offline: no GitHub round-trip or git lazy fetch from a test surface
        + extra_config,
        encoding="utf-8",
    )
    (hermes_home / ".env").write_text("OPENAI_API_KEY=sk-fake-e2e\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------------
# Seeded transcripts
# ---------------------------------------------------------------------------------------------------


@dataclass
class TurnSpec:
    marker: str
    user: Any  # str, or a list of content parts (image turns)
    steps: list[Any]
    kind: str


def _write_file(workdir: Path, name: str, rng: random.Random, chars: int) -> Path:
    line_no = 0
    lines = []
    total = 0
    while total < chars:
        line = f"{name}:{line_no} " + " ".join(rng.choice(("alpha", "bravo", "delta", "omega", "sigma")) for _ in range(14))
        lines.append(line)
        total += len(line) + 1
        line_no += 1
    path = workdir / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_transcript(seed: int, n_turns: int, workdir: Path, *, kinds: Iterable[str] | None = None) -> list[TurnSpec]:
    """A seeded random session: tool-pair heavy (incl. parallel calls), huge single turns, image parts,
    long answers, and turns whose single tool result alone exceeds the compaction threshold."""
    rng = random.Random(seed)
    workdir.mkdir(parents=True, exist_ok=True)
    weights = {"tools": 5, "parallel": 3, "huge_user": 1, "image": 1, "chat": 2, "tail_bomb": 1}
    if kinds is not None:
        weights = {k: weights[k] for k in kinds}
    names, ws = zip(*weights.items())
    specs: list[TurnSpec] = []
    fileno = 0

    def new_file(chars: int) -> str:
        nonlocal fileno
        fileno += 1
        return str(_write_file(workdir, f"s{seed}f{fileno}.txt", rng, chars))

    for i in range(n_turns):
        marker = f"U{seed}x{i}x{rng.getrandbits(24):06x}"
        # Turn 0 is always a plain tool turn: its user message is the session's task anchor.
        kind = "tools" if i == 0 else rng.choices(names, ws)[0]
        steps: list[Any] = []
        user: Any = f"{marker} please read the next file and report"
        if kind == "tools":
            for _ in range(rng.randint(1, 3)):
                steps.append(ToolCall("read_file", {"path": new_file(rng.randint(600, 9000))}))
        elif kind == "parallel":
            for _ in range(rng.randint(1, 2)):
                extra = [("read_file", {"path": new_file(rng.randint(400, 6000))}) for _ in range(rng.randint(1, 3))]
                steps.append(ToolCall("read_file", {"path": new_file(rng.randint(400, 6000))}, parallel=extra))
        elif kind == "huge_user":
            filler = " ".join(f"w{rng.randint(0, 99999)}" for _ in range(rng.randint(2500, 6000)))
            user = f"{marker} here is a large paste to keep in mind: {filler}"
        elif kind == "image":
            user = [
                {"type": "text", "text": f"{marker} what is in this picture?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PNG_1PX}"}},
            ]
        elif kind == "tail_bomb":
            steps.append(ToolCall("read_file", {"path": new_file(int(THRESHOLD_TOKENS * 4 * 1.3))}))
        answer_len = rng.randint(4000, 12000) if kind == "chat" else rng.randint(20, 200)
        answer = f"answer for {marker}: " + " ".join(f"t{rng.randint(0, 9999)}" for _ in range(answer_len // 6))
        steps.append(Text(answer, chunk_chars=256))
        specs.append(TurnSpec(marker=marker, user=user, steps=steps, kind=kind))
    return specs


def job_turn(seed: int, workdir: Path, rounds: int) -> TurnSpec:
    """A cron-shaped run: ONE user message (the job prompt) followed by many tool rounds, so compaction
    has to fire mid-turn with the job prompt as the only real ask."""
    rng = random.Random(seed)
    workdir.mkdir(parents=True, exist_ok=True)
    marker = f"U{seed}x0x{rng.getrandbits(24):06x}"
    steps: list[Any] = []
    for k in range(rounds):
        path = _write_file(workdir, f"job{seed}r{k}.txt", rng, rng.randint(3000, 9000))
        steps.append(ToolCall("read_file", {"path": str(path)}))
    steps.append(Text(f"job report for {marker}: done", chunk_chars=64))
    return TurnSpec(marker=marker, user=f"{marker} daily job: read every report file and summarize", steps=steps, kind="job")


# ---------------------------------------------------------------------------------------------------
# Normalisation + invariants
# ---------------------------------------------------------------------------------------------------


def flatten(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                out.append(str(part.get("text") or ""))
            else:
                out.append(str(part))
        return "\n".join(out)
    return str(content)


def _squash(text: str) -> str:
    return " ".join(text.split())


def _calls(tool_calls: Any) -> tuple:
    if isinstance(tool_calls, str):
        try:
            tool_calls = json.loads(tool_calls)
        except ValueError:
            tool_calls = []
    out = []
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        try:
            args = json.dumps(json.loads(args), sort_keys=True) if isinstance(args, str) else json.dumps(args, sort_keys=True)
        except ValueError:
            pass
        out.append((tc.get("id"), fn.get("name"), args))
    return tuple(out)


def norm(msg: dict[str, Any]) -> tuple:
    """Identity of a message that must survive persistence and request building unchanged.

    A user message is identified by the seeded markers it carries (image turns are rendered to text
    differently for the wire and for storage by design); everything else by its full content."""
    role = msg.get("role")
    text = flatten(msg.get("content"))
    if role == "user":
        marks = tuple(sorted(set(MARKER_RE.findall(text))))
        return ("user", marks if marks else _squash(text))
    if role == "assistant":
        return ("assistant", _squash(text), _calls(msg.get("tool_calls")))
    if role == "tool":
        # Tool output may be shrunk on the wire by designed lossy steps (old-result pruning, oversized-result
        # spill to disk); the pairing, not the bytes, is the invariant here.
        return ("tool", msg.get("tool_call_id"))
    return (role, _squash(text))


def merge_users(items: list[tuple]) -> list[tuple]:
    """Alternation repair merges back-to-back same-role rows (e.g. after an aborted turn) on the wire; the
    merged row carries exactly the same user messages, so compare with the same merge applied."""
    out: list[tuple] = []
    for item in items:
        if (item[0] == "user" and out and out[-1][0] == "user"
                and isinstance(item[1], tuple) and isinstance(out[-1][1], tuple)):
            out[-1] = ("user", tuple(sorted(set(out[-1][1]) | set(item[1]))))
            continue
        # Same repair for assistant->assistant (a split summary row followed by the next assistant).
        if item[0] == "assistant" and out and out[-1][0] == "assistant" and not out[-1][2]:
            out[-1] = ("assistant", _squash(f"{out[-1][1]} {item[1]}"), item[2])
            continue
        out.append(item)
    return out


def norm_list(msgs: Iterable[dict[str, Any]]) -> list[tuple]:
    return merge_users([norm(m) for m in msgs if m.get("role") != "system"])


def db_rows(db_path: Path, session_id: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, role, content, tool_call_id, tool_calls, active, compacted FROM messages "
            "WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            content = d["content"]
            if isinstance(content, str) and content.startswith("\x00json:"):
                d["content"] = flatten(json.loads(content[len("\x00json:"):]))
            out.append(d)
        return out
    finally:
        conn.close()


def active_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["active"]]


def assert_tool_pairs(msgs: list[dict[str, Any]], where: str, *, allow_trailing_pending: bool = False) -> None:
    pending: dict[str, int] = {}
    seen_calls: set[str] = set()
    seen_results: set[str] = set()
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role == "assistant":
            assert not pending, f"{where}: tool call(s) {sorted(pending)} unanswered before assistant #{i}"
            for call_id, _name, _args in _calls(m.get("tool_calls")):
                assert call_id not in seen_calls, f"{where}: tool_call id {call_id} issued twice (#{i})"
                seen_calls.add(call_id)
                pending[call_id] = i
        elif role == "tool":
            cid = m.get("tool_call_id")
            assert cid in pending, f"{where}: orphan tool result {cid!r} at #{i} (no open assistant tool_call)"
            assert cid not in seen_results, f"{where}: tool result {cid} duplicated (#{i})"
            seen_results.add(cid)
            del pending[cid]
        elif role == "user":
            assert not pending, f"{where}: tool call(s) {sorted(pending)} unanswered before user #{i}"
    if not allow_trailing_pending:
        assert not pending, f"{where}: tool call(s) {sorted(pending)} never answered"


def assert_wire_request_ok(body: dict[str, Any], turn_marker: str, base_system: str | None, where: str) -> None:
    msgs = body["messages"]
    assert msgs and msgs[0]["role"] == "system", f"{where}: system prompt missing from the request"
    assert all(m["role"] != "system" for m in msgs[1:]), f"{where}: a second system message in the request"
    if base_system is not None:
        # The system anchor is built once per session; compaction may only append its note.
        assert msgs[0]["content"].startswith(base_system), f"{where}: the session system prompt was rewritten"
    # The in-flight ask (for a cron run: the job prompt) must reach the model AFTER any summary; a summary
    # that swallows it leaves the model told to "respond to the message below" with nothing below (#100818).
    text = "\n".join(flatten(m.get("content")) for m in msgs[1:] if m["role"] in ("user", "assistant", "tool"))
    ask_at = max((i for i, m in enumerate(msgs) if m["role"] == "user" and turn_marker in flatten(m.get("content"))),
                 default=-1)
    assert ask_at > 0, f"{where}: the in-flight user message {turn_marker} is missing from the request"
    if GOOD_SUMMARY_TOKEN in text:
        assert text.rfind(turn_marker) > text.rfind(GOOD_SUMMARY_TOKEN), (
            f"{where}: the in-flight user message {turn_marker} only survives inside/before the summary")
    assert_tool_pairs(msgs, where)
    for a, b in zip(msgs, msgs[1:]):
        assert not (a["role"] == "user" and b["role"] == "user"), f"{where}: two consecutive user messages"
    assert msgs[-1]["role"] in ("user", "tool"), f"{where}: request ends with {msgs[-1]['role']}"


def assert_persisted_prefix(snapshot: list[tuple], body: dict[str, Any], turn_marker: str, first_new_call: int, where: str) -> None:
    """The durable rows at request time are exactly the request's history prefix; whatever follows is
    only the in-flight turn (its own user message and the tool calls it issued)."""
    sent = norm_list(body["messages"])
    expected = list(snapshot)
    if not any(it[0] == "user" and isinstance(it[1], tuple) and turn_marker in it[1] for it in expected):
        expected = merge_users(expected + [("user", (turn_marker,))])
    assert sent[: len(expected)] == expected, (
        f"{where}: request history diverges from state.db (+ the in-flight user message) at the time it "
        f"was sent\n  first mismatch {_diff_at(sent, expected)}\n  sent={_brief(sent)}\n  db  ={_brief(expected)}")
    for item in sent[len(expected):]:
        assert item[0] != "user", f"{where}: unpersisted user message {item[1]} in the request after the in-flight turn"
        if item[0] == "assistant":
            for call_id, _n, _a in item[2]:
                assert int(call_id.rsplit("_", 1)[1]) >= first_new_call, (
                    f"{where}: unpersisted tool call {call_id} from an earlier turn in the request")


def assert_db_recoverable(rows: list[dict[str, Any]], sent_markers: Iterable[str], where: str) -> None:
    live_or_archived = [r for r in rows if r["active"] or r["compacted"]]
    marks_kept: set[str] = set()
    for r in live_or_archived:
        if r["role"] == "user":
            marks_kept.update(MARKER_RE.findall(r["content"] or ""))
    lost = [m for m in sent_markers if m not in marks_kept]
    assert not lost, f"{where}: user message(s) {lost} are neither live nor archived in state.db (data loss)"
    # A carried row may be re-inserted with the summary merged into its content, so the live copy only
    # has to contain the original bytes (same role, same tool linkage).
    keep: dict[tuple, list[str]] = {}
    for r in live_or_archived:
        keep.setdefault(_row_key(r), []).append(r["content"] or "")
    # Tool outputs are exempt: pruning old tool results to stubs is a designed, lossy compaction step.
    for r in rows:
        if not r["active"] and not r["compacted"] and r["role"] in ("user", "assistant"):
            copies = keep.get(_row_key(r), [])
            if r["role"] == "user" and MARKER_RE.search(r["content"] or ""):
                # Multimodal user rows are stored as a placeholder by the turn flush but as the full parts
                # list when carried; identity is the seeded marker.
                marks = set(MARKER_RE.findall(r["content"]))
                covered = any(marks <= set(MARKER_RE.findall(c)) for c in copies)
            else:
                covered = any((r["content"] or "") in c for c in copies)
            assert covered, (
                f"{where}: row {r['id']} ({r['role']}) was retired as a carried duplicate but no live or "
                f"archived copy exists: {str(r['content'])[:80]!r}; same-linkage copies: "
                f"{[c[:200] for c in copies]}")


def assert_db_matches_history(rows: list[dict[str, Any]], history: list[dict[str, Any]], where: str) -> None:
    live = norm_list(_row_msg(r) for r in active_rows(rows))
    mem = norm_list(history)
    assert live == mem, (
        f"{where}: live state.db rows differ from the host's in-memory history at #{_first_mismatch(live, mem)}\n"
        f"  db ={_brief(live)}\n  mem={_brief(mem)}")
    assert_tool_pairs([_row_msg(r) for r in active_rows(rows)], f"{where} (state.db live rows)")


def _row_msg(r: dict[str, Any]) -> dict[str, Any]:
    return {"role": r["role"], "content": r["content"], "tool_calls": r["tool_calls"], "tool_call_id": r["tool_call_id"]}


def _row_key(r: dict[str, Any]) -> tuple:
    return (r["role"], r["tool_call_id"], json.dumps(_calls(r["tool_calls"])))


def _diff_at(a: list, b: list) -> str:
    i = _first_mismatch(a, b)
    x = a[i] if i < len(a) else None
    y = b[i] if i < len(b) else None
    return f"#{i}: {str(x)[:160]!r} ... {str(x)[-160:]!r}\n   vs  {str(y)[:160]!r} ... {str(y)[-160:]!r}"


def _first_mismatch(a: list, b: list) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _brief(items: list[tuple]) -> list[str]:
    out = []
    for it in items:
        if it[0] == "user":
            out.append(f"U{it[1] if isinstance(it[1], tuple) else '(' + it[1][:30] + ')'}")
        elif it[0] == "assistant":
            out.append(f"A[{','.join(c[0] for c in it[2])}]{it[1][:24]!r}")
        else:
            out.append(f"{it[0][0].upper()}:{it[1]}")
    return out


# ---------------------------------------------------------------------------------------------------
# Scenario driver
# ---------------------------------------------------------------------------------------------------


@dataclass
class TurnRecord:
    spec: TurnSpec
    result: dict[str, Any]
    elapsed: float
    main_requests: list[dict[str, Any]]
    summary_calls: int
    statuses: list[tuple]
    aborted: bool = False


@dataclass
class Scenario:
    """One session: its own HERMES_HOME, state.db and agent against the shared provider."""

    server: FakeLLMServer
    dispatch: Dispatch
    home: Path
    mode: str = "good"
    session_id: str = "compaction-e2e"
    platform: str = "cli"
    window: int | None = None  # provider-enforced context window (tokens); None = unlimited
    history: list[dict[str, Any]] = field(default_factory=list)
    sent_markers: list[str] = field(default_factory=list)
    summary_calls: list = field(default_factory=list)
    statuses: list[tuple] = field(default_factory=list)
    turns: list[TurnRecord] = field(default_factory=list)
    base_system: str | None = None

    def __post_init__(self) -> None:
        self.db_path = self.home / "state.db"
        self._queue: list[Any] = []
        self._qlock = threading.Lock()
        self._snapshots: list[tuple[dict[str, Any], list[tuple]]] = []
        self.overflow_rejections = 0
        self.db = None
        self.agent = None

    # provider side ------------------------------------------------------------------------------
    def install(self) -> None:
        self.dispatch.main = self._main
        self.dispatch.aux = summary_responder(self.mode, self.summary_calls)

    def _main(self, rec: dict[str, Any]) -> Any:
        body = rec["body"]
        snap = norm_list(_row_msg(r) for r in active_rows(db_rows(self.db_path, self.session_id))) if self.db_path.exists() else []
        self._snapshots.append((body, snap))
        if self.window is not None and estimate_prompt_tokens(body) > self.window:
            self.overflow_rejections += 1
            return Error(400, f"This model's maximum context length is {self.window} tokens. However, your "
                              f"messages resulted in {estimate_prompt_tokens(body)} tokens. (context_length_exceeded)")
        with self._qlock:
            if self._queue:
                return self._queue.pop(0)
        return Text("(unscripted extra request)")

    # agent side ---------------------------------------------------------------------------------
    def build_agent(self) -> Any:
        from hermes_state import SessionDB
        from run_agent import AIAgent

        if self.db is None:
            self.db = SessionDB(db_path=self.db_path)
        self.agent = AIAgent(
            provider="custom", base_url=self.server.base_url, api_key="sk-fake-e2e", model=MODEL_ID,
            session_db=self.db, session_id=self.session_id, quiet_mode=True, platform=self.platform,
            enabled_toolsets=["file"], skip_context_files=True, skip_memory=True,
            status_callback=lambda *a, **_k: self.statuses.append(tuple(a)),
        )
        return self.agent

    def resume_from_db(self) -> None:
        """Gateway/Desktop shape: a fresh agent whose history is whatever state.db hands back."""
        self.close_agent()
        self.build_agent()
        self.history = self.db.get_messages_as_conversation(self.session_id)

    def close_agent(self) -> None:
        if self.agent is not None:
            close = getattr(self.agent, "close", None)
            if callable(close):
                close()
            self.agent = None

    def close(self) -> None:
        self.close_agent()
        if self.db is not None:
            self.db.close()
            self.db = None

    def run_turn(self, spec: TurnSpec, *, allow_abort: bool = False) -> TurnRecord:
        if self.agent is None:
            self.build_agent()
        self.install()
        with self._qlock:
            self._queue = list(spec.steps)
        first_req = len(self._snapshots)
        summary_before = len(self.summary_calls)
        status_before = len(self.statuses)
        with self.server._lock:
            first_new_call = self.server._tool_seq + 1
        started = time.monotonic()
        result = self.agent.run_conversation(spec.user, conversation_history=self.history)
        elapsed = time.monotonic() - started
        self.sent_markers.append(spec.marker)
        where = f"turn {len(self.turns)} ({spec.kind}, mode={self.mode})"

        # (1) liveness: bounded wall time, every scripted step consumed exactly once.
        assert elapsed < TURN_WALL_BOUND_SECONDS, f"{where}: turn took {elapsed:.1f}s"
        with self._qlock:
            unconsumed = len(self._queue)
            self._queue = []
        if unconsumed:
            # Stopping early is only acceptable as an explicit, user-visible failure.
            assert allow_abort, (
                f"{where}: turn ended with {unconsumed} scripted step(s) unconsumed: "
                f"{ {k: result.get(k) for k in ('failed', 'error', 'turn_exit_reason', 'final_response')} }")
            assert result.get("error") or result.get("failed"), (
                f"{where}: turn stopped early without an explicit error: "
                f"{ {k: result.get(k) for k in ('failed', 'error', 'turn_exit_reason', 'final_response')} }")
        reqs = self._snapshots[first_req:]
        if self.base_system is None and reqs:
            self.base_system = reqs[0][0]["messages"][0]["content"]
        # (2)+(4) every request: well-formed, anchored, pairs matched, persisted prefix + in-flight turn.
        for k, (body, snap) in enumerate(reqs):
            w = f"{where} request {k}"
            assert_wire_request_ok(body, spec.marker, self.base_system, w)
            assert_persisted_prefix(snap, body, spec.marker, first_new_call, w)
        rows = db_rows(self.db_path, self.session_id)
        assert_db_recoverable(rows, self.sent_markers, where)
        self.history = result["messages"]
        assert_db_matches_history(rows, self.history, where)
        rec = TurnRecord(spec, result, elapsed, [b for b, _ in reqs], len(self.summary_calls) - summary_before,
                         self.statuses[status_before:], aborted=bool(unconsumed))
        self.turns.append(rec)
        return rec

    # derived facts ----------------------------------------------------------------------------------
    def committed_compactions(self) -> int:
        return sum(1 for r in db_rows(self.db_path, self.session_id) if r["compacted"])

    def summary_rows(self) -> list[dict[str, Any]]:
        return [r for r in db_rows(self.db_path, self.session_id) if r["active"] and GOOD_SUMMARY_TOKEN in (r["content"] or "")]


def drive(sc: Scenario, seed: int, workdir: Path, n_turns: int = N_TURNS, *, resume_every: int = 4,
          kinds: Iterable[str] | None = None) -> list[TurnSpec]:
    """Run a seeded session through ``sc``; every ``resume_every`` turns the host is replaced by a fresh
    agent resumed from state.db (the gateway/Desktop shape). Every turn asserts the shared invariants."""
    specs = generate_transcript(seed, n_turns, workdir, kinds=kinds)
    for i, spec in enumerate(specs):
        if i and resume_every and i % resume_every == 0:
            sc.resume_from_db()
        rec = sc.run_turn(spec)
        assert rec.summary_calls <= MAX_SUMMARY_CALLS_PER_TURN, (
            f"turn {i}: {rec.summary_calls} summarizer calls in one turn (runaway compaction)")
    return specs


def warned(sc: Scenario) -> bool:
    """The user was told something about context/compaction (status ``warn`` / ``error`` lines)."""
    return any(s and s[0] in ("warn", "error") for s in sc.statuses)
