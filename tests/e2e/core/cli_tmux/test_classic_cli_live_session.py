"""The classic CLI's long live session, with a per-process witness (#134889 T24).

Ported from the local dogfood harness used on #106742 (``dogfood/drive.py``): one real ``hermes``
(``display.interface: cli``) session in tmux against a loopback ``FakeLLMServer`` keyed on the
``LIVEnn`` marker in the latest user message (zero spend), driven through tool turns, approval and
clarify modals, a background process, delegation, a sweep of slash commands, side tasks
(``/bg``, ``/btw``), ``/branch``, ``/compress``, Ctrl+C, ``/new``, ``/quit``, ``-c`` and a piped
``chat -q``.

Every Python process loads ``witness/sitecustomize.py``, which records what each process actually
did. On top of each step's screen check, the session asserts the one-gateway contract (#106742):
the CLI process builds no ``AIAgent``, dispatches no tool and writes no ``state.db`` row; the
gateway daemon does all three.

Left out on purpose: ``/debug`` (uploads a summary, T7), ``/reload-mcp`` (opens a confirmation
modal the sweep would answer), ``/voice`` (probes the host's audio stack), ``/diff`` (prepares an
isolated runtime on first use, an install inside the fake home), and ``/retry``/``/undo``
right after a ``/goal`` continuation (the open T5 flake).
"""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.e2e.core.cli_tmux._classic import ClassicHome, Pane, last_user, new_needle, require_tmux, text_of
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall

# Process discovery, cleanup and the witness's pid accounting read /proc.
pytestmark = pytest.mark.platforms("linux")

WITNESS_DIR = Path(__file__).resolve().parent / "witness"
TITLE = "Live session title"
EXTRA_CONFIG = """memory:
  memory_enabled: true
  user_profile_enabled: false
  nudge_interval: 0
delegation:
  max_iterations: 5
"""


def _marker(user: str) -> str:
    import re
    match = re.search(r"LIVE\d+", user)
    return match.group(0) if match else ""


def _main(record: dict[str, Any]):
    body = record.get("body") or {}
    msgs = body.get("messages") or []
    user = last_user(body)
    if "LIVECHILD" in user:
        return Text("LIVECHILD result: the child counted the files.")
    if "LIVE15" in user:  # /btw: the gateway's side call carries the session's tools
        return Text("LIVE15 side answer: you asked about tea.")
    mark = _marker(user)
    if msgs and msgs[-1].get("role") == "tool":
        return Text(f"{mark} done. Tool said: {text_of(msgs[-1].get('content'))[:120]}")
    table: dict[str, Callable[[], Any]] = {
        "LIVE01": lambda: Text("# LIVE01 heading\n\n- one\n- two\n\n```python\nprint('hi')\n```\n\nLIVE01 end."),
        "LIVE02": lambda: ToolCall("terminal", {"command": 'bash -c "echo live-terminal && pwd"'}),
        "LIVE03": lambda: ToolCall("read_file", {"path": "notes.txt"}),
        "LIVE04": lambda: ToolCall("write_file", {"path": "made_by_agent.txt", "content": "written through the gateway\n"}),
        "LIVE05": lambda: ToolCall("search_files", {"pattern": "fixture", "path": "."}),
        "LIVE07": lambda: ToolCall("memory", {"action": "add", "target": "memory", "content": "LIVE07 fact: likes tea"}),
        "LIVE08": lambda: ToolCall("clarify", {"questions": [
            {"question": "LIVE08 which colour?", "choices": ["red", "blue"]},
            {"question": "LIVE08 which size?", "choices": ["small", "large"]}]}),
        "LIVE09": lambda: ToolCall("delegate_task", {"goal": "LIVECHILD count the files in the work folder"}),
        "LIVE10": lambda: ToolCall("terminal", {"command": "sleep 3; echo live10-bg-finished", "background": True,
                                                "notify_on_complete": True}),
        "LIVE14": lambda: Text("LIVE14 background answer: computed in the background."),
        "LIVE20": lambda: Text("LIVE20 " + " ".join(f"w{i:03d}" for i in range(500)) + " THE-END",
                               chunk_chars=6, delay_per_chunk=0.04),
    }
    fn = table.get(mark)
    return fn() if fn else Text(f"{mark or 'plain'} ok.")


def _aux(record: dict[str, Any]):
    blob = json.dumps((record.get("body") or {}).get("messages") or [])
    if "You name chat sessions" in blob:
        return Text("Live session")
    if "LIVE15" in blob:
        return Text("LIVE15 side answer: you asked about tea.")
    return Text("Fake summary of the earlier conversation.")


# (command, a needle main and this checkout print for it)
SLASH_SWEEP = (
    ("/status", "Session ID: local-"), ("/usage", "Session Token Usage"), ("/tools", "read_file"),
    ("/toolsets", "currently enabled"), ("/reasoning high", "Reasoning effort set to 'high'"),
    ("/verbose", "Tool progress:"), ("/yolo", "YOLO mode ON"), ("/yolo", "YOLO mode OFF"),
    ("/personality", "Personalities"), ("/cron", "Scheduled Tasks"), (f"/title {TITLE}", f"Session title set: {TITLE}"),
    ("/platforms", "Gateway Status"), ("/history", "LIVE01"), ("/save", "Usage: /save"), ("/agents", "Running processes"),
    ("/insights", "Models Used"), ("/memory", "memory.write_approval"),
    # /busy and /footer print the gateway handler's markdown verbatim (main: plain CLI text); the
    # needles stop before the formatting so the sweep pins the behaviour, not that rendering gap.
    ("/config", "Configuration"), ("/busy queue", "Busy input mode set to"), ("/footer status", "Runtime footer"),
    ("/approvals", "Approval mode: manual"),
)


class Session:
    def __init__(self, pane: Pane) -> None:
        self.pane = pane
        self.failures: list[str] = []
        self.timings: list[tuple[str, float]] = []
        self.steps = 0

    def step(self, name: str, action: Callable[[], Any], needles: list[str], *, timeout: float = 60.0,
             quiet: float = 2.0) -> str:
        """Run *action*, then require one of *needles* to appear in output produced after the step's
        latest input (``Pane.wait``): text already in the scrollback never satisfies a step."""
        self.steps += 1
        started = time.monotonic()
        action()
        hit = self.pane.wait(needles, timeout) if needles else True
        self.pane.settle(quiet=quiet, timeout=30)
        text = self.pane.scrollback()
        self.timings.append((name, round(time.monotonic() - started, 1)))
        if not hit:
            self.failures.append(f"{name}: none of {needles!r} appeared; screen:\n{self.pane.screen()[-2500:]}")
        if "Traceback (most recent call last)" in text:
            self.failures.append(f"{name}: a traceback reached the terminal")
        return text


def _witness(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_witness_coverage(events: list[dict[str, Any]], spawned: list[tuple[int, str]]) -> None:
    """Every process the harness spawned, and the gateway daemon, attested at startup.

    The negative checks below read "no event" as "did not happen"; that holds only for a process
    whose witness was loaded. A spawned CLI with no ``startup`` line (``PYTHONPATH`` lost, witness
    not imported) fails here instead of passing those checks vacuously.
    """
    attested = {e["pid"] for e in events if e["kind"] == "startup"}
    assert spawned, "the harness recorded no spawned CLI process"
    missing = [(pid, argv) for pid, argv in spawned if pid not in attested]
    assert not missing, f"spawned CLI processes with no witness startup attestation: {missing}"
    assert any(e["kind"] == "startup" and e["role"] == "gateway" for e in events), \
        "no gateway daemon attested: its tool/agent/state.db events cannot be trusted"


def test_a_step_needs_output_produced_after_its_input() -> None:
    """A no-op /history (transcript unchanged) must not pass on "LIVE01" left from the markdown turn."""
    before = "\u276f LIVE01 show markdown\n# LIVE01 heading\nLIVE01 end.\n\u276f /history\n"
    assert new_needle(["LIVE01"], before, before) is None
    assert new_needle(["LIVE01"], before, before + "  1. user: LIVE01 show markdown\n") == "LIVE01"


def test_witness_coverage_rejects_an_uninstrumented_cli() -> None:
    """The negative ownership checks are not vacuous: a silent, unattested CLI fails coverage."""
    gateway = {"pid": 2, "ppid": 1, "role": "gateway", "kind": "startup", "detail": "-m gateway run"}
    cli = {"pid": 10, "ppid": 1, "role": "cli", "kind": "startup", "detail": "-m chat"}
    check_witness_coverage([gateway, cli], [(10, "chat")])
    with pytest.raises(AssertionError, match="no witness startup attestation"):
        check_witness_coverage([gateway, cli], [(10, "chat"), (11, "chat -c")])
    with pytest.raises(AssertionError, match="no gateway daemon attested"):
        check_witness_coverage([cli], [(10, "chat")])
    with pytest.raises(AssertionError, match="recorded no spawned"):
        check_witness_coverage([gateway], [])


def test_classic_cli_live_session_runs_every_turn_in_the_gateway(tmp_path: Path) -> None:
    require_tmux()
    llm = FakeLLMServer(_main, aux=_aux)
    witness = tmp_path / "witness.jsonl"
    ch = ClassicHome(tmp_path / "run")
    pane = Pane(ch)
    s = Session(pane)
    with llm:
        ch.write(llm.base_url, EXTRA_CONFIG)
        ch.extra_env = {"PYTHONPATH": f"{ch.env()['PYTHONPATH']}:{WITNESS_DIR}", "HERMES_E2E_WITNESS": str(witness)}
        (ch.work / "notes.txt").write_text("fixture line one\nfixture line two\n", encoding="utf-8")
        (ch.work / "other.md").write_text("# other fixture\n", encoding="utf-8")
        try:
            s.step("startup", pane.launch, ["\u2624 fake-model"], timeout=120, quiet=3.0)
            s.step("markdown", lambda: pane.submit("LIVE01 show markdown"), ["LIVE01 end."])
            s.step("approval_modal", lambda: pane.submit("LIVE02 run a command"), ["Dangerous Command"], quiet=1.5)
            s.step("approve_once", lambda: pane.keys("Enter"), ["LIVE02 done"])
            s.step("read_file", lambda: pane.submit("LIVE03 read notes"), ["LIVE03 done"])
            s.step("write_file", lambda: pane.submit("LIVE04 write a file"), ["LIVE04 done"])
            s.step("search_files", lambda: pane.submit("LIVE05 search"), ["LIVE05 done"])
            s.step("memory", lambda: pane.submit("LIVE07 remember"), ["LIVE07 done"])
            s.step("clarify_modal", lambda: pane.submit("LIVE08 ask me"), ["LIVE08 which colour?"], quiet=1.5)
            s.step("clarify_answer", lambda: (pane.keys("Enter"), pane.paste_gap(), pane.keys("Enter")), ["LIVE08 done"])
            s.step("delegate", lambda: pane.submit("LIVE09 delegate"), ["LIVE09 done"], timeout=90)
            s.step("delegate_result", lambda: None, ["LIVECHILD result"], timeout=90, quiet=3.0)
            s.step("bg_process", lambda: pane.submit("LIVE10 background"), ["LIVE10 done"])
            s.step("bg_process_notice", lambda: None, ["live10-bg-finished"], timeout=60, quiet=3.0)
            for command, needle in SLASH_SWEEP:
                s.step(command, lambda c=command: pane.submit(c), [needle], timeout=30)
            s.step("/model", lambda: pane.submit("/model"), ["Model Picker"], timeout=30, quiet=1.5)
            s.step("/model close", lambda: pane.keys("Escape"), [], quiet=1.0)
            s.step("/bg", lambda: pane.submit("/bg LIVE14 compute something"), ["LIVE14 background answer"],
                   timeout=90)
            s.step("/btw", lambda: pane.submit("/btw LIVE15 what did I ask about?"), ["LIVE15 side answer"],
                   timeout=90)
            s.step("/branch", lambda: pane.submit("/branch"), ["Branched session"], timeout=30, quiet=3.0)
            # The sweep's /reasoning retired the CLI's agent (main's _retire_agent, rebuilt on the next
            # turn) and /compress refuses without one, as on main: run a turn first.
            s.step("turn on the branch", lambda: pane.submit("LIVE31 on the branch"), ["LIVE31 ok."])
            s.step("/compress", lambda: pane.submit("/compress"), ["Compressed:", "No changes from compression"],
                   timeout=90, quiet=3.0)
            s.step("interrupt", lambda: (pane.submit("LIVE20 long story"), pane.wait(["w030"], 60), pane.keys("C-c")),
                   ["Interrupted"], timeout=45, quiet=3.0)
            if "THE-END" in pane.scrollback():
                s.failures.append("interrupt: the turn streamed to THE-END")
            s.step("/new", lambda: pane.submit("/new"), ["destroys conversation state"], timeout=30, quiet=1.5)
            s.step("/new confirm", lambda: pane.keys("Enter"), ["New session started"], timeout=30)
            s.step("plain turn after /new", lambda: pane.submit("LIVE30 hello again"), ["LIVE30 ok."])
            s.step("/quit", lambda: pane.submit("/quit"), ["Resume this session with:"], timeout=30)
            if not pane.wait_dead(30):
                s.failures.append("/quit: the CLI was still running 30 s later")
            s.step("resume -c", lambda: pane.launch("-c"), ["Resumed session"], timeout=120, quiet=3.0)
            s.step("/quit after resume", lambda: pane.submit("/quit"), ["Resume this session with:"], timeout=30)
            pane.wait_dead(30)
            s.steps += 1
            piped = ch.run("chat", "-q", "LIVE01 one shot")
            if piped.returncode != 0 or "LIVE01 end." not in piped.stdout:
                s.failures.append(f"chat -q piped: exit={piped.returncode}\n{piped.stdout[-1500:]}\n{piped.stderr[-1500:]}")
        finally:
            pane.kill()
            ch.reap()

    print("step timings (s):", s.timings)
    assert not s.failures, (f"{len(s.failures)}/{s.steps} steps failed:\n\n" + "\n\n".join(s.failures[:5])
                            + f"\n\n{ch.diagnostics()}")

    events = _witness(witness)
    assert not [e for e in events if e["kind"] == "witness_error"], events
    check_witness_coverage(events, ch.spawned)
    by_role: dict[str, Counter] = {"cli": Counter(), "gateway": Counter()}
    for e in events:
        if e["kind"] != "startup":
            by_role[e["role"]][f"{e['kind']} {e['detail']}"] += 1
    cli, gateway = by_role["cli"], by_role["gateway"]
    # The one-gateway contract: the CLI only reads and submits.
    assert not [k for k in cli if k.startswith(("agent_init", "tool "))], f"the CLI process ran agent work: {dict(cli)}"
    assert not [k for k in cli if k.startswith("sqlite_write state.db")], f"the CLI process wrote state.db: {dict(cli)}"
    # Registry-dispatched tools; memory / clarify / delegate_task run inside the agent loop (the gateway's
    # AIAgent, witnessed by agent_init) and their screens were checked above.
    for tool in ("terminal", "read_file", "write_file", "search_files"):
        assert gateway[f"tool {tool}"], f"tool {tool} never ran in the gateway: {sorted(gateway)}"
    assert any(k.startswith("agent_init") for k in gateway), sorted(gateway)
    assert gateway["sqlite_write state.db:messages"], sorted(gateway)
    assert llm.main_requests(), "no request reached the fake provider"
