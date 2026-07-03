"""Test 15 — driver-loop control logic (deterministic, no container, no tokens).

Drives scripts/drive.py::drive() with mocked invoke / read_plan_tree / archive_journal /
sleep / now. Each case is a scripted sequence of (stdout, plan-tree-text) applied per
invoke; read_plan_tree returns the current on-disk tree. This is the primary regression
guard for the control logic — fully deterministic, milliseconds, no model.

Run:  python3 run.py test_15_driver_loop
"""
import itertools
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from drive import drive, fingerprint, parse_state  # noqa: E402

BASE_PROMPT = "INTENT: do the thing. plan-tree at /opt/data/plans/t/plan-tree.md"
PLAN_PATH = "/opt/data/plans/t/plan-tree.md"


def tree(state, dead=(), done=(), frontier="(empty)"):
    lines = [f"# Plan-Tree: t   STATE: {state}", "INTENT: x", "NODES"]
    lines += [f"- {d}  m  ✝ down" for d in dead]
    lines += [f"- {d}  m  ✓ ok" for d in done]
    lines.append(f"FRONTIER: {frontier}")
    return "\n".join(lines)


T_SUCCESS = tree("SUCCESS", done=["S1", "S2"])
T_EXH = tree("EXHAUSTION-STOP", dead=["S1", "S2"])
T_GH = tree("GUARD-HALT", dead=["S1"], frontier="S2")
T_A = tree("active", dead=["S1"], frontier="S2, S3")
T_B = tree("active", dead=["S1", "S2"], frontier="S3")        # distinct fp from T_A


class FakeDisk:
    """current = the on-disk plan-tree; each invoke applies the next scripted (stdout, tree)."""

    def __init__(self, initial, script):
        self.current = initial
        self.script = list(script)
        self.i = 0
        self.prompts = []
        self.archives = []
        self.sleeps = 0

    def read(self, slug):
        return self.current

    def invoke(self, prompt, timeout):
        self.prompts.append(prompt)
        if self.i < len(self.script):
            stdout, t = self.script[self.i]
            self.i += 1
        else:
            stdout, t = "", self.current  # script exhausted -> no change (no-op)
        self.current = t
        return SimpleNamespace(returncode=0, stdout=stdout)

    def archive(self, slug, n):
        self.archives.append(n)

    def sleep(self, _):
        self.sleeps += 1


def run(initial, script, **kw):
    fd = FakeDisk(initial, script)
    now = kw.pop("now", lambda: 0)
    res = drive(BASE_PROMPT, "t", invoke=fd.invoke, read_plan_tree=fd.read,
                archive_journal=fd.archive, plan_path=PLAN_PATH,
                sleep=fd.sleep, now=now, log=lambda *a, **k: None,
                noop_gap=0, **kw)
    return res, fd


# --------------------------------------------------------------------------- terminals
def test_success_first_tick():
    res, fd = run(None, [("done", T_SUCCESS)])
    assert res.status == "SUCCESS" and res.terminal
    assert res.productive_ticks == 1 and res.invokes == 1
    assert fd.archives == []  # no pre-existing tree on the first tick -> nothing archived


def test_active_active_success():
    res, fd = run(None, [("", T_A), ("", T_B), ("", T_SUCCESS)])
    assert res.status == "SUCCESS"
    assert res.productive_ticks == 3 and res.invokes == 3
    # archive runs only when a tree already existed (ticks 1 & 2), with increasing seq.
    assert fd.archives == sorted(fd.archives) and len(fd.archives) == 2


def test_exhaustion():
    res, _ = run(None, [("", T_EXH)])
    assert res.status == "EXHAUSTION" and res.terminal and res.final_state == "EXHAUSTION-STOP"


def test_guard_halt_default_stops():
    res, fd = run(None, [("", T_GH)])
    assert res.status == "GUARD_HALT" and res.terminal
    assert res.invokes == 1 and res.guard_bumps == 0


# --------------------------------------------------------------------------- guard bump
def test_guard_bump_reaches_success():
    # tick0 -> GUARD-HALT (progress) -> bump; tick1 resumes -> SUCCESS.
    res, _ = run(None, [("", T_GH), ("", T_SUCCESS)], bump_guard=True, max_guard_bumps=2)
    assert res.status == "SUCCESS" and res.guard_bumps == 1


def test_guard_bump_capped():
    gh2 = tree("GUARD-HALT", dead=["S1", "S2"], frontier="S3")  # distinct fp -> a real bump
    res, _ = run(None, [("", T_GH), ("", gh2)], bump_guard=True, max_guard_bumps=1)
    assert res.status == "GUARD_HALT" and res.guard_bumps == 1  # one bump then stop


def test_guard_bump_unchanged_is_structural_wall():
    # bump, but the resume produced the SAME tree -> structural wall, stop (don't bump again).
    res, _ = run(None, [("", T_GH), ("", T_GH)], bump_guard=True, max_guard_bumps=3)
    assert res.status == "GUARD_HALT"
    assert "structural wall" in res.detail


# --------------------------------------------------------------------------- no-op budget
def test_noop_does_not_consume_productive_budget():
    res, fd = run(None, [("", None), ("out", T_A), ("done", T_SUCCESS)], max_consecutive_noops=3)
    assert res.status == "SUCCESS"
    assert res.productive_ticks == 2          # the no-op did NOT count as productive
    assert res.invokes == 3                   # but it did consume an invoke
    assert fd.sleeps == 1                      # one no-op backoff


def test_persistent_noop_exhausts():
    res, _ = run(None, [("", None)] * 6, max_consecutive_noops=3)
    assert res.status == "NOOP_EXHAUSTED"


# --------------------------------------------------------------------------- livelock
def test_livelock_repeated_tree_is_stuck():
    res, _ = run(None, [("talk", T_A)] * 6, stuck_n=2, max_ticks=20)
    assert res.status == "STUCK"


def test_oscillation_is_stuck():
    res, _ = run(None, [("talk", T_A), ("talk", T_B)] * 4, stuck_n=2, max_ticks=20)
    assert res.status == "STUCK"


# --------------------------------------------------------------------------- backstops
def test_max_ticks_backstop():
    # always distinct (growing done-set) -> always "progress" -> never STUCK -> hits max_ticks.
    script = [("", tree("active", done=[f"S{i}" for i in range(k + 1)])) for k in range(6)]
    res, _ = run(None, script, max_ticks=3, stuck_n=10)
    assert res.status == "MAX_TICKS" and res.productive_ticks == 3
    assert res.final_state == "active"  # non-terminal -> resumable later


def test_wallclock_backstop():
    # now() jumps past the budget on the first loop check -> stop before any invoke.
    clock = iter([0, 10_000, 10_000, 10_000])
    res, fd = run(None, [("", T_SUCCESS)], wallclock_budget=100, now=lambda: next(clock))
    assert res.status == "WALLCLOCK" and res.invokes == 0 and fd.prompts == []


# --------------------------------------------------------------------------- resume nudge
def test_no_nudge_on_fresh_tick0():
    res, fd = run(None, [("done", T_SUCCESS)])
    assert fd.prompts[0] == BASE_PROMPT  # fresh start -> base prompt verbatim, no suffix


def test_nudge_when_nonterminal_tree_exists():
    # seeded ACTIVE tree at tick 0 -> the very first invoke must carry the resume suffix,
    # and must NAME the ✝ dead node explicitly (T_A's dead node is "S1").
    res, fd = run(T_A, [("done", T_SUCCESS)])
    p = fd.prompts[0]
    assert p.startswith(BASE_PROMPT)                       # base intact + unmodified
    assert PLAN_PATH in p and "✝" in p
    assert "proven dead" in p.lower() and "S1" in p        # dead-set named, not just referenced
    assert res.status == "SUCCESS"


def test_resume_suffix_names_dead_methods():
    from drive import dead_nodes, resume_suffix
    t = tree("active", dead=["S1"], frontier="S2")
    assert dead_nodes(t) == ["S1  m"]                       # id + method, up to the ✝ marker
    s = resume_suffix("/p/plan-tree.md", dead=["S1  alfa (primary)"])
    assert "S1  alfa (primary)" in s and "MUST NOT" in s


def test_seeded_success_not_redone():
    # a seeded terminal tree at tick 0 -> stop immediately, never invoke (test_12 path).
    res, fd = run(T_SUCCESS, [("x", T_SUCCESS)])
    assert res.status == "SUCCESS" and res.invokes == 0 and fd.prompts == []


# --------------------------------------------------------------------------- parsers
def test_parse_state_robustness():
    assert parse_state("# P: t   STATE: SUCCESS") == "SUCCESS"
    assert parse_state("state:  success   (done)") == "SUCCESS"
    assert parse_state("x\r\nSTATE:\tactive\r\n") == "active"
    assert parse_state("...   STATE: EXHAUSTION-STOP — frontier empty") == "EXHAUSTION-STOP"
    assert parse_state("STATE: GUARD-HALT") == "GUARD-HALT"
    assert parse_state("# Plan-Tree: t   STAT") is None         # torn header
    assert parse_state("STATE: WEIRD") is None                  # unrecognized
    assert parse_state(None) is None and parse_state("") is None


def test_fingerprint_ignores_receipt_rewording_but_sees_progress():
    a1 = tree("active", dead=["S1"], frontier="S2")
    a2 = a1.replace("✝ down", "✝ DNS failure, non-transient, re-checked")  # same node, reworded
    assert fingerprint(a1) == fingerprint(a2)                   # cosmetic rewording != progress
    b = tree("active", dead=["S1", "S2"], frontier="S3")        # a node moved to dead
    assert fingerprint(a1) != fingerprint(b)                    # real progress IS seen
    torn = "# Plan-Tree: t   STAT\nINTENT broken"
    assert fingerprint(torn)[0] == "hash"                       # unparseable -> stable hash
