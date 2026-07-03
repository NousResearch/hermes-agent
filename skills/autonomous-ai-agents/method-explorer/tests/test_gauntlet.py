"""Meta-test for the gauntlet gate in run.py — container-free, deterministic, no tokens.

Pins the hardened escalation-gate semantics via dependency injection (mirroring
test_15's fake-invoke style): per-tier retry policy, diagnostics-BEFORE-retry ordering
(a retry self-resets artifacts, destroying the failed attempt's evidence), FLAKY-PASS
continuation, fail-twice red with resume hint, infra abort (exit 2), preflight abort,
--survey no-stop, and the JSONL trend log.

Run:  python3 run.py test_gauntlet
"""
import contextlib
import io
import json
import os
import tempfile
import types

import run as runner

# Fake ladder using REAL tier indices (retry/offline/cost policy is index-based).
FAKE_TIERS = [
    (0, "T0", "offline fakes", ["m0"]),
    (1, "T1", "convention fakes", ["m1"]),
    (2, "T2", "core fakes", ["m2"]),
    (3, "T3", "harder fakes", ["m3"]),
]


def _mk_module(*fn_names):
    mod = types.SimpleNamespace()
    for n in fn_names:
        def f():
            pass
        f.__name__ = n
        setattr(mod, n, f)
    return mod


def _world(script, preflight=(), infra=False):
    """script: fn_name -> list of per-call outcomes ('pass'/'fail'), consumed in order."""
    mods = {m: _mk_module(f"test_{m}") for _, _, _, ms in FAKE_TIERS for m in ms}
    events, logs = [], []

    def run_fn(fn):
        events.append(("call", fn.__name__))
        s = script[fn.__name__].pop(0)
        return s, ("boom" if s == "fail" else "")

    kw = dict(
        tiers_list=FAKE_TIERS,
        get_module=lambda n: mods[n],
        run_fn=run_fn,
        diagnose=lambda mod, modname, err: events.append(("diag", modname)),
        infra_down=lambda: infra,
        log=lambda e: logs.append(dict(e)),
        preflight_fn=lambda: list(preflight),
    )
    return kw, events, logs


def _run(kw, **opts):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = runner.run_gauntlet(**{**opts, **kw})
    return code, out.getvalue()


def test_no_retry_at_t0_t1():
    # A T1 failure would pass on retry — but T1 has NO retry policy: flaky T1 = red.
    kw, events, _ = _world({"test_m0": ["pass"], "test_m1": ["fail", "pass"],
                            "test_m2": ["pass"], "test_m3": ["pass"]})
    code, out = _run(kw)
    assert code == 1, "flaky T1 must be RED (no retry below T2)"
    assert events.count(("call", "test_m1")) == 1, "T1 must not retry"
    assert "--from 1" in out, "red stop must print the exact resume command"


def test_retry_once_flaky_pass_continues():
    # T2 fails then passes on retry -> FLAKY-PASS; the ladder continues to T3 and greens.
    kw, events, logs = _world({"test_m0": ["pass"], "test_m1": ["pass"],
                               "test_m2": ["fail", "pass"], "test_m3": ["pass"]})
    code, out = _run(kw)
    assert code == 0, "FLAKY-PASS must not stop the ladder"
    assert ("call", "test_m3") in events, "ladder must continue past a FLAKY-PASS"
    assert "FLAKY-PASS" in out
    assert any(l.get("outcome") == "flaky-pass" for l in logs), "flake must be logged"


def test_diagnostics_render_before_retry():
    # The failed attempt's artifacts are wiped by a retry — diag MUST come first.
    kw, events, _ = _world({"test_m0": ["pass"], "test_m1": ["pass"],
                            "test_m2": ["fail", "pass"], "test_m3": ["pass"]})
    _run(kw)
    i_diag = events.index(("diag", "m2"))
    i_retry = len(events) - 1 - events[::-1].index(("call", "test_m2"))
    assert i_diag < i_retry, "diagnostics must render BEFORE the retry wipes artifacts"


def test_fail_twice_is_red_and_stops():
    kw, events, _ = _world({"test_m0": ["pass"], "test_m1": ["pass"],
                            "test_m2": ["fail", "fail"], "test_m3": ["pass"]})
    code, out = _run(kw)
    assert code == 1
    assert ("call", "test_m3") not in events, "higher tier must NOT run after a red tier"
    assert "TIER 2" in out and "--from 2" in out


def test_no_retry_when_reps_gt1():
    # --reps N measures pass-rate explicitly: conjunctive, no retries.
    kw, events, _ = _world({"test_m0": ["pass", "pass"], "test_m1": ["pass", "pass"],
                            "test_m2": ["fail", "pass", "pass"], "test_m3": ["pass"]})
    code, _ = _run(kw, reps=2)
    assert code == 1, "a failed rep is red at reps>1 (no retry)"
    assert events.count(("call", "test_m2")) == 2, "exactly reps calls, no retry call"


def test_infra_abort_exits_2():
    kw, events, logs = _world({"test_m0": ["pass"], "test_m1": ["pass"],
                               "test_m2": ["fail", "pass"], "test_m3": ["pass"]},
                              infra=True)
    code, out = _run(kw)
    assert code == 2, "persistent no-op is INFRA (exit 2), not a tier-red"
    assert ("call", "test_m3") not in events
    assert events.count(("call", "test_m2")) == 1, "no retry against a dead backend"
    assert "INFRA" in out and logs[-1]["outcome"] == "infra"


def test_preflight_abort_before_any_module():
    kw, events, logs = _world({"test_m0": ["pass"]}, preflight=["scenario missing: x.json"])
    code, out = _run(kw)
    assert code == 2
    assert not events, "preflight failure must abort before ANY module runs"
    assert logs and logs[0]["module"] == "(preflight)"
    assert "INFRA" in out


def test_survey_never_stops_early():
    # T1 red (no retry) but survey runs every tier and reports a stratified scorecard.
    kw, events, _ = _world({"test_m0": ["pass"], "test_m1": ["fail"],
                            "test_m2": ["pass"], "test_m3": ["pass"]})
    code, out = _run(kw, survey=True)
    assert code == 1, "survey exit reflects reds"
    assert ("call", "test_m3") in events, "survey must not stop at a red tier"
    assert "SURVEY SCORECARD" in out and "RED" in out


def test_jsonl_log_lines_written():
    path = os.path.join(tempfile.gettempdir(), "gauntlet-log-test.jsonl")
    if os.path.exists(path):
        os.remove(path)
    kw, _, _ = _world({"test_m0": ["pass"], "test_m1": ["pass"],
                       "test_m2": ["pass"], "test_m3": ["pass"]})
    kw["log"] = lambda e: runner._log_outcome(e, path=path)
    code, _ = _run(kw)
    assert code == 0
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    assert len(rows) == 4 and all("ts" in r and r["outcome"] == "pass" for r in rows)
    os.remove(path)
