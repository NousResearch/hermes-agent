#!/usr/bin/env python3
"""A/B/C latency eval for the computer_use `sequence` primitive (RFC #112639).

Arms:
  A  reactive: capture, then one computer_use call per action (last with capture_after=true)
  B  sequence with capture_after=true, verify_mode="som"
  C  sequence with capture_after=true, verify_mode="ax_first"

Scenarios (deterministic, real X11):
  1  single_field_submit  click textbox -> type -> submit via Return
  2  two_field_tab        click field1 -> type -> Tab -> type -> Return
  3  shortcut_chain       click text -> ctrl+a -> type -> ctrl+z (undo restores)
  4  abort_mid_slice      click field -> type -> ctrl+alt+delete (hard-blocked: must abort cleanly)

Method: every scenario x arm x run builds a fresh tkinter app on Xvfb, installs X11EvalBackend
into the tool's session cache, and runs a scripted policy that sleeps --model-latency seconds
per "model decision" — the LLM round trip the primitive eliminates. Two clocks are reported:
wall_ms (includes the simulated model latency) and tool_ms (measured handle_computer_use time,
no simulation). The gate in the RFC is about wall time on eligible tasks; tool_ms keeps the
report honest about how much is simulation.

Honesty notes (also in README.md):
- The backend is real pixels + real XTest input, but it is NOT cua-driver (no AX tree, no
  pid-scoped posting). Arm deltas are the evidence; absolute ms are Xvfb-box-specific.
- Aux-vision routing is disabled for the run: the eval must not make network/model calls.
- Requires DISPLAY (Xvfb) and HERMES_YOLO_MODE=1: the policy is scripted, there is no human to
  approve, and the inputs go to a synthetic app on a headless display.
"""

import argparse
import json
import os
import sys
import time

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EVAL_DIR))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, EVAL_DIR)

import tkinter as tk  # noqa: E402

from x11_backend import EvalApp, EvalWidget, X11EvalBackend  # noqa: E402  (no approval side effects)


# --------------------------------------------------------------------------
# scripted policy: stands in for the model; each decide() is one LLM round trip
# --------------------------------------------------------------------------

class ScriptedPolicy:
    def __init__(self, latency_s: float):
        self.latency_s = latency_s
        self.api_calls = 0

    def decide(self, make_call):
        time.sleep(self.latency_s)
        self.api_calls += 1
        return make_call()


class Harness:
    """Counts computer_use calls and measures real tool time around handle_computer_use."""

    def __init__(self, cu_tool, sid: str):
        self.cu_tool = cu_tool
        self.sid = sid
        self.cu_calls = 0
        self.tool_ms = 0.0

    def call(self, args):
        t = time.perf_counter()
        out = self.cu_tool.handle_computer_use(args, session_id=self.sid)
        self.tool_ms += (time.perf_counter() - t) * 1000.0
        self.cu_calls += 1
        return json.loads(out) if isinstance(out, str) else out


def _action_ok(resp) -> bool:
    if not isinstance(resp, dict):
        return False
    if resp.get("_multimodal"):
        inner = resp.get("sequence_result") or resp.get("action_result") or {}
        return _action_ok(inner)
    if "error" in resp:
        return False
    return resp.get("ok", True) is not False


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------

class Scenario:
    name = ""
    expect_abort_step = None

    def build(self):
        raise NotImplementedError

    def steps(self, backend):
        raise NotImplementedError

    def verify(self, state, outcome):
        raise NotImplementedError


class SingleFieldSubmit(Scenario):
    name = "single_field_submit"

    def build(self):
        root = tk.Tk()
        root.title("lat-s1")
        root.geometry("420x220+80+80")
        entry = tk.Entry(root, width=32)
        entry.pack(pady=18, padx=20)
        status = tk.Label(root, text="idle")
        status.pack(pady=6)

        def submit(ev=None):
            status.config(text="submitted:" + entry.get())

        root.bind("<Return>", submit)
        go = tk.Button(root, text="Go", command=submit)
        go.pack(pady=6)
        app = EvalApp("s1", root, [EvalWidget(entry, "AXTextField", "name field"),
                                   EvalWidget(go, "AXButton", "Go")])
        return {"root": root, "app": app, "entry": entry, "status": status}

    def steps(self, backend):
        return [
            {"action": "click", "element": backend.find_element("AXTextField", "name")},
            {"action": "type", "text": "hello hermes"},
            {"action": "key", "keys": "Return"},
        ]

    def verify(self, state, outcome):
        ok = state["status"].cget("text") == "submitted:hello hermes"
        return ok, state["status"].cget("text")


class TwoFieldTab(Scenario):
    name = "two_field_tab"

    def build(self):
        root = tk.Tk()
        root.title("lat-s2")
        root.geometry("420x260+80+80")
        e1 = tk.Entry(root, width=32)
        e1.pack(pady=10, padx=20)
        e2 = tk.Entry(root, width=32)
        e2.pack(pady=10, padx=20)
        status = tk.Label(root, text="idle")
        status.pack(pady=6)

        def submit(ev=None):
            status.config(text=f"submitted:{e1.get()}|{e2.get()}")

        root.bind("<Return>", submit)
        app = EvalApp("s2", root, [EvalWidget(e1, "AXTextField", "first name"),
                                   EvalWidget(e2, "AXTextField", "email")])
        return {"root": root, "app": app, "status": status}

    def steps(self, backend):
        return [
            {"action": "click", "element": backend.find_element("AXTextField", "first")},
            {"action": "type", "text": "ada"},
            {"action": "key", "keys": "Tab"},
            {"action": "type", "text": "x@y.z"},
            {"action": "key", "keys": "Return"},
        ]

    def verify(self, state, outcome):
        ok = state["status"].cget("text") == "submitted:ada|x@y.z"
        return ok, state["status"].cget("text")


class ShortcutChain(Scenario):
    name = "shortcut_chain"

    def build(self):
        root = tk.Tk()
        root.title("lat-s3")
        root.geometry("440x240+80+80")
        txt = tk.Text(root, width=36, height=5, undo=True)
        txt.insert("1.0", "original")
        txt.pack(pady=12, padx=16)
        app = EvalApp("s3", root, [EvalWidget(txt, "AXTextArea", "notes")])
        return {"root": root, "app": app, "txt": txt}

    def steps(self, backend):
        return [
            {"action": "click", "element": backend.find_element("AXTextArea", "notes")},
            {"action": "key", "keys": "ctrl+a"},
            {"action": "type", "text": "v2"},
            {"action": "key", "keys": "ctrl+z"},
        ]

    def verify(self, state, outcome):
        got = state["txt"].get("1.0", "end-1c")
        return got == "original", repr(got)


class AbortMidSlice(Scenario):
    name = "abort_mid_slice"
    expect_abort_step = 2

    def build(self):
        root = tk.Tk()
        root.title("lat-s4")
        root.geometry("420x200+80+80")
        entry = tk.Entry(root, width=32)
        entry.pack(pady=18, padx=20)
        app = EvalApp("s4", root, [EvalWidget(entry, "AXTextField", "name field")])
        return {"root": root, "app": app, "entry": entry}

    def steps(self, backend):
        # The failing step passes V1 validation but is hard-blocked by the per-step safety path
        # (ctrl+alt+delete): this proves safety runs inside the slice, not just at the top-level call.
        return [
            {"action": "click", "element": backend.find_element("AXTextField", "name")},
            {"action": "type", "text": "zz"},
            {"action": "key", "keys": "ctrl+alt+delete"},
        ]

    def verify(self, state, outcome):
        intact = state["entry"].get() == "zz"
        clean = outcome["abort_step"] == self.expect_abort_step
        return intact and clean, f"entry={state['entry'].get()!r} abort_step={outcome['abort_step']}"


SCENARIOS = [SingleFieldSubmit(), TwoFieldTab(), ShortcutChain(), AbortMidSlice()]
ARMS = ("A", "B", "C")


# --------------------------------------------------------------------------
# arm runners
# --------------------------------------------------------------------------

def run_reactive(h, policy, backend, scenario):
    """Arm A: one computer_use call per action, like the current reactive loop."""
    policy.decide(lambda: h.call({"action": "capture", "mode": "som"}))
    steps = scenario.steps(backend)
    executed = 0
    for i, sa in enumerate(steps):
        args = dict(sa)
        if i == len(steps) - 1:
            args["capture_after"] = True
        resp = policy.decide(lambda a=args: h.call(a))
        if not _action_ok(resp):
            return {"finished": False, "abort_step": i, "actions_executed": executed,
                    "slice_length": len(steps)}
        executed += 1
    return {"finished": True, "abort_step": None, "actions_executed": executed,
            "slice_length": len(steps)}


def run_sequenced(h, policy, backend, scenario, arm):
    """Arms B/C: one capture, one sequence call, one final verification capture."""
    policy.decide(lambda: h.call({"action": "capture", "mode": "som"}))
    steps = scenario.steps(backend)
    resp = policy.decide(lambda: h.call({
        "action": "sequence", "steps": steps,
        "capture_after": True, "verify_mode": "som" if arm == "B" else "ax_first"}))
    m = {}
    if isinstance(resp, dict):
        m = ((resp.get("sequence_result") or {}).get("sequence_metrics")
             or resp.get("sequence_metrics") or {})
    return {"finished": bool(m.get("success")), "abort_step": m.get("slice_abort_step"),
            "actions_executed": m.get("actions_executed", 0), "slice_length": len(steps),
            "sequence_metrics": m}


def one_run(scenario, arm, run_idx, latency_s, cu_tool):
    backend = X11EvalBackend()
    sid = f"latency-eval-{scenario.name}-{arm}-{run_idx}"
    state = scenario.build()
    backend.add_app(state["app"])
    backend.start()
    backend.pump()
    time.sleep(0.5)
    backend.pump()
    with cu_tool._backend_lock:
        cu_tool._install_backend(cu_tool._scoped_sid(sid), backend,
                                cu_tool._cua_permission_mode(sid))
    h, policy = Harness(cu_tool, sid), ScriptedPolicy(latency_s)
    wall_t0 = time.perf_counter()
    try:
        if arm == "A":
            outcome = run_reactive(h, policy, backend, scenario)
        else:
            outcome = run_sequenced(h, policy, backend, scenario, arm)
        wall_ms = (time.perf_counter() - wall_t0) * 1000.0
        success, detail = scenario.verify(state, outcome)
    finally:
        cu_tool.release_computer_use_session(sid)
        state["root"].destroy()
    actions = outcome["actions_executed"] or 0
    run = {
        "scenario": scenario.name, "arm": arm, "run": run_idx,
        "wall_ms": wall_ms, "tool_ms": h.tool_ms,
        "api_calls": policy.api_calls, "computer_use_calls": h.cu_calls,
        "captures": backend.captures_taken, "image_captures": backend.images_produced,
        "actions_executed": actions, "slice_length": outcome["slice_length"],
        "abort_step": outcome["abort_step"], "success": bool(success), "detail": detail,
        "llm_round_trips_per_action": (policy.api_calls / actions) if actions else None,
        "ms_per_successful_action": (wall_ms / actions) if (success and actions) else None,
    }
    if arm in ("B", "C"):
        m = outcome["sequence_metrics"]
        run["sequence_tool_ms"] = m.get("tool_ms")
        run["sequence_capture_ms"] = m.get("capture_ms")
        run["sequence_verification_ms"] = m.get("verification_ms")
        run["verify_mode_used"] = m.get("verify_mode_used")
    return run


# --------------------------------------------------------------------------
# aggregation + gate
# --------------------------------------------------------------------------

def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def aggregate(runs):
    n = len(runs)
    ok = [r for r in runs if r["success"]]
    mean = lambda k: sum(r[k] for r in runs) / n
    return {
        "n": n,
        "success_rate": len(ok) / n,
        "wall_ms_p50": _pct([r["wall_ms"] for r in runs], 0.5),
        "wall_ms_p95": _pct([r["wall_ms"] for r in runs], 0.95),
        "tool_ms_mean": mean("tool_ms"),
        "api_calls_mean": mean("api_calls"),
        "computer_use_calls_mean": mean("computer_use_calls"),
        "captures_mean": mean("captures"),
        "image_captures_mean": mean("image_captures"),
        "actions_mean": mean("actions_executed"),
        "ms_per_successful_action_mean": (
            sum(r["ms_per_successful_action"] for r in ok) / len(ok) if ok else None),
        "round_trips_per_action_mean": (
            sum(r["llm_round_trips_per_action"] for r in ok if r["llm_round_trips_per_action"])
            / len(ok) if ok else None),
    }


def main():
    ap = argparse.ArgumentParser(description="A/B/C latency eval for computer_use sequence")
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--scenarios", default="1,2,3,4")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--model-latency", type=float, default=1.5,
                    help="seconds of simulated LLM latency per model decision")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if os.environ.get("HERMES_YOLO_MODE") != "1":
        sys.exit("refusing: re-run with HERMES_YOLO_MODE=1 (scripted policy, synthetic app, headless display)")
    if not os.environ.get("DISPLAY"):
        sys.exit("refusing: set DISPLAY (run under Xvfb, e.g. DISPLAY=:99)")

    from tools.computer_use import tool as cu_tool  # noqa: E402  (after env checks: approval frozen at import)
    # Hermetic: the eval measures orchestration, so it must not make network/model calls.
    cu_tool._should_route_through_aux_vision = lambda: False

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip().upper() in ARMS]
    scen_idxs = {int(s) for s in args.scenarios.split(",") if s.strip()}
    scenarios = [s for i, s in enumerate(SCENARIOS, 1) if i in scen_idxs]
    if not arms or not scenarios:
        sys.exit("nothing to run: check --arms / --scenarios")

    runs = []
    total = len(scenarios) * len(arms) * args.runs
    done = 0
    for sc in scenarios:
        for arm in arms:
            for i in range(args.runs):
                done += 1
                print(f"[{done}/{total}] {sc.name} arm={arm} run={i}", flush=True)
                runs.append(one_run(sc, arm, i, args.model_latency, cu_tool))

    by = {}
    for sc in scenarios:
        by[sc.name] = {arm: aggregate([r for r in runs
                                       if r["scenario"] == sc.name and r["arm"] == arm])
                       for arm in arms}

    # promotion gate (RFC #112639): no success regression, fewer model calls,
    # ~30%+ p50 wall improvement on eligible tasks, clean aborts on the failure slice
    gate = {}
    for sc in scenarios:
        g = {"scenario": sc.name}
        if sc.name == "abort_mid_slice":
            arm_runs = {arm: [r for r in runs if r["scenario"] == sc.name and r["arm"] == arm]
                        for arm in arms}
            g["clean_abort"] = {
                arm: all(r["success"] and r["abort_step"] == sc.expect_abort_step for r in rs)
                for arm, rs in arm_runs.items()}
            g["pass"] = all(g["clean_abort"].values())
        else:
            a, b, c = (by[sc.name].get(x) for x in ("A", "B", "C"))
            g["no_success_regression"] = ((b is None or b["success_rate"] >= a["success_rate"])
                                          and (c is None or c["success_rate"] >= a["success_rate"]))
            g["fewer_api_calls"] = ((b is None or b["api_calls_mean"] < a["api_calls_mean"])
                                    and (c is None or c["api_calls_mean"] < a["api_calls_mean"]))
            g["p50_improvement_B"] = (a["wall_ms_p50"] - b["wall_ms_p50"]) / a["wall_ms_p50"] if b else None
            g["p50_improvement_C"] = (a["wall_ms_p50"] - c["wall_ms_p50"]) / a["wall_ms_p50"] if c else None
            g["p50_gate_30pct_B"] = g["p50_improvement_B"] is not None and g["p50_improvement_B"] >= 0.30
            g["p50_gate_30pct_C"] = g["p50_improvement_C"] is not None and g["p50_improvement_C"] >= 0.30
            g["pass"] = g["no_success_regression"] and g["fewer_api_calls"] and (
                (b is None or g["p50_gate_30pct_B"]) and (c is None or g["p50_gate_30pct_C"]))
        gate[sc.name] = g

    report = {"runs": runs, "aggregates": by, "gate": gate,
              "config": {"model_latency_s": args.model_latency, "arms": arms,
                         "scenarios": [s.name for s in scenarios], "runs_per_arm": args.runs,
                         "aux_vision_routing": "disabled (hermetic)",
                         "backend": "X11EvalBackend (real X11, not cua-driver)"}}
    out_path = args.out or os.path.join(
        EVAL_DIR, "results", f"latency-{time.strftime('%Y%m%d-%H%M%S')}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== aggregates (wall_ms includes simulated model latency; tool_ms does not) ===")
    for sc in scenarios:
        print(f"\n-- {sc.name} --")
        for arm in arms:
            g = by[sc.name][arm]
            msa = f"{g['ms_per_successful_action_mean']:.0f}" if g["ms_per_successful_action_mean"] else "-"
            rt = f"{g['round_trips_per_action_mean']:.2f}" if g["round_trips_per_action_mean"] else "-"
            print(f"  arm {arm}: n={g['n']} ok={g['success_rate']:.0%} "
                  f"p50={g['wall_ms_p50']:.0f}ms p95={g['wall_ms_p95']:.0f}ms "
                  f"tool={g['tool_ms_mean']:.0f}ms api={g['api_calls_mean']:.1f} "
                  f"cu_calls={g['computer_use_calls_mean']:.1f} caps={g['captures_mean']:.1f} "
                  f"img={g['image_captures_mean']:.1f} ms/action={msa} rt/action={rt}")
    print("\n=== promotion gate ===")
    all_pass = True
    for sc in scenarios:
        g = gate[sc.name]
        status = "PASS" if g["pass"] else "FAIL"
        all_pass = all_pass and g["pass"]
        if sc.name == "abort_mid_slice":
            print(f"  {sc.name}: {status} clean_abort={g['clean_abort']}")
        else:
            print(f"  {sc.name}: {status} no_regress={g['no_success_regression']} "
                  f"fewer_calls={g['fewer_api_calls']} "
                  f"p50ΔB={g['p50_improvement_B']:+.0%} p50ΔC={g['p50_improvement_C']:+.0%}")
    print(f"\nreport: {out_path}")
    print("GATE:", "PASS" if all_pass else "FAIL")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
