#!/usr/bin/env python3
"""Reversible activation preflight for agentpod-stop-check.

The previous activation plan was "copy the plugin into ``~/.hermes/plugins/``,
add config, restart". Against an installed core that does not yet carry
``agent.pre_verify_on_no_edit_turns`` that plan **silently ships a dead gate**:
``pre_verify`` never fires on a no-edit turn, only the preemptable
``transform_llm_output`` fallback survives, and the config reads as if the gate
were on. This script exists so that failure cannot happen unnoticed.

It is **read-only**. It writes nothing, installs nothing, restarts nothing, and
never touches the installed tree, live plugins, config, profiles or cron. It
answers one question with an exit code:

    is it safe to enable this plugin against THIS installed runtime, right now?

Checks, in the order the deployment must perform them:

1. **Installed core carries the capability** — not by grepping source, but by
   importing ``agent.verify_hooks`` *in the installed checkout's own
   interpreter* and confirming both that the setting resolver exists and that
   ``agent.conversation_loop.run_conversation`` actually references it (read
   off the compiled code object). An old core fails here.
2. **Default-off is intact** — with ``agent.pre_verify_on_no_edit_turns``
   absent or false the resolver must return False, so step 3 can be staged
   inert.
3. **Config scope is real** — the plugin must be scoped to at least one
   session id, and a configured ``project_id``/``tenant`` must match more than
   zero cards on the board it will sweep. An empty scope is refused (it would
   report a clean board while covering nothing).

Usage (nothing is changed by any of these):

    python activation_preflight.py --core-root ~/.hermes/hermes-agent
    python activation_preflight.py --core-root ~/.hermes/hermes-agent \
        --config ~/.hermes/config.yaml --board-db ~/.hermes/kanban/board.db

Exit codes: 0 = every gate passed (safe to proceed to the NEXT reviewed step);
1 = at least one gate refused; 2 = the preflight could not run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CAPABILITY = "pre_verify_on_no_edit_turns"

# Executed inside the INSTALLED checkout's interpreter. Imports only; it
# resolves the capability from the real modules and prints JSON on stdout.
_PROBE = r"""
import json, sys
out = {"import_error": None, "resolver": False, "call_site": False,
       "default_off": None, "revision": None,
       "verdict_api": False, "verdict_enforced": False, "verdict_inert": None,
       "registry_binding": False}
try:
    import agent.verify_hooks as vh
    out["resolver"] = hasattr(vh, "CAPABILITY_NAME")
    if out["resolver"]:
        # Default-off must hold with the key absent AND explicitly false.
        fn = getattr(vh, "CAPABILITY_NAME")
        out["default_off"] = (fn({}) is False
                              and fn({"agent": {"CAPABILITY_NAME": False}}) is False)
    # Enforced-verdict contract: the API exists, the finalizer calls it, and it
    # is inert with nothing pending.
    out["verdict_api"] = (hasattr(vh, "apply_pre_verify_verdict")
                          and hasattr(vh, "record_pre_verify_verdict"))
    if out["verdict_api"]:
        class _A:
            pass
        out["verdict_inert"] = (
            vh.apply_pre_verify_verdict(_A(), "untouched") == "untouched"
        )
    import agent.turn_finalizer as tf
    seen, stack = set(), [tf.finalize_turn.__code__]
    while stack:
        code = stack.pop()
        if id(code) in seen:
            continue
        seen.add(id(code))
        if "apply_pre_verify_verdict" in (code.co_names + code.co_varnames):
            out["verdict_enforced"] = True
            break
        for const in code.co_consts:
            if hasattr(const, "co_names"):
                stack.append(const)
    import dataclasses
    from tools.process_registry import ProcessSession
    out["registry_binding"] = any(
        f.name == "kanban_task_id" for f in dataclasses.fields(ProcessSession)
    )
    import agent.conversation_loop as cl
    seen, stack = set(), [cl.run_conversation.__code__]
    while stack:
        code = stack.pop()
        if id(code) in seen:
            continue
        seen.add(id(code))
        if "CAPABILITY_NAME" in code.co_names:
            out["call_site"] = True
            break
        for const in code.co_consts:
            if hasattr(const, "co_names"):
                stack.append(const)
except Exception as exc:
    out["import_error"] = "%s: %s" % (type(exc).__name__, exc)
print(json.dumps(out))
""".replace("CAPABILITY_NAME", CAPABILITY)


class Gate:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.lines: list[str] = []

    def check(self, ok: bool, name: str, detail: str) -> bool:
        self.lines.append(f"[{'PASS' if ok else 'REFUSE'}] {name}: {detail}")
        if not ok:
            self.failures.append(name)
        return ok

    def note(self, detail: str) -> None:
        self.lines.append(f"[info] {detail}")


def _interpreter_for(core_root: Path) -> Path:
    """The installed checkout's own interpreter (never this worktree's)."""
    for rel in (".venv/bin/python", "venv/bin/python",
                ".venv/Scripts/python.exe", "venv/Scripts/python.exe"):
        cand = core_root / rel
        if cand.exists():
            return cand
    return Path(sys.executable)


def probe_core(core_root: Path, gate: Gate) -> None:
    """Gate 1+2 — the INSTALLED core's real, imported capability."""
    core_root = core_root.expanduser()
    if not (core_root / "agent" / "verify_hooks.py").exists():
        gate.check(False, "installed core", f"{core_root} is not a Hermes checkout")
        return
    python = _interpreter_for(core_root)
    gate.note(f"probing {core_root} with {python}")
    try:
        rev = subprocess.run(
            ["git", "-C", str(core_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(core_root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        gate.note(f"installed revision: {branch or '?'} @ {rev or '?'}")
    except Exception as exc:  # pragma: no cover - diagnostics only
        gate.note(f"revision unavailable: {exc}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(core_root)
    try:
        proc = subprocess.run(
            [str(python), "-c", _PROBE],
            cwd=str(core_root), env=env, capture_output=True, text=True, timeout=180,
        )
        data = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except Exception as exc:
        gate.check(False, "installed core", f"capability probe failed: {exc}")
        return

    if data.get("import_error"):
        gate.check(False, "installed core", f"import failed: {data['import_error']}")
        return
    gate.check(
        bool(data.get("resolver")),
        "installed core resolver",
        f"agent.verify_hooks.{CAPABILITY} "
        + ("is present" if data.get("resolver")
           else "is MISSING — this core predates the change; enabling the plugin "
                "here ships a dead gate"),
    )
    gate.check(
        bool(data.get("call_site")),
        "installed core call site",
        "run_conversation references the setting"
        if data.get("call_site")
        else "run_conversation does NOT reference the setting — pre_verify would "
             "never fire on a no-edit turn",
    )
    gate.check(
        data.get("default_off") is True,
        "default-off",
        "resolver returns False when absent and when explicitly false"
        if data.get("default_off") is True
        else "default-off could not be confirmed — do not stage against this core",
    )
    gate.check(
        bool(data.get("verdict_api")) and data.get("verdict_inert") is True,
        "enforced-verdict contract",
        "agent.verify_hooks.apply_pre_verify_verdict is present and inert with "
        "nothing pending"
        if data.get("verdict_api") and data.get("verdict_inert") is True
        else "MISSING or not inert — without it a spent continuation budget can "
             "still end a supervision turn quietly",
    )
    gate.check(
        bool(data.get("verdict_enforced")),
        "enforced-verdict call site",
        "finalize_turn applies the verdict after the output transforms"
        if data.get("verdict_enforced")
        else "finalize_turn does NOT apply it — enforcement would depend on "
             "transform ordering again",
    )
    gate.check(
        bool(data.get("registry_binding")),
        "process-registry card binding",
        "ProcessSession records kanban_task_id at spawn"
        if data.get("registry_binding")
        else "ProcessSession has no kanban_task_id — real workers would be "
             "owner_unknown even when correctly pinned",
    )


# Shipped defaults for the two caps the stale-verdict window depends on.
# They are SAFE as shipped: the plugin's own cap (2) is reached first, so the
# gate re-evaluates and a verdict recorded earlier in the turn is cleared when
# the continuation actually resolved the board.
DEFAULT_PLUGIN_MAX_CONTINUATIONS = 2
DEFAULT_CORE_MAX_VERIFY_NUDGES = 3


def evaluate_cap_ordering(cfg: dict) -> tuple[bool, int, int, str]:
    """Gate the core-cap / plugin-cap ordering. Pure function, config in.

    The ``pre_verify`` call site only re-evaluates while
    ``attempt < agent.max_verify_nudges``. If the CORE cap is reached first
    (``max_verify_nudges <= max_continuations``) the verdict recorded at the
    first evaluation is frozen and ships even when the continuation then really
    resolved the board — a stale fail-explicit verdict on finished work.

    This is a configuration foot-gun, not a runtime defect: with the shipped
    defaults (2 vs 3) the plugin's cap is hit first and re-evaluation clears the
    verdict. So the preflight REFUSES the bad ordering with the exact required
    setting instead of adding new runtime behavior.

    Returns ``(ok, nudges, continuations, detail)``.
    """
    sc = (cfg.get("agentpod_stop_check") or {})
    agent_cfg = (cfg.get("agent") or {})

    def _as_int(raw, default):
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return default

    continuations = _as_int(
        sc.get("max_continuations"), DEFAULT_PLUGIN_MAX_CONTINUATIONS)
    nudges = _as_int(
        agent_cfg.get("max_verify_nudges"), DEFAULT_CORE_MAX_VERIFY_NUDGES)
    ok = nudges > continuations
    if ok:
        detail = (
            f"agent.max_verify_nudges={nudges} > "
            f"agentpod_stop_check.max_continuations={continuations}: the "
            "plugin's cap is reached first, so the last evaluation clears a "
            "verdict the continuation resolved"
        )
    else:
        detail = (
            f"agent.max_verify_nudges={nudges} <= "
            f"agentpod_stop_check.max_continuations={continuations}: the CORE "
            "cap stops re-evaluation first, so a verdict recorded earlier in "
            "the turn would ship even after the continuation resolved the "
            f"board. Required: set agent.max_verify_nudges to at least "
            f"{continuations + 1} (or lower "
            f"agentpod_stop_check.max_continuations below {nudges})."
        )
    return ok, nudges, continuations, detail


def probe_scope(config_path: Path | None, board_db: Path | None, gate: Gate) -> None:
    """Gate 3 — the configured scope must select a real, non-empty set."""
    if config_path is None:
        gate.note("no --config given: scope gate skipped (must be re-run before enabling)")
        gate.check(False, "config scope", "not evaluated — rerun with --config")
        return
    config_path = config_path.expanduser()
    if not config_path.exists():
        gate.check(False, "config scope", f"{config_path} does not exist")
        return
    try:
        import yaml

        cfg = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    except Exception as exc:
        gate.check(False, "config scope", f"config unreadable: {exc}")
        return

    sc = (cfg.get("agentpod_stop_check") or {})
    sessions = sc.get("session_ids") or []
    if isinstance(sessions, str):
        sessions = [sessions]
    gate.check(
        bool(sessions),
        "session scope",
        f"{len(sessions)} session id(s) configured"
        if sessions
        else "session_ids is empty — the plugin would be inert, and enabling it "
             "would read as active",
    )

    cap_ok, _nudges, _conts, cap_detail = evaluate_cap_ordering(cfg)
    gate.check(cap_ok, "cap ordering", cap_detail)

    project_id, tenant = sc.get("project_id"), sc.get("tenant")
    if project_id is None and tenant is None:
        gate.note("no project_id/tenant narrowing: the whole configured board is swept")
        return
    db = board_db or (Path(sc["db_path"]).expanduser() if sc.get("db_path") else None)
    if db is None or not Path(db).expanduser().exists():
        gate.check(
            False, "project scope",
            f"project_id={project_id!r}/tenant={tenant!r} configured but the board "
            f"could not be located — cannot prove the scope matches anything",
        )
        return
    try:
        import sqlite3

        conn = sqlite3.connect(f"file:{Path(db).expanduser()}?mode=ro", uri=True)
        total = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        where, params = [], []
        if project_id is not None:
            where.append("project_id IS ?")
            params.append(project_id)
        if tenant is not None:
            where.append("tenant IS ?")
            params.append(tenant)
        matched = conn.execute(
            f"SELECT COUNT(*) FROM tasks WHERE {' AND '.join(where)}", params
        ).fetchone()[0]
        conn.close()
    except Exception as exc:
        gate.check(False, "project scope", f"board unreadable: {exc}")
        return
    gate.check(
        matched > 0,
        "project scope",
        f"scope matches {matched} of {total} cards"
        + ("" if matched else " — it would sweep nothing and report a clean board"),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--core-root", default="~/.hermes/hermes-agent",
                    help="the INSTALLED Hermes checkout the runtime executes")
    ap.add_argument("--config", default=None, help="config.yaml to evaluate (read-only)")
    ap.add_argument("--board-db", default=None, help="board sqlite file (read-only)")
    args = ap.parse_args(argv)

    gate = Gate()
    gate.note("read-only preflight: nothing is installed, written or restarted")
    try:
        probe_core(Path(args.core_root), gate)
        probe_scope(
            Path(args.config) if args.config else None,
            Path(args.board_db) if args.board_db else None,
            gate,
        )
    except Exception as exc:  # pragma: no cover - preflight must not crash silently
        print("\n".join(gate.lines))
        print(f"preflight error: {type(exc).__name__}: {exc}")
        return 2

    print("\n".join(gate.lines))
    if gate.failures:
        print(
            "\nREFUSED: " + ", ".join(gate.failures)
            + "\nDo NOT enable agentpod_stop_check against this runtime. Land and "
            "independently read back the core half first, then re-run this "
            "preflight. Reversal at this point is a no-op: nothing was changed."
        )
        return 1
    print(
        "\nOK — this runtime can run the gate. The next step is still staged and "
        "reversible: copy the plugin with `enabled: false`, confirm no hook fires, "
        "then flip `agent.pre_verify_on_no_edit_turns` and `enabled` in one "
        "separately-authorised change. Reversal is setting both back to false."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
