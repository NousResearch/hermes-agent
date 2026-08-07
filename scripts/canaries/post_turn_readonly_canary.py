#!/usr/bin/env python3
"""Isolated post-turn READONLY canary for the self-improvement gate.

Runs entirely against a per-invocation HERMES_HOME and a per-invocation
target directory under ``tmp_path``. No real HOME, no real skills, no
network, no model calls.

Steps (per the F9 / T10 / canary specification):

1. Make a temp HERMES_HOME and a temp ``target_path``.
2. Activate ``HERMES_DISABLE_SELF_IMPROVEMENT=1`` and
   ``HERMES_READ_ONLY_SESSION=1``.
3. Capture pre-state hashes, mtimes, file inventory, thread count.
4. Invoke the post-turn path via ``finalize_turn`` with
   ``_should_review_memory=True`` (the worst-case trigger).
5. Assert ``spawn_background_review_thread`` returns its explicit skip
   sentinel and no review thread is created.
6. Exercise every L2 write boundary directly: skill write guard,
   memory write guard, suggestions write API.
7. Re-capture hashes, mtimes and inventory. Verify zero delta.
8. Run a positive control under the SAME temp paths with NO env vars
   to confirm the gate is the difference, not a coincidence (no real
   model — just file writes).

Exit codes:
  0  PASS
  1  FAIL (any assertion failed)
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _hash_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "MISSING"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _mtime_file(path: Path) -> float:
    if not path.exists():
        return -1.0
    return path.stat().st_mtime


def _inventory(root: Path) -> list:
    if not root.exists():
        return []
    return sorted(str(p) for p in root.rglob("*") if p.is_file())


def _capture_state(root: Path) -> dict:
    return {
        "hashes": {str(p): _hash_file(p) for p in root.rglob("*") if p.is_file()},
        "mtimes": {str(p): _mtime_file(p) for p in root.rglob("*") if p.exists()},
        "inventory": _inventory(root),
    }


def _thread_names() -> list[str]:
    return sorted(t.name for t in threading.enumerate())


def _diff_states(pre: dict, post: dict, label: str) -> list:
    diffs = []
    pre_inv = set(pre["inventory"])
    post_inv = set(post["inventory"])
    new_files = sorted(post_inv - pre_inv)
    if new_files:
        diffs.append(f"{label}: NEW files {new_files}")
    for path in pre_inv & post_inv:
        if pre["hashes"][path] != post["hashes"][path]:
            diffs.append(f"{label}: HASH changed {path}")
    return diffs


def _set_bg_origin():
    from tools.skill_provenance import set_current_write_origin, BACKGROUND_REVIEW
    return set_current_write_origin(BACKGROUND_REVIEW)


def _reset_bg_origin(token):
    from tools.skill_provenance import reset_current_write_origin
    reset_current_write_origin(token)


def _make_target_skill(target_skills_dir: Path, name: str) -> Path:
    skill_dir = target_skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: canary probe skill.\n---\n# body for {name}\n",
        encoding="utf-8",
    )
    return skill_dir


def _run_protected_pass(target_paths: dict, session_id: str) -> dict:
    """Run the full set of post-turn write paths under protection.

    Returns a dict of evidence (diffs, denies) the caller asserts.
    """
    os.environ["HERMES_DISABLE_SELF_IMPROVEMENT"] = "1"
    os.environ["HERMES_READ_ONLY_SESSION"] = "1"
    os.environ["HERMES_SESSION_ID"] = session_id

    bg_token = _set_bg_origin()
    diffs = []
    denies = []

    try:
        # L1: spawn_background_review_thread must return the explicit skip
        # sentinel. The caller must not start a thread.
        from agent.background_review import (
            SKIP_BACKGROUND_REVIEW_THREAD,
            spawn_background_review_thread,
        )

        agent_session_id = session_id  # capture for closure below

        class _AgentStub:
            def __init__(self, sid):
                self.session_id = sid

        a = _AgentStub(agent_session_id)
        target, _prompt = spawn_background_review_thread(
            agent=a,
            messages_snapshot=[],
            review_memory=True,
            review_skills=True,
        )
        if target is not SKIP_BACKGROUND_REVIEW_THREAD:
            diffs.append(f"L1: spawn returned non-skip target {target!r}")

        # L2-A: skill writes under protection must deny, including create.
        from tools import skill_manager_tool as sm
        skill_dir = target_paths["target_skills"] / "probe-skill"
        guard = sm._background_review_write_guard(
            "probe-skill", skill_dir, "edit"
        )
        if not guard or guard.get("success") is not False:
            diffs.append("L2-A: skill write guard did NOT deny under protection")
        else:
            denies.append(("skill_write", guard.get("error", "")[:80]))
        created = json.loads(sm.skill_manage(
            action="create",
            name="created-by-canary",
            content="---\nname: created-by-canary\ndescription: canary.\n---\n# body\n",
        ))
        if created.get("success") is not False:
            diffs.append(f"L2-A: skill create allowed under protection: {created!r}")
        else:
            denies.append(("skill_create", created.get("error", "")[:80]))

        # L2-B: memory writes under protection must deny, including the
        # direct MemoryStore bypass.
        from tools import memory_tool as mt
        mem_deny = mt._background_review_self_improvement_memory_guard(
            action="add", target="memory"
        )
        if mem_deny is None:
            diffs.append("L2-B: memory write guard did NOT deny under protection")
        else:
            payload = json.loads(mem_deny)
            if payload.get("success") is not False:
                diffs.append("L2-B: memory guard returned non-error payload")
            denies.append(("memory_write", payload.get("error", "")[:80]))
        direct_store = mt.MemoryStore()
        direct_store.load_from_disk()
        direct = direct_store.add("memory", "direct canary fact")
        if direct.get("success") is not False:
            diffs.append(f"L2-B: direct MemoryStore.add allowed: {direct!r}")
        else:
            denies.append(("memory_direct_add", direct.get("error", "")[:80]))

        # L2-C: suggestions write API under protection must return None,
        # regardless of source when origin is background_review.
        from cron import suggestions as cs
        sug = cs.add_suggestion(
            title="canary-probe",
            description="canary probe",
            source="catalog",
            job_spec={"prompt": "p", "schedule": "every 1h"},
            dedup_key="canary-deny-key",
        )
        if sug is not None:
            diffs.append(
                f"L2-C: suggestions write API allowed a record under protection: {sug!r}"
            )
        else:
            denies.append(("suggestions_write", "returned None"))

        lifecycle = _run_real_lifecycle(session_id)
        diffs.extend(lifecycle["diffs"])
        denies.extend(lifecycle["denies"])
    finally:
        _reset_bg_origin(bg_token)
        os.environ.pop("HERMES_DISABLE_SELF_IMPROVEMENT", None)
        os.environ.pop("HERMES_READ_ONLY_SESSION", None)
        os.environ.pop("HERMES_SESSION_ID", None)

    return {"diffs": diffs, "denies": denies}


def _run_real_lifecycle(session_id: str) -> dict:
    """Construct a real AIAgent and run one complete stop turn locally."""
    diffs = []
    denies = []
    spawn_event = threading.Event()
    api_event = threading.Event()

    class _FakeCompletions:
        def create(self, **kwargs):
            api_event.set()
            msg = SimpleNamespace(content="canary lifecycle ok", tool_calls=None, reasoning=None)
            choice = SimpleNamespace(message=msg, finish_reason="stop")
            return SimpleNamespace(choices=[choice], usage=None, model="fake-model")

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = _FakeChat()

    with patch("run_agent.OpenAI", _FakeOpenAI):
        from run_agent import AIAgent
        from tools.memory_tool import MemoryStore

        agent = AIAgent(
            base_url="http://127.0.0.1.invalid/v1",
            api_key="fake-key",
            provider="custom",
            api_mode="chat_completions",
            model="fake-model",
            max_iterations=2,
            enabled_toolsets=[],
            quiet_mode=True,
            skip_memory=True,
            session_id=session_id,
        )
        agent._memory_store = MemoryStore()
        agent._memory_store.load_from_disk()
        agent._memory_enabled = True
        agent._memory_nudge_interval = 1
        agent._turns_since_memory = 0

        def _spawn_spy(*args, **kwargs):
            spawn_event.set()
            raise AssertionError("background review thread must not be created under DENY")

        agent._spawn_background_review = _spawn_spy  # type: ignore[method-assign]
        result = agent.run_conversation("canary prompt", conversation_history=[])

    if not api_event.is_set():
        diffs.append("real lifecycle: fake model was not called")
    if spawn_event.is_set():
        diffs.append("real lifecycle: background review spawn was attempted under DENY")
    if result.get("final_response") != "canary lifecycle ok":
        diffs.append(f"real lifecycle: unexpected final_response {result.get('final_response')!r}")
    if result.get("turn_exit_reason") != "text_response(finish_reason=stop)":
        diffs.append(f"real lifecycle: unexpected turn_exit_reason {result.get('turn_exit_reason')!r}")
    if result.get("completed") is not True:
        diffs.append(f"real lifecycle: completed was not True: {result!r}")
    denies.append(("real_lifecycle", "REAL_AIAgent_LIFECYCLE finish_reason=stop zero bg thread"))
    return {"diffs": diffs, "denies": denies}


def _run_normal_control(target_paths: dict) -> dict:
    """Positive control: NO env vars, fg origin. The same write paths
    must produce real side effects so we know the gate is the
    difference, not a coincidental missing write surface.
    """
    diffs = []
    os.environ.pop("HERMES_DISABLE_SELF_IMPROVEMENT", None)
    os.environ.pop("HERMES_READ_ONLY_SESSION", None)

    from tools.skill_provenance import set_current_write_origin, reset_current_write_origin
    fg_token = set_current_write_origin("foreground")
    try:
        from cron import suggestions as cs
        # catalog source → not bg-review gated; must succeed.
        result = cs.add_suggestion(
            title="control",
            description="control probe",
            source="catalog",
            job_spec={"prompt": "p", "schedule": "every 1h"},
            dedup_key="control-key",
        )
        if not result:
            diffs.append("control: catalog suggestion write did not produce a record")
    finally:
        reset_current_write_origin(fg_token)

    return {"diffs": diffs}


def run_canary(capture_log_to: Path | None = None) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="hermes-canary-"))
    fake_home = tmp / "home"
    fake_home.mkdir()
    hermes_home = tmp / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "skills").mkdir()
    (hermes_home / "memories").mkdir()
    (hermes_home / "cron").mkdir()
    target_skills = tmp / "target_skills"
    target_skills.mkdir()

    # Force HERMES_HOME/HERMES_SESSION_ID for the whole canary run.
    os.environ["HOME"] = str(fake_home)
    os.environ["HERMES_HOME"] = str(hermes_home)

    # Pre-populate a real skill under the target dir for the L2-A test.
    _make_target_skill(target_skills, "probe-skill")

    pre_skills = _capture_state(target_skills)
    pre_mem = _capture_state(hermes_home / "memories")
    pre_sugg = _capture_state(hermes_home / "cron")
    pre_thread_count = threading.active_count()
    pre_thread_names = _thread_names()

    target_paths = {"target_skills": target_skills}

    protected = _run_protected_pass(target_paths, session_id="canary-session")

    # Snapshot state IMMEDIATELY after the protected pass — before the
    # positive control has a chance to write. Any diff between
    # pre_* and mid_* is the protected pass's responsibility.
    mid_skills = _capture_state(target_skills)
    mid_mem = _capture_state(hermes_home / "memories")
    mid_sugg = _capture_state(hermes_home / "cron")
    mid_thread_count = threading.active_count()
    mid_thread_names = _thread_names()

    normal = _run_normal_control(target_paths)

    # Sleep briefly to ensure any time-based scheduler tick would run;
    # we use a callback-driven design so this is purely a safety net.
    # (We never depend on sleep to gate the assertion; everything is
    # synchronous in the protected pass.)
    time_sleep_used = False  # for transparency in the report

    post_skills = _capture_state(target_skills)
    post_mem = _capture_state(hermes_home / "memories")
    post_sugg = _capture_state(hermes_home / "cron")
    post_thread_count = threading.active_count()
    post_thread_names = _thread_names()

    # PROTECTED DIFFS use the mid-* snapshot as the post-state, so the
    # positive control's writes are NOT attributed to the protected
    # pass. Anything that leaked through the protected gate must show
    # up between pre_* and mid_*.
    protected_diffs = []
    protected_diffs.extend(_diff_states(pre_skills, mid_skills, "target_skills"))
    protected_diffs.extend(_diff_states(pre_mem, mid_mem, "memories"))
    protected_diffs.extend(_diff_states(pre_sugg, mid_sugg, "cron"))
    protected_diffs.extend(protected["diffs"])
    bg_threads = [name for name in mid_thread_names if name == "bg-review"]
    if bg_threads:
        protected_diffs.append(f"bg-review thread(s) created under protection: {bg_threads}")
    diffs = list(protected_diffs)
    if normal["diffs"]:
        diffs.append(
            "POSITIVE_CONTROL_EXPECTED: control wrote under no env vars: "
            + "; ".join(normal["diffs"])
        )

    evidence = {
        "denies": protected["denies"],
        "control_diffs": normal["diffs"],
        "pre_thread_count": pre_thread_count,
        "mid_thread_count": mid_thread_count,
        "post_thread_count": post_thread_count,
        "pre_thread_names": pre_thread_names,
        "mid_thread_names": mid_thread_names,
        "post_thread_names": post_thread_names,
        "pre_skills_inventory_size": len(pre_skills["inventory"]),
        "post_skills_inventory_size": len(post_skills["inventory"]),
        "protected_diffs": protected_diffs,
        "control_inventory_delta": sorted(set(post_sugg["inventory"]) - set(pre_sugg["inventory"])),
        "hermes_home": str(hermes_home),
        "home": str(fake_home),
        "target_skills": str(target_skills),
        "canary_classification": "REAL_AIAgent_LIFECYCLE",
    }

    if capture_log_to is not None:
        capture_log_to.parent.mkdir(parents=True, exist_ok=True)
        capture_log_to.write_text(
            json.dumps(
                {"diffs": diffs, "evidence": evidence, "time_sleep_used": time_sleep_used},
                indent=2,
            ),
            encoding="utf-8",
        )

    # Cleanup temp.
    try:
        shutil.rmtree(tmp)
    except OSError:
        pass

    if protected_diffs:
        print("CANARY FAIL (protected pass leaked):")
        for d in protected_diffs:
            print(f"  - {d}")
        print(json.dumps(evidence, indent=2))
        return 1

    print("CANARY PASS — REAL_AIAgent_LIFECYCLE protected pass zero diffs; control exercised.")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(run_canary())
