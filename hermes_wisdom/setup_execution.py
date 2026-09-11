"""Native-approved setup steps using the existing terminal and process registry."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from contextvars import copy_context
from contextlib import contextmanager

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal

from .agent_led.schemas import _no_secret_shapes
from .client import WisdomConflict
from .store import utc_now

KIND = "wisdom_setup"


@contextmanager
def _execution_lock(store, skill_id):
    from hermes_cli.active_sessions import _FileLock

    name = hashlib.sha256(skill_id.encode()).hexdigest()
    lock = _FileLock(store.root / "setup-locks" / name, blocking=False)
    try:
        lock.__enter__()
    except (OSError, RuntimeError) as exc:
        raise WisdomConflict("setup execution is busy or its lock is unavailable; check progress again") from exc
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


class SetupStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    phase: Literal["prerequisite", "setup", "verify"]
    index: int = Field(default=0, ge=0, le=39)
    command: str = Field(default="", max_length=1200)

    @model_validator(mode="after")
    def validate_command(self):
        if self.phase == "prerequisite":
            if self.command:
                raise ValueError("prerequisites use a private acknowledgement, not a command")
        elif not self.command.strip():
            raise ValueError("propose the exact command for separate native approval")
        if self.phase == "verify" and self.index:
            raise ValueError("there is one verification step")
        _no_secret_shapes(self.command)
        if any(ord(char) < 32 or ord(char) == 127 or unicodedata.category(char) == "Cf" for char in self.command):
            raise ValueError("setup commands must be a single visible line")
        return self


def _rows(store, skill_id):
    with store.transaction() as db:
        return [dict(row) for row in db.execute(
            "SELECT * FROM operation_journal WHERE kind=? AND entity_id=? ORDER BY rowid",
            (KIND, skill_id),
        )]


def _save(store, row, phase, payload, *, done=False):
    with store.transaction() as db:
        db.execute(
            "UPDATE operation_journal SET phase=?,state=?,payload_json=?,updated_at=? "
            "WHERE id=? AND phase=? AND payload_json=?",
            (phase, "done" if done else "pending", json.dumps(payload, sort_keys=True),
             utc_now(), row["id"], row["phase"], row["payload_json"]),
        )


def _poll(store, row):
    from tools.process_registry import process_registry

    payload = json.loads(row["payload_json"])
    if row["state"] == "done" or row["phase"] != "running":
        return {**payload, "state": row["phase"]}
    identity = payload["process_id"]
    process = process_registry.get(identity)
    if process is None or (
        process.id != identity
        or process.owner_task_id != payload["task_id"]
        or process.session_key != payload["session_key"]
        or process.command != payload["step"]["command"]
    ):
        # Retain the journal: disappearance does not prove that side effects stopped.
        return {**payload, "state": "unknown"}
    status = process_registry.poll(identity)
    if status.get("session_id") != identity or status.get("status") not in {"running", "exited"}:
        return {**payload, "state": "unknown"}
    if status["status"] == "running":
        return {**payload, "state": "running"}
    code = status.get("exit_code")
    if type(code) is not int or status.get("completion_reason") != "exited":
        return {**payload, "state": "unknown"}
    state = "passed" if code == 0 else "failed"
    payload["exit_code"] = code
    _save(store, row, state, payload, done=True)
    return {**payload, "state": state}


def progress(store, inspection):
    current = []
    for row in _rows(store, inspection["skill_id"]):
        payload = json.loads(row["payload_json"])
        if (payload["org_id"], payload["version"], payload["content_hash"]) != (
            store.active_org_id(), inspection["version"], inspection["content_hash"],
        ):
            continue
        value = _poll(store, row)
        current.append({
            "interaction_id": value["interaction_id"],
            "phase": value["step"]["phase"], "index": value["step"]["index"],
            "state": "unknown" if value["state"] == "starting" else value["state"],
            **({"exit_code": value["exit_code"]} if "exit_code" in value else {}),
        })
    return current


def plan_step(store, reference):
    from .installed_setup import inspect_installed_setup

    step = SetupStep.model_validate(reference["step"])
    info = inspect_installed_setup(store, reference["skill_id"], version=reference["version"])
    guidance = info["guidance"]
    if not guidance:
        raise WisdomConflict("this installed version has no valid setup guidance")
    if any(row["entity_id"] == info["skill_id"] for row in store.pending_operations()):
        raise WisdomConflict("a setup operation is unfinished; inspect its outcome before continuing")
    latest = {(row["phase"], row["index"]): row for row in info["setup_progress"]}
    completed = {key for key, row in latest.items() if row["state"] == "passed"}
    if step.phase == "prerequisite":
        requirements = info["prerequisites"]
        if step.index >= len(requirements):
            raise WisdomConflict("prerequisite is not in the installed guidance")
        item = requirements[step.index]
        label = "Environment variable" if item["kind"] == "env_var" else item["kind"].capitalize()
        instruction = f"{label}: {item['name']}\n{item['purpose']}"
        if item.get("handoff"):
            instruction += "\nPublisher guidance: " + item["handoff"]
        if item["kind"] == "env_var":
            instruction += (
                "\nConfigure this variable privately for the active Hermes profile on the machine running Hermes. "
                "Run hermes config env-path under that same profile to locate its private environment file, then edit it privately. "
                "Reload that profile after editing so new terminal processes receive the change. "
                "Never paste its value into chat, a skill file, or a command argument. "
                "Only presence is checked here; verification must still test whether it works."
            )
        elif item["kind"] == "command":
            instruction += "\nThe command must be available on the local Hermes runtime's PATH. Use the declared setup instructions or your normal installation process."
        else:
            instruction += "\nConfirm only after you have configured this prerequisite."
        if item["status"] == "missing":
            instruction += "\nNot detected. Configure it privately, then select Recheck. No command will run."
    elif step.phase == "setup":
        if step.index >= len(guidance["setup_instructions"]):
            raise WisdomConflict("setup step is not in the installed guidance")
        instruction = guidance["setup_instructions"][step.index]
    else:
        unresolved = any(
            item["status"] == "missing" or (item["status"] == "manual" and ("prerequisite", index) not in completed)
            for index, item in enumerate(info["prerequisites"])
        )
        if unresolved or any(("setup", index) not in completed for index in range(len(guidance["setup_instructions"]))):
            raise WisdomConflict("finish the declared prerequisites and setup steps before verification")
        instruction = guidance["verification_step"]
    installation = store.installation(info["skill_id"])
    identity = [store.active_org_id(), info["skill_id"], info["version"], info["content_hash"], step.model_dump()]
    if step.phase == "verify":
        identity.append([row["interaction_id"] for row in info["setup_progress"] if row["phase"] != "verify"])
    return {
        "skill_id": info["skill_id"], "slug": installation["slug"],
        "version": info["version"], "content_hash": info["content_hash"],
        "setup_key": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        "step": step.model_dump(), "setup_instruction": instruction,
        "allowed": step.phase != "prerequisite" or item["status"] != "missing",
        **({"setup_requirement": item} if step.phase == "prerequisite" else {}),
    }


def execute_step(service, value, actor):
    """Only the authenticated consent handler calls this after native confirmation."""
    # This descriptor lock cannot expire while an approval prompt or spawn is
    # still active. Recovery takes the same lock and never races a live starter.
    with _execution_lock(service.store, value["plan"]["skill_id"]):
        with service.store.transaction() as db:
            current = db.execute("SELECT state FROM wisdom_consent WHERE id=?", (value["id"],)).fetchone()
            if current is None or current["state"] != "applying":
                raise WisdomConflict("setup approval changed before execution")
        return _execute_step(service, value, actor)


def _execute_step(service, value, actor):
    from gateway.session_context import set_session_vars
    from tools.terminal_tool import _get_env_config, terminal_tool

    store, plan = service.store, value["plan"]
    skill_id = plan["skill_id"]
    lock = store.acquire_operation_lock(skill_id)
    if not lock:
        raise WisdomConflict("another operation owns this installed skill")
    try:
        fresh = plan_step(store, {"skill_id": skill_id, "version": plan["version"], "step": plan["step"]})
        if fresh["setup_key"] != plan["setup_key"]:
            raise WisdomConflict("installed package or proposed setup step changed")
        if not fresh["allowed"]:
            raise WisdomConflict("prerequisite is still missing; configure it privately and recheck")
        if plan["step"]["phase"] != "prerequisite" and _get_env_config()["env_type"] != "local":
            raise WisdomConflict("local skill setup requires a local terminal; the configured sandbox was not bypassed")
        payload = {
            "interaction_id": value["id"], "org_id": value["organization_id"],
            "version": plan["version"], "content_hash": plan["content_hash"],
            "step": plan["step"], "session_key": actor.session_key,
            "task_id": "wisdom-setup:" + value["id"],
        }
        operation_id = store.journal(KIND, skill_id, "starting", payload)
        if plan["step"]["phase"] == "prerequisite":
            store.advance(operation_id, "passed", done=True)
            return {"state": "passed"}

        def run():
            from tools.approval_context import set_current_session_key

            # Callback handlers do not necessarily bind tool context. Use a copied
            # context so guards see the authenticated origin without changing its caller.
            set_session_vars(
                platform=actor.platform, session_key=actor.session_key,
                user_id=actor.actor_id, chat_id=actor.chat_id, chat_type="dm",
                thread_id=actor.thread_id, scope_id=actor.scope_id, async_delivery=False,
                cron_session="",
            )
            set_current_session_key(actor.session_key)
            return json.loads(terminal_tool(
                command=plan["step"]["command"], background=True,
                task_id=payload["task_id"], workdir=store.installation(skill_id)["target_path"],
            ))

        # The starting journal is committed first. Any exception, including an
        # interrupted spawn, leaves unknown work visible and forbids blind replay.
        result = copy_context().run(run)
        process_id = result.get("session_id")
        if not process_id:
            if result.get("status") in {"blocked", "pending_approval"}:
                store.advance(operation_id, "blocked", done=True)
                return {"state": "blocked"}
            return {"state": "unknown"}
        payload["process_id"] = process_id
        row = store.operation(operation_id)
        _save(store, row, "running", payload)
        return {"state": "running"}
    finally:
        store.release_operation_lock(skill_id, lock)


def interaction_progress(store, value):
    for row in _rows(store, value["plan"]["skill_id"]):
        payload = json.loads(row["payload_json"])
        if payload["interaction_id"] == value["id"] and payload["org_id"] == value["organization_id"]:
            result = _poll(store, row)
            return {"state": "unknown" if result["state"] == "starting" else result["state"]}
    return {"state": "unknown"}


def recover_step(store, value, *, acknowledged_stopped=False, finish):
    """Review or clear uncertain work; only native, owner-bound controls call this.

    The acknowledgement confirms the user checked side effects and stopped the
    command and its children. It grants no permission to run or verify anything.
    """
    skill_id = value["plan"]["skill_id"]
    with _execution_lock(store, skill_id):
        from .mediation_store import MediationStore

        for row in _rows(store, skill_id):
            payload = json.loads(row["payload_json"])
            if (payload["interaction_id"], payload["org_id"]) != (value["id"], value["organization_id"]):
                continue
            result = _poll(store, row)
            if result["state"] not in {"unknown", "starting"}:
                return {"state": result["state"]}
            if not acknowledged_stopped:
                return {"state": "unknown", "recovery_review": True}
            with store.transaction() as db:
                MediationStore._check_org(db, value["organization_id"])
                current = db.execute("SELECT * FROM operation_journal WHERE id=?", (row["id"],)).fetchone()
                if current["state"] == "done":
                    return {"state": current["phase"]}
                if current["phase"] != row["phase"] or current["payload_json"] != row["payload_json"]:
                    raise WisdomConflict("setup progress changed; check it again before recovery")
                payload["recovery"] = {"acknowledged_stopped": True, "at": utc_now()}
                db.execute(
                    "UPDATE operation_journal SET phase='abandoned',state='done',payload_json=?,updated_at=? WHERE id=?",
                    (json.dumps(payload, sort_keys=True), utc_now(), row["id"]),
                )
                finish(db, {"state": "abandoned"})
            return {"state": "abandoned"}
        # Without a journal there is no starting process protected by this lock,
        # but a legacy/partially written record still needs explicit human review.
        if acknowledged_stopped:
            with store.transaction() as db:
                MediationStore._check_org(db, value["organization_id"])
                finish(db, {"state": "abandoned"})
            return {"state": "abandoned"}
        return {"state": "unknown", "recovery_review": True}
