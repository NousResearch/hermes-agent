"""
WorkflowEngine facade — single object the API layer (Phase 3) calls.

All methods are async. Phase 3 wires these 1:1 to HTTP endpoints.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

from engine.store.run_store import RunStore
from engine.store.definition_store import DefinitionStore
from engine.emitter.bus import EventBus
from engine.runtime.runner import WorkflowRunner
from engine.runtime.manifest import ManifestWriter
from engine.discovery.loader import parse_workflow

logger = logging.getLogger("workflow.engine")


class WorkflowEngine:
    """
    WorkflowEngine facade.

    Lifecycle::

        engine = create_engine()          # wiring.py
        run = await engine.start_run(...)
        await engine.cancel_run(run["id"])
        async for evt in engine.subscribe_events(run["id"]):
            ...
        await engine.shutdown()
    """

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        run_store: RunStore,
        def_store: DefinitionStore,
        bus: EventBus,
        runner: WorkflowRunner,
        manifest_writer: ManifestWriter,
        boot: Dict[str, Any],
    ) -> None:
        self._conn = conn
        self._run_store = run_store
        self._def_store = def_store
        self._bus = bus
        self._runner = runner
        self._manifest_writer = manifest_writer
        self.boot = boot

    # ------------------------------------------------------------------ #
    # Definitions                                                         #
    # ------------------------------------------------------------------ #

    async def list_definitions(self) -> List[Dict[str, Any]]:
        return self._def_store.list_definitions()

    async def get_definition(self, definition_id: str) -> Optional[Dict[str, Any]]:
        return self._def_store.get_definition(definition_id)

    async def upsert_definition(
        self,
        yaml_text: str,
        source_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        row = self._def_store.upsert_definition(
            yaml_text=yaml_text,
            source_path=source_path,
        )
        # Refresh manifest
        self._manifest_writer.write()
        return row

    async def parse_definition(self, definition_id: str) -> Optional[Dict[str, Any]]:
        defn = self._def_store.get_definition(definition_id)
        if defn is None:
            return None
        workflow, error = parse_workflow(defn["yaml"], f"{definition_id}.yaml")
        if error or workflow is None:
            return {"id": definition_id, "error": error.error if error else "parse failed"}
        dag_nodes, _ = workflow.get_dag_nodes()
        return {
            "id": definition_id,
            "name": workflow.name,
            "description": workflow.description,
            "nodes": [
                {"id": n.id, "type": type(n).__name__.replace("Node", "").lower()}
                for n in dag_nodes
            ],
            "kind": workflow.kind or "workflow",
        }

    # ------------------------------------------------------------------ #
    # Runs                                                                #
    # ------------------------------------------------------------------ #

    async def list_runs(
        self,
        *,
        workflow_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        return self._run_store.list_workflow_runs(
            workflow_id=workflow_id,
            limit=limit,
        )

    async def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._run_store.get_workflow_run(run_id)

    async def start_run(
        self,
        workflow_id: str,
        inputs: Dict[str, Any],
        trigger: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._runner.start(workflow_id, inputs, trigger)

    async def cancel_run(self, run_id: str) -> None:
        await self._runner.cancel(run_id)

    # ------------------------------------------------------------------ #
    # Approvals                                                           #
    # ------------------------------------------------------------------ #

    async def approve(
        self,
        run_id: str,
        node_id: str,
        decision: Literal["approve", "reject"],
        comment: Optional[str] = None,
    ) -> None:
        """
        Process an approval decision.

        1. Find the paused node_run for (run_id, node_id).
        2. Atomic CAS: update status paused → completed/failed.
        3. If claimed: emit approval_received, resume the workflow run.
        """
        nr = self._run_store.find_node_run(run_id, node_id)
        if nr is None:
            raise ValueError(f"Node run not found: run={run_id} node={node_id}")

        claimed = self._run_store.try_claim_approval(nr["id"], decision, comment)
        if not claimed:
            logger.warning(
                "approve: node_run %s was not in 'paused' state (already processed?)",
                nr["id"],
            )
            return

        self._bus.emit(
            run_id=run_id,
            event_type="approval_received",
            node_run_id=nr["id"],
            data={
                "node_id": node_id,
                "decision": decision,
                "comment": comment,
            },
        )

        if decision == "approve":
            self._run_store.resume_workflow_run(run_id)
            # Emit so subscribers know the run is live again
            self._bus.emit(
                run_id=run_id,
                event_type="workflow_resumed",
                data={"node_id": node_id},
            )
        else:
            # Reject → fail the run
            self._run_store.update_workflow_run(
                run_id,
                status="failed",
                error=f"Rejected at node {node_id}: {comment or 'no comment'}",
            )
            self._bus.emit(
                run_id=run_id,
                event_type="workflow_failed",
                data={
                    "error": f"Rejected at node {node_id}",
                    "node_id": node_id,
                },
            )

    # ------------------------------------------------------------------ #
    # Extended definitions                                                #
    # ------------------------------------------------------------------ #

    async def delete_definition(self, definition_id: str) -> int:
        """Delete a non-bundled definition. Returns rows deleted."""
        rows = self._def_store.delete_definition(definition_id)
        if rows > 0:
            self._manifest_writer.write()
        return rows

    # ------------------------------------------------------------------ #
    # Extended runs                                                        #
    # ------------------------------------------------------------------ #

    async def find_run_by_conversation_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._run_store.find_run_by_conversation_id(conversation_id)

    async def get_active_run_by_path(self, scope_path: str) -> Optional[Dict[str, Any]]:
        return self._run_store.get_active_run_by_path(scope_path)

    async def resume_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        self._run_store.resume_workflow_run(run_id)
        return self._run_store.get_workflow_run(run_id)

    # ------------------------------------------------------------------ #
    # Extended node runs                                                  #
    # ------------------------------------------------------------------ #

    async def list_node_runs(self, run_id: str) -> List[Dict[str, Any]]:
        return self._run_store.list_node_runs(run_id)

    async def find_node_run_by_id(self, node_run_id: str) -> Optional[Dict[str, Any]]:
        return self._run_store.find_node_run_by_id(node_run_id)

    # ------------------------------------------------------------------ #
    # Extended events                                                      #
    # ------------------------------------------------------------------ #

    async def append_workflow_event(self, event: Dict[str, Any]) -> None:
        self._run_store.append_workflow_event(
            workflow_run_id=event["workflow_run_id"],
            event_type=event["event_type"],
            node_run_id=event.get("node_run_id"),
            step_index=event.get("step_index"),
            step_name=event.get("step_name"),
            data=event.get("data"),
            event_id=event.get("id"),
            created_at=event.get("created_at"),
        )

    async def list_recent_workflow_events(self, run_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        return self._run_store.list_recent_events(run_id, limit=limit)

    # ------------------------------------------------------------------ #
    # Extended phase transitions                                           #
    # ------------------------------------------------------------------ #

    async def record_phase_transition(
        self,
        *,
        run_id: str,
        to_phase: str,
        decided_by: str,
        decision_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._run_store.record_phase_transition(
            run_id=run_id,
            to_phase=to_phase,
            decided_by=decided_by,
            decision_data=decision_data,
        )

    async def list_phase_transitions(self, run_id: str) -> List[Dict[str, Any]]:
        return self._run_store.list_phase_transitions(run_id)

    # ------------------------------------------------------------------ #
    # Extended approvals                                                   #
    # ------------------------------------------------------------------ #

    async def try_claim_approval_for_resume(
        self,
        node_run_id: str,
        decision: Literal["approved", "rejected"],
        approval_response: str,
    ) -> Dict[str, Any]:
        return self._run_store.try_claim_approval_for_resume(
            node_run_id, decision, approval_response
        )

    # ------------------------------------------------------------------ #
    # Events / SSE                                                        #
    # ------------------------------------------------------------------ #

    def subscribe_events(
        self, run_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Return an async iterator of events.
        Replays last 50 DB events then streams live events.
        """
        return self._bus.subscribe(run_id)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def shutdown(self) -> None:
        """Close event bus and DB connection."""
        self._bus.close_all()
        try:
            self._conn.close()
        except Exception:
            pass
        logger.info("WorkflowEngine shut down.")
