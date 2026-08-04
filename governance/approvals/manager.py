import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

from governance.approvals.ledger import ApprovalLedger, generate_fingerprint

try:
    import _board_compat
except ModuleNotFoundError:
    # Operational scripts historically placed ``scripts/`` directly on
    # sys.path. Package imports and isolated test worktrees do not, so retain
    # the same compatibility layer through the checkout-root namespace.
    from scripts import _board_compat
from hermes_cli import kanban_db

class ApprovalWorkflowManager:
    """
    Orchestrates the flow from Discord approval action to Kanban task creation.
    Centralized governance layer for triage approvals.
    """
    def __init__(self, default_board: str = 'default'):
        self.ledger = ApprovalLedger()
        self.default_board = default_board

    def _get_db_path(self, board: str) -> str:
        return _board_compat.resolve_board_db_str(board)

    def _task_exists(self, conn, board: str, fingerprint: str, triage_id: str) -> Tuple[bool, Optional[str]]:
        """
        Checks if a task already exists with the given fingerprint or triage_id.
        """
        cursor = conn.cursor()
        # 1. Check for exact triage_id match in the body
        cursor.execute("SELECT id FROM tasks WHERE body LIKE ?", (f"%{triage_id}%",))
        row = cursor.fetchone()
        if row: return True, row[0]
        
        # 2. Check for fingerprint match
        cursor.execute("SELECT id FROM tasks WHERE body LIKE ?", (f"%{fingerprint}%",))
        row = cursor.fetchone()
        if row: return True, row[0]
        
        return False, None

    def _format_body(self, summary_dict: Dict) -> str:
        """Formats the triage summary dictionary into a structured Kanban body."""
        source = summary_dict.get('source', {})
        prov = source.get('provenance', {})
        
        body_lines = [
            f"# {source.get('title', 'Untitled')}",
            "",
            f"**Source:** {source.get('url', 'N/A')}",
            f"**Type:** {source.get('source_type', 'unknown')}",
            f"**Submitted by:** {prov.get('submitted_by', 'unknown')}",
            f"**Submitted at:** {prov.get('submitted_at', 'N/A')}",
            "",
            "## Triage Summary",
            f"**Category:** {summary_dict.get('classification', {}).get('category', 'unknown')}",
            f"**Confidence:** {summary_dict.get('confidence', 'unknown')}",
            f"**Effort:** {summary_dict.get('effort', 'unknown')}",
            f"**Recommendation:** {summary_dict.get('recommendation', 'unknown')}",
            f"**Routing:** {summary_dict.get('routing', {}).get('specialist', 'unknown')}",
            "",
            "## Reasoning",
            summary_dict.get('reasoning', 'No reasoning provided.'),
            "",
            "## Content Snippet",
            source.get('content_snippet', '')[:1000],
        ]
        return "\n".join(body_lines)

    def handle_approval_action(self, triage_id: str, action: str, actor: str, 
                               reason: Optional[str] = None, 
                               summary_dict: Optional[Dict] = None) -> Dict:
        """
        Processes an approval action and executes state transition + Kanban creation.
        """
        # 1. State Transition (Ledger)
        ok, msg = self.ledger.process_action(triage_id, action, actor, reason)
        if not ok:
            return {"status": "error", "message": msg}

        # 2. On Approval: Create or Link Kanban Task
        if action == "approve":
            req = self.ledger.get_request(triage_id)
            if not req:
                return {"status": "error", "message": "Request not found after approval"}

            # Use provided summary_dict or the one stored in the ledger
            summary_data = summary_dict or req.get('summary', {})
            if isinstance(summary_data, str):
                # Fallback for legacy string summaries
                summary_data = {"reasoning": summary_data, "title": "Approved Triage"}

            board = req["routing"].get("board", self.default_board)
            assignee = req["routing"].get("assignee", "octacon")
            fingerprint = req["fingerprint"]
            
            db_path = self._get_db_path(board)
            if not db_path:
                return {"status": "error", "message": f"Could not resolve board DB for {board}"}

            with sqlite3.connect(db_path) as conn:
                # Deduplication check
                exists, existing_id = self._task_exists(conn, board, fingerprint, triage_id)
                if exists:
                    kanban_db.add_comment(
                        conn, existing_id, 
                        f"Approved duplicate request {triage_id} linked to this task. Actor: {actor}"
                    )
                    return {
                        "status": "ok",
                        "action": "linked",
                        "task_id": existing_id,
                        "message": f"Duplicate detected. Linked to existing task {existing_id}"
                    }

                # Create new task using internal API
                title = f"Triage Approved: {summary_data.get('title', 'Untitled')[:50]}..."
                body = self._format_body(summary_data)
                
                effort_map = {"xs": 5, "s": 4, "m": 3, "l": 2, "xl": 1}
                priority = effort_map.get(summary_data.get('effort'), 3)

                task_id = kanban_db.create_task(
                    conn,
                    title=title,
                    body=body,
                    assignee=assignee,
                    created_by="denji-governance",
                    workspace_kind="scratch",
                    priority=priority,
                    idempotency_key=triage_id,
                    initial_status="backlog"
                )
                
                return {
                    "status": "ok",
                    "action": "created",
                    "task_id": task_id,
                    "message": f"Task {task_id} created successfully"
                }

        return {"status": "ok", "action": action, "message": msg}
