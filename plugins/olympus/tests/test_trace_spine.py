import json
import unittest

from _load_plugin_api import plugin_api


class TraceSpineTest(unittest.TestCase):
    def test_correlates_task_session_run_and_event_refs(self):
        session_ref = plugin_api.safe_id("raw-session-alpha", "session")
        task_ref = plugin_api.safe_id("raw-task-alpha", "task")
        run_ref = plugin_api.safe_id("raw-run-alpha", "run")
        event_ref = plugin_api.safe_id("raw-event-alpha", "event")

        payload = plugin_api.build_trace_spine(
            sessions=[{
                "id": session_ref,
                "session_ref": session_ref,
                "label": "cli",
                "state": "completed",
                "tool_call_count": 44,
                "message_count": 12,
                "handoff_error": None,
                "transcript": "private transcript content must not appear",
            }],
            kanban={
                "recent_tasks": [{
                    "id": task_ref,
                    "board": "default",
                    "title": task_ref,
                    "status": "blocked",
                    "assignee": "profile:default",
                    "session_ref": session_ref,
                    "consecutive_failures": 2,
                    "forced_skill_count": 1,
                }],
                "recent_runs": [{
                    "id": run_ref,
                    "task_id": task_ref,
                    "status": "failed",
                    "outcome": "crashed",
                    "error": "worker failed after retry",
                }],
                "recent_events": [{
                    "id": "default:" + event_ref,
                    "task_id": task_ref,
                    "run_ref": run_ref,
                    "kind": "failed",
                    "task_status": "blocked",
                }],
            },
        )

        self.assertEqual(payload["summary"]["correlated_tasks"], 1)
        self.assertEqual(payload["summary"]["failure_points"], 4)
        self.assertEqual(payload["summary"]["state"], "warning")
        self.assertEqual(payload["items"][0]["task_ref"], task_ref)
        self.assertEqual(payload["items"][0]["session_ref"], session_ref)
        self.assertEqual(payload["items"][0]["run_refs"], [run_ref])
        self.assertEqual(payload["items"][0]["event_refs"], ["default:" + event_ref])
        self.assertEqual(payload["items"][0]["recommended_view"], "/kanban")

    def test_trace_spine_hides_raw_ids_and_transcripts(self):
        session_ref = plugin_api.safe_id("raw-session-beta", "session")
        task_ref = plugin_api.safe_id("raw-task-beta", "task")

        payload = plugin_api.build_trace_spine(
            sessions=[{
                "id": session_ref,
                "session_ref": session_ref,
                "label": "cli",
                "state": "error",
                "tool_call_count": 8,
                "message_count": 4,
                "handoff_error": "private transcript content",
                "messages": ["raw user prompt"],
            }],
            kanban={
                "recent_tasks": [{
                    "id": task_ref,
                    "board": "default",
                    "title": task_ref,
                    "status": "running",
                    "assignee": "profile:default",
                    "session_ref": session_ref,
                }],
                "recent_runs": [],
                "recent_events": [],
            },
        )
        serialized = json.dumps(payload)

        for raw in ("raw-session-beta", "raw-task-beta", "raw user prompt", "private transcript content"):
            self.assertNotIn(raw, serialized)
        for raw_key in ('"session_id"', '"task_id"', '"run_id"', '"event_id"', '"messages"', '"transcript"'):
            self.assertNotIn(raw_key, serialized)
        self.assertIn('"session_ref"', serialized)
        self.assertIn('"task_ref"', serialized)


if __name__ == "__main__":
    unittest.main()
