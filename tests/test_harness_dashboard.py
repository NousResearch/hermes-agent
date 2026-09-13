import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import profiles as profiles_mod
from hermes_cli import web_server


class HarnessDashboardTests(unittest.TestCase):
    def test_harness_endpoint_aggregates_safe_profile_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_home = Path(tmp) / "coder"
            profile_home.mkdir()
            info = SimpleNamespace(
                name="coder",
                path=profile_home,
                is_default=False,
                model="gpt-test",
                provider="openai",
                has_env=True,
                skill_count=4,
            )
            runtime = {
                "state": "running",
                "running": True,
                "pid": 123,
                "updated_at": "2026-09-12T12:00:00+00:00",
                "stale": False,
                "has_exit_reason": False,
                "platforms": [{
                    "name": "telegram",
                    "state": "connected",
                    "needs_attention": False,
                    "updated_at": None,
                }],
            }
            with patch.object(profiles_mod, "list_profiles", return_value=[info]), patch.object(
                web_server, "_read_profile_runtime", return_value=runtime
            ), patch.object(
                web_server,
                "_profile_session_summary",
                return_value={"active": 2, "total": 9, "last_activity_at": None},
            ):
                payload = asyncio.run(web_server.list_harnesses_endpoint())

        self.assertEqual(payload["summary"], {
            "total": 1,
            "running": 1,
            "connected_platforms": 1,
            "attention": 0,
        })
        harness = payload["harnesses"][0]
        self.assertEqual(harness["name"], "coder")
        self.assertEqual(harness["gateway"]["state"], "running")
        self.assertEqual(harness["sessions"]["active"], 2)
        self.assertNotIn("path", harness)
        self.assertNotIn("env", harness)

    def test_stopped_gateway_does_not_report_persisted_platform_as_connected(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_home = Path(tmp) / "stopped"
            profile_home.mkdir()
            info = SimpleNamespace(
                name="stopped",
                path=profile_home,
                is_default=False,
                model=None,
                provider=None,
                has_env=False,
                skill_count=0,
            )
            runtime = {
                "state": "stopped",
                "running": False,
                "pid": None,
                "updated_at": "2020-01-01T00:00:00+00:00",
                "stale": False,
                "has_exit_reason": True,
                "platforms": [{
                    "name": "telegram",
                    "state": "connected",
                    "needs_attention": False,
                    "updated_at": None,
                }],
            }
            with patch.object(profiles_mod, "list_profiles", return_value=[info]), patch.object(
                web_server, "_read_profile_runtime", return_value=runtime
            ), patch.object(
                web_server,
                "_profile_session_summary",
                return_value={"active": 0, "total": 0, "last_activity_at": None},
            ):
                payload = asyncio.run(web_server.list_harnesses_endpoint())

        harness = payload["harnesses"][0]
        self.assertEqual(harness["connected_platforms"], 0)
        self.assertEqual(payload["summary"]["connected_platforms"], 0)
        self.assertTrue(harness["attention"])
        self.assertNotIn("exit_reason", harness["gateway"])

    def test_session_summary_uses_exact_count_beyond_recent_page(self):
        class FakeSessionDB:
            requested_limit = None

            def __init__(self, db_path):
                self.db_path = db_path

            def session_count(self):
                return 123

            def list_sessions_rich(self, *, limit):
                FakeSessionDB.requested_limit = limit
                now = time.time()
                return [{"ended_at": None, "last_active": now, "started_at": now}] * 123

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            profile_home = Path(tmp) / "many-sessions"
            profile_home.mkdir()
            (profile_home / "state.db").touch()
            fake_module = SimpleNamespace(SessionDB=FakeSessionDB)
            with patch.dict(sys.modules, {"hermes_state": fake_module}):
                summary = web_server._profile_session_summary(profile_home)

        self.assertEqual(FakeSessionDB.requested_limit, 123)
        self.assertEqual(summary["total"], 123)
        self.assertEqual(summary["active"], 123)

    def test_running_gateway_is_not_marked_stale_from_lifecycle_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_home = Path(tmp) / "coder"
            profile_home.mkdir()
            (profile_home / "gateway_state.json").write_text(
                json.dumps({
                    "gateway_state": "running",
                    "updated_at": "2020-01-01T00:00:00+00:00",
                    "platforms": {},
                }),
                encoding="utf-8",
            )
            with patch.object(web_server, "get_running_pid", return_value=987):
                runtime = web_server._read_profile_runtime(profile_home)

        self.assertTrue(runtime["running"])
        self.assertEqual(runtime["state"], "running")
        self.assertFalse(runtime["stale"])

    def test_stopped_gateway_is_not_marked_stale_from_lifecycle_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_home = Path(tmp) / "stopped"
            profile_home.mkdir()
            (profile_home / "gateway_state.json").write_text(
                json.dumps({
                    "gateway_state": "stopped",
                    "updated_at": "2020-01-01T00:00:00+00:00",
                    "platforms": {},
                }),
                encoding="utf-8",
            )
            with patch.object(web_server, "get_running_pid", return_value=None):
                runtime = web_server._read_profile_runtime(profile_home)

        self.assertFalse(runtime["running"])
        self.assertEqual(runtime["state"], "stopped")
        self.assertFalse(runtime["stale"])

    def test_malformed_gateway_state_is_safe_and_keeps_the_harness(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_home = Path(tmp) / "malformed"
            profile_home.mkdir()
            (profile_home / "gateway_state.json").write_text(
                json.dumps({
                    "gateway_state": {"unexpected": True},
                    "platforms": [],
                    "updated_at": [],
                }),
                encoding="utf-8",
            )
            with patch.object(web_server, "get_running_pid", return_value=None):
                runtime = web_server._read_profile_runtime(profile_home)

        self.assertEqual(runtime["state"], "unknown")
        self.assertFalse(runtime["running"])
        self.assertEqual(runtime["platforms"], [])
        self.assertIsNone(runtime["updated_at"])

    def test_profile_runtime_platforms_are_sanitized(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_home = Path(tmp)
            (profile_home / "gateway_state.json").write_text(
                json.dumps({
                    "platforms": {
                        "telegram": {
                            "state": "provider leaked detail",
                            "needs_attention": "false",
                            "updated_at": "provider timestamp",
                            "error_message": "secret provider response",
                        },
                        "malformed": [],
                        "https://attacker.invalid/token": {
                            "state": "connected",
                        },
                    },
                    "updated_at": "provider timestamp",
                }),
                encoding="utf-8",
            )
            with patch.object(web_server, "get_running_pid", return_value=None):
                runtime = web_server._read_profile_runtime(profile_home)

        self.assertEqual(runtime["platforms"], [{
            "name": "telegram",
            "state": "unknown",
            "needs_attention": False,
            "updated_at": None,
        }])

    def test_profile_runtime_sanitizes_pid_and_stale_starting_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_home = Path(tmp)
            (profile_home / "gateway_state.json").write_text(
                json.dumps({"gateway_state": "starting"}),
                encoding="utf-8",
            )
            with patch.object(web_server, "get_running_pid", return_value="not-a-pid"):
                runtime = web_server._read_profile_runtime(profile_home)

        self.assertIsNone(runtime["pid"])
        self.assertFalse(runtime["running"])
        self.assertTrue(runtime["stale"])

    def test_status_endpoint_safely_handles_malformed_runtime_state(self):
        malformed = {
            "gateway_state": {"unexpected": True},
            "platforms": [],
            "updated_at": {"unexpected": True},
            "exit_reason": {"sensitive": True},
        }
        with patch.object(web_server, "check_config_version", return_value=(1, 1)), patch.object(
            web_server, "get_running_pid", return_value=None
        ), patch.object(web_server, "read_runtime_status", return_value=malformed):
            status = asyncio.run(web_server.get_status())

        self.assertEqual(status["gateway_state"], "stopped")
        self.assertEqual(status["gateway_platforms"], {})
        self.assertIsNone(status["gateway_updated_at"])
        self.assertTrue(status["gateway_has_exit_reason"])
        self.assertNotIn("gateway_exit_reason", status)

    def test_status_sanitizes_remote_health_pid(self):
        with patch.object(web_server, "check_config_version", return_value=(1, 1)), patch.object(
            web_server, "get_running_pid", return_value=None
        ), patch.object(web_server, "read_runtime_status", return_value=None), patch.object(
            web_server, "_GATEWAY_HEALTH_URL", "http://gateway.invalid"
        ), patch.object(
            web_server,
            "_probe_gateway_health",
            return_value=(True, {
                "gateway_state": "running",
                "pid": "provider response",
                "platforms": {},
            }),
        ):
            status = asyncio.run(web_server.get_status())

        self.assertTrue(status["gateway_running"])
        self.assertIsNone(status["gateway_pid"])
        self.assertEqual(status["gateway_state"], "running")

    def test_status_platform_sanitizer_drops_raw_adapter_errors(self):
        safe = web_server._safe_status_platforms({
            "telegram": {
                "state": "fatal",
                "error_code": "provider_error",
                "error_message": "sensitive provider response",
                "updated_at": {"unexpected": True},
                "needs_attention": True,
            },
            "unknown_state": {
                "state": "provider leaked detail",
                "updated_at": None,
            },
            "malformed": [],
        })

        self.assertEqual(safe, {
            "telegram": {
                "state": "fatal",
                "updated_at": None,
                "needs_attention": True,
            },
            "unknown_state": {
                "state": "unknown",
                "updated_at": None,
                "needs_attention": False,
            },
        })

    def test_gateway_action_requires_confirmation(self):
        body = web_server.HarnessGatewayAction(confirmed=False)
        with self.assertRaises(web_server.HTTPException) as raised:
            asyncio.run(web_server.run_harness_gateway_action("coder", "restart", body))
        self.assertEqual(raised.exception.status_code, 400)

    def test_gateway_action_requires_a_boolean_confirmation(self):
        with self.assertRaises(Exception):
            web_server.HarnessGatewayAction(confirmed="true")

    def test_legacy_restart_requires_confirmation(self):
        with self.assertRaises(web_server.HTTPException) as raised:
            asyncio.run(
                web_server.restart_gateway(
                    body=web_server.HarnessGatewayAction(confirmed=False)
                )
            )
        self.assertEqual(raised.exception.status_code, 400)

    def test_legacy_actions_are_default_profile_scoped_and_tracked(self):
        class FakeProcess:
            pid = 654

            def poll(self):
                return None

        with patch.object(
            web_server,
            "_start_tracked_harness_action",
            return_value=web_server._ActionLaunch(FakeProcess(), "legacy-invocation"),
        ) as start_action:
            restart = asyncio.run(
                web_server.restart_gateway(
                    body=web_server.HarnessGatewayAction(confirmed=True)
                )
            )
            update = asyncio.run(
                web_server.update_hermes(
                    body=web_server.HarnessGatewayAction(confirmed=True)
                )
            )

        self.assertEqual(restart["profile"], "default")
        self.assertEqual(update["profile"], "default")
        self.assertEqual(start_action.call_args_list[0].args, (
            "default", "restart", ["--profile", "default", "gateway", "restart"],
            "gateway-restart",
        ))
        self.assertEqual(start_action.call_args_list[1].args, (
            "default", "update", ["--profile", "default", "update"],
            "hermes-update",
        ))

    def test_default_gateway_action_is_explicitly_profile_scoped(self):
        with patch.object(profiles_mod, "validate_profile_name"):
            name, command = web_server._resolve_harness_action("default", "restart")

        self.assertEqual(name, "gateway-default-restart")
        self.assertEqual(command, ["--profile", "default", "gateway", "restart"])

    def test_gateway_action_validates_profile_and_uses_profile_scoped_command(self):
        class FakeProcess:
            pid = 321

            def poll(self):
                return None

        with patch.object(profiles_mod, "validate_profile_name"), patch.object(
            profiles_mod, "profile_exists", return_value=True
        ), patch.object(
            web_server, "_spawn_hermes_action", return_value=FakeProcess()
        ) as spawn, patch.object(web_server, "_audit_harness_action"):
            payload = asyncio.run(
                web_server.run_harness_gateway_action(
                    "coder", "restart", web_server.HarnessGatewayAction(confirmed=True)
                )
            )

        self.assertEqual(payload["name"], "gateway-coder-restart")
        self.assertEqual(payload["profile"], "coder")
        self.assertRegex(payload["invocation_id"], r"^[0-9a-f]{32}$")
        spawn.assert_called_once_with(
            ["--profile", "coder", "gateway", "restart"],
            "gateway-coder-restart",
            log_file_name="gateway-coder-restart.log",
        )
        web_server._ACTION_PROCS.pop("gateway-coder-restart", None)
        web_server._ACTION_META.pop("gateway-coder-restart", None)
        web_server._ACTION_INVOCATIONS.pop(payload["invocation_id"], None)
        web_server._ACTION_FINALIZED = {
            key for key in web_server._ACTION_FINALIZED
            if key[0] != "gateway-coder-restart"
        }

    def test_gateway_action_blocks_any_concurrent_action_for_same_profile(self):
        class RunningProcess:
            def poll(self):
                return None

        web_server._ACTION_META["gateway-coder-start"] = {"profile": "coder", "action": "start"}
        web_server._ACTION_PROCS["gateway-coder-start"] = RunningProcess()  # type: ignore[assignment]
        try:
            with patch.object(profiles_mod, "validate_profile_name"), patch.object(
                profiles_mod, "profile_exists", return_value=True
            ):
                with self.assertRaises(web_server.HTTPException) as raised:
                    asyncio.run(
                        web_server.run_harness_gateway_action(
                            "coder", "stop", web_server.HarnessGatewayAction(confirmed=True)
                        )
                    )
            self.assertEqual(raised.exception.status_code, 409)
        finally:
            web_server._ACTION_PROCS.pop("gateway-coder-start", None)
            web_server._ACTION_META.pop("gateway-coder-start", None)

    def test_update_serializes_with_gateway_actions_across_profiles(self):
        class RunningProcess:
            def poll(self):
                return None

        web_server._ACTION_META["gateway-coder-restart"] = {
            "profile": "coder", "action": "restart"
        }
        web_server._ACTION_PROCS["gateway-coder-restart"] = RunningProcess()  # type: ignore[assignment]
        try:
            with self.assertRaises(web_server.HTTPException) as raised:
                asyncio.run(
                    web_server.update_hermes(
                        body=web_server.HarnessGatewayAction(confirmed=True)
                    )
                )
            self.assertEqual(raised.exception.status_code, 409)
        finally:
            web_server._ACTION_PROCS.pop("gateway-coder-restart", None)
            web_server._ACTION_META.pop("gateway-coder-restart", None)

        web_server._ACTION_META["hermes-update"] = {
            "profile": "default", "action": "update"
        }
        web_server._ACTION_PROCS["hermes-update"] = RunningProcess()  # type: ignore[assignment]
        try:
            with patch.object(profiles_mod, "validate_profile_name"), patch.object(
                profiles_mod, "profile_exists", return_value=True
            ):
                with self.assertRaises(web_server.HTTPException) as raised:
                    asyncio.run(
                        web_server.run_harness_gateway_action(
                            "coder", "restart", web_server.HarnessGatewayAction(confirmed=True)
                        )
                    )
            self.assertEqual(raised.exception.status_code, 409)
        finally:
            web_server._ACTION_PROCS.pop("hermes-update", None)
            web_server._ACTION_META.pop("hermes-update", None)

    def test_legacy_action_shares_default_profile_lock(self):
        class RunningProcess:
            def poll(self):
                return None

        web_server._ACTION_META["gateway-default-restart"] = {
            "profile": "default", "action": "restart"
        }
        web_server._ACTION_PROCS["gateway-default-restart"] = RunningProcess()  # type: ignore[assignment]
        try:
            with self.assertRaises(web_server.HTTPException) as raised:
                asyncio.run(
                    web_server.update_hermes(
                        body=web_server.HarnessGatewayAction(confirmed=True)
                    )
                )
            self.assertEqual(raised.exception.status_code, 409)
        finally:
            web_server._ACTION_PROCS.pop("gateway-default-restart", None)
            web_server._ACTION_META.pop("gateway-default-restart", None)

    def test_action_status_never_returns_persisted_logs_without_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            (log_dir / "gateway-restart.log").write_text(
                "provider response https://sensitive.invalid/token\\n",
                encoding="utf-8",
            )
            with patch.object(web_server, "_ACTION_LOG_DIR", log_dir):
                web_server._ACTION_PROCS.pop("gateway-restart", None)
                web_server._ACTION_META.pop("gateway-restart", None)
                status = asyncio.run(
                    web_server.get_action_status(name="gateway-restart")
                )

        self.assertEqual(status["lines"], [])

    def test_completion_audit_is_once_per_process_not_once_per_action_name(self):
        class FinishedProcess:
            def __init__(self, pid):
                self.pid = pid

            def wait(self):
                return 0

        with patch.object(web_server, "_audit_harness_action") as audit:
            first = FinishedProcess(901)
            second = FinishedProcess(901)
            web_server._watch_harness_action(
                "gateway-coder-start", "coder", "start", first
            )
            web_server._watch_harness_action(
                "gateway-coder-start", "coder", "start", second
            )

        self.assertEqual(audit.call_count, 2)
        finalized = {
            key for key in web_server._ACTION_FINALIZED
            if key[0] == "gateway-coder-start"
        }
        self.assertIn(("gateway-coder-start", first), finalized)
        self.assertIn(("gateway-coder-start", second), finalized)
        web_server._ACTION_FINALIZED = {
            key for key in web_server._ACTION_FINALIZED
            if key[0] != "gateway-coder-start"
        }

    def test_action_status_can_attribute_repeated_invocations(self):
        class FinishedProcess:
            def __init__(self, pid):
                self.pid = pid

            def poll(self):
                return 0

        first = FinishedProcess(901)
        second = FinishedProcess(902)
        first_id = "a" * 32
        second_id = "b" * 32
        name = "gateway-coder-restart"
        meta_first = {
            "profile": "coder",
            "action": "restart",
            "invocation_id": first_id,
        }
        meta_second = {
            "profile": "coder",
            "action": "restart",
            "invocation_id": second_id,
        }
        web_server._ACTION_LOG_FILES[name] = f"{name}.log"
        web_server._ACTION_INVOCATIONS[first_id] = (name, first, meta_first)
        web_server._ACTION_INVOCATIONS[second_id] = (name, second, meta_second)
        web_server._ACTION_PROCS[name] = second
        web_server._ACTION_META[name] = meta_second
        try:
            first_status = asyncio.run(
                web_server.get_action_status(name, invocation_id=first_id)
            )
            second_status = asyncio.run(
                web_server.get_action_status(name, invocation_id=second_id)
            )
        finally:
            web_server._ACTION_INVOCATIONS.pop(first_id, None)
            web_server._ACTION_INVOCATIONS.pop(second_id, None)
            web_server._ACTION_PROCS.pop(name, None)
            web_server._ACTION_META.pop(name, None)

        self.assertEqual(first_status["pid"], 901)
        self.assertEqual(first_status["invocation_id"], first_id)
        self.assertEqual(second_status["pid"], 902)
        self.assertEqual(second_status["invocation_id"], second_id)


if __name__ == "__main__":
    unittest.main()
