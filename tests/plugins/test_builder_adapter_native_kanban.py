import sys
import types

from plugins.builder_adapter.native import BUILDER_WORKER_POLICY, NativeKanbanBackend
from plugins.builder_adapter.models import ResolvedDispatchRequest
from tests.plugins.test_builder_adapter_schema import request_payload


class Closing:
    def __enter__(self):
        return object()

    def __exit__(self, *args):
        return False


def test_native_backend_uses_argument_safe_kanban_api(monkeypatch, tmp_path):
    calls = {}
    fake = types.SimpleNamespace()
    fake.connect_closing = lambda **kwargs: Closing()

    def create_task(conn, **kwargs):
        calls.update(kwargs)
        return "t_12345678"

    fake.create_task = create_task
    fake_module = types.ModuleType("hermes_cli")
    fake_module.kanban_db = fake
    monkeypatch.setitem(sys.modules, "hermes_cli", fake_module)
    backend = NativeKanbanBackend(board="governed")
    intent = request_payload(tmp_path)
    request = ResolvedDispatchRequest.model_validate(
        {
            **{
                key: intent[key]
                for key in (
                    "schema_version",
                    "dispatch_id",
                    "idempotency_key",
                    "cycle_id",
                    "builder_role",
                    "completion_schema_version",
                )
            },
            "contract": {
                "contract_id": intent["contract_id"],
                "repository_id": "orchestrator",
                "path": "contracts/active/FEAT_TEST_001.json",
                "commit": "1" * 40,
                "sha256": "2" * 64,
            },
            "repository": {
                "repository_id": intent["repository_id"],
                "canonical_remote": "git@example.invalid:hermes.git",
            },
            "worktree_path": str(tmp_path),
            "branch": "feat/test",
            "expected_head_sha": "3" * 40,
            "allowed_path_manifest": {
                "repository_id": "orchestrator",
                "path": "contracts/manifests/test.json",
                "commit": "1" * 40,
                "sha256": "4" * 64,
            },
            "validation_profile": "hermes-builder-adapter-strict.v1",
            "timeout_policy": {
                "max_runtime_seconds": 60,
                "heartbeat_timeout_seconds": 30,
            },
            "retry_policy": {
                "max_attempts": 2,
                "retryable_terminal_states": ["CRASHED"],
            },
        }
    )
    assert backend.create_task("a" * 64, request) == "t_12345678"
    assert calls["assignee"] == "deepseek-builder"
    assert calls["workspace_path"] == str(tmp_path)
    assert calls["idempotency_key"] == request.idempotency_key
    assert calls["worker_policy"] == BUILDER_WORKER_POLICY
    assert "command" not in calls and "argv" not in calls
