"""Offline contract tests for the Agent Sandbox terminal backend."""

from __future__ import annotations

import json
import subprocess

import pytest

from plugins.terminal.agent_sandbox.provider import (
    AgentSandboxEnvironment,
    AgentSandboxError,
    _config_values,
    _safe_task_label,
)


IMAGE = "registry.example/coding@sha256:" + "a" * 64


def config(**overrides):
    values = {
        "kubectl_path": "/usr/bin/kubectl",
        "namespace": "agent-sandbox-tasks",
        "image": IMAGE,
        "create_timeout": 1,
        "ready_timeout": 1,
        "cleanup_timeout": 1,
        "max_output_bytes": 1024,
    }
    values.update(overrides)
    return _config_values(values)


def test_configuration_requires_immutable_image_and_fixed_namespace():
    with pytest.raises(ValueError, match="immutable"):
        _config_values({"kubectl_path": "/usr/bin/kubectl", "image": "busybox:latest"})
    with pytest.raises(ValueError, match="namespace"):
        _config_values({"kubectl_path": "/usr/bin/kubectl", "namespace": "default", "image": IMAGE})
    with pytest.raises(ValueError, match="max_output_bytes"):
        _config_values({"kubectl_path": "/usr/bin/kubectl", "image": IMAGE, "max_output_bytes": 0})


def test_task_ids_are_kubernetes_label_safe():
    label = _safe_task_label("session:profile/123")
    assert label.startswith("session-profile-123-")
    assert len(label) <= 63
    with pytest.raises(ValueError):
        _safe_task_label("bad\nidentity")
    with pytest.raises(ValueError):
        _safe_task_label("x" * 257)


def test_configuration_bounds_creation_timeout():
    values = _config_values({"kubectl_path": "/usr/bin/kubectl", "image": IMAGE, "create_timeout": 7})
    assert values["create_timeout"] == 7
    with pytest.raises(ValueError, match="create_timeout"):
        _config_values({"kubectl_path": "/usr/bin/kubectl", "image": IMAGE, "create_timeout": 0})


def test_manifest_contains_isolation_and_resource_contract():
    env = object.__new__(AgentSandboxEnvironment)
    env.config = config()
    env.task_id = "coding-123"
    env.task_label = "coding-123"
    env.sandbox_name = "hermes-task-example"
    env.namespace = "agent-sandbox-tasks"
    env.container = "task"
    manifest = env._manifest()
    pod = manifest["spec"]["podTemplate"]["spec"]
    container = pod["containers"][0]
    assert manifest["apiVersion"] == "agents.x-k8s.io/v1beta1"
    assert manifest["spec"]["shutdownPolicy"] == "Delete"
    assert pod["automountServiceAccountToken"] is False
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
    assert {volume["name"] for volume in pod["volumes"]} == {"workspace", "tmp"}
    assert container["resources"]["limits"]["memory"] == "256Mi"
    assert env.config["max_output_bytes"] == 1024


def test_workspace_cwd_rejects_host_paths():
    env = object.__new__(AgentSandboxEnvironment)
    assert env._workspace_cwd("/root") == "/workspace"
    assert env._workspace_cwd("/workspace/repo") == "/workspace/repo"
    assert env._workspace_cwd("/workspace/../../tmp") == "/workspace"
    assert env._workspace_cwd("/workspace-other") == "/workspace"


def test_output_collector_uses_backend_limit():
    env = object.__new__(AgentSandboxEnvironment)
    env.config = {"max_output_bytes": 64}
    collector = env._new_output_collector(None, True)
    collector.append("1" * 1000)
    assert len(collector.render()) <= 64
    assert "TRUNCATED" in collector.render()


def test_explicit_timeout_is_clamped_to_backend_limit(monkeypatch):
    env = object.__new__(AgentSandboxEnvironment)
    env.config = {"command_timeout": 7}
    env.timeout = 7
    captured = {}

    def execute(_self, command, cwd="", **kwargs):
        captured.update(kwargs)
        return {"output": "", "returncode": 0}

    monkeypatch.setattr("tools.environments.base.BaseEnvironment.execute", execute)
    AgentSandboxEnvironment.execute(env, "true", timeout=99)
    assert captured["timeout"] == 7


def test_shell_contract_rejects_images_without_bash(monkeypatch):
    env = object.__new__(AgentSandboxEnvironment)
    env.kubectl = "/usr/bin/kubectl"
    env.pod_name = "task-pod"
    env.namespace = "agent-sandbox-tasks"
    env.container = "task"
    env.task_label = "task-pod-label"
    env.sandbox_name = "hermes-task-example"
    env.sandbox_uid = "sandbox-uid"
    env._pod = lambda: {
        "metadata": {"name": "task-pod", "namespace": "agent-sandbox-tasks",
                      "labels": {"agent-sandbox.rbtr.dev/task-id": "task-pod-label",
                                  "agent-sandbox.rbtr.dev/role": "coding-task"},
                      "ownerReferences": [{"apiVersion": "agents.x-k8s.io/v1beta1", "kind": "Sandbox",
                                           "name": "hermes-task-example", "uid": "sandbox-uid"}]},
        "spec": {
            "automountServiceAccountToken": False,
            "containers": [{"name": "task", "securityContext": {"privileged": False, "allowPrivilegeEscalation": False}}],
        },
    }
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 127))
    with pytest.raises(AgentSandboxError, match="must contain bash"):
        env._check_shell()


def test_exec_argv_does_not_use_a_host_shell():
    env = object.__new__(AgentSandboxEnvironment)
    env.kubectl = "/usr/bin/kubectl"
    env.pod_name = "task-pod"
    env.namespace = "agent-sandbox-tasks"
    env.container = "task"
    env.task_label = "task-pod-label"
    env.sandbox_name = "hermes-task-example"
    env.sandbox_uid = "sandbox-uid"
    env._pod = lambda: {
        "metadata": {"name": "task-pod", "namespace": "agent-sandbox-tasks",
                      "labels": {"agent-sandbox.rbtr.dev/task-id": "task-pod-label",
                                  "agent-sandbox.rbtr.dev/role": "coding-task"},
                      "ownerReferences": [{"apiVersion": "agents.x-k8s.io/v1beta1", "kind": "Sandbox",
                                           "name": "hermes-task-example", "uid": "sandbox-uid"}]},
        "spec": {
            "automountServiceAccountToken": False,
            "containers": [{"name": "task", "securityContext": {"privileged": False, "allowPrivilegeEscalation": False}}],
        },
    }
    argv = env._exec_argv("printf '%s' ok", login=False)
    assert argv == [
        "/usr/bin/kubectl", "exec", "task-pod", "-n", "agent-sandbox-tasks", "-c", "task",
        "--", "bash", "-c", "printf '%s' ok",
    ]
    assert "shell=True" not in argv


def test_existing_sandbox_is_not_adopted(monkeypatch):
    env = object.__new__(AgentSandboxEnvironment)
    env.config = config()
    env.task_id = "coding-123"
    env.task_label = "coding-123"
    env.sandbox_name = "hermes-task-example"
    env.namespace = "agent-sandbox-tasks"
    env.container = "task"
    env._owned = False
    env._get_json = lambda *args, **kwargs: {"metadata": {
        "name": env.sandbox_name, "namespace": env.namespace,
        "labels": {"agent-sandbox.rbtr.dev/task-id": env.task_label,
                    "agent-sandbox.rbtr.dev/role": "coding-task"},
        "annotations": {"agent-sandbox.rbtr.dev/full-task-id": env.task_id},
    }}
    with pytest.raises(AgentSandboxError, match="already exists"):
        env._ensure_sandbox()
    assert env._owned is False


def test_readiness_rejects_unexpected_pod_identity(monkeypatch):
    env = object.__new__(AgentSandboxEnvironment)
    env.config = config()
    env.namespace = "agent-sandbox-tasks"
    env.task_label = "coding-123"
    env._kubectl = lambda *args, **kwargs: (0, json.dumps({"items": [{
        "metadata": {"name": "wrong", "namespace": "other", "labels": {"agent-sandbox.rbtr.dev/task-id": "coding-123"}},
    }]}))
    assert env._pod() is None


def test_kubectl_errors_are_bounded_and_redacted(monkeypatch):
    env = object.__new__(AgentSandboxEnvironment)
    env.kubectl = "/usr/bin/kubectl"
    env.config = {"max_output_bytes": 1024}

    class TimeoutProcess:
        pid = 123
        returncode = None
        stdout = stderr = None
        killed = False
        def wait(self, timeout: float | None = None):
            if not self.killed:
                raise subprocess.TimeoutExpired("kubectl", float(timeout or 0))
        def kill(self):
            self.killed = True

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: TimeoutProcess())
    env._kill_process = lambda proc: proc.kill()
    with pytest.raises(AgentSandboxError, match="timed out"):
        env._kubectl(["get", "sandbox"], timeout=1)


def test_validate_sandbox_rejects_changed_image():
    env = object.__new__(AgentSandboxEnvironment)
    env.config = config()
    env.task_id = "coding-123"
    env.task_label = "coding-123"
    env.sandbox_name = "hermes-task-example"
    env.namespace = "agent-sandbox-tasks"
    env.container = "task"
    sandbox = env._manifest()
    sandbox["spec"]["podTemplate"]["spec"]["containers"][0]["image"] = "registry.example/other@sha256:" + "b" * 64
    with pytest.raises(AgentSandboxError, match="image"):
        env._validate_sandbox(sandbox)


def test_validate_sandbox_rejects_host_namespace_and_extra_container():
    env = object.__new__(AgentSandboxEnvironment)
    env.config = config()
    env.task_id = "coding-123"
    env.task_label = "coding-123"
    env.sandbox_name = "hermes-task-example"
    env.namespace = "agent-sandbox-tasks"
    env.container = "task"
    sandbox = env._manifest()
    pod_spec = sandbox["spec"]["podTemplate"]["spec"]
    pod_spec["hostNetwork"] = True
    pod_spec["containers"].append({"name": "unexpected"})
    with pytest.raises(AgentSandboxError, match="host namespace"):
        env._validate_sandbox(sandbox)


def test_validate_pod_identity_rejects_privileged_extra_container_and_host_namespace():
    env = object.__new__(AgentSandboxEnvironment)
    env.namespace = "agent-sandbox-tasks"
    env.pod_name = "task-pod"
    env.container = "task"
    env.task_label = "coding-123"
    env.sandbox_name = "hermes-task-example"
    env.sandbox_uid = "sandbox-uid"
    env._pod = lambda: {
        "metadata": {"name": "task-pod", "namespace": "agent-sandbox-tasks",
                      "labels": {"agent-sandbox.rbtr.dev/task-id": "coding-123",
                                  "agent-sandbox.rbtr.dev/role": "coding-task"},
                      "ownerReferences": [{"apiVersion": "agents.x-k8s.io/v1beta1", "kind": "Sandbox",
                                           "name": "hermes-task-example", "uid": "sandbox-uid"}]},
        "spec": {
            "hostPID": True,
            "containers": [
                {"name": "task", "securityContext": {"privileged": True}},
                {"name": "unexpected"},
            ],
        },
    }
    with pytest.raises(AgentSandboxError, match="host namespace"):
        env._validate_pod_identity()


def test_pod_identity_rejects_unrelated_owner():
    env = object.__new__(AgentSandboxEnvironment)
    env.namespace = "agent-sandbox-tasks"
    env.pod_name = "task-pod"
    env.container = "task"
    env.task_label = "coding-123"
    env.sandbox_name = "hermes-task-example"
    env.sandbox_uid = "sandbox-uid"
    env._pod = lambda: {
        "metadata": {"name": "task-pod", "namespace": "agent-sandbox-tasks",
                      "labels": {"agent-sandbox.rbtr.dev/task-id": "coding-123",
                                  "agent-sandbox.rbtr.dev/role": "coding-task"},
                      "ownerReferences": [{"apiVersion": "agents.x-k8s.io/v1beta1", "kind": "Sandbox",
                                           "name": "other", "uid": "other-uid"}]},
        "spec": {},
    }
    with pytest.raises(AgentSandboxError, match="not owned"):
        env._validate_pod_identity()


def test_constructor_preserves_cleanup_failure_note(monkeypatch):
    env = object.__new__(AgentSandboxEnvironment)
    monkeypatch.setattr(AgentSandboxEnvironment, "_ensure_sandbox", lambda self: (_ for _ in ()).throw(AgentSandboxError("create", "original")))
    monkeypatch.setattr(AgentSandboxEnvironment, "cleanup", lambda self: (_ for _ in ()).throw(AgentSandboxError("cleanup", "orphaned")))
    with pytest.raises(AgentSandboxError, match="original") as caught:
        AgentSandboxEnvironment.__init__(env, config(), "task", "/workspace", 1)
    assert "cleanup also failed" in str(caught.value.__notes__[0])


def test_kubectl_manifest_stdin_is_passed_to_the_child(monkeypatch):
    env = object.__new__(AgentSandboxEnvironment)
    env.kubectl = "/usr/bin/kubectl"
    env.config = {"max_output_bytes": 1024}

    class FakeStream:
        def __init__(self, value=b""):
            self.value = value
        def read(self, _size):
            value, self.value = self.value, b""
            return value

    class FakeStdin:
        def __init__(self):
            self.value = b""
        def write(self, value):
            self.value += value
        def close(self):
            pass

    class FakeProcess:
        pid = 123
        returncode = 0
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = FakeStream(b'{"metadata": {}}')
            self.stderr = FakeStream()
        def wait(self, timeout=None):
            pass
        def kill(self):
            pass

    process = FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    assert env._kubectl(["create", "-f", "-"], stdin="{}", timeout=1)[1] == '{"metadata": {}}'
    assert process.stdin.value == b"{}"


def test_kubectl_rejects_output_before_process_completion(monkeypatch):
    env = object.__new__(AgentSandboxEnvironment)
    env.kubectl = "/usr/bin/kubectl"
    env.config = {"max_output_bytes": 4}

    class FakeProcess:
        pid = 123
        returncode = 0
        stdin = None
        stdout = type("Stream", (), {"read": lambda self, _size: b"12345"})()
        stderr = type("Stream", (), {"read": lambda self, _size: b""})()
        def wait(self, timeout=None):
            pass
        def kill(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    env._kill_process = lambda proc: proc.kill()
    with pytest.raises(AgentSandboxError, match="output limit"):
        env._kubectl(["get", "sandbox"], timeout=1)


def test_bundled_plugin_registers_through_real_discovery():
    from agent import terminal_env_registry
    from hermes_cli import plugins as plugins_module

    manager = plugins_module.PluginManager()
    manager.discover_and_load()
    try:
        loaded = manager._plugins["terminal/agent_sandbox"]
        assert loaded.enabled is True
        assert terminal_env_registry.get_provider("agent_sandbox") is not None
    finally:
        manager.unload()


def test_provider_rejects_model_image_override(monkeypatch):
    class Context:
        def get_config(self, key, default=None):
            return {
                "kubectl_path": "/usr/bin/kubectl",
                "namespace": "agent-sandbox-tasks",
                "image": IMAGE,
            }
    from plugins.terminal.agent_sandbox.provider import AgentSandboxProvider
    provider = AgentSandboxProvider(Context())
    with pytest.raises(ValueError, match="overrides"):
        provider.create_environment(cwd="/workspace", timeout=1, task_id="default", image="other@sha256:" + "b" * 64)


def test_cleanup_is_idempotent_after_success(monkeypatch):
    env = object.__new__(AgentSandboxEnvironment)
    env._deleted = False
    env._owned = True
    env.sandbox_name = "hermes-task-example"
    env.namespace = "agent-sandbox-tasks"
    env.config = config()
    calls = []
    env._kubectl = lambda *args, **kwargs: calls.append(args) or (0, "")
    env._get_json = lambda *args, **kwargs: None
    env._pod = lambda: None
    env._configmaps = lambda: []
    env.cleanup()
    env.cleanup()
    assert len(calls) == 1
