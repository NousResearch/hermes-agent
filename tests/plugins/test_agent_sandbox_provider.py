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
        "ready_timeout": 1,
        "cleanup_timeout": 1,
    }
    values.update(overrides)
    return _config_values(values)


def test_configuration_requires_immutable_image_and_fixed_namespace():
    with pytest.raises(ValueError, match="immutable"):
        _config_values({"kubectl_path": "/usr/bin/kubectl", "image": "busybox:latest"})
    with pytest.raises(ValueError, match="namespace"):
        _config_values({"kubectl_path": "/usr/bin/kubectl", "namespace": "default", "image": IMAGE})


def test_task_ids_are_kubernetes_label_safe():
    label = _safe_task_label("session:profile/123")
    assert label.startswith("session-profile-123-")
    assert len(label) <= 63
    with pytest.raises(ValueError):
        _safe_task_label("bad\nidentity")
    with pytest.raises(ValueError):
        _safe_task_label("x" * 257)


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


def test_exec_argv_does_not_use_a_host_shell():
    env = object.__new__(AgentSandboxEnvironment)
    env.kubectl = "/usr/bin/kubectl"
    env.pod_name = "task-pod"
    env.namespace = "agent-sandbox-tasks"
    env.container = "task"
    env.task_label = "task-pod-label"
    env._pod = lambda: {
        "metadata": {"name": "task-pod", "namespace": "agent-sandbox-tasks",
                      "labels": {"agent-sandbox.rbtr.dev/task-id": "task-pod-label",
                                  "agent-sandbox.rbtr.dev/role": "coding-task"}},
        "spec": {"containers": [{"name": "task"}]},
    }
    argv = env._exec_argv("printf '%s' ok", login=False)
    assert argv == [
        "/usr/bin/kubectl", "exec", "task-pod", "-n", "agent-sandbox-tasks", "-c", "task",
        "--", "bash", "-c", "printf '%s' ok",
    ]
    assert "shell=True" not in argv


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
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("kubectl", 1)
    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(AgentSandboxError, match="timed out"):
        env._kubectl(["get", "sandbox"], timeout=1)


def test_kubectl_manifest_stdin_does_not_mix_input_and_stdin(monkeypatch):
    env = object.__new__(AgentSandboxEnvironment)
    env.kubectl = "/usr/bin/kubectl"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write(b'{"metadata": {}}')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert env._kubectl(["create", "-f", "-"], stdin="{}", timeout=1)[1] == '{"metadata": {}}'
    assert "input" in calls[0][1]
    assert "stdin" not in calls[0][1]


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
