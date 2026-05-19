import json
import subprocess

import plugins.kubernetes as k8s


def test_registers_read_only_kubernetes_tools():
    calls = []

    class Ctx:
        def register_tool(self, **kwargs):
            calls.append(kwargs)

    k8s.register(Ctx())

    assert [call["name"] for call in calls] == [
        "k8s_contexts",
        "k8s_get",
        "k8s_describe",
        "k8s_logs",
        "k8s_events",
    ]
    assert {call["toolset"] for call in calls} == {"kubernetes"}
    assert all("check_fn" not in call for call in calls)


def test_missing_kubectl_returns_structured_tool_error(monkeypatch):
    monkeypatch.setattr(k8s, "_kubectl_binary", lambda: None)

    result = json.loads(k8s._handle_contexts({}))

    assert result["ok"] is False
    assert "kubectl was not found" in result["error"]


def test_get_pods_returns_compact_summary(monkeypatch):
    captured = {}

    def fake_load_json(cmd):
        captured["cmd"] = cmd
        return {
            "items": [
                {
                    "metadata": {
                        "name": "api-1",
                        "namespace": "default",
                        "creationTimestamp": "2026-01-01T00:00:00Z",
                        "labels": {"app": "api"},
                    },
                    "spec": {"nodeName": "node-a"},
                    "status": {
                        "phase": "Running",
                        "podIP": "10.0.0.12",
                        "conditions": [{"type": "Ready", "status": "True"}],
                        "containerStatuses": [
                            {
                                "name": "api",
                                "ready": True,
                                "restartCount": 2,
                                "image": "example/api:v1",
                            }
                        ],
                    },
                }
            ]
        }

    monkeypatch.setattr(k8s, "_kubectl_binary", lambda: "kubectl")
    monkeypatch.setattr(k8s, "_load_json", fake_load_json)

    result = json.loads(k8s._handle_get({
        "resource_type": "po",
        "namespace": "default",
        "selector": "app=api,tier=backend",
    }))

    assert result["ok"] is True
    assert result["resource_type"] == "pods"
    assert result["count"] == 1
    assert result["result"][0]["ready"] == "True"
    assert result["result"][0]["restarts"] == 2
    assert captured["cmd"] == [
        "kubectl", "get", "pods", "-n", "default", "-l", "app=api,tier=backend", "-o", "json",
    ]


def test_secret_resources_are_denied():
    result = json.loads(k8s._handle_get({"resource_type": "secrets"}))

    assert result["ok"] is False
    assert "secrets is not allowed" in result["error"]


def test_logs_clamps_tail_and_builds_safe_args(monkeypatch):
    captured = {}

    def fake_run(cmd, timeout=k8s._KUBECTL_TIMEOUT):
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(cmd, 0, stdout="token=abc123\nhello\n", stderr="")

    monkeypatch.setattr(k8s, "_kubectl_binary", lambda: "kubectl")
    monkeypatch.setattr(k8s, "_run_kubectl", fake_run)

    result = json.loads(k8s._handle_logs({
        "pod": "api-1",
        "namespace": "default",
        "container": "api",
        "tail": 99999,
        "previous": True,
        "since": "10m",
    }))

    assert result["ok"] is True
    assert result["tail"] == 2000
    assert "token=[REDACTED]" in result["logs"]
    assert captured["timeout"] == 45
    assert captured["cmd"] == [
        "kubectl", "logs", "api-1", "--tail", "2000", "-n", "default",
        "-c", "api", "--previous", "--since", "10m",
    ]


def test_invalid_selector_is_rejected(monkeypatch):
    monkeypatch.setattr(k8s, "_kubectl_binary", lambda: "kubectl")

    result = json.loads(k8s._handle_get({
        "resource_type": "pods",
        "selector": "app=$(rm -rf /)",
    }))

    assert result["ok"] is False
    assert "Invalid selector" in result["error"]
