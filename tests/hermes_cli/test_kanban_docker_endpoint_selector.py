"""P1-C regressions: immutable Docker endpoint selector for Kanban task runtimes.

Matrix (ACCEPTANCE_MATRIX P1-C):
- explicit DOCKER_CONTEXT beats DOCKER_HOST
- remote DOCKER_CONTEXT classifies as remote
- sticky current context resolves and pins
- local context + stale path map does NOT translate
- resolution failure fails closed
- one frozen selector drives translation AND every lifecycle argv
"""
from __future__ import annotations

import pytest

from hermes_cli.kanban_runtime import (
    DockerEndpointSelectorError,
    resolve_docker_endpoint_selector,
)


def _fake_inspect(endpoints: dict[str, str]):
    """Build an inspect_context hook returning canned endpoints per context."""

    def inspect(argv: list[str]) -> tuple[str, str, int]:
        if "show" in argv:
            return ("default\n", "", 0)
        if "inspect" in argv:
            name = argv[-1]
            if name in endpoints:
                import json

                return (json.dumps(endpoints[name]), "", 0)
            return ("", f"no context {name}", 1)
        return ("", "unsupported", 2)

    return inspect


def test_context_beats_docker_host():
    sel = resolve_docker_endpoint_selector(
        environ={"DOCKER_CONTEXT": "remote-ctx", "DOCKER_HOST": "ssh://ignored@host"},
        inspect_context=_fake_inspect(
            {"remote-ctx": "ssh://docker@real-remote"}
        ),
    )
    assert sel.selection == "context"
    assert sel.daemon_host == "ssh://docker@real-remote"
    assert sel.is_remote
    assert sel.cli_args == ("--context", "remote-ctx")


def test_remote_context_is_remote():
    # Sticky path with a remote current context.
    def inspect(argv):
        if "show" in argv:
            return ("my-remote\n", "", 0)
        import json

        return (json.dumps("tcp://daemons.example:2375"), "", 0)

    sel = resolve_docker_endpoint_selector(environ={}, inspect_context=inspect)
    assert sel.selection == "sticky-context"
    assert sel.is_remote
    assert sel.cli_args == ("--context", "my-remote")


def test_explicit_host_selected():
    sel = resolve_docker_endpoint_selector(
        environ={"DOCKER_HOST": "ssh://docker@box"},
        inspect_context=lambda argv: ("default\n", "", 0) if "show" in argv else ("", "", 0),
    )
    # DOCKER_CONTEXT absent -> DOCKER_HOST wins even over a sticky context.
    assert sel.selection == "host"
    assert sel.daemon_host == "ssh://docker@box"
    assert sel.is_remote
    assert sel.cli_args == ("--host", "ssh://docker@box")


def test_implicit_default_is_explicitly_pinned():
    sel = resolve_docker_endpoint_selector(
        environ={},
        inspect_context=lambda argv: ("", "context unavailable", 1),
    )
    assert sel.selection == "default"
    assert sel.cli_args == ("--context", "default")


def test_local_context_with_stale_path_map_does_not_translate(tmp_path):
    from hermes_cli.kanban_runtime import translate_runtime_mounts

    src = tmp_path / "ws"
    src.mkdir()
    runtime = {
        "version": 1,
        "task_id": "t_map",
        "workspace_kind": "dir",
        "workspace": str(src),
        "authorized_roots": [str(tmp_path)],
        "container_cwd": "/workspace",
        "mounts": [
            {"source": str(src), "target": "/workspace",
             "read_only": False, "purpose": "workspace"}
        ],
    }
    stale_map = [{"local_root": str(tmp_path / "unrelated"), "host_root": "/elsewhere"}]
    (tmp_path / "unrelated").mkdir()
    mounts = translate_runtime_mounts(
        runtime, path_map=stale_map,
        docker_host="unix:///var/run/docker.sock",
    )
    assert mounts[0]["source"] == str(src)  # identity on a local daemon


def test_resolution_failure_fails_closed():
    def broken_inspect(argv):
        return ("", "", 0) if "show" in argv else ("", "daemon gone", 1)

    with pytest.raises(DockerEndpointSelectorError):
        resolve_docker_endpoint_selector(
            environ={"DOCKER_CONTEXT": "gone"}, inspect_context=broken_inspect
        )


def test_missing_context_mapping_fails_closed_for_remote_translation():
    """Remote endpoint without a valid mapping must fail closed."""
    from hermes_cli.kanban_runtime import KanbanRuntimeError, translate_host_path

    # Use an existing local directory so the failure is purely the missing map.
    with pytest.raises(KanbanRuntimeError, match="docker_host_path_map"):
        translate_host_path(
            "/tmp",
            path_map=[],
            docker_host="ssh://docker@remote",
        )


def _selector():
    return resolve_docker_endpoint_selector(
        environ={"DOCKER_CONTEXT": "pin-ctx"},
        inspect_context=_fake_inspect({"pin-ctx": "unix:///var/run/docker.sock"}),
    )


def test_frozen_selector_pins_lifecycle_argv(monkeypatch, tmp_path):
    """probe/inspect/run/start/ps/rm all carry the same pinned selector."""
    from tools.environments.docker import DockerEnvironment

    ws = tmp_path / "ws"
    ws.mkdir()
    captured_cmds: list[list[str]] = []

    class FakeProc:
        def __init__(self):
            self.stdout = ""
            self.stderr = ""
            self.returncode = 0

    def recording_run(cmd, *a, **kw):
        cmd = list(cmd)
        captured_cmds.append(cmd)
        result = FakeProc()
        if "version" in cmd:
            result.stdout = "Client: fake\nServer: fake\n"  # availability probe passes
        elif "image" in cmd[:3]:
            result.returncode = 1  # entrypoint inspect failure -> safe defaults
        elif "run" in cmd[:4]:
            result.stdout = "abc123containerid\n"
        elif "ps" in cmd[:4]:
            result.stdout = ""  # no reusable container
        elif "NetworkMode" in " ".join(cmd):
            result.stdout = "bridge\n"
        return result

    monkeypatch.setattr("subprocess.run", recording_run)

    sel = _selector()
    DockerEnvironment(
        image="busybox",
        cwd="/workspace",
        task_id="kbt-pin",
        persist_across_processes=False,
        runtime_mounts=[
            {"source": str(ws), "target": "/workspace",
             "read_only": False, "purpose": "workspace"}
        ],
        endpoint_selector=sel,
    )

    # Every docker invocation starts with the same pinned prefix.
    docker_cmds = [c for c in captured_cmds if c and c[0].endswith("docker")]
    assert len(docker_cmds) >= 3, captured_cmds
    for c in docker_cmds:
        assert c[1:3] == ["--context", "pin-ctx"], c
    # The run command specifically carries the pin right after the exe.
    run_cmd = next(c for c in docker_cmds if "run" in c[1:4])
    assert run_cmd[1:3] == ["--context", "pin-ctx"]
    assert any("version" in c for c in docker_cmds)
