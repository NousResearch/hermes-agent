"""Security tests for Terminal-Bench 2 archive extraction."""

import base64
import importlib
import io
import sys
import tarfile
import types

import pytest


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_terminalbench_module(monkeypatch):
    class _EvalHandlingEnum:
        STOP_TRAIN = "stop_train"

    class _APIServerConfig:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _AgentResult:
        pass

    class _HermesAgentLoop:
        pass

    class _HermesAgentBaseEnv:
        pass

    class _HermesAgentEnvConfig:
        pass

    class _ToolContext:
        pass

    stub_modules = {
        "atroposlib": _stub_module("atroposlib"),
        "atroposlib.envs": _stub_module("atroposlib.envs"),
        "atroposlib.envs.base": _stub_module(
            "atroposlib.envs.base",
            EvalHandlingEnum=_EvalHandlingEnum,
        ),
        "atroposlib.envs.server_handling": _stub_module("atroposlib.envs.server_handling"),
        "atroposlib.envs.server_handling.server_manager": _stub_module(
            "atroposlib.envs.server_handling.server_manager",
            APIServerConfig=_APIServerConfig,
        ),
        "environments.agent_loop": _stub_module(
            "environments.agent_loop",
            AgentResult=_AgentResult,
            HermesAgentLoop=_HermesAgentLoop,
        ),
        "environments.hermes_base_env": _stub_module(
            "environments.hermes_base_env",
            HermesAgentBaseEnv=_HermesAgentBaseEnv,
            HermesAgentEnvConfig=_HermesAgentEnvConfig,
        ),
        "environments.tool_context": _stub_module(
            "environments.tool_context",
            ToolContext=_ToolContext,
        ),
        "tools.terminal_tool": _stub_module(
            "tools.terminal_tool",
            register_task_env_overrides=lambda *args, **kwargs: None,
            clear_task_env_overrides=lambda *args, **kwargs: None,
            cleanup_vm=lambda *args, **kwargs: None,
        ),
    }

    stub_modules["atroposlib"].envs = stub_modules["atroposlib.envs"]
    stub_modules["atroposlib.envs"].base = stub_modules["atroposlib.envs.base"]
    stub_modules["atroposlib.envs"].server_handling = stub_modules["atroposlib.envs.server_handling"]
    stub_modules["atroposlib.envs.server_handling"].server_manager = stub_modules[
        "atroposlib.envs.server_handling.server_manager"
    ]

    for name, module in stub_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "environments.benchmarks.terminalbench_2.terminalbench2_env"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _build_tar_b64(entries):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for entry in entries:
            kind = entry["kind"]
            info = tarfile.TarInfo(entry["name"])

            if kind == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
                continue

            if kind == "file":
                data = entry["data"].encode("utf-8")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                continue

            if kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = entry["target"]
                tar.addfile(info)
                continue

            raise ValueError(f"Unknown tar entry kind: {kind}")

    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_extract_base64_tar_allows_safe_files(tmp_path, monkeypatch):
    module = _load_terminalbench_module(monkeypatch)
    archive = _build_tar_b64(
        [
            {"kind": "dir", "name": "nested"},
            {"kind": "file", "name": "nested/hello.txt", "data": "hello"},
        ]
    )

    target = tmp_path / "extract"
    module._extract_base64_tar(archive, target)

    assert (target / "nested" / "hello.txt").read_text(encoding="utf-8") == "hello"


def test_extract_base64_tar_rejects_path_traversal(tmp_path, monkeypatch):
    module = _load_terminalbench_module(monkeypatch)
    archive = _build_tar_b64(
        [
            {"kind": "file", "name": "../escape.txt", "data": "owned"},
        ]
    )

    target = tmp_path / "extract"
    with pytest.raises(ValueError, match="Unsafe archive member path"):
        module._extract_base64_tar(archive, target)

    assert not (tmp_path / "escape.txt").exists()


def test_extract_base64_tar_rejects_symlinks(tmp_path, monkeypatch):
    module = _load_terminalbench_module(monkeypatch)
    archive = _build_tar_b64(
        [
            {"kind": "symlink", "name": "link", "target": "../../escape.txt"},
        ]
    )

    target = tmp_path / "extract"
    with pytest.raises(ValueError, match="Unsupported archive member type"):
        module._extract_base64_tar(archive, target)

    assert not (target / "link").exists()
