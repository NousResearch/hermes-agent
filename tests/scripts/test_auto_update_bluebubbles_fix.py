from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "auto_update_bluebubbles_fix.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location(
        "auto_update_bluebubbles_fix", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_remote_layout_supports_current_local_names():
    module = load_script_module()

    layout = module.resolve_remote_layout(
        [
            ("cherry", "https://github.com/1960697431/hermes-agent.git", "fetch"),
            ("cherry", "https://github.com/1960697431/hermes-agent.git", "push"),
            ("origin", "https://github.com/NousResearch/hermes-agent.git", "fetch"),
            ("origin", "https://github.com/NousResearch/hermes-agent.git", "push"),
        ]
    )

    assert layout.upstream == "origin"
    assert layout.fork == "cherry"


def test_tracked_status_paths_ignore_untracked_local_config_and_skills():
    module = load_script_module()

    paths = module.tracked_status_paths(
        [
            " M gateway/platforms/bluebubbles.py",
            "M  pyproject.toml",
            "?? config.yaml",
            "?? skills/devops/hermes-update-fix-cn/",
        ]
    )

    assert paths == ["gateway/platforms/bluebubbles.py", "pyproject.toml"]


def test_needs_dependency_sync_only_for_environment_files():
    module = load_script_module()

    assert module.needs_dependency_sync(["gateway/platforms/bluebubbles.py"]) is False
    assert module.needs_dependency_sync(["uv.lock"]) is True
    assert module.needs_dependency_sync(["pyproject.toml"]) is True


def test_build_bluebubbles_validation_commands_are_narrow_and_deterministic():
    module = load_script_module()

    commands = module.build_bluebubbles_validation_commands(
        python="/repo/venv/bin/python"
    )

    assert commands == [
        [
            "/repo/venv/bin/python",
            "-m",
            "py_compile",
            "gateway/platforms/bluebubbles.py",
        ],
        ["scripts/run_tests.sh", "tests/gateway/test_bluebubbles.py"],
    ]


def test_sync_dependencies_installs_dev_extra_for_pytest(monkeypatch, tmp_path):
    module = load_script_module()
    calls = []

    monkeypatch.setattr(module, "find_uv", lambda: "/bin/uv")

    def fake_run_command(cmd, cwd, logger, *, check=True, env=None):
        calls.append((cmd, cwd, env))

    monkeypatch.setattr(module, "run_command", fake_run_command)

    module.sync_dependencies(tmp_path / "project", tmp_path / "repo", lambda msg: None)

    assert calls[0][0] == [
        "/bin/uv",
        "sync",
        "--extra",
        "all",
        "--extra",
        "dev",
        "--locked",
    ]
