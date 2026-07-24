import json
import os
import stat
import subprocess
from pathlib import Path

import pytest


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


def test_project_identity_is_stable_for_subdirs_and_symlinks(tmp_path):
    from agent.project_local import project_identity

    repo = tmp_path / "repo"
    nested = repo / "pkg"
    nested.mkdir(parents=True)
    _git(repo, "init")

    linked = tmp_path / "linked"
    try:
        linked.symlink_to(repo, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")

    root_id, root_path = project_identity(repo)
    nested_id, nested_path = project_identity(nested)
    linked_id, linked_path = project_identity(linked / "pkg")

    assert root_id == nested_id == linked_id
    assert root_path == nested_path == linked_path == repo.resolve()


def test_candidate_detection_requires_recognized_project_files(tmp_path):
    from agent.project_local import (
        clear_project_local_cache,
        has_project_local_candidate,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / ".hermes" / "plans").mkdir(parents=True)

    clear_project_local_cache()
    assert has_project_local_candidate(repo) is False

    skill = repo / ".hermes" / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\n---\nbody\n", encoding="utf-8")

    clear_project_local_cache()
    assert has_project_local_candidate(repo) is True


def test_symlinked_project_hermes_dir_is_ignored(tmp_path):
    from agent.project_local import (
        clear_project_local_cache,
        has_project_local_candidate,
    )

    repo = tmp_path / "repo"
    external = tmp_path / "external-hermes"
    (external / "skills" / "demo").mkdir(parents=True)
    (external / "skills" / "demo" / "SKILL.md").write_text("body", encoding="utf-8")
    repo.mkdir()
    _git(repo, "init")
    try:
        (repo / ".hermes").symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")

    clear_project_local_cache()
    assert has_project_local_candidate(repo) is False


def test_project_mcp_trust_sidecar_is_profile_local_and_mode_600(tmp_path, monkeypatch):
    from agent.project_local import (
        resolve_project_local_state,
        trust_path,
        trust_project_mcp,
        trusted_project_mcp_servers,
    )

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo = tmp_path / "repo"
    config_dir = repo / ".hermes"
    config_dir.mkdir(parents=True)
    _git(repo, "init")
    (config_dir / "config.yaml").write_text(
        "mcp_servers:\n  demo:\n    command: node\n",
        encoding="utf-8",
    )

    state = resolve_project_local_state(repo)
    assert state is not None
    assert state.mcp.servers == ("demo",)
    assert state.mcp.trusted is False
    assert trusted_project_mcp_servers(repo) == {}

    trusted = trust_project_mcp(repo)
    assert trusted is not None
    assert trusted.mcp.trusted is True
    assert trusted_project_mcp_servers(repo)["demo"]["command"] == "node"

    mode = stat.S_IMODE(trust_path().stat().st_mode)
    if os.name != "nt":
        assert mode == 0o600
    data = json.loads(trust_path().read_text(encoding="utf-8"))
    assert trusted.canonical_id in data["projects"]

    (config_dir / "config.yaml").write_text(
        "mcp_servers:\n  changed:\n    command: node\n",
        encoding="utf-8",
    )
    assert trusted_project_mcp_servers(repo) == {}


def test_stored_prompt_runtime_match_includes_project_manifest(tmp_path, monkeypatch):
    from agent.conversation_loop import _stored_prompt_matches_runtime
    from agent.project_local import resolve_project_local_state, trust_project_mcp

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo = tmp_path / "repo"
    config_dir = repo / ".hermes"
    config_dir.mkdir(parents=True)
    _git(repo, "init")
    (config_dir / "config.yaml").write_text(
        "mcp_servers:\n  demo:\n    command: node\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TERMINAL_CWD", str(repo))

    class Agent:
        model = "model-a"
        provider = "provider-a"

    state = resolve_project_local_state(repo)
    assert state is not None
    prompt = (
        "Conversation started: Thursday, July 2, 2026\n"
        "Model: model-a\n"
        "Provider: provider-a\n"
        f"Project-Local-ID: {state.canonical_id}\n"
        f"Project-Local-Manifest: {state.manifest_hash}"
    )
    assert _stored_prompt_matches_runtime(Agent(), prompt) is True

    trusted = trust_project_mcp(repo)
    assert trusted is not None
    assert trusted.manifest_hash != state.manifest_hash
    assert _stored_prompt_matches_runtime(Agent(), prompt) is False
