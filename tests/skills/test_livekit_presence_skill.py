from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "communication"
    / "livekit-presence"
    / "scripts"
    / "livekit_presence.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("livekit_presence_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_export_persona_writes_snapshot(tmp_path: Path, monkeypatch):
    mod = load_module()
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "SOUL.md").write_text("You are Hermes.\nStay concise.\n", encoding="utf-8")

    output = tmp_path / "project" / "docs" / "hermes-persona.md"
    result = mod.export_persona(output)

    content = output.read_text(encoding="utf-8")
    assert result["success"] is True
    assert "Hermes Persona Snapshot" in content
    assert "Stay concise." in content
    assert str(hermes_home / "SOUL.md") in content


def test_write_env_local_preserves_existing_keys_when_not_overwriting(tmp_path: Path, monkeypatch):
    mod = load_module()
    project = tmp_path / "livekit-project"
    project.mkdir()
    env_path = project / ".env.local"
    env_path.write_text("LIVEKIT_URL=wss://existing.example\nKEEP=1\n", encoding="utf-8")
    monkeypatch.setenv("LIVEKIT_URL", "wss://new.example")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key123")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret123")

    result = mod.write_env_local(project, overwrite=False)

    content = env_path.read_text(encoding="utf-8")
    assert result["success"] is True
    assert "LIVEKIT_URL=wss://existing.example" in content
    assert "LIVEKIT_API_KEY=key123" in content
    assert "LIVEKIT_API_SECRET=secret123" in content
    assert "KEEP=1" in content


def test_bootstrap_project_falls_back_to_git_and_writes_docs(tmp_path: Path, monkeypatch):
    mod = load_module()
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "SOUL.md").write_text("Hermes soul.\n", encoding="utf-8")
    monkeypatch.setenv("LIVEKIT_URL", "wss://cloud.livekit.example")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key123")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret123")

    def fake_which(binary: str):
        if binary == "git":
            return "/usr/bin/git"
        return None

    def fake_run(cmd, cwd=None):
        target = Path(cmd[-1])
        target.mkdir(parents=True, exist_ok=True)
        return None

    monkeypatch.setattr(mod, "_which", fake_which)
    monkeypatch.setattr(mod, "_run", fake_run)

    project = tmp_path / "workspace" / "hermes-livekit"
    result = mod.bootstrap_project(project)

    assert result["success"] is True
    assert result["method"] == "git"
    assert (project / "docs" / "hermes-persona.md").exists()
    assert (project / "docs" / "hermes-bootstrap.md").exists()
    assert (project / ".env.local").exists()
