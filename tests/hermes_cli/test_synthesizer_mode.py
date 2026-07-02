from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_swarm import SwarmWorkerSpec, create_swarm


POLISH_TEXT = "Za\u017c\u00f3\u0142\u0107 g\u0119\u015bl\u0105 ja\u017a\u0144"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _local_skills_root() -> Path:
    explicit = os.environ.get("HERMES_LOCAL_SKILLS_DIR")
    if explicit:
        return Path(explicit)
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        pytest.skip("LOCALAPPDATA is not available for local Hermes skill checks")
    return Path(local_appdata) / "hermes" / "skills"


def _read_local_skill(relative: str) -> str:
    root = _local_skills_root()
    if not root.exists():
        pytest.skip(f"local Hermes skills root is not present: {root}")
    path = root / relative / "SKILL.md"
    assert path.exists(), f"missing local Hermes skill: {path}"
    return path.read_text(encoding="utf-8")


def test_swarm_synthesizer_card_declares_curated_evidence_mode(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Workers research, reviewer verifies, writer synthesizes.",
            workers=[
                SwarmWorkerSpec(profile="researcher", title="Research", body="Find proof"),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )

        synthesizer = kb.get_task(conn, created.synthesizer_id)
        assert synthesizer is not None
        assert "curated parent summaries" in (synthesizer.body or "")
        assert "explicit artifacts" in (synthesizer.body or "")
        assert "capped worker logs" in (synthesizer.body or "")
        assert "missing-evidence request" in (synthesizer.body or "")
        assert "kanban-synthesizer" in (synthesizer.skills or [])
    finally:
        conn.close()


def test_local_kanban_synthesizer_skill_contains_evidence_boundary():
    text = _read_local_skill("kanban-synthesizer")

    required_phrases = [
        "curated parent summaries",
        "explicit artifacts",
        "capped worker logs",
        "missing-evidence request",
        "Do not run broad find",
        "Do not run unbounded rg",
        "Do not run huge terminal commands",
    ]

    missing = [phrase for phrase in required_phrases if phrase not in text]
    assert not missing, "missing synthesizer skill phrases: " + ", ".join(missing)


def test_synthesizer_worker_log_status_uses_capped_safe_decode(kanban_home):
    task_id = "t_synth_log"
    path = kb.worker_log_path(task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = b"SECRET-PREFIX-SHOULD-NOT-LEAK\n"
    suffix = POLISH_TEXT.encode("cp1250")
    path.write_bytes(prefix + suffix)

    status = kb.synthesizer_worker_log_status(task_id, tail_bytes=len(suffix))

    assert status is not None
    assert status["artifact_path"] == str(path)
    assert status["encoding"].lower() == "cp1250"
    assert status["used_fallback"] is True
    assert status["truncated"] is True
    assert status["excerpt"] == POLISH_TEXT
    assert "SECRET-PREFIX" not in status["excerpt"]


def test_synthesizer_worker_log_status_redacts_obvious_secrets(kanban_home):
    task_id = "t_synth_secret"
    path = kb.worker_log_path(task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "upload failed with token=sk-proj-abcdefghijklmnopqrstuvwxyz123456",
        encoding="utf-8",
    )

    status = kb.synthesizer_worker_log_status(task_id)

    assert status is not None
    assert "sk-proj-" not in status["excerpt"]
    assert "[REDACTED_SECRET]" in status["excerpt"]
