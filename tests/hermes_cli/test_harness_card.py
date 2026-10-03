"""Tests for the opt-in, secret-safe Harness Card + run export (issue #110680).

Covers the four acceptance properties the card has to hold: fields come from real config, secrets
never reach the artifact, the opt-in gate produces nothing when off, and the serialized key set is
stable (so a paired-arm differ sees config drift, not schema churn).
"""

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.harness_card import (
    HARNESS_CARD_SCHEMA_ID,
    HARNESS_CARD_SCHEMA_VERSION,
    build_harness_card,
    build_run_export,
    cmd_harness_card,
    harness_card_enabled,
)

_FAKE_KEY = "sk-ant-api03-FAKEKEYFAKEKEYFAKEKEYFAKEKEY1234"
_FAKE_MCP_TOKEN = "sk-live-MCPTOKEN1234567890ABCDEF"
_FAKE_URL = f"https://user:{_FAKE_KEY}@api.anthropic.com/v1"

_CARD_KEYS = {
    "schema", "generated_at", "card_id", "hermes", "profile", "model", "context", "toolsets",
    "limits", "compression", "delegation", "retries", "evaluator", "verification",
}
_RUN_KEYS = {
    "session_id", "source", "started_at", "ended_at", "wall_seconds", "end_reason", "outcome",
    "model", "tokens", "cost", "messages", "turns", "verification", "usage_by_task", "coverage",
}


def _seed_skill(home: Path, name: str, description: str) -> None:
    skill_dir = home / "skills" / "demo" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n# {name}\nbody\n", encoding="utf-8")


def _write_config(home: Path, *, enabled: bool) -> None:
    (home / "config.yaml").write_text(
        f"harness_card:\n  enabled: {str(enabled).lower()}\n"
        "model:\n"
        "  default: claude-sonnet-4-5\n"
        "  provider: anthropic\n"
        f'  base_url: "{_FAKE_URL}"\n'
        f'  api_key: "{_FAKE_KEY}"\n'
        "toolsets: [hermes-cli]\n"
        "agent:\n  max_turns: 42\n  api_max_retries: 7\n"
        "compression:\n  threshold: 0.25\n"
        "memory:\n  memory_char_limit: 1234\n"
        "mcp_servers:\n  demo:\n    command: npx\n"
        f"    env:\n      DEMO_TOKEN: {_FAKE_MCP_TOKEN}\n"
        "security:\n  redact_secrets: false\n",
        encoding="utf-8")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with the opt-in enabled; skills seeded before the first prompt build."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.chdir(tmp_path)  # avoid picking up the repo's own AGENTS.md
    _seed_skill(hermes_home, "card-skill", "a skill the card must name")
    _write_config(hermes_home, enabled=True)
    return hermes_home


def _card(session_id=None, **kwargs):
    return build_harness_card(session_id=session_id, **kwargs)


# --------------------------------------------------------------------------- #
# Fields come from real config / state
# --------------------------------------------------------------------------- #

def test_card_fields_come_from_real_config(home):
    marker = "PROJECT-CONTEXT-MARKER-12345"
    (home.parent / "AGENTS.md").write_text(f"# Project\n{marker}\n", encoding="utf-8")

    card = _card()
    blob = json.dumps(card)

    assert card["schema"] == {"id": HARNESS_CARD_SCHEMA_ID, "version": HARNESS_CARD_SCHEMA_VERSION}
    assert card["model"]["name"] == "claude-sonnet-4-5"
    assert card["model"]["provider"] == "anthropic"
    assert card["model"]["pricing"]["billing_mode"] == "official_docs_snapshot"

    # Toolsets come from the platform resolver, not from the raw key alone.
    assert "hermes-cli" in card["toolsets"]["declared"]
    assert card["toolsets"]["enabled"]
    assert card["toolsets"]["mcp_servers"] == ["demo"]

    # Effective limits / retries / compression / memory are the merged config values.
    assert card["limits"]["max_turns"] == 42
    assert card["retries"]["api_max_retries"] == 7
    assert card["compression"]["threshold"] == 0.25
    assert card["context"]["memory"]["memory_char_limit"] == 1234

    # Profile + identity are read live.
    assert card["profile"]["config_path"] == str(home / "config.yaml")
    assert card["hermes"]["version"]
    assert card["hermes"]["short_commit"]
    assert card["hermes"]["runtime"]["python"]

    # Prompt/skill/project context is identified by hash and name, never by raw text.
    assert card["context"]["system_prompt_sha256"].startswith("sha256:")
    assert card["context"]["skills"]["count"] >= 1
    assert "card-skill" in card["context"]["skills"]["names"]
    project_file = next(f for f in card["context"]["project_files"] if f["path"] == "AGENTS.md")
    assert project_file["sha256"].startswith("sha256:")
    assert marker not in blob, "project context content leaked into the card"


def test_card_is_deterministic(home):
    def body(card):
        return {key: value for key, value in card.items() if key != "generated_at"}

    first, second = _card(), _card()
    assert first["card_id"] == second["card_id"]
    assert first["card_id"].startswith("sha256:")
    assert body(first) == body(second), "same harness in, same card out"


def test_card_key_set_is_stable(home):
    """A paired-arm differ must trip on config drift, not on schema churn."""
    assert set(_card()) == _CARD_KEYS
    assert set(_card()) == _CARD_KEYS


def test_context_section_reports_zero_skills_as_a_real_zero(home, monkeypatch):
    """No skills installed -> ``0``, not ``null``; only a failed prompt build leaves ``null``."""
    import hermes_cli.harness_card as hc
    import hermes_cli.prompt_size as prompt_size
    from hermes_cli.config import load_config

    monkeypatch.setattr(prompt_size, "_build_inspection_agent", lambda platform: object())
    monkeypatch.setattr("agent.system_prompt.build_system_prompt_parts",
                        lambda agent: {"stable": "stable", "context": "context", "volatile": "volatile"})
    monkeypatch.setattr("agent.system_prompt.build_system_prompt", lambda agent: "prompt")

    section = hc._context_section(load_config(), "cli", None)

    assert section["skills"] == {"count": 0, "names": [], "index_sha256": None}
    assert section["system_prompt_sha256"].startswith("sha256:")


def test_context_section_failure_is_unavailable_not_zero(home, monkeypatch):
    import hermes_cli.harness_card as hc
    import hermes_cli.prompt_size as prompt_size
    from hermes_cli.config import load_config

    (home.parent / "AGENTS.md").write_text("# Project\n", encoding="utf-8")

    def boom(_platform):
        raise RuntimeError("prompt build unavailable")

    monkeypatch.setattr(prompt_size, "_build_inspection_agent", boom)

    section = hc._context_section(load_config(), "cli", None)

    assert section["system_prompt_sha256"] is None
    assert section["tiers"] == {"stable_sha256": None, "context_sha256": None, "volatile_sha256": None}
    assert section["skills"]["count"] is None
    assert [f["path"] for f in section["project_files"]] == ["AGENTS.md"]  # filesystem half survives


# --------------------------------------------------------------------------- #
# Secret safety
# --------------------------------------------------------------------------- #

def test_card_redacts_secrets_even_with_redaction_disabled(home):
    blob = json.dumps(_card())

    assert _FAKE_KEY not in blob, "API key leaked into the card"
    assert "FAKEKEYFAKEKEY" not in blob
    assert _FAKE_MCP_TOKEN not in blob, "MCP server env leaked into the card"
    assert _FAKE_URL not in blob
    # The credential-bearing base_url is kept but its userinfo is masked, not dropped silently.
    assert "***" in _card()["model"]["base_url"]


def test_card_never_copies_raw_config_and_env(home):
    card = _card()
    dumped = json.dumps(card)

    # Field selection, not whole-section dumping: an api_key has nowhere to land.
    assert "api_key" not in dumped
    # MCP servers contribute names only — definitions carry env vars and headers.
    assert card["toolsets"]["mcp_servers"] == ["demo"]
    assert "command" not in json.dumps(card["toolsets"])


# --------------------------------------------------------------------------- #
# Opt-in gate
# --------------------------------------------------------------------------- #

def test_opt_in_off_produces_nothing(home, tmp_path, capsys):
    _write_config(home, enabled=False)
    assert harness_card_enabled() is False

    out = tmp_path / "card.json"
    with pytest.raises(SystemExit) as exit_info:
        cmd_harness_card(SimpleNamespace(
            platform="cli", cwd=None, session=None, solved=None, failure_class=None,
            out=str(out), json=True))

    captured = capsys.readouterr()
    assert exit_info.value.code == 1, "an automated caller must see a non-zero exit"
    assert captured.out == "", "no card may be printed while opt-in is off"
    assert not out.exists(), "no artifact may be written while opt-in is off"
    assert "harness_card.enabled" in captured.err


def test_opt_in_on_writes_json_with_owner_only_permissions(home, tmp_path):
    out = tmp_path / "nested" / "card.json"
    cmd_harness_card(SimpleNamespace(
        platform="cli", cwd=None, session=None, solved=None, failure_class=None,
        out=str(out), json=True))

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema"]["id"] == HARNESS_CARD_SCHEMA_ID
    assert list(written) == sorted(written), "sorted keys keep the artifact diff-friendly"
    if os.name != "nt":  # chmod is a no-op on Windows
        assert stat.S_IMODE(out.stat().st_mode) == 0o600


def test_opt_in_on_prints_json_to_stdout(home, capsys):
    cmd_harness_card(SimpleNamespace(
        platform="cli", cwd=None, session=None, solved=None, failure_class=None,
        out=None, json=True))
    assert json.loads(capsys.readouterr().out)["schema"]["version"] == HARNESS_CARD_SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# Per-task run export (reuses usage accounting)
# --------------------------------------------------------------------------- #

def _seed_session(db, session_id="sess-1"):
    db.create_session(session_id, source="cli", model="claude-sonnet-4-5")
    db.append_message(session_id, "user", "solve it")
    db.append_message(session_id, "assistant", "done")
    db.update_token_counts(
        session_id, input_tokens=1000, output_tokens=200, model="claude-sonnet-4-5",
        billing_provider="anthropic", billing_base_url="https://api.anthropic.com/v1",
        api_call_count=2, estimated_cost_usd=0.01, cost_status="estimated",
        cost_source="official_docs_snapshot", pricing_version="anthropic-docs-2026-08")
    db.record_auxiliary_usage(
        session_id, "compression", model="claude-haiku-4-5", billing_provider="anthropic",
        input_tokens=500, output_tokens=50, estimated_cost_usd=0.002)
    db.flush_token_counts()
    return session_id


@pytest.fixture
def session_db(home):
    from hermes_state import SessionDB
    db = SessionDB()
    try:
        yield db
    finally:
        db.close()


def test_run_export_carries_per_task_usage_rows(session_db):
    _seed_session(session_db)
    run = build_run_export("sess-1", solved=True, db=session_db)

    assert set(run) == _RUN_KEYS
    assert run["session_id"] == "sess-1"
    assert run["model"]["provider"] == "anthropic"
    assert run["tokens"]["input"] == 1000
    assert run["cost"]["estimated_usd"] == pytest.approx(0.01)
    assert run["cost"]["pricing_version"] == "anthropic-docs-2026-08"

    by_task = {row["task"]: row for row in run["usage_by_task"]}
    assert set(by_task) == {"", "compression"}
    assert by_task[""]["model"] == "claude-sonnet-4-5"
    assert by_task["compression"]["model"] == "claude-haiku-4-5"
    assert by_task["compression"]["input_tokens"] == 500


def test_run_export_distinguishes_unavailable_from_reported_zero(session_db):
    _seed_session(session_db)
    run = build_run_export("sess-1", db=session_db)

    # Unavailable telemetry: null + a coverage label, never a fabricated zero.
    assert run["outcome"]["solved"] is None
    assert run["coverage"]["outcome_solved"] == "unavailable"
    assert run["turns"] == {"count": None, "productive": None, "no_action": None, "repeated_action": None}
    assert run["coverage"]["turns"] == "unavailable"

    # Reported zero: the session really made no compression call.
    assert run["verification"]["compression_count"] == 0
    assert run["tokens"]["cache_read"] == 0
    assert run["coverage"]["tokens"] == "reported"

    solved = build_run_export("sess-1", solved=True, failure_class=None, db=session_db)
    assert solved["outcome"]["solved"] is True
    assert solved["coverage"]["outcome_solved"] == "reported"


def test_run_export_unknown_session_is_an_error(session_db):
    with pytest.raises(LookupError):
        build_run_export("nope", db=session_db)


def test_session_usage_breakdown_is_stably_ordered(session_db):
    """The reader the export depends on: same rows, same order, every call."""
    _seed_session(session_db, "sess-2")
    first = session_db.session_usage_breakdown("sess-2")
    second = session_db.session_usage_breakdown("sess-2")
    assert first == second
    assert [row["task"] for row in first] == ["", "compression"]
    assert session_db.session_usage_breakdown("") == []


def test_card_includes_run_section_when_session_given(session_db, home):
    _seed_session(session_db, "sess-3")
    card = build_harness_card(session_id="sess-3", solved=False, failure_class="test_failure", db=session_db)

    assert card["run"]["session_id"] == "sess-3"
    assert card["run"]["outcome"] == {"solved": False, "failure_class": "test_failure"}
    # card_id keys the *harness*, so a different run must not change it.
    assert card["card_id"] == build_harness_card(db=session_db)["card_id"]
