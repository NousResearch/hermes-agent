"""Real-writer coverage for bounded API request-dump retention."""

import os
import sys
import types
from pathlib import Path

import pytest
import yaml

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent
from agent import agent_runtime_helpers


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "sessions").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def agent(hermes_home, monkeypatch):
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    built = run_agent.AIAgent(
        model="gpt-4o",
        base_url="http://127.0.0.1:9208/v1",
        api_key="test-key",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    built.logs_dir = hermes_home / "sessions"
    built._vprint = lambda *args, **kwargs: None
    return built


def _write_config(home: Path, value) -> None:
    (home / "config.yaml").write_text(
        f"sessions:\n  request_dump_retention: {value}\n",
        encoding="utf-8",
    )


def _dump(agent, marker: str, session_id: str):
    agent.session_id = session_id
    path = agent._dump_api_request_debug(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": marker}]},
        reason="non_retryable_client_error",
        error=ValueError("HTTP 400"),
    )
    assert path is not None and path.exists()
    return path


def _real_dumps(directory: Path):
    return sorted(
        path
        for path in directory.glob("request_dump_*.json")
        if not path.is_symlink() and path.is_file()
    )


def test_real_writer_keeps_newest_dumps_globally_after_every_write(agent, hermes_home):
    keep = 3
    _write_config(hermes_home, keep)
    written = []

    for index in range(9):
        session_id = "zzzz-session" if index < 5 else "aaaa-session"
        written.append(_dump(agent, f"turn {index}", session_id))
        assert len(_real_dumps(agent.logs_dir)) == min(index + 1, keep)

    assert {path.name for path in _real_dumps(agent.logs_dir)} == {
        path.name for path in written[-keep:]
    }


def test_equal_mtime_and_inverse_session_names_never_prune_returned_dump(
    agent, hermes_home, monkeypatch
):
    keep = 2
    _write_config(hermes_home, keep)
    tied_mtime = 1_700_000_000.0
    for index in range(4):
        stale = agent.logs_dir / f"request_dump_zzzz_{index}.json"
        stale.write_text("{}", encoding="utf-8")
        os.utime(stale, (tied_mtime, tied_mtime))

    real_atomic_write = agent_runtime_helpers.atomic_json_write

    def atomic_write_with_coarse_mtime(path, *args, **kwargs):
        real_atomic_write(path, *args, **kwargs)
        os.utime(path, (tied_mtime, tied_mtime))

    monkeypatch.setattr(
        agent_runtime_helpers, "atomic_json_write", atomic_write_with_coarse_mtime
    )

    current = _dump(agent, "fresh", "aaaa-session")

    assert current.exists()
    assert len(_real_dumps(agent.logs_dir)) == keep


def test_pruning_leaves_non_dump_entries_and_symlink_untouched(
    agent, hermes_home, tmp_path
):
    _write_config(hermes_home, 1)
    state_db = agent.logs_dir / "state.db"
    transcript = agent.logs_dir / "session.jsonl"
    archive = agent.logs_dir / "archive"
    archived_dump = archive / "request_dump_archived.json"
    outside = tmp_path / "outside.json"
    symlink = agent.logs_dir / "request_dump_link.json"
    state_db.write_text("db", encoding="utf-8")
    transcript.write_text("{}\n", encoding="utf-8")
    archive.mkdir()
    archived_dump.write_text("{}", encoding="utf-8")
    outside.write_text("outside", encoding="utf-8")
    try:
        symlink.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    for index in range(4):
        _dump(agent, f"turn {index}", f"session-{index % 2}")

    assert len(_real_dumps(agent.logs_dir)) == 1
    assert state_db.exists()
    assert transcript.exists()
    assert archived_dump.exists()
    assert symlink.is_symlink()
    assert outside.read_text(encoding="utf-8") == "outside"


@pytest.mark.parametrize("keep", [0, -1])
def test_non_positive_retention_opts_out(agent, hermes_home, keep):
    _write_config(hermes_home, keep)

    for index in range(5):
        _dump(agent, f"turn {index}", f"session-{index % 2}")

    assert len(_real_dumps(agent.logs_dir)) == 5


def test_malformed_retention_falls_back_to_bounded_default(
    agent, hermes_home
):
    _write_config(hermes_home, "not-a-number")
    keep = agent_runtime_helpers._REQUEST_DUMP_DEFAULT_KEEP

    for index in range(keep + 3):
        _dump(agent, f"turn {index}", f"session-{index % 2}")

    assert len(_real_dumps(agent.logs_dir)) == keep


def test_request_dump_default_is_one_invariant_across_code_and_config():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    example = Path(__file__).resolve().parents[2] / "cli-config.yaml.example"
    example_config = yaml.safe_load(example.read_text(encoding="utf-8"))
    declared = DEFAULT_CONFIG["sessions"]["request_dump_retention"]

    assert agent_runtime_helpers._REQUEST_DUMP_DEFAULT_KEEP == declared
    assert example_config["sessions"]["request_dump_retention"] == declared


def test_unlink_failure_is_nonfatal_and_other_stale_dump_is_still_pruned(
    agent, hermes_home, monkeypatch
):
    _write_config(hermes_home, 1)
    blocked = agent.logs_dir / "request_dump_oldest.json"
    removable = agent.logs_dir / "request_dump_later.json"
    for index, path in enumerate((blocked, removable)):
        path.write_text("{}", encoding="utf-8")
        os.utime(path, (1_700_000_000.0 + index, 1_700_000_000.0 + index))
    real_unlink = Path.unlink

    def fail_one_unlink(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("in use")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_one_unlink)

    current = _dump(agent, "fresh", "current-session")

    assert current.exists()
    assert blocked.exists()
    assert not removable.exists()


def test_local_pruner_returns_successful_delete_count(tmp_path):
    directory = tmp_path / "sessions"
    directory.mkdir()
    paths = []
    for index in range(4):
        path = directory / f"request_dump_{index}.json"
        path.write_text("{}", encoding="utf-8")
        os.utime(path, (1_700_000_000.0 + index, 1_700_000_000.0 + index))
        paths.append(path)

    deleted = agent_runtime_helpers._prune_request_dumps(
        directory, 2, protect=paths[-1]
    )

    assert deleted == 2
    assert {path.name for path in _real_dumps(directory)} == {
        paths[-1].name,
        paths[-2].name,
    }
