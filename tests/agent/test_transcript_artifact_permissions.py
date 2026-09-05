"""Permission-policy tests for real trajectory and MoA append writers.

These exercise the real \`save_trajectory\` / \`save_moa_turn\` file-write
paths (no mocking of \`open\`/\`mkdir\`), asserting the artifacts they create
pick up \`artifact_file_mode()\` / \`secure_artifact_dir\` from #77472 instead of
the previous plain \`open(..., "a")\` / bare \`mkdir\` (world- and group-readable
under a permissive umask).
"""

import json
import os
import stat
from pathlib import Path

import pytest

from agent.trajectory import save_trajectory
from agent.moa_trace import save_moa_turn


posix_only = pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX permission bits are advisory on Windows",
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@posix_only
def test_legacy_internal_transcripts_are_private_on_append(tmp_path, monkeypatch):
    from agent.trajectory import default_trajectory_path

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text("moa:\n  save_traces: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    monkeypatch.chdir(tmp_path)
    paths = [default_trajectory_path(True), home / "moa-traces" / "legacy.jsonl"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o755)
        path.write_text("{}\n", encoding="utf-8")
        path.chmod(0o644)
    save_trajectory([{"from": "human", "value": "secret remains replayable"}], "m", True)
    save_moa_turn(session_id="legacy", preset_name="test", reference_outputs=[],
                  aggregator_label="agg", aggregator_model="m", aggregator_provider="p",
                  aggregator_temperature=0.4, aggregator_input_messages=[],
                  aggregator_output="output", aggregator_streamed=False)
    for path in paths:
        assert not _mode(path) & 0o077
        assert not _mode(path.parent) & 0o077
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2


@posix_only
@pytest.mark.parametrize("completed", [True, False])
def test_save_trajectory_creates_fresh_file_owner_only(
    tmp_path, monkeypatch, completed
):
    """A freshly created trajectory file must have no group/other bits."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    monkeypatch.chdir(tmp_path)

    trajectory = [
        {
            "from": "human",
            "value": "hello <REASONING_SCRATCHPAD>secret</REASONING_SCRATCHPAD>",
        },
        {
            "from": "gpt",
            "value": "world",
            "tool_calls": [{"id": "call_1", "name": "terminal"}],
        },
    ]

    old_umask = os.umask(0o022)
    try:
        save_trajectory(trajectory, "test-model", completed, filename="traj.jsonl")
    finally:
        os.umask(old_umask)

    path = tmp_path / "traj.jsonl"
    assert path.exists()
    assert not _mode(path) & 0o077, (
        f"fresh trajectory file mode {oct(_mode(path))} leaks to group/other"
    )

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["conversations"] == trajectory
    assert entry["model"] == "test-model"
    assert entry["completed"] is completed


@posix_only
def test_save_trajectory_preserves_pre_existing_file_mode(tmp_path, monkeypatch):
    """An existing trajectory file at a broader mode keeps that mode on append."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    monkeypatch.chdir(tmp_path)

    path = tmp_path / "traj.jsonl"
    path.write_text("", encoding="utf-8")
    os.chmod(path, 0o640)

    save_trajectory(
        [{"from": "human", "value": "hi"}], "m", True, filename="traj.jsonl"
    )

    assert _mode(path) == 0o640


@posix_only
def test_save_moa_turn_unmanaged_leaf_and_file_owner_only(tmp_path, monkeypatch):
    """Real config.yaml-gated MoA trace: unmanaged leaf and file are owner-only."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "moa:\n  save_traces: true\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)

    old_umask = os.umask(0o022)
    try:
        save_moa_turn(
            session_id="sess-1",
            preset_name="closed",
            reference_outputs=[],
            aggregator_label="agg",
            aggregator_model="m",
            aggregator_provider="p",
            aggregator_temperature=0.4,
            aggregator_input_messages=[{"role": "user", "content": "hi"}],
            aggregator_output="out",
            aggregator_streamed=False,
        )
    finally:
        os.umask(old_umask)

    trace_dir = hermes_home / "moa-traces"
    trace_path = trace_dir / "sess-1.jsonl"
    assert trace_path.exists()
    assert not _mode(trace_dir) & 0o077, (
        f"unmanaged trace dir mode {oct(_mode(trace_dir))} leaks to group/other"
    )
    assert not _mode(trace_path) & 0o077, (
        f"unmanaged trace file mode {oct(_mode(trace_path))} leaks to group/other"
    )

    record = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert record["session_id"] == "sess-1"
    assert record["preset"] == "closed"
    assert record["aggregator"]["output"] == "out"


@posix_only
def test_save_moa_turn_managed_setgid_parent_group_mode(tmp_path, monkeypatch):
    """Managed + setgid parent: trace dir stays group-writable, file is 0o660."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    os.chmod(hermes_home, 0o2770)
    # Managed mode requires these to already exist (normally created by
    # the NixOS activation script); without them ensure_hermes_home()
    # raises RuntimeError, which _traces_enabled_and_dir() swallows, and
    # save_moa_turn silently no-ops before ever writing the trace.
    for subdir in ("cron", "sessions", "logs", "memories"):
        (hermes_home / subdir).mkdir()
    (hermes_home / "config.yaml").write_text(
        "moa:\n  save_traces: true\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_MANAGED", "nixos")

    probe = hermes_home / "kernel-probe"
    probe.mkdir()
    if not _mode(probe) & stat.S_ISGID:
        pytest.skip("kernel does not inherit setgid on new directories")

    old_umask = os.umask(0o007)
    try:
        save_moa_turn(
            session_id="sess-2",
            preset_name="closed",
            reference_outputs=[],
            aggregator_label="agg",
            aggregator_model="m",
            aggregator_provider="p",
            aggregator_temperature=0.4,
            aggregator_input_messages=[{"role": "user", "content": "hi"}],
            aggregator_output="out",
            aggregator_streamed=False,
        )
    finally:
        os.umask(old_umask)

    trace_dir = hermes_home / "moa-traces"
    trace_path = trace_dir / "sess-2.jsonl"
    assert trace_path.exists()
    assert _mode(trace_dir) & 0o777 == 0o770
    assert _mode(trace_path) == 0o660
