"""The upgrade cells' log reader returns every line written after its snapshot, exactly once.

``tests/e2e/core/upgrade/_logs`` feeds the engine-error and manual-step checks of the e2e-upgrade
lane, which runs only on release and dispatch runs; its contract is pinned here so PR CI covers it.
A line that a rollover moved into ``<log>.1`` before the check reads must still be seen.
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e.core.upgrade import _logs as L


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)


def _rollover(log: Path) -> None:
    """What stdlib ``RotatingFileHandler.doRollover`` does with backupCount >= 2."""
    for n in (2, 1):
        src = log.with_name(f"{log.name}.{n - 1}") if n > 1 else log
        if src.exists():
            src.replace(log.with_name(f"{log.name}.{n}"))
    log.touch()


def _new_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln and not ln.startswith("--- ")]


def test_a_rollover_after_the_snapshot_loses_no_line(tmp_path: Path) -> None:
    home = tmp_path / "home"
    log = home / "logs" / "agent.log"
    _write(log, "old 1\n")
    _rollover(log)
    _write(log, "old 2\n")
    snap = L.log_offsets([home])

    _write(log, "engine error before the rollover\n")
    _rollover(log)
    _write(log, "after the first rollover\n")
    _rollover(log)  # deletes the snapshot's agent.log.1: the new agent.log may reuse its inode
    _write(log, "after the second rollover\n")

    got = _new_lines(L.log_text_since(snap, [home]))
    assert sorted(got) == sorted(
        ["engine error before the rollover", "after the first rollover", "after the second rollover"]
    )


def test_profile_logs_and_extra_process_logs_rotate_too(tmp_path: Path) -> None:
    home = tmp_path / "home"
    plog = home / "profiles" / "work" / "logs" / "gateway.log"
    extra = tmp_path / "proc" / "gateway.out.log"
    _write(plog, "old\n")
    _write(extra, "old\n")
    snap = L.log_offsets([home], [extra])

    _write(plog, "profile error\n")
    _write(extra, "process error\n")
    _rollover(plog)
    _rollover(extra)

    assert sorted(_new_lines(L.log_text_since(snap, [home], [extra]))) == ["process error", "profile error"]


def test_without_rotation_only_the_appended_text(tmp_path: Path) -> None:
    home = tmp_path / "home"
    log = home / "logs" / "agent.log"
    _write(log, "old\n")
    snap = L.log_offsets([home])
    _write(log, "new\n")
    _write(home / "logs" / "errors.log", "created since\n")

    assert sorted(_new_lines(L.log_text_since(snap, [home]))) == ["created since", "new"]


def test_a_truncated_log_is_read_whole(tmp_path: Path) -> None:
    home = tmp_path / "home"
    log = home / "logs" / "agent.log"
    _write(log, "a long line written before the snapshot\n")
    snap = L.log_offsets([home])
    log.write_text("short\n", encoding="utf-8")

    assert _new_lines(L.log_text_since(snap, [home])) == ["short"]
