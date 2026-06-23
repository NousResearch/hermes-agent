"""Tests for ``hermes kanban boards owner …`` — delivery-channel registration.

Covers the CLI surface added to let users register ``(platform, chat_id)``
owners for a board so the task-loop engine and convergence summaries have
a persistent delivery target (the gap that left alpha-chat/beta-chat
silent: no owner → no injection).

Verifies:
  * ``owner add`` registers a row visible to ``get_board_owners``.
  * ``owner add`` is idempotent (repeat only bumps updated_at).
  * ``owner list`` prints registered owners.
  * ``owner rm`` removes a specific owner.
  * ``owner show`` lists owners for the current board.
  * Multiple owners fan out (the multi-channel fix).

Run:
  cd /home/zml/workspace/hermes-agent
  venv/bin/python -m pytest tests/hermes_cli/test_kanban_boards_owner.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb


def _cli(args: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_WORKTREE)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban"] + args,
        env=env,
        capture_output=True,
        text=True,
        cwd=str(_WORKTREE),
        timeout=30,
    )


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    return home


class TestOwnerCLI:
    def test_add_registers_owner_visible_to_get_board_owners(self, fresh_home):
        env = {"HERMES_HOME": str(fresh_home)}
        assert _cli(["boards", "create", "alpha", "--switch"], env_extra=env).returncode == 0
        r = _cli(["boards", "owner", "add", "alpha", "feishu", "chat_f1"], env_extra=env)
        assert r.returncode == 0, r.stderr
        assert "Owner registered" in r.stdout
        with kb.connect_closing(board="alpha") as conn:
            owners = kb.get_board_owners(conn, "alpha")
        assert owners == [("feishu", "chat_f1")]

    def test_add_is_idempotent(self, fresh_home):
        env = {"HERMES_HOME": str(fresh_home)}
        _cli(["boards", "create", "beta", "--switch"], env_extra=env)
        _cli(["boards", "owner", "add", "beta", "feishu", "chat_b"], env_extra=env)
        _cli(["boards", "owner", "add", "beta", "feishu", "chat_b"], env_extra=env)
        with kb.connect_closing(board="beta") as conn:
            owners = kb.get_board_owners(conn, "beta")
        assert owners == [("feishu", "chat_b")], "duplicate add must not create a second row"

    def test_list_prints_owners(self, fresh_home):
        env = {"HERMES_HOME": str(fresh_home)}
        _cli(["boards", "create", "gamma", "--switch"], env_extra=env)
        _cli(["boards", "owner", "add", "gamma", "feishu", "chat_g1"], env_extra=env)
        _cli(["boards", "owner", "add", "gamma", "weixin", "chat_g2"], env_extra=env)
        r = _cli(["boards", "owner", "list", "gamma"], env_extra=env)
        assert r.returncode == 0, r.stderr
        assert "feishu" in r.stdout
        assert "weixin" in r.stdout
        assert "chat_g1" in r.stdout
        assert "chat_g2" in r.stdout

    def test_list_empty_board_prints_guidance(self, fresh_home):
        env = {"HERMES_HOME": str(fresh_home)}
        _cli(["boards", "create", "empty", "--switch"], env_extra=env)
        r = _cli(["boards", "owner", "list", "empty"], env_extra=env)
        assert r.returncode == 0, r.stderr
        assert "no owners" in r.stdout.lower()
        assert "hermes kanban boards owner add" in r.stdout

    def test_rm_removes_specific_owner(self, fresh_home):
        env = {"HERMES_HOME": str(fresh_home)}
        _cli(["boards", "create", "delta", "--switch"], env_extra=env)
        _cli(["boards", "owner", "add", "delta", "feishu", "chat_d1"], env_extra=env)
        _cli(["boards", "owner", "add", "delta", "weixin", "chat_d2"], env_extra=env)
        r = _cli(["boards", "owner", "rm", "delta", "feishu", "chat_d1"], env_extra=env)
        assert r.returncode == 0, r.stderr
        assert "Removed" in r.stdout
        with kb.connect_closing(board="delta") as conn:
            owners = kb.get_board_owners(conn, "delta")
        assert owners == [("weixin", "chat_d2")], "rm should leave the other owner intact"

    def test_rm_nonexistent_returns_nonzero(self, fresh_home):
        env = {"HERMES_HOME": str(fresh_home)}
        _cli(["boards", "create", "eps", "--switch"], env_extra=env)
        r = _cli(["boards", "owner", "rm", "eps", "feishu", "ghost"], env_extra=env)
        # main.py doesn't propagate return codes today (see test_kanban_boards.py);
        # assert the user-visible signal instead.
        assert "No matching owner" in r.stdout

    def test_show_lists_current_board_owners(self, fresh_home):
        env = {"HERMES_HOME": str(fresh_home)}
        _cli(["boards", "create", "zeta", "--switch"], env_extra=env)
        _cli(["boards", "owner", "add", "zeta", "telegram", "chat_tz"], env_extra=env)
        r = _cli(["boards", "owner", "show"], env_extra=env)
        assert r.returncode == 0, r.stderr
        assert "telegram" in r.stdout
        assert "chat_tz" in r.stdout

    def test_add_rejects_unknown_board(self, fresh_home):
        env = {"HERMES_HOME": str(fresh_home)}
        r = _cli(["boards", "owner", "add", "ghost-board", "feishu", "chat_x"], env_extra=env)
        assert "does not exist" in r.stderr

    def test_multiple_owners_fan_out_for_delivery_targets(self, fresh_home):
        from gateway.kanban_watchers import GatewayKanbanWatchersMixin
        env = {"HERMES_HOME": str(fresh_home)}
        _cli(["boards", "create", "multi", "--switch"], env_extra=env)
        _cli(["boards", "owner", "add", "multi", "feishu", "chat_m1"], env_extra=env)
        _cli(["boards", "owner", "add", "multi", "weixin", "chat_m2"], env_extra=env)

        class FakeRunner(GatewayKanbanWatchersMixin):
            def __init__(self):
                self.adapters = {}

        runner = FakeRunner()
        targets = runner._kanban_delivery_targets("multi")
        plats = {p for p, _ in targets}
        assert plats == {"feishu", "weixin"}, (
            f"delivery targets must fan out to all registered owners, got {targets}"
        )

    def test_auto_detect_single_most_recent_dm(self, fresh_home):
        env = {"HERMES_HOME": str(fresh_home)}
        (fresh_home / "channel_directory.json").write_text(json.dumps({
            "updated_at": "2026-06-21T00:00:00",
            "platforms": {
                "feishu": [{"id": "oc_feishu_dm", "type": "dm"}],
                "weixin": [{"id": "wx_dm", "type": "dm"},
                           {"id": "epoch:kanban", "type": "system"}],
            },
        }))
        _cli(["boards", "create", "auto1", "--switch"], env_extra=env)
        r = _cli(["boards", "owner", "add", "auto1", "--auto"], env_extra=env)
        assert r.returncode == 0, r.stderr
        assert "registered" in r.stdout.lower()
        with kb.connect_closing(board="auto1") as conn:
            owners = kb.get_board_owners(conn, "auto1")
        assert len(owners) == 1, f"--auto without --all should register exactly 1, got {owners}"

    def test_auto_all_registers_every_dm_channel(self, fresh_home):
        env = {"HERMES_HOME": str(fresh_home)}
        (fresh_home / "channel_directory.json").write_text(json.dumps({
            "updated_at": "2026-06-21T00:00:00",
            "platforms": {
                "feishu": [{"id": "oc_f1", "type": "dm"}],
                "weixin": [{"id": "wx1", "type": "dm"},
                           {"id": "HomeChannel(platform=x)", "type": "private"}],
            },
        }))
        _cli(["boards", "create", "auto2", "--switch"], env_extra=env)
        r = _cli(["boards", "owner", "add", "auto2", "--auto", "--all"], env_extra=env)
        assert r.returncode == 0, r.stderr
        with kb.connect_closing(board="auto2") as conn:
            owners = kb.get_board_owners(conn, "auto2")
        plats = {p for p, _ in owners}
        assert plats == {"feishu", "weixin"}, (
            f"--auto --all should register both DM platforms, got {owners}"
        )
        chats = {c for _, c in owners}
        assert "HomeChannel" not in str(chats), "repr garbage must be filtered out"

    def test_auto_no_channels_reports_error(self, fresh_home):
        env = {"HERMES_HOME": str(fresh_home)}
        _cli(["boards", "create", "auto3", "--switch"], env_extra=env)
        r = _cli(["boards", "owner", "add", "auto3", "--auto"], env_extra=env)
        assert "no dm channels" in r.stderr.lower(), (
            f"should report no channels detected, got stderr={r.stderr!r}"
        )

    def test_auto_falls_back_to_default_board_owners(self, fresh_home):
        env = {"HERMES_HOME": str(fresh_home)}
        _cli(["boards", "create", "srcboard", "--switch"], env_extra=env)
        with kb.connect_closing(board=kb.DEFAULT_BOARD) as conn:
            kb.set_board_owner(conn, kb.DEFAULT_BOARD, "telegram", "tg_chat")
        (fresh_home / "channel_directory.json").write_text(json.dumps({"platforms": {}}))
        _cli(["boards", "create", "auto4", "--switch"], env_extra=env)
        r = _cli(["boards", "owner", "add", "auto4", "--auto"], env_extra=env)
        assert r.returncode == 0, r.stderr
        with kb.connect_closing(board="auto4") as conn:
            owners = kb.get_board_owners(conn, "auto4")
        assert ("telegram", "tg_chat") in owners, (
            f"fallback to default board owners failed, got {owners}"
        )
