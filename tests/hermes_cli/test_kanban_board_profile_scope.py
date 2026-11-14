"""In-process kanban tool calls must resolve the board from the ACTIVE PROFILE's
pin (profile .env HERMES_KANBAN_BOARD), not only the gateway process env / global
``current`` symlink. A multiplex gateway serves every profile in one process; the
profile pin exists precisely for per-team boards and never reaches os.environ."""
import os
from pathlib import Path

import pytest


def test_profile_env_pin_wins_for_in_process_calls(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kbd
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    home = tmp_path / "home"
    (home / "kanban" / "boards" / "board-a").mkdir(parents=True)
    (home / "kanban" / "boards" / "board-a" / "kanban.db").write_bytes(b"")
    (home / "kanban" / "boards" / "board-b").mkdir(parents=True)
    (home / "kanban" / "boards" / "board-b" / "kanban.db").write_bytes(b"")
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(home)
    try:
        # Global default points at board-b (e.g. a `boards switch` months ago)
        (home / "kanban" / "current").write_text("board-b\n")
        monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
        assert kbd.get_current_board() == "board-b"

        # The profile's pin lives in profiles/<name>/.env — the same place the CLI
        # child loads it from. An in-process call made while THAT profile is active
        # (multiplex gateway) must resolve to the profile's board, not the global.
        prof = home / "profiles" / "worker-a"
        prof.mkdir(parents=True)
        (prof / ".env").write_text("HERMES_KANBAN_BOARD=board-a\n")

        # Simulate the multiplex gateway serving this profile: the session ContextVar
        # names the active profile; the pin must win over the global `current`.
        from gateway.session_context import set_session_vars, clear_session_vars
        tokens = set_session_vars(platform="api_server", chat_id="20260104_x", profile="worker-a")
        try:
            assert kbd.get_current_board() == "board-a", (
                "profile .env pin must win for in-process calls under multiplex")
        finally:
            clear_session_vars(tokens)
    finally:
        reset_hermes_home_override(token)


def test_pin_survives_kanban_home_split_and_dotenv_edge_cases(tmp_path, monkeypatch):
    """Review follow-ups (PR #132625): the pin must resolve identically to a CLI child
    on (a) a split layout with HERMES_KANBAN_HOME pointing elsewhere, (b) an
    ``export``-prefixed line, (c) an inline comment, (d) a UTF-8 BOM."""
    from hermes_cli import kanban_db as kbd
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from gateway.session_context import set_session_vars, clear_session_vars

    home = tmp_path / "home"
    for slug in ("board-a", "board-b"):
        (home / "kanban" / "boards" / slug).mkdir(parents=True)
        (home / "kanban" / "boards" / slug / "kanban.db").write_bytes(b"")
    (home / "kanban" / "current").write_text("board-b\n")
    # Split layout: boards live under a SEPARATE kanban home (Docker/mounted boards).
    # HERMES_KANBAN_HOME is the root containing kanban/boards (see kanban_home()).
    separate = tmp_path / "kanban-root"
    (separate / "kanban" / "boards" / "board-a").mkdir(parents=True)
    (separate / "kanban" / "boards" / "board-a" / "kanban.db").write_bytes(b"")
    (separate / "kanban" / "boards" / "board-b").mkdir(parents=True)
    (separate / "kanban" / "boards" / "board-b" / "kanban.db").write_bytes(b"")
    (separate / "kanban" / "current").write_text("board-b\n")
    prof = home / "profiles" / "worker-a"
    prof.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(separate))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    # The profiles root anchors to get_default_hermes_root(); a scratch-TMPDIR test env
    # can make that collapse to the developer's real ~/.hermes (env under native). Pin it
    # so the fixture's profiles/ dir is authoritative, as in a real deployment.
    from hermes_cli import profiles as profiles_mod
    monkeypatch.setattr(profiles_mod, "_get_default_hermes_home", lambda: home)
    token = set_hermes_home_override(home)
    try:
        tokens = set_session_vars(platform="api_server", chat_id="20260104_y", profile="worker-a")
        try:
            for content in (
                "HERMES_KANBAN_BOARD=board-a\n",                    # plain
                "export HERMES_KANBAN_BOARD=board-a\n",             # export prefix
                "HERMES_KANBAN_BOARD=board-a  # ops board\n",       # inline comment
                "\ufeffHERMES_KANBAN_BOARD=board-a\n",              # UTF-8 BOM
            ):
                (prof / ".env").write_text(content, encoding="utf-8")
                kbd._PROFILE_BOARD_PIN_CACHE["key"] = None  # bypass the mtime cache per case
                assert kbd.get_current_board() == "board-a", content
        finally:
            clear_session_vars(tokens)
    finally:
        reset_hermes_home_override(token)
