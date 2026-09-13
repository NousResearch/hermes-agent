"""Project-to-gateway-turn cwd and Kanban context isolation tests."""

from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
import gateway.run_turn as gateway_turn
import gateway.session_context as session_context
from agent.prompt_builder import build_context_files_prompt
from agent.runtime_cwd import resolve_agent_cwd, resolve_context_cwd
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource, build_session_key
from hermes_cli import kanban_db, projects_db
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@contextmanager
def _profile_home(home: Path):
    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _create_profile_project(
    home: Path,
    *,
    primary_path: str | None = None,
    board_slug: str | None = None,
) -> projects_db.Project:
    home.mkdir(parents=True, exist_ok=True)
    with _profile_home(home):
        with projects_db.connect_closing() as conn:
            project_id = projects_db.create_project(
                conn,
                name="Gateway project",
                slug="gateway-project",
                primary_path=primary_path,
                board_slug=board_slug,
            )
            projects_db.set_active(conn, project_id)
            project = projects_db.get_project(conn, project_id)
            assert project is not None
            return project


def _session_context(profile: str) -> SessionContext:
    source = SessionSource(
        platform=Platform.MATRIX,
        chat_id=f"room-{profile}",
        chat_type="group",
        user_id="user",
        user_name="User",
        profile=profile,
    )
    return SessionContext(
        source=source,
        connected_platforms=[Platform.MATRIX],
        home_channels={},
        session_key=build_session_key(source),
        session_id=f"session-{profile}",
    )


def _runner() -> GatewayRunner:
    return object.__new__(GatewayRunner)


def _bind_session_env(runner: GatewayRunner, context: SessionContext):
    tokens = runner._set_session_env(context)
    return tokens


def test_legacy_profile_project_lookup_does_not_create_projects_db(tmp_path):
    home = tmp_path / "profiles" / "legacy"
    home.mkdir(parents=True)
    db_path = home / "projects.db"

    with _profile_home(home):
        assert not db_path.exists()
        assert gateway_run._resolve_active_project_for_gateway() is None

    assert not db_path.exists()


def test_missing_projects_db_is_fail_open(tmp_path):
    home = tmp_path / "profiles" / "alpha"
    home.mkdir(parents=True)

    with _profile_home(home):
        assert gateway_run._resolve_active_project_for_gateway() is None


def test_malformed_projects_db_is_fail_open(tmp_path):
    home = tmp_path / "profiles" / "alpha"
    home.mkdir(parents=True)
    (home / "projects.db").write_text("not a sqlite database", encoding="utf-8")

    with _profile_home(home):
        assert gateway_run._resolve_active_project_for_gateway() is None

        class _MalformedProject:
            @property
            def primary_path(self):
                raise RuntimeError("malformed primary path")

            @property
            def board_slug(self):
                raise RuntimeError("malformed board slug")

        malformed = _MalformedProject()
        with gateway_run._scoped_gateway_project_cwd(malformed):
            with gateway_run._scoped_gateway_project_board(malformed):
                assert resolve_agent_cwd().is_dir()


def test_archived_active_project_is_ignored(tmp_path):
    home = tmp_path / "profiles" / "alpha"
    project = _create_profile_project(
        home, primary_path=str(tmp_path), board_slug="alpha-board"
    )

    with _profile_home(home):
        with projects_db.connect_closing() as conn:
            assert projects_db.archive_project(conn, project.id)
        assert gateway_run._resolve_active_project_for_gateway() is None


def test_stale_active_project_id_is_ignored(tmp_path):
    home = tmp_path / "profiles" / "alpha"
    home.mkdir(parents=True)

    with _profile_home(home):
        with projects_db.connect_closing() as conn:
            projects_db.set_active(conn, "missing-project-id")
        assert gateway_run._resolve_active_project_for_gateway() is None


def test_active_project_primary_path_reaches_session_env_and_prompt_context(tmp_path, monkeypatch):
    project_path = tmp_path / "project"
    project_path.mkdir()
    (project_path / "AGENTS.md").write_text(
        "# Project context\n\nPROJECT_CONTEXT_MARKER\n",
        encoding="utf-8",
    )
    home = tmp_path / "profiles" / "alpha"
    project = _create_profile_project(
        home, primary_path=str(project_path), board_slug=None
    )
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    runner = _runner()
    context = _session_context("alpha")
    captured = {}
    original_set_session_vars = session_context.set_session_vars

    def capture_set_session_vars(**kwargs):
        captured["cwd"] = kwargs.get("cwd")
        return original_set_session_vars(**kwargs)

    monkeypatch.setattr(session_context, "set_session_vars", capture_set_session_vars)
    with _profile_home(home), gateway_run._scoped_gateway_project_cwd(project):
        tokens = _bind_session_env(runner, context)
        try:
            assert captured["cwd"] == str(project_path)
            assert resolve_agent_cwd() == project_path
            assert resolve_context_cwd() == project_path
            prompt = build_context_files_prompt(
                cwd=str(resolve_context_cwd()), skip_soul=True
            )
        finally:
            runner._clear_session_env(tokens)

    assert "PROJECT_CONTEXT_MARKER" in prompt


@pytest.mark.asyncio
async def test_hmwa_prepare_turn_binds_project_cwd_before_prompt_rendering(tmp_path, monkeypatch):
    project_path = tmp_path / "project"
    project_path.mkdir()
    home = tmp_path / "profiles" / "alpha"
    project = _create_profile_project(home, primary_path=str(project_path))
    context = _session_context("alpha")
    runner = _runner()
    runner.config = SimpleNamespace()
    runner._hmwa_open_session = _async_open_session
    runner._hmwa_acquire_turn_lease = _async_noop
    runner._mark_durable_active_turn = _async_noop
    runner._hmwa_run_session_hygiene = _async_hygiene
    runner._hmwa_first_contact_notes = _async_noop
    runner._prepare_profile_scoped_inbound_message_text = _async_message_text
    runner._hmwa_apply_message_timestamp = lambda _event, text: (text, None, None)
    runner._voice_channel_sidecar_note = lambda *_args: None
    runner._bind_adapter_run_generation = lambda *_args: None
    runner._hmwa_deliver_auto_reset_notice = _async_noop
    runner._hmwa_auto_load_skills = lambda *_args: None

    class _Store:
        async def load_transcript(self, _session_id):
            return []

    store = _Store()
    monkeypatch.setattr(
        GatewayRunner,
        "async_session_store",
        property(lambda _self: store),
    )
    monkeypatch.setattr(
        gateway_turn,
        "build_session_context",
        lambda *_args: context,
    )

    captured = {"prompt_cwd": None, "session_env_cwd": None}
    original_set_session_vars = session_context.set_session_vars

    def capture_set_session_vars(**kwargs):
        captured["session_env_cwd"] = kwargs.get("cwd")
        return original_set_session_vars(**kwargs)

    def _capture_prompt_cwd(_context, _redact_pii, _session_key):
        captured["prompt_cwd"] = resolve_context_cwd()
        return "prompt"

    monkeypatch.setattr(session_context, "set_session_vars", capture_set_session_vars)
    runner._pinned_session_context_prompt = _capture_prompt_cwd
    event = SimpleNamespace(text="hello", internal=False, auto_skill=None)
    session_entry = SimpleNamespace(
        was_auto_reset=False,
        session_key=context.session_key,
        session_id=context.session_id,
    )

    with _profile_home(home), gateway_run._scoped_gateway_project_cwd(project):
        prepared, tokens = await runner._hmwa_prepare_turn(
            event,
            context.source,
            session_entry,
            context.session_key,
            "quick-key",
            1,
        )
        try:
            assert isinstance(prepared, runner._PreparedTurn)
            assert captured["session_env_cwd"] == str(project_path)
            assert captured["prompt_cwd"] == project_path
        finally:
            runner._clear_session_env(tokens)


async def _async_open_session(*_args, **_kwargs):
    return False, False


async def _async_noop(*_args, **_kwargs):
    return None


async def _async_hygiene(*args, **_kwargs):
    return args[4] if len(args) > 4 else None


async def _async_message_text(*, event, **_kwargs):
    return event.text


def test_invalid_primary_path_uses_profile_terminal_cwd_without_env_mutation(
    tmp_path, monkeypatch
):
    configured_path = tmp_path / "configured"
    configured_path.mkdir()
    home = tmp_path / "profiles" / "alpha"
    project = _create_profile_project(
        home,
        primary_path=str(tmp_path / "removed-project"),
        board_slug=None,
    )
    (home / "config.yaml").write_text(
        f"terminal:\n  cwd: {configured_path}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "launch-cwd"))
    env_before = {
        key: os.environ.get(key)
        for key in ("TERMINAL_CWD", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB")
    }

    runner = _runner()
    context = _session_context("alpha")
    with (
        _profile_runtime_scope_without_secret_hydration(home),
        gateway_run._scoped_gateway_project_cwd(project),
    ):
        tokens = _bind_session_env(runner, context)
        try:
            assert resolve_agent_cwd() == configured_path
        finally:
            runner._clear_session_env(tokens)

    assert {
        key: os.environ.get(key)
        for key in ("TERMINAL_CWD", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB")
    } == env_before


def _profile_runtime_scope_without_secret_hydration(home: Path):
    return gateway_run._profile_runtime_scope(home, hydrate_secrets=False)


def test_invalid_unset_profile_cwd_falls_back_to_native_cwd(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "alpha"
    project = _create_profile_project(
        home,
        primary_path=str(tmp_path / "missing-project"),
        board_slug=None,
    )
    (home / "config.yaml").write_text(
        "terminal:\n  cwd: /path/that/does/not/exist\n",
        encoding="utf-8",
    )
    native_path = tmp_path / "native"
    native_path.mkdir()
    monkeypatch.chdir(native_path)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "ambient-terminal-cwd"))

    runner = _runner()
    context = _session_context("alpha")
    with (
        _profile_runtime_scope_without_secret_hydration(home),
        gateway_run._scoped_gateway_project_cwd(project),
    ):
        tokens = _bind_session_env(runner, context)
        try:
            assert resolve_agent_cwd() == native_path
        finally:
            runner._clear_session_env(tokens)


def test_project_board_is_scoped_to_agent_turn_and_restored(tmp_path, monkeypatch):
    board_root = tmp_path / "kanban"
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(board_root))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    kanban_db.init_db(board="alpha-board")
    home = tmp_path / "profiles" / "alpha"
    project = _create_profile_project(
        home, primary_path=str(tmp_path), board_slug="alpha-board"
    )

    runner = _runner()
    observed = {}
    lookup_calls = []
    original_lookup = gateway_run._resolve_active_project_for_gateway

    def count_lookup():
        lookup_calls.append(True)
        return original_lookup()

    monkeypatch.setattr(gateway_run, "_resolve_active_project_for_gateway", count_lookup)

    async def inner(event, source, quick_key, generation):
        context = _session_context(source.profile or "alpha")
        tokens = runner._set_session_env(context)
        try:
            observed["board"] = kanban_db.get_current_board()
            observed["cwd"] = str(resolve_agent_cwd())
            await asyncio.sleep(0)
            return "ok"
        finally:
            runner._clear_session_env(tokens)

    runner._handle_message_with_agent_inner = inner
    event = SimpleNamespace(text="hello", source=_session_context("alpha").source)
    with _profile_home(home):
        result = asyncio.run(
            runner._handle_message_with_agent(event, event.source, "key", 1)
        )
        after = kanban_db.get_current_board()

    assert result == "ok"
    assert observed == {"board": "alpha-board", "cwd": str(tmp_path)}
    assert len(lookup_calls) == 1
    assert after == "default"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_project_context_scopes_restore_after_unwind(
    tmp_path, monkeypatch, failure
):
    board_root = tmp_path / "kanban"
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(board_root))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    kanban_db.init_db(board="alpha-board")
    native_path = tmp_path / "native"
    native_path.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(native_path))
    env_before = {
        key: os.environ.get(key)
        for key in ("TERMINAL_CWD", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB")
    }
    home = tmp_path / "profiles" / "alpha"
    project = _create_profile_project(
        home, primary_path=str(tmp_path), board_slug="alpha-board"
    )
    runner = _runner()

    async def inner(event, source, quick_key, generation):
        context = _session_context(source.profile or "alpha")
        tokens = runner._set_session_env(context)
        try:
            assert resolve_agent_cwd() == tmp_path
            assert kanban_db.get_current_board() == "alpha-board"
            await asyncio.sleep(0)
            raise failure("abort project turn")
        finally:
            runner._clear_session_env(tokens)

    runner._handle_message_with_agent_inner = inner
    event = SimpleNamespace(text="hello", source=_session_context("alpha").source)
    with _profile_home(home):
        with pytest.raises(failure):
            await runner._handle_message_with_agent(event, event.source, "key", 1)
        assert resolve_agent_cwd() == native_path
        assert kanban_db.get_current_board() == "default"

    assert {
        key: os.environ.get(key)
        for key in ("TERMINAL_CWD", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB")
    } == env_before


@pytest.mark.asyncio
async def test_overlapping_profile_turns_keep_project_cwd_and_board_isolated(
    tmp_path, monkeypatch
):
    board_root = tmp_path / "kanban"
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(board_root))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    kanban_db.init_db(board="alpha-board")
    kanban_db.init_db(board="beta-board")

    alpha_path = tmp_path / "alpha-project"
    beta_path = tmp_path / "beta-project"
    alpha_path.mkdir()
    beta_path.mkdir()
    alpha_home = tmp_path / "profiles" / "alpha"
    beta_home = tmp_path / "profiles" / "beta"
    alpha = _create_profile_project(
        alpha_home, primary_path=str(alpha_path), board_slug="alpha-board"
    )
    beta = _create_profile_project(
        beta_home, primary_path=str(beta_path), board_slug="beta-board"
    )
    runner = _runner()

    async def inner(event, source, quick_key, generation):
        context = _session_context(source.profile or "default")
        tokens = runner._set_session_env(context)
        try:
            await asyncio.sleep(0)
            return str(resolve_agent_cwd()), kanban_db.get_current_board()
        finally:
            runner._clear_session_env(tokens)

    runner._handle_message_with_agent_inner = inner

    async def observe(home, profile):
        source = _session_context(profile).source
        event = SimpleNamespace(text="hello", source=source)
        with _profile_home(home):
            return await runner._handle_message_with_agent(event, source, "key", 1)

    observed_alpha, observed_beta = await asyncio.gather(
        observe(alpha_home, "alpha"),
        observe(beta_home, "beta"),
    )

    assert observed_alpha == (str(alpha_path), "alpha-board")
    assert observed_beta == (str(beta_path), "beta-board")
    assert kanban_db.get_current_board() == "default"


def test_explicit_board_pin_wins_over_project_association(tmp_path, monkeypatch):
    board_root = tmp_path / "kanban"
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(board_root))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "explicit-board")
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    kanban_db.init_db(board="explicit-board")
    kanban_db.init_db(board="project-board")
    home = tmp_path / "profiles" / "alpha"
    project = _create_profile_project(
        home, primary_path=str(tmp_path), board_slug="project-board"
    )

    with _profile_home(home):
        with gateway_run._scoped_gateway_project_board(project):
            assert kanban_db.get_current_board() == "explicit-board"

    assert os.environ["HERMES_KANBAN_BOARD"] == "explicit-board"


def test_explicit_db_pin_wins_over_project_association(tmp_path, monkeypatch):
    board_root = tmp_path / "kanban"
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(board_root))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    kanban_db.init_db(board="project-board")

    home = tmp_path / "profiles" / "alpha"
    project = _create_profile_project(
        home, primary_path=str(tmp_path), board_slug="project-board"
    )
    pinned_db = tmp_path / "pinned-kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned_db))

    with _profile_home(home):
        with gateway_run._scoped_gateway_project_board(project):
            assert kanban_db.kanban_db_path() == pinned_db
            assert kanban_db.get_current_board() == "default"


def test_direct_board_argument_remains_authoritative(tmp_path, monkeypatch):
    board_root = tmp_path / "kanban"
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(board_root))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    kanban_db.init_db(board="project-board")
    kanban_db.init_db(board="direct-board")
    home = tmp_path / "profiles" / "alpha"
    project = _create_profile_project(
        home, primary_path=str(tmp_path), board_slug="project-board"
    )

    direct_db = kanban_db.kanban_db_path(board="direct-board")
    with _profile_home(home), kanban_db.scoped_current_board("direct-board"):
        with gateway_run._scoped_gateway_project_board(project):
            assert kanban_db.get_current_board() == "direct-board"
            assert kanban_db.kanban_db_path(board="direct-board") == direct_db


@pytest.mark.parametrize("placeholder", [".", "auto", "cwd"])
def test_placeholder_profile_cwd_preserves_native_fallback(
    tmp_path, monkeypatch, placeholder
):
    native_path = tmp_path / "native"
    native_path.mkdir()
    home = tmp_path / "profiles" / "alpha"
    project = _create_profile_project(
        home,
        primary_path=str(tmp_path / "removed-project"),
        board_slug=None,
    )
    (home / "config.yaml").write_text(
        f"terminal:\n  cwd: {placeholder}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(native_path)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "ambient-terminal-cwd"))

    runner = _runner()
    context = _session_context("alpha")
    with (
        _profile_runtime_scope_without_secret_hydration(home),
        gateway_run._scoped_gateway_project_cwd(project),
    ):
        tokens = _bind_session_env(runner, context)
        try:
            assert resolve_agent_cwd() == native_path
        finally:
            runner._clear_session_env(tokens)
