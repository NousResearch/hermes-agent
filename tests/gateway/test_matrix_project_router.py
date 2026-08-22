"""Focused Matrix project-router vertical-slice coverage."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from agent.runtime_cwd import resolve_agent_cwd, resolve_context_cwd
from agent.prompt_builder import build_context_files_prompt
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.project_router import (
    ProjectSetupPlan,
    SetupRecommendation,
    add_project_alias,
    active_project_path,
    analyze_project_setup,
    apply_project_setup,
    bootstrap_registry,
    project_details,
    project_keys,
    project_path,
    register_project,
    resolve_project_reference,
)
from gateway.session import SessionContext, SessionEntry, SessionSource, build_session_key
from hermes_state import SessionDB


NEWMOON_PATH = Path("/home/rle/projects/NewMoonNailsAndSpa")
FIVEHOURS_PATH = Path("/home/rle/projects/savefivehours")


def _make_project(tmp_path: Path, name: str, *, agents: bool = True) -> Path:
    project = tmp_path / name
    project.mkdir()
    (project / ".git").mkdir()
    (project / "README.md").write_text("# Test project\n")
    if agents:
        (project / "AGENTS.md").write_text("# Agent context\n")
    return project


def _repository_snapshot(path: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(candidate.relative_to(path)): (candidate.read_bytes(), candidate.stat().st_mtime_ns)
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file()
    }


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.MATRIX,
        user_id="matrix-user",
        chat_id="matrix-room",
        user_name="tester",
        chat_type="room",
    )


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="message-1",
        internal=True,
    )


def _session_entry() -> SessionEntry:
    source = _source()
    return SessionEntry(
        session_key=build_session_key(source),
        session_id="matrix-session",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.MATRIX,
        chat_type="room",
        total_tokens=0,
    )


def _runner(tmp_path):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.MATRIX: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._pending_messages = {}
    runner.adapters = {Platform.MATRIX: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]), loaded_hooks=False
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = _session_entry()
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._queued_events = {}
    runner._session_db = SimpleNamespace(_db=SessionDB(db_path=tmp_path / "state.db"))
    runner._session_db._db.get_session_title = MagicMock(return_value=None)
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._update_prompt_pending = {}
    runner._busy_input_mode = "interrupt"
    runner._draining = False
    runner._session_run_generation = {}
    runner._session_sources = {}
    runner._pending_native_image_paths_by_session = {}
    runner._background_tasks = {}
    runner._background_task_counter = 0
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._service_tier = None
    runner._fast_mode_by_session = {}
    runner._goal_state_by_session = {}
    runner._goal_runs_in_progress = set()
    runner._goal_queued_by_session = set()
    runner._is_telegram_topic_root_lobby = lambda _source: False
    runner._should_send_telegram_lobby_reminder = lambda _source: False
    runner._check_slash_access = lambda _source, _command: None
    runner._begin_session_run_generation = lambda _key: 1
    runner._release_running_agent_state = lambda key: runner._running_agents.pop(key, None)
    runner._evict_cached_agent = MagicMock()
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "path"),
    [("newmoon", NEWMOON_PATH), ("fivehours", FIVEHOURS_PATH)],
)
async def test_project_selection_intercepts_persists_and_evicts_cached_agent(tmp_path, key, path):
    runner = _runner(tmp_path)
    session_key = build_session_key(_source())
    runner._handle_message_with_agent = AsyncMock()

    response = await runner._handle_message(_event(f"!project {key}"))

    assert response == f"Active project: {key} ({path})"
    assert runner._session_db._db.get_meta("matrix_project_router:" + session_key) == key
    assert active_project_path(runner._session_db._db, session_key) == project_path(
        runner._session_db._db, key
    )
    runner._evict_cached_agent.assert_called_once_with(session_key)
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.parametrize(
    ("key", "path", "agents_text"),
    [
        ("newmoon", NEWMOON_PATH, "Authoritative context"),
        ("fivehours", FIVEHOURS_PATH, "Authoritative sources"),
    ],
)
def test_selected_matrix_session_binds_project_cwd_and_discovers_agents_md(
    tmp_path, key, path, agents_text
):
    runner = _runner(tmp_path)
    session_key = build_session_key(_source())
    runner._session_db._db.set_meta("matrix_project_router:" + session_key, key)
    context = SessionContext(
        source=_source(), connected_platforms=[], home_channels={}, session_key=session_key
    )

    tokens = runner._set_session_env(context)
    try:
        assert resolve_agent_cwd() == Path(path)
        assert resolve_context_cwd() == Path(path)
        prompt = build_context_files_prompt(cwd=str(resolve_context_cwd()), skip_soul=True)
        assert "# AGENTS.md" in prompt
        assert agents_text in prompt
    finally:
        runner._clear_session_env(tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first", "second"),
    [("newmoon", "fivehours"), ("fivehours", "newmoon")],
)
async def test_project_selection_switches_active_project_and_evicts_each_time(tmp_path, first, second):
    runner = _runner(tmp_path)
    session_key = build_session_key(_source())
    runner._handle_message_with_agent = AsyncMock()

    await runner._handle_message(_event(f"!project {first}"))
    response = await runner._handle_message(_event(f"!project {second}"))

    expected_path = FIVEHOURS_PATH if second == "fivehours" else NEWMOON_PATH
    assert response == f"Active project: {second} ({expected_path})"
    assert runner._session_db._db.get_meta("matrix_project_router:" + session_key) == second
    assert active_project_path(runner._session_db._db, session_key) == project_path(
        runner._session_db._db, second
    )
    assert runner._evict_cached_agent.call_args_list == [call(session_key), call(session_key)]
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "path"),
    [("newmoon", NEWMOON_PATH), ("fivehours", FIVEHOURS_PATH)],
)
async def test_project_status_reports_active_project(tmp_path, key, path):
    runner = _runner(tmp_path)
    runner._handle_message_with_agent = AsyncMock()

    await runner._handle_message(_event(f"!project {key}"))
    runner._evict_cached_agent.reset_mock()
    response = await runner._handle_message(_event("!project status"))

    assert response == f"Active project: {key}\nPath: {path}"
    runner._evict_cached_agent.assert_not_called()
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_project_status_reports_none_when_no_project_is_active(tmp_path):
    runner = _runner(tmp_path)
    runner._handle_message_with_agent = AsyncMock()

    response = await runner._handle_message(_event("!project status"))

    assert response == "Active project: none"
    runner._evict_cached_agent.assert_not_called()
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_project_clear_removes_state_evicts_agent_and_unbinds_follow_up(tmp_path):
    runner = _runner(tmp_path)
    session_key = build_session_key(_source())
    expected = {"final_response": "ordinary dispatch", "messages": []}
    runner._handle_message_with_agent = AsyncMock(return_value=expected)

    await runner._handle_message(_event("!project fivehours"))
    context = SessionContext(
        source=_source(), connected_platforms=[], home_channels={}, session_key=session_key
    )
    tokens = runner._set_session_env(context)
    assert resolve_context_cwd() == Path(FIVEHOURS_PATH)
    runner._evict_cached_agent.reset_mock()
    response = await runner._handle_message(_event("!project clear"))

    assert response == "Project context cleared."
    assert runner._session_db._db.get_meta("matrix_project_router:" + session_key) is None
    assert active_project_path(runner._session_db._db, session_key) is None
    runner._evict_cached_agent.assert_called_once_with(session_key)
    try:
        assert resolve_context_cwd() is None
        assert resolve_agent_cwd() not in {Path(NEWMOON_PATH), Path(FIVEHOURS_PATH)}
        prompt = build_context_files_prompt(cwd=None, skip_soul=True)
        assert "Authoritative context" not in prompt
        assert "Authoritative sources" not in prompt
    finally:
        runner._clear_session_env(tokens)

    result = await runner._handle_message(_event("ordinary Matrix message"))
    assert result == expected
    runner._handle_message_with_agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_project_selection_works_after_clear(tmp_path):
    runner = _runner(tmp_path)
    session_key = build_session_key(_source())
    runner._handle_message_with_agent = AsyncMock()

    await runner._handle_message(_event("!project newmoon"))
    await runner._handle_message(_event("!project clear"))
    response = await runner._handle_message(_event("!project fivehours"))

    assert response == f"Active project: fivehours ({FIVEHOURS_PATH})"
    assert runner._session_db._db.get_meta("matrix_project_router:" + session_key) == "fivehours"
    assert runner._evict_cached_agent.call_args_list == [
        call(session_key),
        call(session_key),
        call(session_key),
    ]
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_project_does_not_dispatch_and_lists_valid_keys(tmp_path):
    runner = _runner(tmp_path)
    runner._handle_message_with_agent = AsyncMock()

    response = await runner._handle_message(_event("!project unknown"))

    assert response == "Project selection failed: unknown project 'unknown'. Valid projects: fivehours, newmoon"
    runner._evict_cached_agent.assert_not_called()
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_unbound_matrix_session_dispatches_normally(tmp_path):
    runner = _runner(tmp_path)
    expected = {"final_response": "ordinary dispatch", "messages": []}
    runner._handle_message_with_agent = AsyncMock(return_value=expected)

    result = await runner._handle_message(_event("ordinary Matrix message"))

    assert result == expected
    runner._handle_message_with_agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_natural_project_reference_switches_and_continues_same_message(tmp_path):
    runner = _runner(tmp_path)
    session_key = build_session_key(_source())
    expected = {"final_response": "ordinary dispatch", "messages": []}
    runner._handle_message_with_agent = AsyncMock(return_value=expected)

    result = await runner._handle_message(_event("For Five Hours, update the homepage."))

    assert result == expected
    assert active_project_path(runner._session_db._db, session_key) == FIVEHOURS_PATH
    runner._evict_cached_agent.assert_called_once_with(session_key)
    runner._handle_message_with_agent.assert_awaited_once()
    runner.adapters[Platform.MATRIX].send.assert_awaited_once_with(
        chat_id="matrix-room", content="Using project: fivehours"
    )


@pytest.mark.asyncio
async def test_active_project_stays_bound_for_generic_follow_up_and_switches_for_other_reference(tmp_path):
    runner = _runner(tmp_path)
    session_key = build_session_key(_source())
    runner._handle_message_with_agent = AsyncMock(return_value={"final_response": "ok", "messages": []})

    await runner._handle_message(_event("!project newmoon"))
    runner._evict_cached_agent.reset_mock()
    await runner._handle_message(_event("Change the CTA to Get Started."))
    assert active_project_path(runner._session_db._db, session_key) == NEWMOON_PATH
    runner._evict_cached_agent.assert_not_called()

    await runner._handle_message(_event("Let's go back to Five Hours and change pricing."))
    assert active_project_path(runner._session_db._db, session_key) == FIVEHOURS_PATH
    runner._evict_cached_agent.assert_called_once_with(session_key)


@pytest.mark.asyncio
async def test_ambiguous_natural_project_reference_asks_without_dispatch(tmp_path):
    runner = _runner(tmp_path)
    runner._handle_message_with_agent = AsyncMock()
    db = runner._session_db._db
    project = _make_project(tmp_path, "Other Project")
    register_project(db, str(project))
    registry = json.loads(db.get_meta("matrix_project_router:registry"))
    registry["projects"]["newmoon"]["metadata"]["aliases"] = ["website"]
    registry["projects"]["otherproject"]["metadata"]["aliases"] = ["website"]
    db.set_meta("matrix_project_router:registry", json.dumps(registry))

    result = await runner._handle_message(_event("I want to work on the website."))

    assert result == "I found multiple possible projects:\n- newmoon\n- otherproject\nWhich one do you want to use?"
    runner._handle_message_with_agent.assert_not_awaited()
    runner._evict_cached_agent.assert_not_called()


def test_registry_bootstraps_legacy_projects_once_without_overwriting_additions(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")

    bootstrap_registry(db)
    assert project_keys(db) == ("fivehours", "newmoon")

    project = _make_project(tmp_path, "custom-project")
    registered = register_project(db, str(project))
    assert registered.key == "customproject"

    bootstrap_registry(db)
    assert project_keys(db) == ("customproject", "fivehours", "newmoon")


def test_registry_migration_adds_default_metadata_without_overwriting_custom_metadata(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.set_meta(
        "matrix_project_router:registry",
        json.dumps(
            {
                "version": 1,
                "projects": {
                    "fivehours": {
                        "path": str(FIVEHOURS_PATH),
                        "metadata": {"display_name": "Custom Hours", "aliases": ["custom hours"]},
                    }
                },
            }
        ),
    )

    bootstrap_registry(db)

    assert resolve_project_reference(db, "CUSTOM HOURS") == ("fivehours",)
    assert resolve_project_reference(db, "five hours") == ()
    registry = json.loads(db.get_meta("matrix_project_router:registry"))
    assert registry["version"] == 2
    assert registry["projects"]["fivehours"]["metadata"] == {
        "aliases": ["custom hours"],
        "display_name": "Custom Hours",
    }


def test_resolve_project_reference_matches_key_display_name_and_aliases_with_boundaries(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    trinity = _make_project(tmp_path, "trinity-water")
    register_project(db, str(trinity))

    assert resolve_project_reference(db, "Let's work on Five Hours.") == ("fivehours",)
    assert resolve_project_reference(db, "Go back to New Moon Nails") == ("newmoon",)
    assert resolve_project_reference(db, "I want to change something on the nail site.") == ("newmoon",)
    assert resolve_project_reference(db, "For Trinity Water, change the homepage") == ("trinitywater",)
    assert resolve_project_reference(db, "FIVE HOURS should work") == ("fivehours",)
    assert resolve_project_reference(db, "fivehoursly is not a project reference") == ()


def test_project_aliases_persist_and_conflicts_never_reassign(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)

    assert add_project_alias(db, "newmoon", "Nail Website") == "nail website"
    assert resolve_project_reference(db, "Let's work on the nail website.") == ("newmoon",)
    with pytest.raises(ValueError, match="conflicts with project 'newmoon'"):
        add_project_alias(db, "fivehours", "nail website")

    reopened = SessionDB(db_path=db_path)
    assert resolve_project_reference(reopened, "NAIL WEBSITE") == ("newmoon",)
    assert project_details(reopened, "newmoon")[2] == (
        "nail site",
        "nail website",
        "nails site",
        "new moon",
        "new moon nails",
    )
    fresh = SessionDB(db_path=tmp_path / "fresh-state.db")
    assert resolve_project_reference(fresh, "nail website") == ()


@pytest.mark.asyncio
async def test_project_alias_commands_mutate_registry_show_sorted_aliases_and_bypass_routing(tmp_path):
    runner = _runner(tmp_path)
    runner._handle_message_with_agent = AsyncMock()

    added = await runner._handle_message(_event("!project alias add newmoon Nail Website"))
    removed = await runner._handle_message(_event("!project alias remove newmoon nail website"))

    assert added == (
        "Alias added: nail website\n"
        "Project: newmoon\n"
        "Aliases:\n"
        "- nail site\n"
        "- nail website\n"
        "- nails site\n"
        "- new moon\n"
        "- new moon nails"
    )
    assert removed == (
        "Alias removed: nail website\n"
        "Project: newmoon\n"
        "Aliases:\n"
        "- nail site\n"
        "- nails site\n"
        "- new moon\n"
        "- new moon nails"
    )
    assert resolve_project_reference(runner._session_db._db, "nail website") == ()
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_project_aliases_lists_all_or_one_and_unknown_key_is_safe(tmp_path):
    runner = _runner(tmp_path)
    runner._handle_message_with_agent = AsyncMock()
    trinity = _make_project(tmp_path, "trinity-water")
    register_project(runner._session_db._db, str(trinity))

    all_aliases = await runner._handle_message(_event("!project aliases"))
    one_aliases = await runner._handle_message(_event("!project aliases newmoon"))
    no_aliases = await runner._handle_message(_event("!project aliases trinitywater"))
    unknown = await runner._handle_message(_event("!project aliases missing"))

    assert all_aliases == (
        "Project aliases:\n\n"
        "- fivehours (Five Hours)\n"
        "  - five hours\n"
        "  - save five hours\n\n"
        "- newmoon (New Moon Nails)\n"
        "  - nail site\n"
        "  - nails site\n"
        "  - new moon\n"
        "  - new moon nails\n\n"
        "- trinitywater (Trinity Water)\n"
        "  - none"
    )
    assert one_aliases == (
        "Project: newmoon\n"
        "Name: New Moon Nails\n"
        "Aliases:\n"
        "- nail site\n"
        "- nails site\n"
        "- new moon\n"
        "- new moon nails"
    )
    assert no_aliases == "Project: trinitywater\nName: Trinity Water\nAliases:\n- none"
    assert unknown == (
        "Project aliases failed: unknown project 'missing'. "
        "Valid projects: fivehours, newmoon, trinitywater"
    )
    runner._handle_message_with_agent.assert_not_awaited()


def test_registered_project_uses_canonical_path_and_survives_reopening_state(tmp_path):
    db_path = tmp_path / "state.db"
    project = _make_project(tmp_path, "My Cool App")
    db = SessionDB(db_path=db_path)

    registered = register_project(db, str(project / "."))

    assert registered.key == "mycoolapp"
    assert registered.path == project.resolve()
    reopened = SessionDB(db_path=db_path)
    assert project_path(reopened, "mycoolapp") == project.resolve()


def test_register_project_resolves_exact_short_name_under_injected_projects_root(tmp_path):
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    project = _make_project(projects_root, "trinity-water")
    db = SessionDB(db_path=tmp_path / "state.db")

    registered = register_project(db, "trinity-water", projects_root=projects_root)

    assert registered.key == "trinitywater"
    assert registered.path == project.resolve()
    assert project_path(db, "trinitywater") == project.resolve()


def test_register_project_accepts_absolute_path_with_injected_projects_root(tmp_path):
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    project = _make_project(tmp_path, "outside-project")
    db = SessionDB(db_path=tmp_path / "state.db")

    registered = register_project(db, str(project / "."), projects_root=projects_root)

    assert registered.path == project.resolve()


@pytest.mark.parametrize("short_name", ["trinity", "trin*"])
def test_register_project_rejects_nonexistent_short_name(tmp_path, short_name):
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    db = SessionDB(db_path=tmp_path / "state.db")

    with pytest.raises(ValueError, match="project path does not exist"):
        register_project(db, short_name, projects_root=projects_root)


@pytest.mark.parametrize("relative_path", ["../outside-project", "foo/../../../outside-project"])
def test_register_project_rejects_relative_path_that_escapes_projects_root(tmp_path, relative_path):
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    escaped_project = _make_project(tmp_path, "outside-project")
    db = SessionDB(db_path=tmp_path / "state.db")

    with pytest.raises(ValueError, match="must remain beneath projects root"):
        register_project(db, relative_path, projects_root=projects_root)

    assert project_path(db, "outsideproject") is None
    assert (escaped_project / "README.md").read_text() == "# Test project\n"


def test_register_project_supports_exact_nested_relative_path(tmp_path):
    projects_root = tmp_path / "projects"
    nested_root = projects_root / "experiments"
    nested_root.mkdir(parents=True)
    project = _make_project(nested_root, "my-app")
    db = SessionDB(db_path=tmp_path / "state.db")

    registered = register_project(db, "experiments/my-app", projects_root=projects_root)

    assert registered.key == "myapp"
    assert registered.path == project.resolve()


@pytest.mark.asyncio
async def test_project_add_resolves_short_name_from_default_projects_root(tmp_path, monkeypatch):
    import gateway.project_router as project_router

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    project = _make_project(projects_root, "trinity-water", agents=False)
    monkeypatch.setattr(project_router, "DEFAULT_PROJECTS_ROOT", projects_root)
    runner = _runner(tmp_path)
    runner._handle_message_with_agent = AsyncMock()

    response = await runner._handle_message(_event("!project add trinity-water"))

    assert response.startswith(
        f"Project registered: trinitywater\nPath: {project.resolve()}\n\nContext:\n"
    )
    assert project_path(runner._session_db._db, "trinitywater") == project.resolve()
    assert (project / "README.md").read_text() == "# Test project\n"


def test_register_project_rejects_duplicate_keys_and_paths(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    first = _make_project(tmp_path, "first")
    second = _make_project(tmp_path, "second")
    register_project(db, str(first), key="shared")

    with pytest.raises(ValueError, match="already registered for this path"):
        register_project(db, str(first), key="shared")
    with pytest.raises(ValueError, match="key 'shared'.*already registered"):
        register_project(db, str(second), key="shared")
    with pytest.raises(ValueError, match="already registered as 'shared'"):
        register_project(db, str(first), key="other")


@pytest.mark.parametrize("raw_path", ["relative-project", "/does/not/exist"])
def test_register_project_rejects_invalid_paths(tmp_path, raw_path):
    db = SessionDB(db_path=tmp_path / "state.db")

    with pytest.raises(ValueError):
        register_project(db, raw_path)


def test_register_project_rejects_directory_that_is_not_a_project(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    directory = tmp_path / "not-a-project"
    directory.mkdir()

    with pytest.raises(ValueError, match="does not appear to be a project or repository"):
        register_project(db, str(directory))


@pytest.mark.asyncio
async def test_project_add_registers_without_modifying_repository_and_can_be_selected(tmp_path):
    runner = _runner(tmp_path)
    runner._handle_message_with_agent = AsyncMock()
    project = _make_project(tmp_path, "My Cool App", agents=False)
    readme_before = (project / "README.md").read_text()

    response = await runner._handle_message(_event(f"!project add {project}"))

    assert response == (
        f"Project registered: mycoolapp\nPath: {project.resolve()}\n\nContext:\n"
        "- AGENTS.md: missing\n- README*: found\n- CONTRIBUTING.md: missing\n"
        "- package.json: missing\n- pyproject.toml: missing\n- Cargo.toml: missing\n"
        "- go.mod: missing\n- docs/: missing\n- docs/STATUS.md: missing\n"
        "- docs/decisions/: missing\n\n"
        "Project routing is available, but repository agent context is incomplete."
    )
    assert (project / "README.md").read_text() == readme_before
    assert not (project / "AGENTS.md").exists()

    selected = await runner._handle_message(_event("!project mycoolapp"))
    assert selected == f"Active project: mycoolapp ({project.resolve()})"
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_project_list_is_deterministic_and_unknown_keys_are_dynamic(tmp_path):
    runner = _runner(tmp_path)
    runner._handle_message_with_agent = AsyncMock()
    project = _make_project(tmp_path, "Zebra App")
    await runner._handle_message(_event(f"!project add {project}"))

    listed = await runner._handle_message(_event("!project list"))
    unknown = await runner._handle_message(_event("!project unknown"))

    assert listed == (
        "Registered projects:\n\n"
        "- fivehours\n"
        "  Name: Five Hours\n"
        "  Aliases: five hours, save five hours\n"
        "  Path: /home/rle/projects/savefivehours\n\n"
        "- newmoon\n"
        "  Name: New Moon Nails\n"
        "  Aliases: nail site, nails site, new moon, new moon nails\n"
        "  Path: /home/rle/projects/NewMoonNailsAndSpa\n\n"
        "- zebraapp\n"
        "  Name: Zebra App\n"
        "  Aliases: none\n"
        f"  Path: {project.resolve()}"
    )
    assert "Valid projects: fivehours, newmoon, zebraapp" in unknown


def test_setup_analysis_for_minimal_repo_recommends_agent_context_without_writes(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    project = _make_project(tmp_path, "minimal", agents=False)
    (project / "package.json").write_text('{"scripts":{"prepare":"touch executed"}}\n')
    before = _repository_snapshot(project)
    register_project(db, str(project))

    plan = analyze_project_setup(db, "minimal")

    assert plan.key == "minimal"
    assert plan.path == project.resolve()
    assert plan.found == ("README.md", "package.json")
    assert [(item.action, item.target, item.category) for item in plan.recommendations] == [
        ("create", "AGENTS.md", "recommended"),
    ]
    assert plan.not_needed == (
        "docs/STATUS.md — no documentation convention detected",
        "docs/decisions/ — no documentation or ADR convention detected",
    )
    assert plan.authoritative_sources == (("README.md", "project overview"),)
    assert _repository_snapshot(project) == before
    assert not (project / "executed").exists()


@pytest.mark.asyncio
async def test_project_setup_known_project_returns_deterministic_read_only_plan(tmp_path):
    runner = _runner(tmp_path)
    runner._handle_message_with_agent = AsyncMock()
    project = _make_project(tmp_path, "Mature App")
    (project / "CONTRIBUTING.md").write_text("# Contributing\n")
    (project / "docs").mkdir()
    (project / "docs" / "adr").mkdir()
    (project / "docs" / "adr" / "0001-record.md").write_text("# ADR\n")
    register_project(runner._session_db._db, str(project))
    before = _repository_snapshot(project)

    response = await runner._handle_message(_event("!project setup matureapp"))

    assert response == (
        f"Project setup analysis: matureapp\nPath: {project.resolve()}\n\n"
        "Found:\n- AGENTS.md\n- README.md\n- CONTRIBUTING.md\n- docs/\n- docs/adr/\n\n"
        "Recommended:\n- docs/STATUS.md — concise current-state snapshot would aid ongoing work\n\n"
        "Not currently needed:\n- AGENTS.md — existing repository agent instructions are present\n"
        "- docs/decisions/ — existing ADR convention: docs/adr/\n\n"
        "Authoritative sources:\n- AGENTS.md — repository agent instructions\n"
        "- README.md — project overview\n- CONTRIBUTING.md — contribution conventions\n\n"
        "No repository files were changed."
    )
    assert _repository_snapshot(project) == before
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_project_setup_unknown_key_lists_dynamic_keys_without_dispatch(tmp_path):
    runner = _runner(tmp_path)
    runner._handle_message_with_agent = AsyncMock()
    project = _make_project(tmp_path, "Zebra App")
    register_project(runner._session_db._db, str(project))

    response = await runner._handle_message(_event("!project setup missing"))

    assert response == (
        "Project setup failed: unknown project 'missing'. "
        "Valid projects: fivehours, newmoon, zebraapp"
    )
    runner._handle_message_with_agent.assert_not_awaited()


def test_setup_analysis_respects_claude_and_existing_personal_context_conventions(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    project = _make_project(tmp_path, "personal")
    (project / "CLAUDE.md").write_text("# Repository instructions\n")
    (project / "docs").mkdir()
    (project / "docs" / "STATUS.md").write_text("# Current status\n")
    (project / "docs" / "decisions").mkdir()
    (project / "docs" / "decisions" / "001.md").write_text("# Decision\n")
    register_project(db, str(project))

    plan = analyze_project_setup(db, "personal")

    assert plan.found == (
        "AGENTS.md",
        "README.md",
        "CLAUDE.md",
        "docs/",
        "docs/STATUS.md",
        "docs/decisions/",
    )
    assert plan.recommendations == ()
    assert plan.not_needed == (
        "AGENTS.md — existing repository agent instructions are present",
        "docs/STATUS.md — current-state context already exists",
        "docs/decisions/ — existing ADR convention: docs/decisions/",
    )
    assert plan.authoritative_sources == (
        ("AGENTS.md", "repository agent instructions"),
        ("README.md", "project overview"),
        ("CLAUDE.md", "repository agent instructions"),
        ("docs/STATUS.md", "current implementation state"),
    )


def test_setup_analysis_detects_readme_variants_and_claude_instruction_convention(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    project = _make_project(tmp_path, "variant", agents=False)
    (project / "README.md").unlink()
    (project / "README.rst").write_text("Variant project overview\n")
    (project / "CLAUDE.md").write_text("Repository agent instructions\n")
    (project / "requirements-dev.txt").write_text("pytest\n")
    (project / "docs").mkdir()
    register_project(db, str(project))

    first = analyze_project_setup(db, "variant")
    second = analyze_project_setup(db, "variant")

    assert first == second
    assert first.found == ("README.rst", "CLAUDE.md", "requirements-dev.txt", "docs/")
    assert all(item.target != "AGENTS.md" for item in first.recommendations)
    assert first.not_needed[0] == "AGENTS.md — existing repository agent instructions are present"
    assert first.authoritative_sources[:2] == (
        ("README.rst", "project overview"),
        ("CLAUDE.md", "repository agent instructions"),
    )


def test_setup_apply_with_no_recommendations_does_not_mutate(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    project = _make_project(tmp_path, "complete")
    (project / "docs" / "decisions").mkdir(parents=True)
    (project / "docs" / "STATUS.md").write_text("# Status\n")
    register_project(db, str(project))
    before = _repository_snapshot(project)

    result = apply_project_setup(db, "complete")

    assert result.created == ()
    assert result.skipped == ()
    assert result.plan.recommendations == ()
    assert _repository_snapshot(project) == before


@pytest.mark.asyncio
async def test_project_setup_apply_reports_no_changes_for_empty_current_plan(tmp_path):
    runner = _runner(tmp_path)
    runner._handle_message_with_agent = AsyncMock()
    project = _make_project(tmp_path, "Complete App")
    (project / "docs" / "decisions").mkdir(parents=True)
    (project / "docs" / "STATUS.md").write_text("# Status\n")
    register_project(runner._session_db._db, str(project))

    response = await runner._handle_message(_event("!project setup completeapp --apply"))

    assert response == (
        "Project setup analysis: completeapp\n\n"
        "No setup changes are currently recommended.\nNothing to apply.\n\n"
        "No repository files were changed."
    )


@pytest.mark.asyncio
async def test_project_setup_apply_renders_deterministic_created_and_skipped_sections(tmp_path):
    runner = _runner(tmp_path)
    runner._handle_message_with_agent = AsyncMock()
    project = _make_project(tmp_path, "Apply App", agents=False)
    register_project(runner._session_db._db, str(project))

    response = await runner._handle_message(_event("!project setup applyapp --apply"))

    assert response == (
        "Project setup applied: applyapp\n\nCreated:\n- AGENTS.md\n\nSkipped:\n- none\n\n"
        "No existing files were overwritten.\nChanges remain uncommitted for review."
    )
    assert (project / "AGENTS.md").is_file()


def test_setup_apply_creates_recommended_agents_from_static_evidence_without_scripts(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    project = _make_project(tmp_path, "minimal", agents=False)
    (project / "package.json").write_text('{"scripts":{"prepare":"touch executed"}}\n')
    register_project(db, str(project))

    result = apply_project_setup(db, "minimal")

    assert result.created == ("AGENTS.md",)
    assert result.skipped == ()
    content = (project / "AGENTS.md").read_text()
    assert "# Agent Instructions" in content
    assert "README.md" in content
    assert "package.json" in content
    assert "must be confirmed" in content
    assert not (project / "executed").exists()
    assert all(item.target != "AGENTS.md" for item in result.plan.recommendations)


def test_setup_apply_reanalyzes_and_never_overwrites_agents_created_after_a_stale_plan(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    project = _make_project(tmp_path, "stale", agents=False)
    register_project(db, str(project))
    stale_plan = analyze_project_setup(db, "stale")
    assert any(item.target == "AGENTS.md" for item in stale_plan.recommendations)
    agents = project / "AGENTS.md"
    agents.write_text("# Existing instructions\n")

    result = apply_project_setup(db, "stale")

    assert result.created == ()
    assert agents.read_text() == "# Existing instructions\n"
    assert result.plan.recommendations == ()


def test_setup_apply_creates_only_recommended_status_file(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    project = _make_project(tmp_path, "status")
    (project / "docs").mkdir()
    register_project(db, str(project))

    result = apply_project_setup(db, "status")

    assert result.created == ("docs/STATUS.md",)
    assert (project / "docs" / "STATUS.md").read_text().startswith("# Current Status\n")
    assert not (project / "docs" / "decisions").exists()
    assert all(item.target != "docs/STATUS.md" for item in result.plan.recommendations)


def test_setup_apply_creates_decision_scaffold_only_when_recommended(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    project = _make_project(tmp_path, "decisions")
    (project / "CONTRIBUTING.md").write_text("# Contributing\n")
    (project / "package.json").write_text("{}\n")
    (project / "docs").mkdir()
    register_project(db, str(project))

    result = apply_project_setup(db, "decisions")

    assert result.created == ("docs/STATUS.md", "docs/decisions/README.md")
    assert (project / "docs" / "decisions" / "README.md").is_file()
    assert not list((project / "docs" / "decisions").glob("[0-9]*"))
    assert result.plan.recommendations == ()


def test_setup_apply_refuses_a_target_that_appears_after_current_analysis(tmp_path, monkeypatch):
    import gateway.project_router as project_router

    db = SessionDB(db_path=tmp_path / "state.db")
    project = _make_project(tmp_path, "race", agents=False)
    register_project(db, str(project))
    original_write = project_router._write_new_file

    def create_conflict(path, content):
        if path.name == "AGENTS.md":
            path.write_text("# Concurrent instructions\n")
        return original_write(path, content)

    monkeypatch.setattr(project_router, "_write_new_file", create_conflict)
    result = apply_project_setup(db, "race")

    assert result.created == ()
    assert result.skipped == (("AGENTS.md", "target already exists"),)
    assert (project / "AGENTS.md").read_text() == "# Concurrent instructions\n"


def test_setup_apply_never_applies_non_recommended_items(tmp_path, monkeypatch):
    import gateway.project_router as project_router

    db = SessionDB(db_path=tmp_path / "state.db")
    project = _make_project(tmp_path, "categories", agents=False)
    register_project(db, str(project))
    plan = ProjectSetupPlan(
        key="categories",
        path=project,
        found=(),
        recommendations=(
            SetupRecommendation("create", "AGENTS.md", "not selected", "optional"),
            SetupRecommendation("create", "docs/STATUS.md", "not selected", "found"),
        ),
        not_needed=(),
        authoritative_sources=(),
    )
    monkeypatch.setattr(project_router, "analyze_project_setup", lambda _db, _key: plan)

    result = apply_project_setup(db, "categories")

    assert result.created == ()
    assert not (project / "AGENTS.md").exists()
    assert not (project / "docs").exists()


def test_setup_apply_preserves_unrelated_existing_repository_changes(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    project = _make_project(tmp_path, "unrelated", agents=False)
    unrelated = project / "notes.txt"
    unrelated.write_text("uncommitted local note\n")
    register_project(db, str(project))

    result = apply_project_setup(db, "unrelated")

    assert result.created == ("AGENTS.md",)
    assert unrelated.read_text() == "uncommitted local note\n"


def test_setup_apply_preserves_unrelated_git_changes_without_staging_or_branch_mutation(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    project = _make_project(tmp_path, "git-unrelated", agents=False)
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=project, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=project, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=project, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=project, check=True)
    unrelated = project / "local-note.txt"
    unrelated.write_text("preserve me\n")
    branch_before = subprocess.check_output(["git", "branch", "--show-current"], cwd=project, text=True)
    head_before = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True)
    register_project(db, str(project))

    result = apply_project_setup(db, "gitunrelated")

    assert result.had_unrelated_changes is True
    assert unrelated.read_text() == "preserve me\n"
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=project, text=True) == branch_before
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True) == head_before
    assert subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=project).returncode == 0
