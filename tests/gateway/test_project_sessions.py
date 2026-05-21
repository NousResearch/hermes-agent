from pathlib import Path
from types import SimpleNamespace

from gateway import project_sessions as ps
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, coerce_plaintext_gateway_command
from gateway.session import SessionSource


def _source(chat_id="chat1"):
    return SimpleNamespace(platform=SimpleNamespace(value="telegram"), chat_id=chat_id, user_id="u1", thread_id=None)


def test_plaintext_commute_commands_are_dm_only():
    dm = MessageEvent(
        text="출근모드",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm", user_id="u1"),
    )
    coerce_plaintext_gateway_command(dm)
    assert dm.text == "/office all"

    group = MessageEvent(
        text="출근모드",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="g1", chat_type="group", user_id="u1"),
    )
    coerce_plaintext_gateway_command(group)
    assert group.text == "출근모드"


def test_parse_project_prefix():
    assert ps.parse_project_prefix("[dev-1] 다음 작업 진행해줘") == ("dev-1", "다음 작업 진행해줘")
    assert ps.parse_project_prefix("일반 메시지") is None


def test_commute_help_text_includes_core_commands():
    text = ps.commute_help_text()
    assert "/projects" in text
    assert "퇴근모드 또는 /afterwork all" in text
    assert "출근모드 또는 /office all" in text
    assert "/afterwork current" in text
    assert "고급/테스트 옵션" in text
    assert "/psend" in text
    assert "[label]" in text
    assert "/approve" in text


def test_load_state_migrates_legacy_active_by_user(tmp_path):
    path = tmp_path / "project_sessions.json"
    path.write_text(
        '{"active_by_user": {"8584626899": "HUD", "test": "work", "old": "home"}, "pinned": {}}',
        encoding="utf-8",
    )

    state = ps.load_state(path)

    assert state.mode == "office"
    assert state.active_by_chat == {"telegram:8584626899": "HUD"}
    assert state.projects == {}


def test_switch_and_current_project_with_alias(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "discover_tmux_projects", lambda: [])
    state = ps.ProjectRoutingState(
        projects={
            "dev-1": ps.ProjectSession(
                label="dev-1",
                tmux_target="dev:1.0",
                workdir=str(tmp_path),
                aliases=["HUD"],
            )
        }
    )
    path = tmp_path / "project_sessions.json"
    ps.save_state(state, path)

    msg = ps.switch_project(_source(), "HUD", path=path)
    assert "[dev-1]" in msg

    current = ps.current_project(_source(), path=path)
    assert "[dev-1]" in current
    assert "dev:1.0" in current


def test_handle_prefixed_message_routes_via_steer(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ps,
        "discover_tmux_projects",
        lambda: [
            ps.ProjectSession(
                label="dev-1",
                tmux_target="dev:1.0",
                workdir=str(tmp_path),
                aliases=["HUD"],
                notify=True,
                status="ready",
                command="python3.11",
                routable=True,
            )
        ],
    )
    sent = []
    monkeypatch.setattr(ps, "_send_to_tmux", lambda project, prompt: sent.append((project.label, prompt)))
    state = ps.ProjectRoutingState(
        mode="away",
        projects={
            "dev-1": ps.ProjectSession(
                label="dev-1",
                tmux_target="dev:1.0",
                workdir=str(tmp_path),
                aliases=["HUD"],
                notify=True,
                status="ready",
                command="python3.11",
                routable=True,
            )
        }
    )
    path = tmp_path / "project_sessions.json"
    ps.save_state(state, path)

    result = ps.handle_prefixed_message(_source(), "[HUD] 다음 작업 진행해줘", path=path)

    assert "[dev-1] 전달 완료" in result
    assert sent == [("dev-1", "/steer 다음 작업 진행해줘")]


def test_handle_prefixed_message_ignores_office_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "discover_tmux_projects", lambda: [])
    sent = []
    monkeypatch.setattr(ps, "_send_to_tmux", lambda project, prompt: sent.append((project.label, prompt)))
    state = ps.ProjectRoutingState(
        mode="office",
        projects={
            "dev-1": ps.ProjectSession(
                label="dev-1",
                tmux_target="dev:1.0",
                workdir=str(tmp_path),
                aliases=["HUD"],
                notify=False,
            )
        },
    )
    path = tmp_path / "project_sessions.json"
    ps.save_state(state, path)

    result = ps.handle_prefixed_message(_source(), "[HUD] 다음 작업 진행해줘", path=path)

    assert result is None
    assert sent == []


def test_set_mode_toggles_notify_and_sends_instructions(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ps,
        "discover_tmux_projects",
        lambda: [
            ps.ProjectSession(label="dev-1", tmux_target="dev:1.0", workdir=str(tmp_path), status="ready", command="python3.11", routable=True),
            ps.ProjectSession(label="dev-2", tmux_target="dev:2.0", workdir=str(tmp_path), status="ready", command="hermes", routable=True),
        ],
    )
    sent = []
    monkeypatch.setattr(ps, "_send_to_tmux", lambda project, prompt: sent.append((project.label, prompt)))
    state = ps.ProjectRoutingState(
        projects={
            "dev-1": ps.ProjectSession(label="dev-1", tmux_target="dev:1.0", workdir=str(tmp_path), status="ready", command="python3.11", routable=True),
            "dev-2": ps.ProjectSession(label="dev-2", tmux_target="dev:2.0", workdir=str(tmp_path), status="ready", command="hermes", routable=True),
        }
    )
    path = tmp_path / "project_sessions.json"
    ps.save_state(state, path)

    away = ps.set_mode(_source(), "away", "all", path=path)
    assert "[퇴근모드 시작]" in away
    assert len(sent) == 2
    reloaded = ps.load_state(path)
    assert reloaded.mode == "away"
    assert all(p.notify for p in reloaded.projects.values())

    sent.clear()
    office = ps.set_mode(_source(), "office", "all", path=path)
    assert "[출근모드 전환 완료]" in office
    reloaded = ps.load_state(path)
    assert reloaded.mode == "office"
    assert not any(p.notify for p in reloaded.projects.values())


def test_set_mode_rejects_invalid_scope(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "discover_tmux_projects", lambda: [])
    sent = []
    monkeypatch.setattr(ps, "_send_to_tmux", lambda project, prompt: sent.append((project.label, prompt)))
    state = ps.ProjectRoutingState(
        projects={"dev-1": ps.ProjectSession(label="dev-1", tmux_target="dev:1.0", workdir=str(tmp_path))}
    )
    path = tmp_path / "project_sessions.json"
    ps.save_state(state, path)

    msg = ps.set_mode(_source(), "away", "curent", path=path)

    assert "사용: /afterwork" in msg
    assert sent == []
    assert ps.load_state(path).mode == "office"


def test_set_mode_does_not_mark_failed_project_notify(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ps,
        "discover_tmux_projects",
        lambda: [ps.ProjectSession(label="dev-1", tmux_target="dev:1.0", workdir=str(tmp_path), status="ready", command="python3.11", routable=True)],
    )

    def fail(project, prompt):
        raise RuntimeError("tmux missing")

    monkeypatch.setattr(ps, "_send_to_tmux", fail)
    state = ps.ProjectRoutingState(
        projects={"dev-1": ps.ProjectSession(label="dev-1", tmux_target="dev:1.0", workdir=str(tmp_path), status="ready", command="python3.11", routable=True)}
    )
    path = tmp_path / "project_sessions.json"
    ps.save_state(state, path)

    msg = ps.set_mode(_source(), "away", "all", path=path)

    assert "전달 실패" in msg
    reloaded = ps.load_state(path)
    assert reloaded.mode == "office"
    assert reloaded.projects["dev-1"].notify is False


def test_discover_tmux_projects_filters_broad_roots(monkeypatch, tmp_path):
    project = tmp_path / "real-project"
    project.mkdir()
    (project / ".hermes").mkdir()
    broad = tmp_path / "work"
    broad.mkdir()
    (broad / "AGENTS.md").write_text("x", encoding="utf-8")
    output = "dev\t1\tHUD\t0\t{}\tpython\n".format(project)
    output += "dev\t2\twork\t0\t{}\tzsh\n".format(broad)
    monkeypatch.setattr(ps, "_run_tmux", lambda args: output)

    projects = ps.discover_tmux_projects()

    assert [p.label for p in projects] == ["HUD"]
    assert projects[0].tmux_target == "dev:1.0"



def test_shell_only_project_is_listed_but_not_routable(monkeypatch, tmp_path):
    project = tmp_path / "shell-project"
    project.mkdir()
    (project / ".hermes").mkdir()
    output = "dev\t1\tshell-project\t0\t{}\tzsh\n".format(project)
    monkeypatch.setattr(ps, "_run_tmux", lambda args: output)
    sent = []
    monkeypatch.setattr(ps, "_send_to_tmux", lambda project, prompt: sent.append((project.label, prompt)))
    path = tmp_path / "project_sessions.json"
    state = ps.refresh_from_tmux(ps.load_state(path))
    ps.save_state(state, path)

    listing = ps.format_projects(state, _source())
    assert "SHELL-ONLY" in listing
    assert "Telegram 지시: 불가" in listing

    switch = ps.switch_project(_source(), "shell-project", path=path)
    assert "전환 완료: 주의" in switch
    blocked = ps.send_prompt_to_project(_source(), "상태 요약", path=path)
    assert "[차단]" in blocked
    assert "Hermes 세션이 아닙니다" in blocked
    assert sent == []


def test_afterwork_all_excludes_shell_only_and_stale_projects(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setattr(ps, "_send_to_tmux", lambda project, prompt: sent.append((project.label, prompt)))
    monkeypatch.setattr(ps, "_capture_pane", lambda project, lines=80: "recent output")
    monkeypatch.setattr(
        ps,
        "discover_tmux_projects",
        lambda: [
            ps.ProjectSession(label="ready", tmux_target="dev:1.0", workdir=str(tmp_path / "ready"), status="ready", command="python3.11", routable=True),
            ps.ProjectSession(label="shell", tmux_target="dev:2.0", workdir=str(tmp_path / "shell"), status="shell-only", command="zsh", routable=False),
        ],
    )
    (tmp_path / "ready").mkdir()
    (tmp_path / "shell").mkdir()
    state = ps.ProjectRoutingState(
        projects={
            "ready": ps.ProjectSession(label="ready", tmux_target="dev:1.0", workdir=str(tmp_path / "ready"), status="ready", command="python3.11", routable=True),
            "stale": ps.ProjectSession(label="stale", tmux_target="old:1.0", workdir=str(tmp_path / "stale"), status="ready", command="python3.11", routable=True),
        }
    )
    path = tmp_path / "project_sessions.json"
    ps.save_state(state, path)

    msg = ps.set_mode(_source(), "away", "all", path=path)

    assert "[ready]" in msg
    assert "[shell] SHELL-ONLY" in msg
    assert "[stale] STALE" in msg
    assert sent == [("ready", sent[0][1])]
    reloaded = ps.load_state(path)
    assert reloaded.projects["ready"].notify is True
    assert reloaded.projects["shell"].notify is False
    assert reloaded.projects["stale"].notify is False
    snapshot = Path(reloaded.projects["ready"].last_away_snapshot_path)
    assert snapshot.exists()
    assert "steer_sent: True" in snapshot.read_text(encoding="utf-8")


def test_refresh_marks_missing_projects_stale(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "discover_tmux_projects", lambda: [])
    state = ps.ProjectRoutingState(
        projects={"old": ps.ProjectSession(label="old", tmux_target="old:1.0", workdir=str(tmp_path), status="ready", command="python3.11", routable=True, notify=True)}
    )

    refreshed = ps.refresh_from_tmux(state)

    assert refreshed.projects["old"].status == "stale"
    assert refreshed.projects["old"].routable is False
    assert refreshed.projects["old"].notify is False
