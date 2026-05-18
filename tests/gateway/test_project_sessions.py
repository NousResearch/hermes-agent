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
    monkeypatch.setattr(ps, "discover_tmux_projects", lambda: [])
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
    monkeypatch.setattr(ps, "discover_tmux_projects", lambda: [])
    sent = []
    monkeypatch.setattr(ps, "_send_to_tmux", lambda project, prompt: sent.append((project.label, prompt)))
    state = ps.ProjectRoutingState(
        projects={
            "dev-1": ps.ProjectSession(label="dev-1", tmux_target="dev:1.0", workdir=str(tmp_path)),
            "dev-2": ps.ProjectSession(label="dev-2", tmux_target="dev:2.0", workdir=str(tmp_path)),
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
    monkeypatch.setattr(ps, "discover_tmux_projects", lambda: [])

    def fail(project, prompt):
        raise RuntimeError("tmux missing")

    monkeypatch.setattr(ps, "_send_to_tmux", fail)
    state = ps.ProjectRoutingState(
        projects={"dev-1": ps.ProjectSession(label="dev-1", tmux_target="dev:1.0", workdir=str(tmp_path))}
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
