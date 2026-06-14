from __future__ import annotations

from types import SimpleNamespace

from hermes_cli import codex_learning as cl


def _session(**overrides):
    command = (
        "mkdir -p /tmp/root && "
        "git -C /tmp/repo worktree add -b codex/fix-thing /tmp/wt origin/main && "
        "codex exec -C /tmp/wt --model gpt-test --sandbox workspace-write 'fix thing'"
    )
    values = {
        "id": "proc_123",
        "task_id": "codex_fix_thing",
        "session_key": "session-1",
        "pid": 1234,
        "command": command,
        "cwd": "/tmp/repo",
        "started_at": 10.0,
        "exit_code": 0,
        "output_buffer": "0123456789abcdef",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_learning_config_defaults_auto_promote():
    cfg = cl.load_learning_config({})

    assert cfg.enabled is False
    assert cfg.harvest_launches is True
    assert cfg.auto_promote_memory is True
    assert cfg.auto_promote_skills is True
    assert cfg.min_confidence == 0.75


def test_build_learning_packet_extracts_codex_launch_metadata(monkeypatch):
    monkeypatch.setattr(cl, "_resolve_repo_root", lambda _cwd: "/tmp/repo")
    packet = cl.build_learning_packet(
        _session(),
        {"codex_cockpit": {"context_helper": {"max_output_chars": 6}}},
    )

    assert packet["id"] == "pkt_proc_123"
    assert packet["process_id"] == "proc_123"
    assert packet["repo"] == "/tmp/repo"
    assert packet["branch"] == "codex/fix-thing"
    assert packet["worktree"] == "/tmp/wt"
    assert packet["exit_code"] == 0
    assert packet["output_tail"] == "abcdef"
    assert packet["cockpit_status"]["context_helper_enabled"] is False


def test_review_packet_links_new_pending_memory_write(monkeypatch):
    from tools import write_approval as wa

    packet = cl.build_learning_packet(
        _session(),
        {"codex_cockpit": {"context_helper": {"enabled": True}}},
    )
    cl.save_packet(packet)

    def fake_reviewer(_packet, _learning):
        wa.stage_write(
            wa.MEMORY,
            {"action": "add", "target": "memory", "content": "Codex learned x"},
            summary="add to memory: Codex learned x",
            origin="codex_learning",
        )

    monkeypatch.setattr(cl, "_run_hermes_reviewer", fake_reviewer)

    reviewed = cl.review_packet(
        packet["id"],
        {
            "codex_cockpit": {
                "context_helper": {
                    "auto_promote_memory": False,
                    "auto_promote_skills": False,
                }
            }
        },
    )
    proposals = cl.list_proposals()

    assert reviewed["status"] == "staged"
    assert len(reviewed["proposal_ids"]) == 1
    assert len(proposals) == 1
    assert proposals[0]["subsystem"] == "memory"
    assert proposals[0]["approval_state"] == "pending"
    assert proposals[0]["repo"] == packet["repo"]


def test_review_packet_auto_promotes_memory_by_default(monkeypatch):
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore

    packet = cl.build_learning_packet(_session(id="proc_mem"), {})
    cl.save_packet(packet)

    def fake_reviewer(_packet, _learning):
        wa.stage_write(
            wa.MEMORY,
            {"action": "add", "target": "memory", "content": "Codex auto memory"},
            summary="add to memory: Codex auto memory",
            origin="codex_learning",
        )

    monkeypatch.setattr(cl, "_run_hermes_reviewer", fake_reviewer)

    reviewed = cl.review_packet(packet["id"], {"codex_cockpit": {"context_helper": {}}})
    proposals = cl.list_proposals()

    assert reviewed["status"] == "applied"
    assert len(proposals) == 1
    assert proposals[0]["subsystem"] == wa.MEMORY
    assert proposals[0]["approval_state"] == "applied"
    assert wa.pending_count(wa.MEMORY) == 0
    store = MemoryStore()
    store.load_from_disk()
    assert "Codex auto memory" in store.memory_entries


def test_review_packet_auto_promotes_skill_and_seeds_curator(monkeypatch):
    from tools import write_approval as wa
    from tools import skill_usage

    skill_content = (
        "---\n"
        "name: codex-auto-skill\n"
        "description: Codex auto skill\n"
        "version: 1.0.0\n"
        "---\n"
        "# Codex Auto Skill\n"
    )
    packet = cl.build_learning_packet(_session(id="proc_skill"), {})
    cl.save_packet(packet)

    def fake_reviewer(_packet, _learning):
        wa.stage_write(
            wa.SKILLS,
            {
                "action": "create",
                "name": "codex-auto-skill",
                "content": skill_content,
            },
            summary="create skill: codex-auto-skill",
            origin="codex_learning",
        )

    monkeypatch.setattr(cl, "_run_hermes_reviewer", fake_reviewer)

    reviewed = cl.review_packet(packet["id"], {"codex_cockpit": {"context_helper": {}}})
    proposals = cl.list_proposals()

    assert reviewed["status"] == "applied"
    assert proposals[0]["subsystem"] == wa.SKILLS
    assert proposals[0]["approval_state"] == "applied"
    assert wa.pending_count(wa.SKILLS) == 0
    assert skill_usage.get_record("codex-auto-skill")["created_by"] == "agent"
    assert "codex-auto-skill" in skill_usage.list_agent_created_skill_names()


def test_apply_and_discard_learning_proposals(monkeypatch):
    from tools import write_approval as wa

    memory_rec = wa.stage_write(
        wa.MEMORY,
        {"action": "add", "target": "memory", "content": "Apply me"},
        summary="add to memory: Apply me",
        origin="codex_learning",
    )
    memory_proposal = {
        "id": "learn_apply",
        "packet_id": "pkt_proc",
        "process_id": "proc",
        "subsystem": wa.MEMORY,
        "pending_id": memory_rec["id"],
        "status": "staged",
        "approval_state": "pending",
        "confidence": 1.0,
        "summary": memory_rec["summary"],
        "repo": "/tmp/repo",
        "branch": "codex/test",
        "source_command": "codex exec",
        "created_at": 1.0,
        "updated_at": 1.0,
    }
    cl.save_proposal(memory_proposal)

    out = cl.apply_learning("learn_apply", {"codex_cockpit": {"context_helper": {}}})

    assert "Applied 1" in out
    assert wa.get_pending(wa.MEMORY, memory_rec["id"]) is None
    assert cl.load_proposal("learn_apply")["status"] == "applied"

    discard_rec = wa.stage_write(
        wa.MEMORY,
        {"action": "add", "target": "memory", "content": "Discard me"},
        summary="add to memory: Discard me",
        origin="codex_learning",
    )
    discard_proposal = dict(memory_proposal)
    discard_proposal.update(
        {
            "id": "learn_discard",
            "pending_id": discard_rec["id"],
            "status": "staged",
            "approval_state": "pending",
        }
    )
    cl.save_proposal(discard_proposal)

    out = cl.discard_learning("learn_discard")

    assert "Discarded 1" in out
    assert wa.get_pending(wa.MEMORY, discard_rec["id"]) is None
    assert cl.load_proposal("learn_discard")["status"] == "discarded"


def test_handle_process_completed_honors_enabled_flag(monkeypatch):
    started = []
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"codex_cockpit": {"context_helper": {"enabled": True}}},
    )
    monkeypatch.setattr(cl, "start_review_thread", lambda packet_id, cfg: started.append(packet_id))
    monkeypatch.setattr(cl, "_resolve_repo_root", lambda _cwd: "/tmp/repo")

    packet = cl.handle_process_completed(_session())

    assert packet is not None
    assert packet["id"] == "pkt_proc_123"
    assert started == ["pkt_proc_123"]


def test_render_pending_lists_control_commands():
    from tools import write_approval as wa

    pending = wa.stage_write(
        wa.MEMORY,
        {"action": "add", "target": "memory", "content": "Pending"},
        summary="add to memory: Pending",
        origin="codex_learning",
    )
    cl.save_proposal(
        {
            "id": "learn_pending",
            "packet_id": "pkt_proc",
            "process_id": "proc",
            "subsystem": wa.MEMORY,
            "pending_id": pending["id"],
            "status": "staged",
            "approval_state": "pending",
            "confidence": 1.0,
            "summary": pending["summary"],
            "repo": "/tmp/repo",
            "branch": "codex/test",
            "source_command": "codex exec -C /tmp/wt test",
            "created_at": 1.0,
            "updated_at": 1.0,
        }
    )

    rendered = cl.render_learn_pending()

    assert "learn_pending" in rendered
    assert "/codex learn apply <id|all>" in rendered
    assert "codex/test" in rendered
