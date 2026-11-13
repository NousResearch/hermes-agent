"""Regression for #155007: approval preserves durable curator deletion semantics."""

import json

import pytest


@pytest.fixture
def skill_home(tmp_path, monkeypatch):
    from hermes_cli.config import load_config, save_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = load_config()
    config.setdefault("skills", {})["write_approval"] = True
    save_config(config)
    return tmp_path


def _package(home, name):
    from tools.skill_usage import record_created

    root = home / "skills" / name
    root.mkdir(parents=True)
    files = {
        "SKILL.md": (f"---\nname: {name}\ndescription: Use when checking a sample task.\n"
                     "---\n\nFollow the original procedure.\n").encode(),
        "references/guide.md": b"Keep the complete support file.\n",
    }
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    record_created(name, agent_created=True)
    return root, files


def _manage(payload):
    from tools.skill_manager_tool import registry

    return json.loads(registry.dispatch("skill_manage", payload))


def _ledger(home):
    path = home / "skills" / ".curator_ledger.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _fail_archive(*args, **kwargs):
    raise OSError("sample archive failure")


@pytest.mark.parametrize("origin,shape,failure", [
    ("background_review", "flat", None),
    ("background_review", "operations", None),
    ("foreground", "flat", None),
    ("foreground", "operations", None),
    (None, "flat", None),
    (None, "operations", None),
    ("background_review", "flat", "pinned"),
    ("background_review", "flat", "locked"),
    ("background_review", "flat", "ownership"),
    ("background_review", "operations", "exception"),
])
def test_approved_delete_preserves_origin_and_restores_scope(skill_home, monkeypatch, origin, shape, failure):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import skill_usage, write_approval as wa
    from tools.skill_manager_tool import apply_skill_pending
    from tools.skill_provenance import reset_current_write_origin, set_current_write_origin

    source, files = _package(skill_home, "sample-leaf")
    umbrella, umbrella_files = _package(skill_home, "sample-umbrella")
    token = set_current_write_origin(origin or "foreground")
    try:
        staged = _manage({"action": "delete", "name": source.name, "absorbed_into": umbrella.name})
    finally:
        reset_current_write_origin(token)
    assert staged["staged"] and source.exists(), staged
    pending_id = staged["pending_id"]
    record_path = wa._pending_path(wa.SKILLS, pending_id)
    record = wa.get_pending(wa.SKILLS, pending_id)
    assert record["origin"] == (origin or "foreground")
    # Older queues may omit origin; operations[] is also a supported persisted replay shape.
    if origin is None:
        record.pop("origin")
    if shape == "operations":
        record["payload"] = {"operations": [record["payload"]]}
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with monkeypatch.context() as patch:
        setup_failure = {
            "pinned": lambda: skill_usage.set_pinned(source.name, True),
            "locked": lambda: skill_usage.set_locked(source.name, True),
            "ownership": lambda: skill_usage.record_created(source.name, agent_created=False),
            "exception": lambda: patch.setattr(skill_usage, "archive_skill", _fail_archive),
        }
        if failure:
            setup_failure[failure]()
        message = handle_pending_subcommand(wa.SKILLS, ["approve", pending_id])

    archived = skill_home / "skills" / ".archive" / source.name
    rows = [row for row in _ledger(skill_home) if row["skill"] == source.name]
    if failure:
        assert "Approved 0" in message, message
        assert wa.get_pending(wa.SKILLS, pending_id) is not None
        assert source.exists() and not archived.exists()
        assert not rows
        for rel, data in files.items():
            assert (source / rel).read_bytes() == data
    else:
        assert "Approved 1" in message, message
        assert wa.get_pending(wa.SKILLS, pending_id) is None
        assert not source.exists()
        curator = origin == "background_review"
        assert archived.exists() == curator
        deletion = next(row for row in rows if row["action"] == "delete")
        assert deletion["actor"] == ("curator" if curator else "agent")
        assert deletion["evidence"] == {"absorbed_into": umbrella.name, "archived": curator}
        if curator:
            assert all(row["actor"] == "curator" for row in rows)
            for rel, data in files.items():
                assert (archived / rel).read_bytes() == data
            ok, message = skill_usage.restore_skill(source.name)
            assert ok, message
            for rel, data in files.items():
                assert (source / rel).read_bytes() == data
            assert not archived.exists()
    for rel, data in umbrella_files.items():
        assert (umbrella / rel).read_bytes() == data

    # The next direct caller has no proposal metadata, even after failed approval.
    unrelated, _ = _package(skill_home, "sample-foreground")
    result = json.loads(apply_skill_pending({"action": "delete", "name": unrelated.name}))
    assert result["success"], result
    assert not unrelated.exists()
    assert not (archived.parent / unrelated.name).exists()
    assert _ledger(skill_home)[-1]["actor"] == "agent"
    assert _ledger(skill_home)[-1]["evidence"]["archived"] is False
    # Gate bypass is scoped too.
    assert _manage({"action": "delete", "name": umbrella.name})["staged"]


@pytest.mark.parametrize("shape", ["flat", "operations"])
def test_approved_patch_keeps_audit_origin_without_review_read_marks(skill_home, shape):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import skill_ledger, write_approval as wa
    from tools.skill_manager_guards import _reset_background_review_read_marks
    from tools.skill_provenance import (
        is_background_review, reset_current_write_origin, set_current_write_origin,
    )
    from tools.skills_tool import skill_view

    source, files = _package(skill_home, "sample-patch")
    op = {"action": "patch", "name": source.name, "old_string": "original", "new_string": "updated"}
    token = set_current_write_origin("background_review")
    try:
        assert json.loads(skill_view(source.name))["success"]
        staged = _manage(op if shape == "flat" else {"operations": [op]})
    finally:
        reset_current_write_origin(token)
        _reset_background_review_read_marks()
    assert staged["staged"], staged
    # Approval must override the approving caller's audit actor, then restore it.
    actor_token = skill_ledger.set_ledger_actor("user")
    try:
        message = handle_pending_subcommand(wa.SKILLS, ["approve", staged["pending_id"]])
        assert "Approved 1" in message, message
        assert skill_ledger.derive_actor() == "user"
    finally:
        skill_ledger.reset_ledger_actor(actor_token)
    assert not is_background_review()
    assert wa.get_pending(wa.SKILLS, staged["pending_id"]) is None
    assert (source / "SKILL.md").read_bytes() == files["SKILL.md"].replace(b"original", b"updated")
    assert (source / "references/guide.md").read_bytes() == files["references/guide.md"]
    assert _ledger(skill_home)[-1]["actor"] == "curator"
    assert skill_ledger.derive_actor() == "agent"

    # An actual autonomous write still requires a read in its own review turn.
    config = load_config()
    config["skills"]["write_approval"] = False
    save_config(config)
    token = set_current_write_origin("background_review")
    try:
        result = _manage({**op, "old_string": "updated", "new_string": "unread"})
    finally:
        reset_current_write_origin(token)
    assert result["success"] is False and result["_read_before_write_required"] is True
    assert b"updated" in (source / "SKILL.md").read_bytes()
