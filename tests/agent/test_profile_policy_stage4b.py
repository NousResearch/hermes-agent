import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agent.profile_policy import (
    ProfilePolicyError,
    check_inbound_event,
    check_outbound,
    check_tool_call,
)
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _setup_hefa(monkeypatch, tmp_path, *, allowed_chat="123"):
    root = tmp_path / "hermes-root"
    profile = root / "profiles" / "hephaestus-h"
    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    profile.mkdir(parents=True)
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HERMES_PROFILE_GUARD_EXPECTED", "hephaestus-h")
    (profile / "policy.yaml").write_text(
        f"""
identity: hephaestus-h
allowed_write_roots:
  - {workspace}
forbidden_basenames:
  - .env
  - .session
forbidden_path_substrings:
  - token
  - secret
  - key
state_dir: {state}
telegram:
  dm_only: true
  allowed_chat_ids:
    - "{allowed_chat}"
  allow_bots: false
loop:
  idempotency_ttl_seconds: 86400
""".strip()
    )
    return profile, workspace, state


def _event(*, chat_id="123", chat_type="dm", is_bot=False, update_id=1):
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=str(chat_id),
            chat_type=chat_type,
            user_id="u1",
            user_name="Vitaliy",
            is_bot=is_bot,
        ),
        message_id=str(update_id),
        platform_update_id=update_id,
    )


def test_profile_policy_blocks_forbidden_path(monkeypatch, tmp_path):
    _profile, workspace, _state = _setup_hefa(monkeypatch, tmp_path)
    with pytest.raises(ProfilePolicyError):
        check_tool_call("write_file", {"path": str(workspace / ".env"), "content": "x"})
    with pytest.raises(ProfilePolicyError):
        check_tool_call("write_file", {"path": str(tmp_path / "outside.txt"), "content": "x"})


def test_profile_policy_blocks_token_leak_text_and_media(monkeypatch, tmp_path):
    _profile, _workspace, _state = _setup_hefa(monkeypatch, tmp_path)
    token = "123456789:" + "A" * 35
    with pytest.raises(ProfilePolicyError):
        check_outbound("telegram", "123", f"leak {token}", kind="text")
    high_entropy = "B6jT9qN2vX8mP5rL0sY4wC7zK1aD3fG6hJ9kM2nQ5pR8tV1x"
    with pytest.raises(ProfilePolicyError):
        check_outbound("telegram", "123", f"leak {high_entropy}", kind="text")
    media = tmp_path / "report.txt"
    media.write_text(f"secret {token}")
    with pytest.raises(ProfilePolicyError):
        check_outbound("telegram", "123", str(media), kind="media")


def test_profile_policy_accepts_allowed_dm(monkeypatch, tmp_path):
    _setup_hefa(monkeypatch, tmp_path)
    check_inbound_event(_event(chat_id="123", chat_type="dm", update_id=11))


def test_profile_policy_drops_group(monkeypatch, tmp_path):
    _setup_hefa(monkeypatch, tmp_path)
    with pytest.raises(ProfilePolicyError):
        check_inbound_event(_event(chat_id="123", chat_type="group", update_id=12))


def test_profile_policy_drops_bot_sender(monkeypatch, tmp_path):
    _setup_hefa(monkeypatch, tmp_path)
    with pytest.raises(ProfilePolicyError):
        check_inbound_event(_event(chat_id="123", chat_type="dm", is_bot=True, update_id=13))


def test_profile_policy_concurrent_dm_idempotency(monkeypatch, tmp_path):
    _setup_hefa(monkeypatch, tmp_path)

    def run_once():
        try:
            check_inbound_event(_event(chat_id="123", chat_type="dm", update_id=77))
            return "accepted"
        except ProfilePolicyError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run_once(), range(2)))
    assert results.count("accepted") == 1
    assert results.count("blocked") == 1


def test_profile_policy_blocks_patch_payload_outside_allowed_roots(monkeypatch, tmp_path):
    _profile, workspace, _state = _setup_hefa(monkeypatch, tmp_path)
    allowed_patch = f"""*** Begin Patch
*** Update File: {workspace / 'ok.txt'}
@@
+ok
*** End Patch
"""
    check_tool_call("patch", {"mode": "patch", "patch": allowed_patch})

    outside_patch = f"""*** Begin Patch
*** Update File: {tmp_path / 'outside.txt'}
@@
+bad
*** End Patch
"""
    with pytest.raises(ProfilePolicyError):
        check_tool_call("patch", {"mode": "patch", "patch": outside_patch})


def test_profile_policy_blocks_terminal_write_destinations(monkeypatch, tmp_path):
    _profile, workspace, _state = _setup_hefa(monkeypatch, tmp_path)
    allowed = workspace / "allowed.txt"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside = outside_dir / "outside.txt"
    check_tool_call("terminal", {"command": f"cp {allowed} {workspace / 'copy.txt'}"})
    check_tool_call("terminal", {"command": "printf ok | tee allowed.txt", "workdir": str(workspace)})
    check_tool_call("terminal", {"command": "printf ok > 'space name.txt'", "workdir": str(workspace)})

    blocked_commands = [
        ({"command": f"cp {allowed} {outside}"}),
        ({"command": f"mv {allowed} {outside}"}),
        ({"command": f"printf ok | tee {outside}"}),
        ({"command": f"printf ok > {outside}"}),
        ({"command": f"printf ok >> {outside}"}),
        ({"command": "printf ok > outside.txt", "workdir": str(outside_dir)}),
        ({"command": "printf ok >> outside.txt", "workdir": str(outside_dir)}),
        ({"command": "printf ok | tee outside.txt", "workdir": str(outside_dir)}),
        ({"command": "cp allowed.txt outside.txt", "workdir": str(outside_dir)}),
        ({"command": "mv allowed.txt outside.txt", "workdir": str(outside_dir)}),
        ({"command": "printf ok > 'space name.txt'", "workdir": str(outside_dir)}),
        ({"command": "touch outside.txt", "workdir": str(outside_dir)}),
        ({"command": "mkdir outside-dir", "workdir": str(outside_dir)}),
        ({"command": "install src.txt outside.txt", "workdir": str(outside_dir)}),
        ({"command": "install -t . src.txt", "workdir": str(outside_dir)}),
        ({"command": "cp --target-directory=. src.txt", "workdir": str(outside_dir)}),
        ({"command": "mv -t . src.txt", "workdir": str(outside_dir)}),
        ({"command": "rsync src.txt outside.txt", "workdir": str(outside_dir)}),
        ({"command": "dd if=/dev/zero of=outside.img", "workdir": str(outside_dir)}),
    ]
    for args in blocked_commands:
        with pytest.raises(ProfilePolicyError):
            check_tool_call("terminal", args)


def test_profile_policy_blocks_unparsed_terminal_write_programs(monkeypatch, tmp_path):
    _profile, workspace, _state = _setup_hefa(monkeypatch, tmp_path)
    check_tool_call("terminal", {"command": "git status --short", "workdir": str(workspace)})
    check_tool_call("terminal", {"command": "python -m pytest tests/agent/test_profile_policy_stage4b.py", "workdir": str(workspace)})

    blocked = [
        "python -c 'open(\"outside.txt\", \"w\").write(\"x\")'",
        "bash -lc 'touch outside.txt'",
        "node -e 'require(\"fs\").writeFileSync(\"outside.txt\", \"x\")'",
    ]
    for command in blocked:
        with pytest.raises(ProfilePolicyError):
            check_tool_call("terminal", {"command": command, "workdir": str(workspace)})


def test_profile_policy_blocks_nested_terminal_execution(monkeypatch, tmp_path):
    _profile, workspace, _state = _setup_hefa(monkeypatch, tmp_path)
    blocked = [
        "printf $(touch /tmp/outside) > allowed.txt",
        "echo `touch /tmp/outside` > allowed.txt",
        "find . -exec touch /tmp/outside ;",
        "find . -delete",
        "find . -fprint0 /tmp/outside",
        "find . -ok touch /tmp/outside ;",
        "find . -okdir touch /tmp/outside ;",
        "cat $(touch /tmp/outside)",
        "git status `touch /tmp/outside`",
    ]
    for command in blocked:
        with pytest.raises(ProfilePolicyError):
            check_tool_call("terminal", {"command": command, "workdir": str(workspace)})
