"""A stored receipt's cwd is a record of the machine that created it, so a foreign spelling restores.

Sessions carried from a Windows install keep ``C:\\Users\\<user>\\...`` as their frozen cwd. That
spelling is not ``Path.is_absolute()`` on POSIX, so ``restore_policy`` rejected the whole (otherwise
well-formed) policy and every control that reloads it answered ``storage_unavailable`` — a storage
failure for a session whose store was healthy. The turn still refuses, and names the folder, through
``workspace_ready``/``workspace_unavailable``: only the receipt validation was wrong. Same pairing as
``test_session_policy_workspace_readiness``: readable and movable, the turn is what refuses.
"""
import json
from dataclasses import asdict, replace

import pytest

from gateway.session_policy import build_policy, policy_scope, restore_policy, workspace_ready

WINDOWS_CWD = r"C:\Users\markl\AppData\Local\hermes"


def _stored_receipt(tmp_path, cwd):
    """A policy as an install stores it, with its frozen terminal scope in step with ``cwd``."""
    policy = build_policy({"source": "gui", "cwd": str(tmp_path), "model": "fixture"},
                          {"platform_toolsets": {"gui": ["terminal"]}})
    terminal = json.loads(policy.terminal_json)
    terminal["TERMINAL_CWD"] = cwd
    return asdict(replace(policy, cwd=cwd, terminal_json=json.dumps(terminal)))


def test_a_windows_cwd_receipt_restores_on_posix(tmp_path):
    restored = restore_policy(_stored_receipt(tmp_path, WINDOWS_CWD))
    assert restored.cwd == WINDOWS_CWD
    assert restored.model == "fixture"


def test_the_foreign_workspace_only_refuses_the_turn(tmp_path):
    from hermes_state_runtime import RuntimeStoreError

    policy = restore_policy(_stored_receipt(tmp_path, WINDOWS_CWD))
    assert workspace_ready(policy) is False
    with pytest.raises(RuntimeStoreError, match="workspace_unavailable"):
        with policy_scope(policy):
            pass


@pytest.mark.parametrize("cwd", ["relative/path", "C:hermes"])
def test_a_non_absolute_cwd_is_still_refused(tmp_path, cwd):
    from hermes_state_runtime import RuntimeStoreError

    with pytest.raises(RuntimeStoreError, match="storage_unavailable"):
        restore_policy(_stored_receipt(tmp_path, cwd))
