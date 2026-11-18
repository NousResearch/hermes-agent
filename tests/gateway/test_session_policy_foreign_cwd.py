"""A stored receipt's cwd is a record of the machine that created it, so a foreign spelling restores.

Sessions carried from a Windows install keep ``C:\\Users\\<user>\\...`` as their frozen cwd. That
spelling is not ``Path.is_absolute()`` on POSIX, so ``restore_policy`` rejected the whole (otherwise
well-formed) policy and every control that reloads it answered ``storage_unavailable`` — a storage
failure for a session whose store was healthy (#154192). The turn still refuses, and names the
folder, through ``workspace_ready``/``workspace_unavailable``: only the receipt validation was wrong.
"""
import json
from dataclasses import asdict, replace

import pytest

from gateway.session_policy import build_policy, policy_scope, restore_policy, workspace_ready
from persistence.core.errors import AuthorityRefusal

WINDOWS_CWD = r"C:\Users\markl\AppData\Local\hermes"


def _stored_receipt(tmp_path, cwd):
    """A policy as an install stores it, with its frozen terminal scope in step with ``cwd``."""
    policy = build_policy({"source": "gui", "cwd": str(tmp_path), "model": "fixture"},
                          {"platform_toolsets": {"gui": ["terminal"]}})
    terminal = json.loads(policy.terminal_json)
    terminal["TERMINAL_CWD"] = cwd
    return asdict(replace(policy, cwd=cwd, terminal_json=json.dumps(terminal)))


@pytest.mark.platforms("linux", "macos")
def test_a_windows_cwd_receipt_restores_and_only_its_turn_refuses(tmp_path):
    policy = restore_policy(_stored_receipt(tmp_path, WINDOWS_CWD))
    assert (policy.cwd, policy.model) == (WINDOWS_CWD, "fixture")
    assert workspace_ready(policy) is False
    with pytest.raises(AuthorityRefusal, match="workspace_unavailable"):
        with policy_scope(policy):
            pass


# ``\Users\markl`` is rooted but has no drive: ``ntpath.isabs`` accepted it before Python 3.13.
@pytest.mark.parametrize("cwd", ["relative/path", "C:hermes", r"\Users\markl"])
def test_a_non_absolute_cwd_is_still_refused(tmp_path, cwd):
    with pytest.raises(AuthorityRefusal, match="storage_unavailable") as caught:
        restore_policy(_stored_receipt(tmp_path, cwd))
    assert caught.value.field == "cwd"
