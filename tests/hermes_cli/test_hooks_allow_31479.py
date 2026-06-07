"""Regression tests for #31479 — ``hermes hooks allow``.

Service accounts and headless deployments have no TTY for the
first-use consent prompt and don't want the blanket
``hooks_auto_accept: true`` switch.  These tests pin the new
``allow`` subcommand so the documented non-interactive workflow
keeps working and so future refactors cannot quietly drop the
public ``approve()`` API the CLI builds on.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import shell_hooks
from hermes_cli import hooks as hooks_cli


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_ACCEPT_HOOKS", raising=False)
    shell_hooks.reset_for_tests()
    yield
    shell_hooks.reset_for_tests()


def _hook_script(tmp_path: Path, body: str = "#!/usr/bin/env bash\nprintf '{}\\n'\n", name: str = "hook.sh") -> Path:
    p = tmp_path / name
    p.write_text(body)
    p.chmod(0o755)
    return p


def _run(sub_args: SimpleNamespace) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        hooks_cli.hooks_command(sub_args)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# agent.shell_hooks.approve — public API contract
# ---------------------------------------------------------------------------


class TestApprovePublicApi:
    """``approve`` is the seam non-CLI callers (deployment scripts,
    operators, future TUI flows) build on.  Pin its contract."""

    def test_writes_new_entry_and_returns_true(self, tmp_path):
        script = _hook_script(tmp_path)
        assert shell_hooks.approve("post_llm_call", str(script)) is True
        # Allowlist now contains exactly the documented schema.
        data = json.loads(shell_hooks.allowlist_path().read_text())
        approvals = data["approvals"]
        assert len(approvals) == 1
        entry = approvals[0]
        assert entry["event"] == "post_llm_call"
        assert entry["command"] == str(script)
        assert "approved_at" in entry
        # mtime is captured so doctor can flag drift.
        assert "script_mtime_at_approval" in entry

    def test_idempotent_repeat_returns_false(self, tmp_path):
        script = _hook_script(tmp_path)
        assert shell_hooks.approve("post_llm_call", str(script)) is True
        # Second call must NOT duplicate the entry and must report
        # "no change" so the CLI can render a useful summary.
        assert shell_hooks.approve("post_llm_call", str(script)) is False
        data = json.loads(shell_hooks.allowlist_path().read_text())
        assert len(data["approvals"]) == 1

    def test_distinct_events_get_distinct_entries(self, tmp_path):
        script = _hook_script(tmp_path)
        shell_hooks.approve("pre_tool_call", str(script))
        shell_hooks.approve("post_tool_call", str(script))
        data = json.loads(shell_hooks.allowlist_path().read_text())
        events = sorted(e["event"] for e in data["approvals"])
        assert events == ["post_tool_call", "pre_tool_call"]

    def test_byte_for_byte_command_match(self, tmp_path):
        # The runtime registration gate compares the literal command
        # string from config.yaml against the literal allowlist entry.
        # A trailing-space variant must NOT collide with the canonical
        # form, otherwise an operator could quietly approve a
        # different command than the one their config declares.
        script = _hook_script(tmp_path)
        shell_hooks.approve("post_llm_call", str(script))
        # Trailing-space variant counts as a different command.
        assert shell_hooks.approve("post_llm_call", f"{script} ") is True
        data = json.loads(shell_hooks.allowlist_path().read_text())
        assert len(data["approvals"]) == 2


# ---------------------------------------------------------------------------
# hermes hooks allow — config-driven default
# ---------------------------------------------------------------------------


class TestHooksAllowConfigDriven:
    """Without --event, the CLI walks ~/.hermes/config.yaml and
    approves every event that mentions the command."""

    def test_approves_every_configured_event(self, tmp_path):
        script = _hook_script(tmp_path)
        cfg = {
            "hooks": {
                "pre_tool_call": [{"matcher": "terminal", "command": str(script)}],
                "post_tool_call": [{"matcher": "terminal", "command": str(script)}],
                "post_llm_call": [{"command": str(script)}],
            }
        }
        with patch("hermes_cli.config.load_config", return_value=cfg):
            out = _run(SimpleNamespace(
                hooks_action="allow", command=str(script), event=None,
            ))

        # All three configured events should be allowlisted now.
        for ev in ("pre_tool_call", "post_tool_call", "post_llm_call"):
            assert shell_hooks.allowlist_entry_for(ev, str(script)) is not None
            assert ev in out

        assert "Approved 3 entry/entries" in out

    def test_idempotent_second_run_reports_no_change(self, tmp_path):
        script = _hook_script(tmp_path)
        cfg = {"hooks": {"post_llm_call": [{"command": str(script)}]}}

        with patch("hermes_cli.config.load_config", return_value=cfg):
            _run(SimpleNamespace(
                hooks_action="allow", command=str(script), event=None,
            ))
            out = _run(SimpleNamespace(
                hooks_action="allow", command=str(script), event=None,
            ))

        assert "already allowlisted" in out
        assert "Nothing to do" in out
        # Still exactly one entry — no duplicate from the rerun.
        data = json.loads(shell_hooks.allowlist_path().read_text())
        assert len(data["approvals"]) == 1

    def test_command_not_in_config_errors_out(self, tmp_path):
        # Service-account scenarios where the operator typo'd the
        # path or forgot to update config.yaml should NOT silently
        # write a dead allowlist entry.
        script = _hook_script(tmp_path)
        cfg = {"hooks": {"post_llm_call": [{"command": "/some/other/script.sh"}]}}

        with patch("hermes_cli.config.load_config", return_value=cfg):
            out = _run(SimpleNamespace(
                hooks_action="allow", command=str(script), event=None,
            ))

        assert "No hook in ~/.hermes/config.yaml" in out
        assert "--event" in out
        # Nothing was written.
        assert shell_hooks.allowlist_entry_for(
            "post_llm_call", str(script),
        ) is None


# ---------------------------------------------------------------------------
# hermes hooks allow --event — direct entry mode
# ---------------------------------------------------------------------------


class TestHooksAllowDirectEvent:
    """``--event`` writes a single entry without consulting config.yaml,
    so operators can stage allowlist + config together."""

    def test_writes_single_entry(self, tmp_path):
        script = _hook_script(tmp_path)
        with patch("hermes_cli.config.load_config", return_value={}):
            out = _run(SimpleNamespace(
                hooks_action="allow",
                command=str(script),
                event="post_llm_call",
            ))

        assert shell_hooks.allowlist_entry_for(
            "post_llm_call", str(script),
        ) is not None
        # No incidental approvals leaked.
        assert shell_hooks.allowlist_entry_for(
            "pre_tool_call", str(script),
        ) is None
        assert "Approved 1 entry/entries" in out

    def test_unknown_event_aborts_cleanly(self, tmp_path):
        script = _hook_script(tmp_path)
        with patch("hermes_cli.config.load_config", return_value={}):
            out = _run(SimpleNamespace(
                hooks_action="allow",
                command=str(script),
                event="totally_made_up_event",
            ))

        assert "Unknown event" in out
        assert "Valid events" in out
        # Nothing written.
        data = json.loads(
            shell_hooks.allowlist_path().read_text()
        ) if shell_hooks.allowlist_path().exists() else {"approvals": []}
        assert data.get("approvals") == []

    def test_approve_alias_dispatches_to_allow(self, tmp_path):
        # We intentionally accept both verbs — operators familiar with
        # the API name (``shell_hooks.approve``) often type
        # ``hermes hooks approve``.  Both must work.
        script = _hook_script(tmp_path)
        with patch("hermes_cli.config.load_config", return_value={}):
            out = _run(SimpleNamespace(
                hooks_action="approve",
                command=str(script),
                event="post_llm_call",
            ))

        assert shell_hooks.allowlist_entry_for(
            "post_llm_call", str(script),
        ) is not None
        assert "allowlisted" in out


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


class TestHooksAllowArgparseWiring:
    """The bug report quotes the literal argparse error.  Pin the
    argparse plumbing so a refactor cannot regress to it again."""

    def test_allow_is_a_recognised_subcommand(self):
        # build_top_level_parser + the rest of main()'s subparser
        # registration is too entangled to invoke directly, so we
        # exercise the public CLI surface with --help and assert the
        # expected token shows up in the choices list.  Anything
        # short of "argparse knows the word" reproduces #31479.
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "hooks", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        # Both verbs are surfaced in the help.
        assert "allow" in result.stdout
        assert "approve" in result.stdout
        # The original error listed only these — make sure the new
        # subcommand actually joined the family rather than printing
        # alongside it as documentation.
        assert "{list,ls,test,allow,approve,revoke,remove,rm,doctor}" in result.stdout
