"""Tests for ``hermes approvals test`` — dry-run approval verdict CLI.

The tester must compose the REAL runtime evaluators from ``tools.approval``
(detect_hardline_command, _match_user_deny_rule, detect_dangerous_command,
the container-skip gate, and the same ``_command_detection_variants``
normalization/de-obfuscation path) — never reimplement them. It is strictly
read-only: nothing is executed, no prompt fires, nothing is persisted.

Exit-code contract (script-friendly, documented in the CLI help):
    0 = allow, 2 = ask-approval, 3 = deny (hardline / user deny rule).
"""

import argparse
import json

import pytest

import tools.approval as A
from tools import approval_detection
from tools import approval_prompt
from tools import approval_context
from hermes_cli import approvals_test as at


def _args(command, env_type="local", as_json=False):
    return argparse.Namespace(
        command_words=list(command) if isinstance(command, (list, tuple)) else [command],
        env_type=env_type,
        json=as_json,
    )


@pytest.fixture
def isolated_approvals(monkeypatch):
    """Isolate the evaluators from the dev machine's real config/state."""
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"mode": "manual"})
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(A, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(A, "load_permanent_allowlist", lambda: set())
    saved = set(A._permanent_approved)
    A._permanent_approved.clear()
    # The tester must NEVER prompt or persist — make any attempt explode.
    def _boom(*_a, **_kw):  # pragma: no cover - failure path
        raise AssertionError("read-only tester touched a prompt/persistence path")
    monkeypatch.setattr(A, "prompt_dangerous_approval", _boom)
    monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval", _boom)
    monkeypatch.setattr(A, "save_permanent_allowlist", _boom)
    monkeypatch.setattr(A, "submit_pending", _boom, raising=False)
    yield A
    A._permanent_approved.clear()
    A._permanent_approved.update(saved)


class TestVerdicts:
    def test_benign_command_allows_with_exit_0(self, isolated_approvals, capsys):
        rc = at.approvals_test_command(_args(["ls", "-la"]))
        out = capsys.readouterr().out
        assert rc == 0
        assert "allow" in out

    def test_hardline_command_denies_with_rule_name(self, isolated_approvals, capsys):
        rc = at.approvals_test_command(_args(["sudo", "re" + "boot"]))
        out = capsys.readouterr().out
        assert rc == 3
        assert "hardline-deny" in out
        assert "system shutdown/reboot" in out

    def test_dangerous_command_asks_with_exit_2(self, isolated_approvals, capsys):
        rc = at.approvals_test_command(_args(["rm", "-rf", "~/project/build"]))
        out = capsys.readouterr().out
        assert rc == 2
        assert "ask-approval" in out
        assert "recursive delete" in out

    def test_user_deny_rule_from_config_honored(self, isolated_approvals, capsys,
                                                monkeypatch):
        monkeypatch.setattr(
            approval_context, "_get_approval_config",
            lambda: {"mode": "manual", "deny": ["git push *"]})
        rc = at.approvals_test_command(_args(["git", "push", "origin", "main"]))
        out = capsys.readouterr().out
        assert rc == 3
        assert "user-deny" in out
        assert "git push *" in out

    def test_container_env_type_skips_guards_like_runtime(self, isolated_approvals,
                                                          capsys):
        # Mirrors check_all_command_guards: isolated docker skips BEFORE the
        # hardline floor, so even a catastrophic command reports allow.
        rc = at.approvals_test_command(_args(["rm", "-rf", "/"], env_type="docker"))
        out = capsys.readouterr().out
        assert rc == 0
        assert "allow" in out
        assert "container" in out or "isolated" in out

    def test_mode_off_bypasses_dangerous_but_not_hardline(self, isolated_approvals,
                                                          capsys, monkeypatch):
        monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"mode": "off"})
        rc = at.approvals_test_command(_args(["rm", "-rf", "~/project/build"]))
        out = capsys.readouterr().out
        assert rc == 0
        assert "off" in out
        rc = at.approvals_test_command(_args(["sudo", "re" + "boot"]))
        assert rc == 3


class TestNormalizationParity:
    """The tester must run the same de-obfuscation path as the runtime."""

    def test_obfuscated_command_matches_plain_verdict(self, isolated_approvals,
                                                      capsys):
        rc_plain = at.approvals_test_command(_args(["rm", "-rf", "/"]))
        out_plain = capsys.readouterr().out
        rc_obf = at.approvals_test_command(_args(["r\\m", "-rf", "/"]))
        out_obf = capsys.readouterr().out
        assert rc_plain == rc_obf == 3
        assert "recursive delete of root filesystem" in out_plain
        assert "recursive delete of root filesystem" in out_obf
        # The trace must show the de-obfuscated form the runtime evaluated.
        assert "rm -rf /" in out_obf

    def test_normalized_trace_shown_when_command_normalizes(self,
                                                            isolated_approvals,
                                                            capsys):
        rc = at.approvals_test_command(_args(['git', 'st""atus']))
        out = capsys.readouterr().out
        assert rc == 0
        assert "git status" in out


class TestReadOnly:
    def test_nothing_executed(self, isolated_approvals, capsys, tmp_path):
        sentinel = tmp_path / "must_not_exist"
        rc = at.approvals_test_command(_args(["touch", str(sentinel)]))
        capsys.readouterr()
        assert rc == 0
        assert not sentinel.exists()

    def test_dangerous_command_never_prompts_or_persists(self, isolated_approvals,
                                                         capsys):
        # isolated_approvals wires prompt/persistence to AssertionError; a
        # dangerous command must complete without touching either.
        rc = at.approvals_test_command(_args(["rm", "-rf", "~/project/build"]))
        capsys.readouterr()
        assert rc == 2


class TestOutputAndWiring:
    def test_json_output_is_machine_readable(self, isolated_approvals, capsys):
        rc = at.approvals_test_command(_args(["sudo", "re" + "boot"], as_json=True))
        payload = json.loads(capsys.readouterr().out)
        assert rc == 3
        assert payload["verdict"] == "hardline-deny"
        assert payload["exit_code"] == 3
        assert payload["rule"]
        assert payload["command"] == "sudo re" + "boot"
        assert isinstance(payload["normalized_variants"], list)

    def test_empty_command_is_usage_error(self, isolated_approvals, capsys):
        rc = at.approvals_test_command(_args([]))
        assert rc == 1

    def test_dispatcher_routes_test_subcommand(self, isolated_approvals, capsys):
        from hermes_cli.approvals_suggest import approvals_command
        args = _args(["ls"])
        args.approvals_command = "test"
        rc = approvals_command(args)
        out = capsys.readouterr().out
        assert rc == 0
        assert "allow" in out

    def test_parser_wires_test_subcommand(self, isolated_approvals, capsys):
        from hermes_cli.subcommands.approvals import build_approvals_parser
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        sentinel = []
        build_approvals_parser(sub, cmd_approvals=lambda a: sentinel.append(a) or 0)
        args = parser.parse_args(
            ["approvals", "test", "--env-type", "ssh", "--", "ls", "-la"])
        assert args.approvals_command == "test"
        assert args.env_type == "ssh"
        # argparse REMAINDER keeps the leading "--"; the handler strips it.
        # dest is command_words (NOT command) so main.py's startup path can
        # keep reading args.command as the top-level subcommand name.
        assert args.command_words == ["--", "ls", "-la"]
        # The subparser must NOT claim the "command" dest — main.py's startup
        # path reads args.command as the top-level subcommand name.
        assert getattr(args, "command", None) != ["--", "ls", "-la"]
        args.func(args)
        assert sentinel

    def test_leading_separator_stripped_from_command(self, isolated_approvals,
                                                     capsys):
        rc = at.approvals_test_command(_args(["--", "ls", "-la"]))
        out = capsys.readouterr().out
        assert rc == 0
        assert "ls -la" in out
        assert "-- ls" not in out


class TestPermanentAllowlistConsentKinds:
    """Step 6 must consent exactly the way the RUNTIME does.

    ``command_allowlist`` holds two kinds of entry: manual command TEXT (possibly a
    shell-style glob such as ``podman *``) and the PATTERN KEY every ``[a]lways`` choice
    persists. The runtime consents to a flagged command by key —
    ``approval.is_approved(session_key, pattern_key)`` in the human gate,
    ``approval._is_permanently_approved(pattern_key)`` under cron/-q deny mode — so a
    text/glob-only check renders ``ask-approval`` for a standing approval the runtime
    would auto-approve. Both kinds must keep working.
    """

    FORCE_PUSH = "git push --force-with-lease origin my-own-branch"

    @staticmethod
    def _key(command):
        is_dangerous, pattern_key, _ = approval_detection.detect_dangerous_command(command)
        assert is_dangerous, f"{command!r} is expected to be a flagged pattern"
        return pattern_key

    def test_pattern_key_entry_allows_the_dangerous_command(self, isolated_approvals,
                                                            capsys):
        key = self._key(self.FORCE_PUSH)
        A._permanent_approved.add(key)
        rc = at.approvals_test_command(_args(self.FORCE_PUSH.split()))
        out = capsys.readouterr().out
        assert rc == 0
        assert "allow" in out
        assert key in out

    def test_pattern_key_absent_still_asks(self, isolated_approvals, capsys):
        rc = at.approvals_test_command(_args(self.FORCE_PUSH.split()))
        out = capsys.readouterr().out
        assert rc == 2
        assert "ask-approval" in out

    def test_a_different_pattern_key_does_not_leak_into_the_verdict(
            self, isolated_approvals, capsys):
        A._permanent_approved.add("recursive delete")
        rc = at.approvals_test_command(_args(self.FORCE_PUSH.split()))
        assert rc == 2

    def test_text_glob_entry_allows_a_matching_command(self, isolated_approvals,
                                                       capsys):
        A._permanent_approved.add("podman *")
        rc = at.approvals_test_command(_args(["podman", "run", "--rm", "alpine", "true"]))
        out = capsys.readouterr().out
        assert rc == 0
        assert "command_allowlist" in out

    def test_text_glob_entry_does_not_allow_a_non_matching_command(
            self, isolated_approvals, capsys):
        A._permanent_approved.add("podman *")
        rc = at.approvals_test_command(_args(["rm", "-rf", "~/project/build"]))
        assert rc == 2
