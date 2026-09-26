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
import tools.approval_prompt as approval_prompt
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
        # One argv word keeps the command string verbatim, so '""' reaches the
        # normalizer as empty-quote syntax rather than literal characters.
        rc = at.approvals_test_command(_args(['git st""atus']))
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


class TestShellQuotingFidelity:
    """``test -- <words>`` arrives post-shell-split; reconstruction must
    restore the quoting the user's shell erased so the dry-run verdict matches
    the runtime verdict for the real command string."""

    def test_quoted_separator_stays_data(self, isolated_approvals, capsys):
        # hermes approvals test -- git commit -m "x; rm -rf /"
        rc = at.approvals_test_command(
            _args(["git", "commit", "-m", "x; rm -rf /"]))
        out = capsys.readouterr().out
        assert rc == 2
        assert "ask-approval" in out
        # Same verdict the runtime gives the real quoted string.
        assert at.evaluate_command(
            'git commit -m "x; rm -rf /"')["exit_code"] == rc

    def test_quoted_pipe_stays_data(self, isolated_approvals, capsys):
        # hermes approvals test -- echo "a | reboot"
        rc = at.approvals_test_command(_args(["echo", "a | reboot"]))
        capsys.readouterr()
        assert rc == 0

    def test_quoted_shell_payload_scanned_as_code(self, isolated_approvals,
                                                  capsys):
        # hermes approvals test -- sh -c "rm -rf /": the dequoted join would
        # report only ask-approval while the runtime hardline-denies.
        rc = at.approvals_test_command(_args(["sh", "-c", "rm -rf /"]))
        capsys.readouterr()
        assert rc == 3

    def test_single_word_command_evaluated_verbatim(self, isolated_approvals,
                                                    capsys):
        # hermes approvals test 'git commit -m "x; rm -rf /"': one argv word is
        # already the complete command string; re-quoting it would turn the
        # command into data.
        rc = at.approvals_test_command(
            _args(['git commit -m "x; rm -rf /"'], as_json=True))
        payload = json.loads(capsys.readouterr().out)
        assert rc == 2
        assert payload["command"] == 'git commit -m "x; rm -rf /"'

    def test_requoted_command_shown_in_json(self, isolated_approvals, capsys):
        # The reported command is the faithful reconstruction, not the
        # dequoted join, so what the user sees is what was evaluated.
        rc = at.approvals_test_command(
            _args(["git", "commit", "-m", "x; rm -rf /"], as_json=True))
        payload = json.loads(capsys.readouterr().out)
        assert payload["command"] == "git commit -m 'x; rm -rf /'"

    def test_embedded_quote_word_round_trips(self, isolated_approvals, capsys):
        # hermes approvals test -- git commit -m "don't": the apostrophe is
        # data inside the word and must survive reconstruction.
        rc = at.approvals_test_command(_args(["git", "commit", "-m", "don't"]))
        capsys.readouterr()
        assert rc == 0
        assert at.evaluate_command('git commit -m "don\'t"')["exit_code"] == rc

    def test_parser_pipeline_restores_quoting(self, isolated_approvals, capsys):
        # e2e through the real parser + dispatcher, exactly as the shell
        # delivers the post-split words via argparse REMAINDER.
        from hermes_cli.approvals_suggest import approvals_command
        from hermes_cli.subcommands.approvals import build_approvals_parser
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        build_approvals_parser(sub, cmd_approvals=approvals_command)
        args = parser.parse_args(
            ["approvals", "test", "--", "git", "commit", "-m", "x; rm -rf /"])
        rc = args.func(args)
        capsys.readouterr()
        assert rc == 2
