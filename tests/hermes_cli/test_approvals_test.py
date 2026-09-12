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
from tools import approval_detection, approval_floors, tirith_security
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
    saved_sessions = {key: set(value) for key, value in A._session_approved.items()}
    A._permanent_approved.clear()
    A._session_approved.clear()
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
    A._session_approved.clear()
    A._session_approved.update(saved_sessions)


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

    def test_permanent_class_key_allows_like_runtime(self, isolated_approvals,
                                                     capsys):
        isolated_approvals._permanent_approved.add("recursive delete")

        rc = at.approvals_test_command(_args(["rm", "-rf", "~/project/build"]))
        out = capsys.readouterr().out

        assert rc == 0
        assert "recursive delete" in out
        assert "already approved" in out

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


class TestRuntimeParity:
    """The dry-run's answer must equal what the runtime gate actually does.

    This is the contract the command exists to keep: `hermes approvals test`
    predicts `check_all_command_guards`. Both are driven for real here — the
    only stub is the human prompt, which stands in for "a prompt happened".
    """

    @staticmethod
    def _runtime_would_prompt(command, monkeypatch):
        """Run the REAL runtime gate; True when it reaches the human prompt."""
        prompted = []

        def _fake_prompt(*_a, **_kw):
            prompted.append(command)
            return "deny"

        monkeypatch.setattr(A, "prompt_dangerous_approval", _fake_prompt)
        monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval", _fake_prompt)
        # Interactive CLI presence, so the gate reaches the prompt instead of
        # resolving through an unattended context.
        monkeypatch.setattr(A, "_presence", lambda cb=None: (None, True, False, False))
        result = A.check_all_command_guards(command, env_type="local")
        return bool(prompted), result

    @pytest.mark.parametrize("command,setup,label", [
        # No approval recorded: both must predict a prompt.
        ("rm -rf ~/project/build", None, "unapproved dangerous pattern"),
        # 'always' persists the PATTERN KEY into command_allowlist; the runtime
        # then allows silently, so the dry-run must not predict a prompt.
        ("rm -rf ~/project/build",
         lambda: A._permanent_approved.add("recursive delete"),
         "permanent class key"),
        # Session-scoped approval of the same key.
        ("rm -rf ~/project/build",
         lambda: A._session_approved.setdefault(
             approval_context.get_current_session_key(), set()).add("recursive delete"),
         "session class key"),
        # Another session's approval must NOT leak into this one.
        ("rm -rf ~/project/build",
         lambda: A._session_approved.setdefault("other-session", set()).add(
             "recursive delete"),
         "other session's class key"),
    ])
    def test_dry_run_prompt_prediction_matches_runtime(
            self, isolated_approvals, monkeypatch, command, setup, label):
        if setup is not None:
            setup()

        runtime_prompts, runtime_result = self._runtime_would_prompt(command, monkeypatch)
        dry = at.evaluate_command(command, env_type="local")
        dry_prompts = dry["verdict"] == "ask-approval"

        assert dry_prompts == runtime_prompts, (
            f"{label}: dry-run says {dry['verdict']!r} but the runtime "
            f"{'prompts' if runtime_prompts else 'allows silently'}"
        )
        # And when neither prompts, the runtime really did approve.
        if not runtime_prompts:
            assert runtime_result["approved"] is True

    @pytest.mark.parametrize("approve_tirith_key", [False, True])
    def test_tirith_only_finding_matches_runtime(self, isolated_approvals,
                                                 monkeypatch, approve_tirith_key):
        """A security-scan finding with no DANGEROUS_PATTERNS match must not read as `allow`.

        The runtime folds tirith findings and dangerous-pattern hits into ONE
        warning list and prompts if anything survives the approved-key filter, so a
        command the static patterns miss can still prompt. conftest disables tirith
        for the suite, so the scanner is stubbed here (the documented opt-in).
        """
        command = "definitely-not-a-dangerous-pattern --flag"
        # Precondition: the static classifier does NOT flag this, so tirith is the
        # only reason a prompt could fire. Without this the test could pass for the
        # wrong reason.
        assert approval_detection.detect_dangerous_command(command)[0] is False

        monkeypatch.setattr(tirith_security, "check_command_security", lambda _c: {
            "action": "warn", "summary": "stubbed finding",
            "findings": [{"rule_id": "stub_rule", "severity": "HIGH",
                          "title": "stubbed finding", "description": "for test"}],
        })
        if approve_tirith_key:
            A._permanent_approved.add("tirith:stub_rule")

        runtime_prompts, runtime_result = self._runtime_would_prompt(command, monkeypatch)
        dry = at.evaluate_command(command, env_type="local")

        assert (dry["verdict"] == "ask-approval") == runtime_prompts
        # Sanity-check the arms are actually different, so neither is vacuous.
        assert runtime_prompts is not approve_tirith_key
        if not runtime_prompts:
            assert runtime_result["approved"] is True

    @pytest.mark.parametrize("command,deny_globs,expected", [
        ("rm -rf /", [], "hardline-deny"),
        ("rm -rf ~/project/build", ["rm -rf *"], "user-deny"),
    ])
    def test_approved_class_key_cannot_punch_through_the_floors(
            self, isolated_approvals, monkeypatch, command, deny_globs, expected):
        """An approved pattern key must never downgrade an unconditional block.

        Both floors fire before the allowlist at runtime; the dry-run must keep
        that ordering, or the command would advertise a bypass that does not exist.
        """
        _dangerous, pattern_key, _desc = approval_detection.detect_dangerous_command(command)
        A._permanent_approved.add(pattern_key)
        monkeypatch.setattr(
            approval_context, "_get_approval_config",
            lambda: {"mode": "manual", "deny": deny_globs})

        dry = at.evaluate_command(command, env_type="local")
        runtime = A.check_all_command_guards(command, env_type="local")

        assert dry["verdict"] == expected
        assert dry["exit_code"] == 3
        assert runtime["approved"] is False


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

    def test_composes_real_runtime_detectors(self, isolated_approvals, capsys,
                                             monkeypatch):
        """Prove the tester calls the real evaluators, not a reimplementation."""
        calls = {}

        def _spy(name, real):
            def wrapper(c):
                calls[name] = c
                return real(c)
            return wrapper

        monkeypatch.setattr(approval_detection, "detect_hardline_command",
                            _spy("hardline", approval_detection.detect_hardline_command))
        monkeypatch.setattr(approval_detection, "detect_dangerous_command",
                            _spy("dangerous", approval_detection.detect_dangerous_command))
        monkeypatch.setattr(approval_floors, "_match_user_deny_rule",
                            _spy("deny", approval_floors._match_user_deny_rule))
        monkeypatch.setattr(approval_detection, "_command_detection_variants",
                            _spy("variants", approval_detection._command_detection_variants))
        cmd = "rm -rf ~/project/build"
        at.approvals_test_command(_args(cmd.split()))
        capsys.readouterr()
        assert calls.get("hardline") == cmd
        assert calls.get("dangerous") == cmd
        assert calls.get("deny") == cmd
        assert calls.get("variants") == cmd


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
        assert payload["rule"] == "system shutdown/reboot"
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
