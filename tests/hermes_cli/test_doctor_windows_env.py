"""Tests for the Windows Environment section of hermes doctor (#91942).

Probe functions are dependency-injectable so privilege/registry failure
modes are exercised as data on any host. Assertions that depend on the
real Windows filesystem/registry are marked ``windows_only``.
"""

import os
import subprocess
import sys
import types

import pytest

import hermes_cli.doctor as doctor_mod


# ── pure classification: git interactive risk ────────────────────────────


class TestGitInteractiveRisk:
    def test_explicit_prompt_disabled_is_ok(self):
        level, detail = doctor_mod._classify_git_interactive_risk(
            {"GIT_TERMINAL_PROMPT": "0"}, ""
        )
        assert level == "ok"
        assert detail == ""

    def test_no_helper_and_no_opt_out_warns_with_fix_command(self):
        level, detail = doctor_mod._classify_git_interactive_risk({}, "")
        assert level == "warn"
        assert "setx GIT_TERMINAL_PROMPT 0" in detail
        assert "hangs" in detail or "hang" in detail

    def test_gui_manager_helper_warns_even_when_configured(self):
        level, detail = doctor_mod._classify_git_interactive_risk(
            {}, "manager-core"
        )
        assert level == "warn"
        assert "manager-core" in detail
        assert "setx GIT_TERMINAL_PROMPT 0" in detail

    def test_noninteractive_helpers_are_ok(self):
        for helper in ("cache", "store"):
            level, detail = doctor_mod._classify_git_interactive_risk({}, helper)
            assert level == "ok", helper
            assert helper in detail

    def test_multiline_helper_uses_first_entry(self):
        level, _ = doctor_mod._classify_git_interactive_risk(
            {}, "store\nhttps://example.com/helper"
        )
        assert level == "ok"


# ── symlink probe (injected failure modes + real host behavior) ─────────


def _raise_eperm(target, link):
    err = OSError(13, "A required privilege is not held by the client.")
    err.winerror = 1314
    raise err


class TestWindowsSymlinkProbe:
    def test_injected_privilege_failure_reports_developer_mode_remedy(self, tmp_path):
        ok, detail = doctor_mod._windows_symlink_probe(
            base_dir=tmp_path, symlink_to=_raise_eperm
        )
        assert ok is False
        assert "Developer Mode" in detail

    def test_injected_generic_failure_reports_error(self, tmp_path):
        def boom(target, link):
            raise OSError(22, "invalid argument")

        ok, detail = doctor_mod._windows_symlink_probe(base_dir=tmp_path, symlink_to=boom)
        assert ok is False
        assert "symlink probe failed" in detail

    def test_probe_cleans_up_artifacts(self, tmp_path):
        doctor_mod._windows_symlink_probe(base_dir=tmp_path, symlink_to=_raise_eperm)
        names = {p.name for p in tmp_path.iterdir()}
        assert not ({"doctor_symlink_target.txt", "doctor_symlink_link"} & names)

    @pytest.mark.windows_only
    def test_real_symlink_succeeds_on_privileged_host(self, tmp_path):
        ok, detail = doctor_mod._windows_symlink_probe(base_dir=tmp_path)
        if not ok and "Developer Mode" in detail:
            pytest.skip("host lacks symlink privilege (Developer Mode off)")
        assert ok is True, f"symlink probe failed on privileged host: {detail}"


# ── bash path round-trip (fake runner = data, not a fake OS) ─────────────


def _fake_runner(outputs_by_arg):
    calls = []

    def run(cmd, capture_output=None, text=None, timeout=None, env=None):
        sample = cmd[-1]
        calls.append((sample, env))
        return types.SimpleNamespace(stdout=outputs_by_arg[sample])

    run.calls = calls
    return run


class TestBashPathRoundtrip:
    SAMPLE_NATIVE = "C:/Users/dev/project"
    SAMPLE_MSYS = "/c/Users/dev/project"

    def test_passthrough_both_forms_passes(self):
        runner = _fake_runner({self.SAMPLE_NATIVE: self.SAMPLE_NATIVE,
                               self.SAMPLE_MSYS: self.SAMPLE_MSYS})
        ok, detail = doctor_mod._windows_bash_path_roundtrip(
            "bash", self.SAMPLE_NATIVE, self.SAMPLE_MSYS, run=runner
        )
        assert ok is True
        assert detail == ""

    def test_rewritten_msys_form_fails_and_names_the_rewrite(self):
        rewritten = "C:/Program Files/Git/Users/dev/project"
        runner = _fake_runner({self.SAMPLE_NATIVE: self.SAMPLE_NATIVE,
                               self.SAMPLE_MSYS: rewritten})
        ok, detail = doctor_mod._windows_bash_path_roundtrip(
            "bash", self.SAMPLE_NATIVE, self.SAMPLE_MSYS, run=runner
        )
        assert ok is False
        assert rewritten in detail
        assert "/c/ form was rewritten" in detail

    def test_runner_receives_msys_opt_out_env(self):
        runner = _fake_runner({self.SAMPLE_NATIVE: self.SAMPLE_NATIVE,
                               self.SAMPLE_MSYS: self.SAMPLE_MSYS})
        doctor_mod._windows_bash_path_roundtrip(
            "bash", self.SAMPLE_NATIVE, self.SAMPLE_MSYS, run=runner
        )
        _, env = runner.calls[0]
        if sys.platform == "win32":
            # opt-outs are set by default; user overrides still win
            assert env.get("MSYS_NO_PATHCONV", "1") == "1"
            assert env.get("MSYS2_ARG_CONV_EXCL", "*") == "*"

    def test_runner_crash_reports_probe_failed(self):
        def run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="bash", timeout=15)

        ok, detail = doctor_mod._windows_bash_path_roundtrip(
            "bash", self.SAMPLE_NATIVE, self.SAMPLE_MSYS, run=run
        )
        assert ok is False
        assert "probe failed to run" in detail


# ── long paths (registry read as injected data) ──────────────────────────


class TestLongPathsEnabled:
    def test_value_one_reads_as_enabled(self):
        state, value = doctor_mod._read_long_paths_enabled(query=lambda k, n: 1)
        assert (state, value) == ("enabled", 1)

    def test_value_zero_reads_as_disabled(self):
        state, value = doctor_mod._read_long_paths_enabled(query=lambda k, n: 0)
        assert (state, value) == ("disabled", 0)

    def test_unreadable_registry_degrades_to_unknown(self):
        def denied(key_path, value_name):
            raise PermissionError("access denied")

        state, value = doctor_mod._read_long_paths_enabled(query=denied)
        assert state is None
        assert value is None


# ── native → msys conversion ──────────────────────────────────────────────


class TestNativePathToMsys:
    def test_windows_path_converts(self):
        assert doctor_mod._native_path_to_msys("C:\\Users\\dev") == "/c/Users/dev"

    def test_forward_slash_windows_path_converts(self):
        assert doctor_mod._native_path_to_msys("D:/work/repo") == "/d/work/repo"

    def test_posix_path_unchanged(self):
        assert doctor_mod._native_path_to_msys("/usr/bin") == "/usr/bin"


# ── full section against the real host (E2E, windows lane) ───────────────


@pytest.mark.windows_only
class TestWindowsSectionEndToEnd:
    def test_section_reports_all_four_checks_with_consistent_summary(self, capsys):
        issues, manual_issues = [], []
        doctor_mod._check_windows_environment(issues, manual_issues)
        out = capsys.readouterr().out
        # every check reports exactly one of its outcomes, never silence
        assert "Symlink creation allowed" in out or "Symlink creation denied" in out
        # check 2 stays silent when git itself is absent (External Tools covers that)
        if doctor_mod._safe_which("git"):
            assert "Background git safety" in out or "credential prompt" in out
        assert ("native paths through untouched" in out
                or "rewrites path arguments" in out
                or "round-trip skipped" in out)
        assert ("Long paths enabled" in out or "Long paths disabled" in out
                or "skipping long-path check" in out)
        # contract: hard failures must land in the actionable summary lists
        if "Symlink creation denied" in out:
            assert any("Symlinks cannot be created" in i for i in issues)
            assert any("Developer Mode" in i for i in issues)
        else:
            assert not any("Symlinks cannot be created" in i for i in issues)
        if "Long paths disabled" in out:
            assert any("reg add" in i for i in manual_issues)

    def test_registry_read_matches_host_policy_domain(self):
        state, value = doctor_mod._read_long_paths_enabled()
        assert state in ("enabled", "disabled", None)
        if state is not None:
            assert value in (0, 1)
            assert state == ("enabled" if value else "disabled")

    def test_os_environment_is_a_plain_mapping_for_classifier(self):
        level, _ = doctor_mod._classify_git_interactive_risk(os.environ, "store")
        assert level == "ok"
