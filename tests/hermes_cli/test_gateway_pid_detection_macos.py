"""Regression guard for #15225 — macOS gateway-pid detection.

Two independent bugs in ``hermes_cli/gateway.py`` caused
``find_gateway_pids()`` to return an empty list on macOS, which made
``hermes cron list`` print a false-positive "Gateway is not running"
warning even when launchd had the service loaded and cron firing was
working normally.

Bug 1 — ``_get_service_pids()`` called ``launchctl list <label>`` and
then parsed the output as the tab-separated table ``PID\\tStatus\\tLabel``.
With a label argument, macOS ``launchctl`` instead returns a plist-dict
``{ "PID" = 855; "Label" = "ai.hermes.gateway"; ...}`` — the old loop's
``parts[2] == label`` check never matched the quoted/semicolon'd token
so no PID was ever extracted.

Bug 2 — ``_scan_gateway_pids()`` invoked ``ps -A eww -o pid=,command=``.
Darwin's ``ps`` rejects ``eww`` as an illegal positional argument
(``exit=1``, empty stdout), so the fallback PID scan also produced
nothing.

These tests pin the fixes at the production-code entry points:

* ``_parse_launchd_list_output`` (the new helper) is exercised directly
  with the exact plist-dict output from the bug report AND with the
  tab-separated format the old code expected — both must surface the
  correct PID.
* ``_get_service_pids`` is exercised end-to-end with ``subprocess.run``
  patched to return the plist-dict payload, proving the macOS branch
  no longer comes back empty for a running service.
* ``_scan_gateway_pids`` (and ``find_gateway_pids`` through it) is
  exercised with a patched ``subprocess.run`` that records the exact
  command list the implementation passed — failing loudly if the
  invocation ever regresses to ``eww``.
"""
from types import SimpleNamespace

import pytest

from hermes_cli import gateway


# Real-world plist-dict payload taken from the #15225 reproduction,
# minus shell-specific paths (so the fixture is portable across contributors).
_MACOS_LAUNCHCTL_LIST_LABEL_OUTPUT = """\
{
\t"StandardOutPath" = "/Users/someuser/.hermes/logs/gateway.log";
\t"LimitLoadToSessionType" = "Aqua";
\t"StandardErrorPath" = "/Users/someuser/.hermes/logs/gateway.error.log";
\t"Label" = "ai.hermes.gateway";
\t"OnDemand" = true;
\t"LastExitStatus" = 0;
\t"PID" = 855;
\t"Program" = "/Users/someuser/.hermes/hermes-agent/venv/bin/python";
\t"ProgramArguments" = (
\t\t"/Users/someuser/.hermes/hermes-agent/venv/bin/python";
\t\t"-m";
\t\t"hermes_cli";
\t\t"gateway";
\t\t"run";
\t);
};
"""


# Tab-separated format — what ``launchctl list`` (no label) produces.  The
# old code path assumed this format regardless of whether a label was
# passed; the helper still has to handle it so callers that change their
# mind about whether to pass a label don't break.
_MACOS_LAUNCHCTL_LIST_TABLE_OUTPUT = """\
PID\tStatus\tLabel
855\t0\tai.hermes.gateway
-\t0\tcom.apple.somethingelse
1024\t0\tcom.other.service
"""


# ---------------------------------------------------------------------------
# Bug 1 — _parse_launchd_list_output (new helper)
# ---------------------------------------------------------------------------


class TestParseLaunchdListOutput:
    def test_extracts_pid_from_plist_dict_format(self):
        """macOS returns a plist-dict when a label is passed to
        ``launchctl list``.  This is the exact format from the #15225
        reproduction.  The old regex-less string.split()-based code saw
        ``parts[2] == '"ai.hermes.gateway";'`` (quoted + semicolon'd)
        and never matched the bare label — so it returned the empty
        set.  The new helper parses ``"PID" = N;`` directly."""
        pids = gateway._parse_launchd_list_output(
            _MACOS_LAUNCHCTL_LIST_LABEL_OUTPUT, "ai.hermes.gateway",
        )
        assert pids == {855}

    def test_extracts_pid_from_tab_separated_format(self):
        """When the caller runs ``launchctl list`` without a label, the
        output is a tab-separated table.  The helper must still extract
        only PIDs for the requested label (the table may list many
        services)."""
        pids = gateway._parse_launchd_list_output(
            _MACOS_LAUNCHCTL_LIST_TABLE_OUTPUT, "ai.hermes.gateway",
        )
        assert pids == {855}

    def test_skips_tab_separated_rows_with_dash_pid(self):
        """In the tab-separated format, an unloaded service appears as
        ``-\\t0\\tlabel``.  ``int("-")`` would raise; the helper must
        skip such rows cleanly and return an empty set if the requested
        label is only present as a dash row."""
        pids = gateway._parse_launchd_list_output(
            "PID\tStatus\tLabel\n-\t0\tai.hermes.gateway\n",
            "ai.hermes.gateway",
        )
        assert pids == set()

    def test_empty_output_returns_empty_set(self):
        """A 0-length stdout (e.g. launchctl hanging, or a non-zero exit
        where the caller already filtered) must never crash the helper
        or synthesise fake PIDs."""
        assert gateway._parse_launchd_list_output("", "ai.hermes.gateway") == set()

    def test_plist_dict_wins_over_tab_separated_when_both_look_plausible(self):
        """Defensive: if some future macOS release returned *both*
        shapes concatenated, trust the plist-dict (the label-scoped
        answer) rather than the table (which might list sibling
        services for the same user).  The helper picks the plist
        matches first and only falls back to the tab-separated path
        when none were found."""
        merged = (
            _MACOS_LAUNCHCTL_LIST_LABEL_OUTPUT
            + "\nPID\tStatus\tLabel\n9999\t0\tai.hermes.gateway\n"
        )
        pids = gateway._parse_launchd_list_output(merged, "ai.hermes.gateway")
        assert pids == {855}  # from the plist-dict block, not the table

    def test_pid_zero_is_rejected(self):
        """``"PID" = 0;`` would be either kernel or "service not running"
        depending on the OS — either way, never a real gateway PID.
        Accepting it would crash downstream ``os.kill`` calls that
        interpret 0 as 'this process group'."""
        pids = gateway._parse_launchd_list_output(
            '{\n\t"PID" = 0;\n\t"Label" = "ai.hermes.gateway";\n};\n',
            "ai.hermes.gateway",
        )
        assert pids == set()

    def test_other_label_entry_in_table_ignored(self):
        """The tab-separated path must filter on the third column
        (label) — a different service at the top of the table must not
        contribute its PID to our result."""
        text = (
            "PID\tStatus\tLabel\n"
            "1234\t0\tcom.apple.unrelated\n"
            "855\t0\tai.hermes.gateway\n"
        )
        pids = gateway._parse_launchd_list_output(text, "ai.hermes.gateway")
        assert pids == {855}

    @pytest.mark.parametrize("spacing", [
        '"PID" = 855;',                  # canonical
        '"PID"=855;',                    # no spaces
        '"PID"   =   855   ;',           # extra spaces
        '\t\t"PID" = 855;',              # leading tabs (real output)
    ])
    def test_pid_regex_tolerates_whitespace_variants(self, spacing):
        """launchd's plist dumper is not a stable format — prior Apple
        releases have shifted spacing.  The regex must not hinge on an
        exact ``"PID" = N;`` with a single space or we'll regress
        silently on a future macOS update."""
        text = f'{{\n\t{spacing}\n\t"Label" = "ai.hermes.gateway";\n}};\n'
        pids = gateway._parse_launchd_list_output(text, "ai.hermes.gateway")
        assert pids == {855}


# ---------------------------------------------------------------------------
# Bug 1 — _get_service_pids wiring (macOS branch)
# ---------------------------------------------------------------------------


class TestGetServicePidsMacOS:
    """End-to-end check that the macOS branch of ``_get_service_pids``
    actually picks up the PID for a running gateway now — the symptom
    #15225 complained about."""

    def _patch_macos(self, monkeypatch, launchctl_stdout, launchctl_rc=0):
        monkeypatch.setattr(gateway, "is_macos", lambda: True)
        # Avoid running systemd probes on a macOS-emulation test —
        # ``supports_systemd_services`` already returns False on Darwin
        # but tests don't run on Darwin by default.
        monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway, "get_launchd_label", lambda: "ai.hermes.gateway")

        def fake_run(cmd, **kwargs):
            if cmd[:3] == ["launchctl", "list", "ai.hermes.gateway"]:
                return SimpleNamespace(
                    returncode=launchctl_rc,
                    stdout=launchctl_stdout,
                    stderr="",
                )
            raise AssertionError(f"Unexpected subprocess call: {cmd!r}")

        monkeypatch.setattr(gateway.subprocess, "run", fake_run)

    def test_returns_pid_for_running_service(self, monkeypatch):
        self._patch_macos(monkeypatch, _MACOS_LAUNCHCTL_LIST_LABEL_OUTPUT)
        assert gateway._get_service_pids() == {855}

    def test_returns_empty_set_when_launchctl_exits_nonzero(self, monkeypatch):
        """The caller should not record PIDs from a failed command —
        e.g. launchctl returning "Could not find service" as rc=3."""
        self._patch_macos(monkeypatch, "", launchctl_rc=3)
        assert gateway._get_service_pids() == set()

    def test_survives_launchctl_missing_on_path(self, monkeypatch):
        """If ``launchctl`` isn't installed (hermes CLI running on a
        nonstandard macOS image), we must not propagate the
        ``FileNotFoundError`` — return an empty set and let the ps
        fallback try next."""
        monkeypatch.setattr(gateway, "is_macos", lambda: True)
        monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway, "get_launchd_label", lambda: "ai.hermes.gateway")

        def boom(cmd, **kwargs):
            raise FileNotFoundError("launchctl")

        monkeypatch.setattr(gateway.subprocess, "run", boom)
        assert gateway._get_service_pids() == set()


# ---------------------------------------------------------------------------
# Bug 2 — _scan_gateway_pids ps invocation
# ---------------------------------------------------------------------------


class TestPsInvocationPortability:
    """The ``ps`` fallback invocation must be portable across Linux,
    Darwin, and FreeBSD.  ``eww`` was illegal on Darwin (#15225) and
    leaked env vars (including API keys) on FreeBSD (#9069) — the
    portable replacement is ``-A -ww`` (list all processes, wide
    wide) with no ``e``.
    """

    def _invoke(self, monkeypatch, *, returncode=0, stdout=""):
        recorded = {"cmd": None}

        def fake_run(cmd, **kwargs):
            recorded["cmd"] = list(cmd)
            return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

        monkeypatch.setattr(gateway, "is_windows", lambda: False)
        monkeypatch.setattr(gateway.subprocess, "run", fake_run)

        # Exercise the production path.  ``_scan_gateway_pids`` takes an
        # exclude-set; we pass empty so nothing is filtered out.
        pids = gateway._scan_gateway_pids(set(), all_profiles=True)
        return recorded["cmd"], pids

    def test_never_passes_eww_positional(self, monkeypatch):
        """The central regression guard: neither ``"eww"`` nor any
        ``e``-containing flag stack that would attach env vars should
        end up on the command line.  Parametrise in case a future
        refactor adds an equivalent flag."""
        cmd, _ = self._invoke(monkeypatch)
        assert cmd is not None, "production code never invoked ps"
        assert "eww" not in cmd, (
            f"ps was invoked with 'eww' (illegal on Darwin, leaks env on "
            f"FreeBSD).  Full cmd: {cmd!r}"
        )
        # ``-A -ww`` is the expected replacement — each flag should appear.
        assert "-A" in cmd
        assert "-ww" in cmd

    def test_invocation_shape_matches_expected(self, monkeypatch):
        """Pin the exact invocation shape so a refactor doesn't
        accidentally alter argv ordering.  ``pid=,command=`` is important
        — the trailing ``=`` on each field suppresses the header line
        that would otherwise confuse the parser."""
        cmd, _ = self._invoke(monkeypatch)
        assert cmd == ["ps", "-A", "-ww", "-o", "pid=,command="], (
            f"ps invocation drifted: {cmd!r}"
        )

    def test_parses_macos_style_output(self, monkeypatch):
        """Verify the parse loop still works for realistic Darwin ps
        output — ``pid=,command=`` suppresses the header and lines are
        space-separated PID + command text.  Command string must contain
        one of the gateway-scan patterns (e.g. ``hermes_cli.main gateway``)
        to be picked up — same as the production matcher."""
        sample = (
            "  855 /Users/someuser/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main gateway run\n"
            " 1024 /usr/sbin/cupsd -l\n"
        )
        monkeypatch.setattr(gateway, "is_windows", lambda: False)

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(returncode=0, stdout=sample, stderr="")

        monkeypatch.setattr(gateway.subprocess, "run", fake_run)
        pids = gateway._scan_gateway_pids(set(), all_profiles=True)
        assert 855 in pids, (
            f"expected PID 855 extracted from ps output, got {pids}"
        )
        assert 1024 not in pids, (
            "non-gateway process (cupsd) must not be matched"
        )

    def test_returncode_nonzero_returns_empty_list(self, monkeypatch):
        """Darwin used to return exit=1 for the old ``eww`` invocation.
        Even with the fix, a future platform that errors on ``-ww``
        must not crash — the caller returns []."""
        cmd, pids = self._invoke(monkeypatch, returncode=1, stdout="")
        assert pids == []
        # Still invoked with the expected shape, just failed.
        assert cmd == ["ps", "-A", "-ww", "-o", "pid=,command="]
