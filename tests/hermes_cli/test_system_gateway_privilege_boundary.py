"""Privilege-boundary regression tests for the system gateway home override.

Closes the five late-review REJECT findings against commit 90ebe58c5e
(root-home recreation guard):

1.  P0 — privileged environment injection: a root process that adopted a
    system unit home must not honour that (user-writable) home's ``.env``
    for PATH, and must not execute PATH-resolved privileged tools.
2.  P1 — early argument detection must agree with the real CLI: global
    flags before the subcommand, ``--`` passthrough, argparse ``--sys``
    abbreviation, and only gateway verbs that own ``--system``.
3.  P1 — unit parsing must follow systemd ordered semantics (later
    ``Environment=`` wins, empty value resets, drop-ins in filename
    order) and fail closed on anything unmodellable.
4.  P1 — malformed paths (embedded NUL, symlink loops) and temporary
    roots (including fixed ``/tmp`` regardless of ``TMPDIR``) must fail
    closed without crashing startup.
5.  P1 — fresh-process coverage proving the import-time bootstrap picks
    the unit home before any state creation.

All privileged-tool tests execute under a *simulated* root (monkeypatched
``os.geteuid``); no test requires real root and no test executes a real
privileged operation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ELEVATED_SENTINEL = "_HERMES_SYSTEM_GATEWAY_ELEVATED"
VERBS_WITH_SYSTEM_FLAG = {
    "start",
    "stop",
    "restart",
    "status",
    "install",
    "uninstall",
}


def _pop_cli_modules() -> None:
    """Drop cached hermes_cli modules so import-time state can be re-run."""
    for name in ("hermes_cli.main", "hermes_cli.gateway", "hermes_cli.env_loader"):
        sys.modules.pop(name, None)


@pytest.fixture
def sudo_root_env(monkeypatch, tmp_path):
    """Environment of a ``sudo hermes gateway <verb> --system`` process."""
    root_home = tmp_path / "root-home"
    root_home.mkdir()
    monkeypatch.setenv("HOME", str(root_home))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv(ELEVATED_SENTINEL, raising=False)
    monkeypatch.setenv("PATH", SAFE_PATH)
    monkeypatch.delenv("TMPDIR", raising=False)
    return root_home


class TestSystemGatewayArgDetection:
    """Finding 2 — the override must fire exactly for real CLI invocations."""

    def _override(self, monkeypatch, tmp_path, argv, dropins=None):
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text(
            '[Service]\nEnvironment="HERMES_HOME=/home/hermes/.hermes"\n',
            encoding="utf-8",
        )
        for name, text in (dropins or {}).items():
            d = unit_path.parent / (unit_path.name + ".d")
            d.mkdir(exist_ok=True)
            (d / name).write_text(text, encoding="utf-8")
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv(ELEVATED_SENTINEL, raising=False)

        from hermes_cli.main import _apply_system_gateway_home_override

        _apply_system_gateway_home_override(unit_path=unit_path)
        return os.environ.get("HERMES_HOME")

    def test_global_flags_before_gateway_still_trigger(self, tmp_path, monkeypatch):
        """`hermes --yolo gateway status --system` is a valid invocation."""
        result = self._override(
            monkeypatch,
            tmp_path,
            argv=["hermes", "--yolo", "gateway", "status", "--system"],
        )
        assert result == "/home/hermes/.hermes"

    def test_global_value_flag_before_gateway_still_triggers(self, tmp_path, monkeypatch):
        """`hermes --reasoning high gateway status --system` is valid too."""
        result = self._override(
            monkeypatch,
            tmp_path,
            argv=["hermes", "--reasoning", "high", "gateway", "status", "--system"],
        )
        assert result == "/home/hermes/.hermes"

    def test_double_dash_separator_does_not_trigger(self, tmp_path, monkeypatch):
        """`hermes -- gateway status --system` treats everything as positional."""
        result = self._override(
            monkeypatch,
            tmp_path,
            argv=["hermes", "--", "gateway", "status", "--system"],
        )
        assert result is None

    def test_argparse_abbreviation_triggers(self, tmp_path, monkeypatch):
        """`hermes gateway status --sys` parses as `--system` (allow_abbrev)."""
        result = self._override(
            monkeypatch,
            tmp_path,
            argv=["hermes", "gateway", "status", "--sys"],
        )
        assert result == "/home/hermes/.hermes"

    @pytest.mark.parametrize("verb", sorted(VERBS_WITH_SYSTEM_FLAG))
    def test_verbs_owning_system_flag_trigger(self, tmp_path, monkeypatch, verb):
        result = self._override(
            monkeypatch,
            tmp_path,
            argv=["hermes", "gateway", verb, "--system"],
        )
        assert result == "/home/hermes/.hermes", verb

    def test_run_does_not_own_system_flag(self, tmp_path, monkeypatch):
        """`gateway run --system` is rejected by argparse — no override."""
        result = self._override(
            monkeypatch,
            tmp_path,
            argv=["hermes", "gateway", "run", "--system"],
        )
        assert result is None

    def test_positional_after_double_dash_does_not_trigger(self, tmp_path, monkeypatch):
        """`gateway status -- --system` passes --system as output positional."""
        result = self._override(
            monkeypatch,
            tmp_path,
            argv=["hermes", "gateway", "status", "--", "--system"],
        )
        assert result is None

    def test_non_gateway_command_does_not_trigger(self, tmp_path, monkeypatch):
        result = self._override(
            monkeypatch,
            tmp_path,
            argv=["hermes", "chat", "--system"],
        )
        assert result is None

    def test_install_run_as_user_value_not_misread_as_positional(self, tmp_path, monkeypatch):
        """`gateway install --run-as-user bob --system` is valid argparse."""
        result = self._override(
            monkeypatch,
            tmp_path,
            argv=["hermes", "gateway", "install", "--run-as-user", "bob", "--system"],
        )
        assert result == "/home/hermes/.hermes"


class TestArgvScannerArgparseParity:
    """The early scanner must agree with the REAL argparse on every
    ``gateway <verb> --s…`` abbreviation (drift-proof parity).

    Builds the actual gateway subparser tree and compares, for every verb
    and every prefix of ``--system`` down to ``--s``, whether argparse
    accepts the invocation vs whether ``_system_gateway_request`` fires.
    Also verifies the ambiguous-prefix table matches the real parser's
    ``--s…`` option inventory per verb.
    """

    def _real_gateway_parser(self):
        import argparse

        from hermes_cli.subcommands.gateway import build_gateway_parser

        root = argparse.ArgumentParser(prog="hermes", allow_abbrev=True)
        sub = root.add_subparsers(dest="cmd")
        build_gateway_parser(
            sub,
            cmd_gateway=lambda: None,
            cmd_proxy=lambda: None,
            cmd_gateway_enroll=lambda: None,
        )
        gateway = sub.choices["gateway"]
        return gateway

    def test_scanner_matches_argparse_for_every_abbreviation(self, monkeypatch):
        from hermes_cli import main as cli_main

        gateway = self._real_gateway_parser()
        verbs = sorted(cli_main._SYSTEM_GATEWAY_VERBS_WITH_SYSTEM_FLAG)
        mismatches = []

        for verb in verbs:
            for abbrev in ("--system", "--syste", "--syst", "--sys", "--sy", "--s"):
                argv_tail = ["gateway", verb, abbrev]
                argv = ["hermes"] + argv_tail
                parser_accepts = True
                try:
                    gateway.parse_args(argv_tail[1:])
                except SystemExit:
                    parser_accepts = False
                scanner_fires = cli_main._system_gateway_request(argv[1:])
                if parser_accepts != scanner_fires:
                    mismatches.append(
                        f"{verb} {abbrev}: argparse={parser_accepts} scanner={scanner_fires}"
                    )
        assert not mismatches, "\n".join(mismatches)

    def test_ambiguous_prefix_table_matches_real_parser(self, monkeypatch):
        """The static collision table must mirror the live parser tree."""
        import argparse

        from hermes_cli import main as cli_main

        gateway = self._real_gateway_parser()
        sub_action = next(
            a
            for a in gateway._actions
            if isinstance(a, argparse._SubParsersAction)
        )
        verbs = sub_action.choices
        table = cli_main._SYSTEM_GATEWAY_AMBIGUOUS_PREFIXES

        # Every verb owning --system must be in the table…
        owning = {
            name
            for name, parser in verbs.items()
            if any("--system" in a.option_strings for a in parser._actions)
        }
        assert owning == set(table.keys()), (owning, set(table.keys()))

        # …and the collisions listed must be exactly the other --s… options.
        for name in owning:
            parser = verbs[name]
            s_options = {
                opt
                for a in parser._actions
                for opt in a.option_strings
                if opt.startswith("--s") and opt != "--system"
            }
            assert table[name] == s_options, (name, table[name], s_options)

    def test_systemx_and_system_value_do_not_trigger(self, tmp_path, monkeypatch):
        """`--systemx` is a different (invalid) option; `--system=x` is a
        store_true so argparse rejects it — neither may fire the override."""
        from hermes_cli.main import _system_gateway_request

        assert (
            _system_gateway_request(["gateway", "status", "--systemx"]) is False
        )
        assert (
            _system_gateway_request(["gateway", "status", "--system=x"]) is False
        )
        assert _system_gateway_request(["gateway", "status", "-system"]) is False


class TestSystemdUnitParsing:
    """Finding 3 — systemd Environment= ordered/reset semantics, drop-ins."""

    def _override(self, monkeypatch, tmp_path, unit_text, dropins=None):
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text(unit_text, encoding="utf-8")
        for name, text in (dropins or {}).items():
            d = unit_path.parent / (unit_path.name + ".d")
            d.mkdir(exist_ok=True)
            (d / name).write_text(text, encoding="utf-8")
        monkeypatch.setattr(
            sys, "argv", ["hermes", "gateway", "status", "--system"]
        )
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv(ELEVATED_SENTINEL, raising=False)

        from hermes_cli.main import _apply_system_gateway_home_override

        _apply_system_gateway_home_override(unit_path=unit_path)
        return os.environ.get("HERMES_HOME")

    def test_later_assignment_wins(self, tmp_path, monkeypatch):
        result = self._override(
            monkeypatch,
            tmp_path,
            unit_text=(
                "[Service]\n"
                'Environment="HERMES_HOME=/srv/old-home"\n'
                'Environment="HERMES_HOME=/srv/actual-home"\n'
            ),
        )
        assert result == "/srv/actual-home"

    def test_empty_assignment_resets(self, tmp_path, monkeypatch):
        result = self._override(
            monkeypatch,
            tmp_path,
            unit_text=(
                "[Service]\n"
                'Environment="HERMES_HOME=/srv/old-home"\n'
                'Environment="HERMES_HOME="\n'
            ),
        )
        assert result is None

    def test_hermes_home_in_wrong_section_fails_closed(self, tmp_path, monkeypatch):
        result = self._override(
            monkeypatch,
            tmp_path,
            unit_text='[Unit]\nEnvironment="HERMES_HOME=/srv/unit-section-home"\n',
        )
        assert result is None

    def test_environmentfile_fails_closed(self, tmp_path, monkeypatch):
        result = self._override(
            monkeypatch,
            tmp_path,
            unit_text=(
                "[Service]\n"
                'Environment="HERMES_HOME=/srv/envfile-home"\n'
                "EnvironmentFile=/etc/default/hermes\n"
            ),
        )
        assert result is None

    def test_unsetenvironment_fails_closed(self, tmp_path, monkeypatch):
        result = self._override(
            monkeypatch,
            tmp_path,
            unit_text=(
                "[Service]\n"
                'Environment="HERMES_HOME=/srv/base-home"\n'
                "UnsetEnvironment=HERMES_HOME\n"
            ),
        )
        assert result is None

    def test_specifier_fails_closed(self, tmp_path, monkeypatch):
        result = self._override(
            monkeypatch,
            tmp_path,
            unit_text='[Service]\nEnvironment="HERMES_HOME=%h/.hermes"\n',
        )
        assert result is None

    def test_dropin_overrides_base_unit(self, tmp_path, monkeypatch):
        result = self._override(
            monkeypatch,
            tmp_path,
            unit_text='[Service]\nEnvironment="HERMES_HOME=/srv/base-home"\n',
            dropins={
                "10-later.conf": (
                    '[Service]\nEnvironment="HERMES_HOME=/srv/dropin-home"\n'
                )
            },
        )
        assert result == "/srv/dropin-home"

    def test_dropin_ordering_is_lexicographic(self, tmp_path, monkeypatch):
        result = self._override(
            monkeypatch,
            tmp_path,
            unit_text='[Service]\nEnvironment="HERMES_HOME=/srv/base-home"\n',
            dropins={
                "20-second.conf": (
                    '[Service]\nEnvironment="HERMES_HOME=/srv/second"\n'
                ),
                "10-first.conf": (
                    '[Service]\nEnvironment="HERMES_HOME=/srv/first"\n'
                ),
            },
        )
        assert result == "/srv/second"

    def test_unreadable_dropin_fails_closed(self, tmp_path, monkeypatch):
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text(
            '[Service]\nEnvironment="HERMES_HOME=/srv/base-home"\n', encoding="utf-8"
        )
        dropin = unit_path.parent / (unit_path.name + ".d") / "10-secret.conf"
        dropin.parent.mkdir(parents=True, exist_ok=True)
        dropin.write_text(
            '[Service]\nEnvironment="HERMES_HOME=/srv/hidden-home"\n', encoding="utf-8"
        )
        dropin.chmod(0o000)
        monkeypatch.setattr(
            sys, "argv", ["hermes", "gateway", "status", "--system"]
        )
        monkeypatch.delenv("HERMES_HOME", raising=False)

        from hermes_cli.main import _apply_system_gateway_home_override

        _apply_system_gateway_home_override(unit_path=unit_path)
        assert os.environ.get("HERMES_HOME") is None

    # ── P2 case-insensitive systemd directive/section matching ──────────

    def test_lowercase_service_section_still_parsed(self, tmp_path, monkeypatch):
        """systemd sections are case-insensitive; [service] must be recognised."""
        result = self._override(
            monkeypatch,
            tmp_path,
            unit_text='[service]\nEnvironment="HERMES_HOME=/srv/lower-section"\n',
        )
        assert result == "/srv/lower-section"

    def test_lowercase_environment_directive_still_parsed(self, tmp_path, monkeypatch):
        """systemd directives are case-insensitive; environment= must be recognised."""
        result = self._override(
            monkeypatch,
            tmp_path,
            unit_text='[Service]\nenvironment="HERMES_HOME=/srv/lower-directive"\n',
        )
        assert result == "/srv/lower-directive"

    def test_lowercase_environmentfile_fails_closed(self, tmp_path, monkeypatch):
        """Lowercase environmentfile= must still fail closed (unmodellable)."""
        result = self._override(
            monkeypatch,
            tmp_path,
            unit_text=(
                "[Service]\n"
                'Environment="HERMES_HOME=/srv/base-home"\n'
                "environmentfile=/etc/default/hermes\n"
            ),
        )
        assert result is None

    def test_lowercase_unsetenvironment_fails_closed(self, tmp_path, monkeypatch):
        """Lowercase unsetenvironment= must still fail closed (unmodellable)."""
        result = self._override(
            monkeypatch,
            tmp_path,
            unit_text=(
                "[Service]\n"
                'Environment="HERMES_HOME=/srv/base-home"\n'
                "unsetenvironment=HERMES_HOME\n"
            ),
        )
        assert result is None

    def test_mixed_case_section_and_directives(self, tmp_path, monkeypatch):
        """Mixed case [Service] with lowercase environmentfile= still fails closed."""
        result = self._override(
            monkeypatch,
            tmp_path,
            unit_text=(
                "[Service]\n"
                'Environment="HERMES_HOME=/srv/base-home"\n'
                "environmentfile=/etc/default/hermes\n"
            ),
        )
        assert result is None

    def test_uppercase_section_still_parsed(self, tmp_path, monkeypatch):
        """[SERVICE] (all-caps) is legal per systemd case-insensitivity."""
        result = self._override(
            monkeypatch,
            tmp_path,
            unit_text='[SERVICE]\nEnvironment="HERMES_HOME=/srv/upper-section"\n',
        )
        assert result == "/srv/upper-section"

    def test_valid_unit_home_still_adopted(self, tmp_path, monkeypatch):
        """Control: the canonical single-assignment unit keeps working."""
        result = self._override(
            monkeypatch,
            tmp_path,
            unit_text='[Service]\nEnvironment="HERMES_HOME=/home/hermes/.hermes"\n',
        )
        assert result == "/home/hermes/.hermes"


class TestOverridePathValidation:
    """Finding 4 — malformed and temporary homes fail closed, never crash."""

    def _override(self, monkeypatch, tmp_path, unit_home):
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text(
            f'[Service]\nEnvironment="HERMES_HOME={unit_home}"\n', encoding="utf-8"
        )
        monkeypatch.setattr(
            sys, "argv", ["hermes", "gateway", "status", "--system"]
        )
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv(ELEVATED_SENTINEL, raising=False)

        from hermes_cli.main import _apply_system_gateway_home_override

        _apply_system_gateway_home_override(unit_path=unit_path)
        return os.environ.get("HERMES_HOME")

    def test_embedded_nul_fails_closed(self, tmp_path, monkeypatch):
        assert self._override(monkeypatch, tmp_path, "/home/\x00evil") is None

    def test_symlink_loop_fails_closed(self, tmp_path, monkeypatch):
        loop = tmp_path / "loop-home"
        loop.symlink_to(loop)
        assert self._override(monkeypatch, tmp_path, str(loop)) is None

    def test_literal_tmp_refused_even_when_tmpdir_elsewhere(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("TMPDIR", str(tmp_path / "elsewhere-tmp"))
        assert self._override(monkeypatch, tmp_path, "/tmp/transient-hermes") is None

    def test_macos_private_tmp_refused(self, tmp_path, monkeypatch):
        assert (
            self._override(monkeypatch, tmp_path, "/private/tmp/transient-hermes")
            is None
        )

    def test_configured_tmpdir_still_refused(self, tmp_path, monkeypatch):
        elsewhere = tmp_path / "elsewhere-tmp"
        elsewhere.mkdir()
        monkeypatch.setenv("TMPDIR", str(elsewhere))
        assert self._override(monkeypatch, tmp_path, str(elsewhere / "hermes")) is None

    def test_valid_absolute_home_outside_temp_still_adopted(self, tmp_path, monkeypatch):
        assert (
            self._override(monkeypatch, tmp_path, "/home/hermes/.hermes")
            == "/home/hermes/.hermes"
        )


class TestPrivilegedSystemToolBoundary:
    """Finding 1 (P0) — no user-controlled env may reach privileged tools."""

    @pytest.fixture
    def hostile_unit_home(self):
        shm = Path("/dev/shm")
        if not shm.is_dir() or not os.access(shm, os.W_OK):
            pytest.skip("/dev/shm not writable")
        unit_home = shm / f"hermes-boundary-{uuid.uuid4().hex[:12]}"
        (unit_home / "bin").mkdir(parents=True)
        (unit_home / ".env").write_text(f"PATH={unit_home}/bin\n", encoding="utf-8")
        marker = unit_home / "systemctl-ran"
        fake = unit_home / "bin" / "systemctl"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys\n"
            f"pathlib.Path({str(marker)!r}).write_text(' '.join(sys.argv[1:]))\n"
            "sys.exit(0)\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        yield {"unit_home": unit_home, "marker": marker}
        import shutil as _shutil

        _shutil.rmtree(unit_home, ignore_errors=True)

    def test_elevated_context_cannot_execute_unit_home_systemctl(
        self, tmp_path, monkeypatch, hostile_unit_home
    ):
        """Privileged tool chain: override -> sentinel -> absolute systemctl.

        The full import-time chain as real root is proven by the
        TestFreshProcessBootstrap root tests; this exercises the same seam
        in-process: a root process adopting a hostile unit home must not
        honour that home's .env, and systemctl must resolve absolutely.
        """
        unit_home = hostile_unit_home["unit_home"]
        marker = hostile_unit_home["marker"]

        monkeypatch.setattr(sys, "argv", ["hermes", "gateway", "status", "--system"])
        monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv(ELEVATED_SENTINEL, raising=False)
        # Simulate the attacker's goal: PATH already pointing at the hijack
        # dir BEFORE any dotenv (worst case even without the .env load).
        monkeypatch.setenv("PATH", f"{unit_home}/bin:{SAFE_PATH}")

        _pop_cli_modules()
        try:
            from hermes_cli.main import _apply_system_gateway_home_override

            unit_file = unit_home / "hermes-gateway.service"
            unit_file.write_text(
                f'[Service]\nEnvironment="HERMES_HOME={unit_home}"\n',
                encoding="utf-8",
            )
            _apply_system_gateway_home_override(unit_path=unit_file)

            assert os.environ.get("HERMES_HOME") == str(unit_home)
            assert os.environ.get(ELEVATED_SENTINEL) == "1"

            from hermes_cli import gateway as gateway_cli

            assert gateway_cli._privileged_system_gateway_context() is True
            cmd = gateway_cli._systemctl_cmd(system=True)
            assert cmd[0] == "/usr/bin/systemctl", cmd
            proc = gateway_cli._run_systemctl(
                ["--version"], system=True, capture_output=True, text=True
            )
            assert proc.returncode == 0, proc.stderr[-500:]
            assert not marker.exists(), "fake systemctl from unit home executed!"
        finally:
            monkeypatch.setenv("PATH", SAFE_PATH)
            os.environ.pop("HERMES_HOME", None)
            os.environ.pop(ELEVATED_SENTINEL, None)
            _pop_cli_modules()

    def test_non_elevated_context_keeps_which_lookup(
        self, tmp_path, monkeypatch, hostile_unit_home
    ):
        """Without the elevated sentinel the legacy PATH lookup still works."""
        unit_home = hostile_unit_home["unit_home"]
        monkeypatch.setattr(os, "geteuid", lambda: 1000, raising=False)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv(ELEVATED_SENTINEL, raising=False)
        monkeypatch.setenv(
            "PATH", f"{unit_home}/bin:{SAFE_PATH}"
        )

        _pop_cli_modules()
        try:
            from hermes_cli import gateway as gateway_cli

            assert gateway_cli._privileged_system_gateway_context() is False
            # Unprivileged callers keep PATH resolution — verify via the
            # resolver seam rather than the host-probing helper (which may
            # legitimately answer differently inside containers).
            resolved = gateway_cli.privileged_system_tool_path("systemctl")
            assert resolved == f"{unit_home}/bin/systemctl", resolved
        finally:
            monkeypatch.setenv("PATH", SAFE_PATH)
            _pop_cli_modules()

    def test_sentinel_without_root_cannot_lock_tools(self, monkeypatch):
        """A forged sentinel in a non-root process must be ignored."""
        monkeypatch.setattr(os, "geteuid", lambda: 1000, raising=False)
        monkeypatch.setenv(ELEVATED_SENTINEL, "1")

        from hermes_cli import gateway as gateway_cli

        assert gateway_cli._privileged_system_gateway_context() is False


class TestFreshProcessBootstrap:
    """Finding 5 — import-time behaviour proven in real subprocesses.

    The privileged scenario runs as real root via ``sudo`` (the live host's
    unit carries a root-only-readable drop-in, so a non-root reader must
    correctly fail closed — that behaviour is covered by the unit-parsing
    tests; here we prove the actual privileged bootstrap).
    """

    CAN_RUN_AS_ROOT = os.geteuid() == 0 or subprocess.run(
        ["sudo", "-n", "true"], capture_output=True
    ).returncode == 0

    def _bootstrap(self, tmp_path, argv_tail):
        root_home = tmp_path / "fresh-root"
        root_home.mkdir()
        code = (
            "import os, sys\n"
            "sys.argv = ['hermes'] + sys.argv[1:]\n"
            "import hermes_cli.main  # import-time overrides run\n"
            "print('ENV=' + (os.environ.get('HERMES_HOME') or '-'))\n"
            "from hermes_cli.config import get_hermes_home\n"
            "print('EFF=' + str(get_hermes_home()))\n"
        )
        env = {"HOME": str(root_home), "PATH": SAFE_PATH}
        if os.geteuid() == 0:
            return (
                subprocess.run(
                    [sys.executable, "-c", code, *argv_tail],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    cwd=str(REPO_ROOT),
                    env=env,
                ),
                root_home,
            )
        if not self.CAN_RUN_AS_ROOT:
            pytest.skip("no passwordless sudo for root-subprocess test")
        return (
            subprocess.run(
                [
                    "sudo", "-n", "-E", "--preserve-env=PATH",
                    sys.executable, "-c", code, *argv_tail,
                ],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=str(REPO_ROOT),
                env=env,
            ),
            root_home,
        )

    @staticmethod
    def _lines(proc):
        return dict(
            line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line
        )

    def test_yolo_prefixed_status_selects_unit_home(self, tmp_path):
        proc, root_home = self._bootstrap(tmp_path, ["--yolo", "gateway", "status", "--system"])
        assert proc.returncode == 0, proc.stderr[-2000:]
        out = self._lines(proc)
        assert out["ENV"] == "/home/hermes/.hermes", out
        assert out["EFF"] == "/home/hermes/.hermes", out
        assert not (root_home / ".hermes").exists(), "root scaffold created!"

    def test_run_system_does_not_adopt_unit_home(self, tmp_path):
        proc, root_home = self._bootstrap(tmp_path, ["gateway", "run", "--system"])
        assert proc.returncode == 0, proc.stderr[-2000:]
        out = self._lines(proc)
        assert out["ENV"] == "-", out
        assert out["EFF"] != "/home/hermes/.hermes", out

    def test_double_dash_form_does_not_adopt_unit_home(self, tmp_path):
        proc, _ = self._bootstrap(tmp_path, ["--", "gateway", "status", "--system"])
        assert proc.returncode == 0, proc.stderr[-2000:]
        out = self._lines(proc)
        assert out["ENV"] == "-", out

    def test_canonical_status_adopts_unit_home(self, tmp_path):
        proc, root_home = self._bootstrap(tmp_path, ["gateway", "status", "--system"])
        assert proc.returncode == 0, proc.stderr[-2000:]
        out = self._lines(proc)
        assert out["ENV"] == "/home/hermes/.hermes", out
        assert not (root_home / ".hermes").exists(), "root scaffold created!"
