"""Tests for the creative/pencil skill.

Covers the SKILL.md authoring contract and the two helper scripts
(pencil_repl.py argv/stdin assembly + missing-binary handling, and
pencil_doctor.py status reporting). No live network or real `pencil` calls —
subprocess and PATH lookups are mocked.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "creative"
    / "pencil"
)


def _load(module_name: str, filename: str):
    path = SKILL_DIR / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def repl():
    return _load("pencil_repl", "pencil_repl.py")


@pytest.fixture
def doctor():
    return _load("pencil_doctor", "pencil_doctor.py")


# --------------------------------------------------------------------------
# SKILL.md authoring contract
# --------------------------------------------------------------------------


class TestSkillContract:
    def test_skill_md_exists(self):
        assert (SKILL_DIR / "SKILL.md").is_file()

    def test_description_under_60_chars(self):
        text = (SKILL_DIR / "SKILL.md").read_text()
        m = re.search(r'^description:\s*"?(.*?)"?\s*$', text, re.MULTILINE)
        assert m, "description frontmatter missing"
        desc = m.group(1)
        assert len(desc) <= 60, f"{len(desc)} chars: {desc!r}"
        assert desc.endswith("."), "description should end with a period"

    def test_scripts_and_reference_present(self):
        assert (SKILL_DIR / "scripts" / "pencil_repl.py").is_file()
        assert (SKILL_DIR / "scripts" / "pencil_doctor.py").is_file()
        assert (SKILL_DIR / "references" / "mcp-tools.md").is_file()

    def test_documents_runtime_discovery_and_start_path(self):
        skill = (SKILL_DIR / "SKILL.md").read_text()
        # Feedback from the Pencil team: learn the CLI at runtime, and prefer
        # the emerging headless `pencil start` + MCP path when available.
        assert "pencil --help" in skill
        assert "pencil interactive --help" in skill
        assert "pencil start" in skill
        ref = (SKILL_DIR / "references" / "mcp-tools.md").read_text()
        assert "pencil interactive --help" in ref


# --------------------------------------------------------------------------
# pencil_repl.py
# --------------------------------------------------------------------------


class TestReplArgv:
    def test_headless_requires_out(self, repl):
        with pytest.raises(ValueError):
            repl.build_pencil_argv()

    def test_headless_argv(self, repl):
        argv = repl.build_pencil_argv(out="design.pen", in_path="base.pen")
        assert argv == ["pencil", "interactive", "--in", "base.pen", "--out", "design.pen"]

    def test_app_argv_no_out_needed(self, repl):
        argv = repl.build_pencil_argv(app="desktop")
        assert argv == ["pencil", "interactive", "--app", "desktop"]

    def test_custom_binary(self, repl):
        argv = repl.build_pencil_argv(out="x.pen", pencil_bin="/opt/pencil")
        assert argv[0] == "/opt/pencil"


class TestReplStdin:
    def test_appends_save_and_exit(self, repl):
        payload = repl.build_stdin(["batch_get()"], save=True)
        assert payload == "batch_get()\nsave()\nexit()\n"

    def test_no_save(self, repl):
        payload = repl.build_stdin(["batch_get()"], save=False)
        assert payload == "batch_get()\nexit()\n"

    def test_blank_lines_dropped(self, repl):
        payload = repl.build_stdin(["  ", "a()", ""], save=False)
        assert payload == "a()\nexit()\n"


class TestReplRun:
    def test_missing_binary_raises_with_hint(self, repl):
        with pytest.raises(FileNotFoundError) as ei:
            repl.run_repl(["batch_get()"], out="x.pen", _which=lambda _b: None)
        assert "npm install -g @pencil.dev/cli" in str(ei.value)

    def test_app_mode_forces_no_save(self, repl):
        captured = {}

        def fake_run(argv, **kw):
            captured["argv"] = argv
            captured["input"] = kw.get("input")

            class _P:
                returncode = 0
                stdout = ""
                stderr = ""

            return _P()

        repl.run_repl(
            ["batch_get()"],
            app="desktop",
            save=True,  # requested, but app mode must override to no-save
            _run=fake_run,
            _which=lambda _b: "/usr/bin/pencil",
        )
        # save() must NOT be present in app mode
        assert "save()" not in captured["input"]
        assert captured["input"].endswith("exit()\n")


# --------------------------------------------------------------------------
# pencil_doctor.py
# --------------------------------------------------------------------------


class TestDoctor:
    def _runner(self, mapping):
        """Build a fake subprocess.run keyed by the first argv token tail."""

        class _P:
            def __init__(self, rc, out=""):
                self.returncode = rc
                self.stdout = out
                self.stderr = ""

        def run(argv, **kw):
            key = argv[1] if len(argv) > 1 else argv[0]
            rc, out = mapping.get(key, (1, ""))
            return _P(rc, out)

        return run

    def test_all_present_and_authed(self, doctor):
        which = {"node": "/n/node", "pencil": "/n/pencil"}.get
        runner = self._runner(
            {"--version": (0, "v20.10.0"), "version": (0, "0.2.7"), "status": (0, "ok")}
        )
        st = doctor.check(which=which, runner=runner)
        assert st["node"]["ok"] is True
        assert st["pencil"]["present"] is True
        assert st["auth"]["ok"] is True

    def test_pencil_missing(self, doctor):
        which = {"node": "/n/node"}.get
        runner = self._runner({"--version": (0, "v20.0.0")})
        st = doctor.check(which=which, runner=runner)
        assert st["pencil"]["present"] is False
        assert doctor.main is not None

    def test_old_node_not_ok(self, doctor):
        which = {"node": "/n/node", "pencil": "/n/pencil"}.get
        runner = self._runner(
            {"--version": (0, "v16.0.0"), "version": (0, "0.2.7"), "status": (1, "no")}
        )
        st = doctor.check(which=which, runner=runner)
        assert st["node"]["ok"] is False
        assert st["auth"]["ok"] is False

    def test_major_parse(self, doctor):
        assert doctor._major("v20.10.0") == 20
        assert doctor._major("18.0.0") == 18
        assert doctor._major(None) is None
        assert doctor._major("weird") is None

    def test_capabilities_detected_from_help(self, doctor):
        which = {"node": "/n/node", "pencil": "/n/pencil"}.get
        help_text = "Usage:\n  pencil interactive  ...\n  pencil start --output <f>\n"
        runner = self._runner(
            {
                "--version": (0, "v22.0.0"),
                "version": (0, "0.3.0"),
                "status": (0, "ok"),
                "--help": (0, help_text),
            }
        )
        st = doctor.check(which=which, runner=runner)
        assert st["capabilities"] == {"start": True, "interactive": True}

    def test_capabilities_interactive_only(self, doctor):
        which = {"node": "/n/node", "pencil": "/n/pencil"}.get
        runner = self._runner(
            {
                "--version": (0, "v22.0.0"),
                "version": (0, "0.2.7"),
                "status": (0, "ok"),
                "--help": (0, "Usage:\n  pencil interactive  Start shell\n"),
            }
        )
        st = doctor.check(which=which, runner=runner)
        assert st["capabilities"]["interactive"] is True
        assert st["capabilities"]["start"] is False

    def test_has_subcommand_is_whole_word(self, doctor):
        assert doctor._has_subcommand("pencil start now", "start") is True
        assert doctor._has_subcommand("please restart it", "start") is False
        assert doctor._has_subcommand("", "start") is False
        assert doctor._has_subcommand(None, "start") is False
