from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "autonomous-ai-agents"
    / "qodercli"
)
SKILL_PATH = SKILL_DIR / "SKILL.md"
SCRIPT_PATH = SKILL_DIR / "scripts" / "qodercli_delegate.sh"

# Extensions tools/skills_tool.py globs when building linked_files.scripts.
DISCOVERED_SCRIPT_SUFFIXES = {".py", ".sh", ".bash", ".js", ".ts", ".rb"}

# The JSON contract the helper promises on stdout.
RESULT_KEYS = {
    "exit_code",
    "error_class",
    "files_changed",
    "diff_stat",
    "output_tail",
    "workdir",
    "timeout_used",
    "git_before",
    "git_after",
}

BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(
    BASH is None or sys.platform == "win32",
    reason="the delegate helper is Bash-only (skill is gated to linux/macos)",
)


# ── SKILL.md content contracts ────────────────────────────────────────────


@pytest.fixture
def skill_text() -> str:
    assert SKILL_PATH.exists(), f"SKILL.md not found at {SKILL_PATH}"
    return SKILL_PATH.read_text(encoding="utf-8")


@pytest.fixture
def frontmatter(skill_text: str) -> dict[str, str]:
    match = re.match(r"^---\n(.*?)\n---", skill_text, re.DOTALL)
    assert match, "No YAML frontmatter found"
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "-")):
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip().strip('"')
    return fields


REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]


def test_required_sections(skill_text: str):
    for section in REQUIRED_SECTIONS:
        assert section in skill_text, f"Missing required section: {section}"


def test_section_order(skill_text: str):
    positions = [skill_text.index(s) for s in REQUIRED_SECTIONS]
    assert positions == sorted(positions), "Sections are out of order"


def test_description_length(frontmatter: dict[str, str]):
    desc = frontmatter.get("description", "")
    assert len(desc) <= 60, f"Description is {len(desc)} chars (max 60)"
    assert desc.endswith("."), "Description must end with a period"


def test_no_marketing_words(frontmatter: dict[str, str]):
    desc = frontmatter.get("description", "").lower()
    for word in ("powerful", "comprehensive", "seamless", "advanced", "robust"):
        assert word not in desc, f"Marketing word '{word}' in description"


def test_author_not_hermes_agent(frontmatter: dict[str, str]):
    assert frontmatter.get("author") != "Hermes Agent", (
        "Author must credit the human contributor first"
    )


def test_required_env_vars_declared(skill_text: str):
    assert "required_environment_variables" in skill_text
    assert "QODER_PERSONAL_ACCESS_TOKEN" in skill_text


def test_uses_terminal_tool(skill_text: str):
    assert "terminal(" in skill_text, "Skill must reference the terminal tool"


def test_line_count(skill_text: str):
    lines = skill_text.count("\n")
    assert lines <= 250, f"Skill is {lines} lines (target ~200, hard cap 250)"


def test_no_raw_shell_utilities_in_prose(skill_text: str):
    in_fence = False
    prose_lines = []
    for line in skill_text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and not line.startswith(("terminal(", "|", "#")):
            prose_lines.append(line)
    prose = " ".join(prose_lines)
    assert "grep " not in prose, "Use search_files instead of grep"
    assert "cat " not in prose, "Use read_file instead of cat"


def _parse_related_skills(text: str) -> list[str]:
    match = re.search(r"related_skills:\s*\[([^\]]*)\]", text)
    if not match:
        return []
    return [s.strip() for s in match.group(1).split(",") if s.strip()]


def test_related_skills_bidirectional(skill_text: str):
    related = _parse_related_skills(skill_text)
    assert related, "qodercli must declare related_skills"
    skills_dir = SKILL_DIR.parent
    for name in related:
        sibling = skills_dir / name / "SKILL.md"
        assert sibling.exists(), f"Related skill '{name}' not found at {sibling}"
        assert "qodercli" in _parse_related_skills(sibling.read_text(encoding="utf-8")), (
            f"Skill '{name}' does not list qodercli in its related_skills"
        )


# ── Platform gating ───────────────────────────────────────────────────────


def test_platforms_match_helper_shebang(frontmatter: dict[str, str]):
    """The declared platforms must not outrun what the bundled helper can run on.

    Derived from the helper's shebang rather than a hardcoded expectation, so
    the assertion still holds if the helper is ever ported.
    """
    listed = {p.strip() for p in frontmatter.get("platforms", "").strip("[]").split(",")}
    listed.discard("")
    assert listed, "platforms must be declared"
    assert listed <= {"linux", "macos", "windows"}, f"Unknown platforms: {listed}"

    shebang = SCRIPT_PATH.read_text(encoding="utf-8").splitlines()[0]
    if shebang.startswith("#!") and ("bash" in shebang or shebang.rstrip().endswith("sh")):
        assert "windows" not in listed, (
            f"Helper shebang is {shebang!r} but platforms include windows"
        )


def test_helper_is_discoverable_by_skill_view():
    """Guard the progressive-disclosure contract.

    tools/skills_tool.py only surfaces files matching DISCOVERED_SCRIPT_SUFFIXES
    inside a scripts/ directory. A helper anywhere else, or without one of those
    extensions, is invisible to skill_view and the agent can never find it.
    """
    assert SCRIPT_PATH.parent.name == "scripts", (
        "Helper must live in scripts/ to appear in linked_files.scripts"
    )
    assert SCRIPT_PATH.suffix in DISCOVERED_SCRIPT_SUFFIXES, (
        f"Suffix {SCRIPT_PATH.suffix!r} is not globbed by skills_tool.py"
    )


def test_skill_invokes_helper_by_relative_path(skill_text: str):
    assert "scripts/qodercli_delegate.sh" in skill_text, (
        "SKILL.md must invoke the helper by its scripts/-relative path"
    )


def test_helper_is_executable():
    mode = SCRIPT_PATH.stat().st_mode
    assert mode & stat.S_IXUSR, "Helper lost its executable bit"


# ── Behavioral coverage of the helper ─────────────────────────────────────


@pytest.fixture
def delegate(tmp_path: Path):
    """Run the helper against a stub qodercli. No network, no real Qoder."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    workdir = tmp_path / "proj"
    workdir.mkdir()

    def run(
        prompt: str = "do the thing",
        *,
        stub: str = "echo ok\n",
        wd: Path | None = None,
        timeout_arg: str = "30",
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        fake = bin_dir / "qodercli"
        fake.write_text("#!/bin/bash\n" + stub, encoding="utf-8")
        fake.chmod(0o755)

        child_env = dict(os.environ)
        child_env["HERMES_QODERCLI_BIN"] = str(fake)
        child_env["TMPDIR"] = str(tmp_path)
        if env:
            child_env.update(env)

        argv = [BASH, str(SCRIPT_PATH), prompt, str(wd or workdir), timeout_arg]
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            env=child_env,
            timeout=120,
        )

    return run


@needs_bash
def test_success_emits_full_json_contract(delegate):
    proc = delegate(stub='printf "all done\\n"')
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert RESULT_KEYS <= set(data), f"Missing keys: {RESULT_KEYS - set(data)}"
    assert data["error_class"] == "none"
    assert data["exit_code"] == 0
    assert data["timeout_used"] == 30
    assert "all done" in data["output_tail"]


@needs_bash
@pytest.mark.parametrize(
    ("stub", "expected"),
    [
        ('echo "Error: Not logged in. Please run /login" >&2; exit 1', "auth_failure"),
        ('echo "HTTP 402: credit limit exhausted" >&2; exit 1', "credit_exhausted"),
        ('echo "Permission confirmation required" >&2; exit 1', "permission_blocked"),
        ('echo "dial tcp: connect ECONNREFUSED" >&2; exit 1', "network_error"),
        ('echo "unmodelled explosion" >&2; exit 1', "unknown_failure"),
    ],
)
def test_error_classification(delegate, stub, expected):
    proc = delegate(stub=stub)
    assert proc.returncode == 1, proc.stderr
    data = json.loads(proc.stdout)
    assert data["error_class"] == expected
    assert data["exit_code"] == 1


@needs_bash
def test_hostile_output_still_parses_as_json(delegate):
    """Real qodercli output carries ANSI escapes, tabs, form feeds, backslashes
    and quotes. All of them must survive into valid JSON."""
    proc = delegate(
        stub=r"""printf '\033[31mERR\033[0m bs=\\ tab\t ff\f quote="x" nl\n'""" + "\n"
    )
    data = json.loads(proc.stdout)  # raises if escaping regresses
    tail = data["output_tail"]
    assert data["error_class"] == "none"
    assert "ERR" in tail
    assert "\\" in tail, "a literal backslash must survive as data"
    assert '"' in tail, "a literal double quote must survive as data"
    assert all(ord(c) >= 32 for c in tail), f"control byte leaked: {tail!r}"


@needs_bash
def test_timeout_is_classified(delegate):
    if shutil.which("timeout") is None and shutil.which("gtimeout") is None:
        pytest.skip("no timeout(1) on this platform")
    proc = delegate(stub="sleep 30\n", timeout_arg="1")
    assert proc.returncode == 124, proc.stderr
    data = json.loads(proc.stdout)
    assert data["error_class"] == "timeout"
    assert data["exit_code"] == 124


@needs_bash
def test_honors_tmpdir(delegate, tmp_path):
    """The scratch file must land in TMPDIR, not a hardcoded /tmp."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    # While the stub runs, the helper's own scratch file is live in TMPDIR.
    proc = delegate(
        stub='ls -1 "${TMPDIR:-/tmp}"/qodercli-delegate.* 2>/dev/null | wc -l | tr -d " "\n',
        env={"TMPDIR": str(scratch)},
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert int(data["output_tail"].strip()) >= 1, (
        f"No scratch file created under TMPDIR; output_tail={data['output_tail']!r}"
    )


@needs_bash
def test_workdir_not_found(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "qodercli"
    fake.write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    fake.chmod(0o755)

    env = dict(os.environ, HERMES_QODERCLI_BIN=str(fake))
    proc = subprocess.run(
        [BASH, str(SCRIPT_PATH), "p", str(tmp_path / "nope"), "30"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["error_class"] == "workdir_not_found"


@needs_bash
def test_binary_not_found(tmp_path):
    empty = tmp_path / "emptybin"
    empty.mkdir()
    env = dict(os.environ, PATH=str(empty))
    env.pop("HERMES_QODERCLI_BIN", None)

    proc = subprocess.run(
        [BASH, str(SCRIPT_PATH), "p", str(tmp_path), "30"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert data["error_class"] == "binary_not_found"
    assert data["exit_code"] == 127


@needs_bash
@pytest.mark.parametrize("argv", [[], [""]])
def test_missing_prompt_is_preflight_failure(argv):
    """A usage error is a preflight failure (exit 2), and must still be JSON."""
    proc = subprocess.run(
        [BASH, str(SCRIPT_PATH), *argv],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert data["error_class"] == "usage_error"
    assert "Usage:" in data["message"]
