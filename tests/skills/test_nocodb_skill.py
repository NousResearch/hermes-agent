"""
Smoke tests for the nocodb optional skill.

Generic frontmatter conformance is not repeated here — test_authoring_
standards.py already sweeps every skill in the repo for it. These are the
nocodb-specific checks: the two platform scripts staying in lockstep, the
SKILL.md command reference staying true to them, and the vendored scripts
keeping their origin and their attribution.

No network. Static analysis of the shipped files only.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "productivity"
    / "nocodb"
)
SH = SKILL_DIR / "scripts" / "nocodb.sh"
PS1 = SKILL_DIR / "scripts" / "nocodb.ps1"

# Bash `case` labels, which may alternate: `where:help|filter:help)`.
_SH_COMMAND = re.compile(r"^([a-z][a-z:|-]*)\)", re.MULTILINE)
# PowerShell `switch` labels: a plain literal, or a script-block condition
# (`{ $_ -eq "where:help" -or $_ -eq "filter:help" }`) covering aliases.
_PS1_COMMAND = re.compile(r'^\s+"([a-z][a-z:-]*)"\s*\{', re.MULTILINE)
_PS1_ALIAS = re.compile(r'\$_ -eq "([a-z][a-z:-]*)"')
_DOC_COMMAND = re.compile(r"`(?:scripts/nocodb\.sh )?([a-z][a-z-]+:[a-z:-]+)`")


def _sh_commands_from(src: str) -> set[str]:
    return {c for label in _SH_COMMAND.findall(src) for c in label.split("|")}


def _ps1_commands_from(src: str) -> set[str]:
    return set(_PS1_COMMAND.findall(src)) | set(_PS1_ALIAS.findall(src))


@pytest.fixture(scope="module")
def skill_src() -> str:
    return (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_src: str) -> dict:
    m = re.search(r"^---\n(.*?)\n---", skill_src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


@pytest.fixture(scope="module")
def sh_commands() -> set[str]:
    return _sh_commands_from(SH.read_text(encoding="utf-8"))


def test_skill_dir_and_scripts_exist() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"
    assert SH.is_file(), "missing scripts/nocodb.sh"
    assert PS1.is_file(), "missing scripts/nocodb.ps1"


def test_bash_script_is_executable() -> None:
    assert SH.stat().st_mode & 0o111, "scripts/nocodb.sh is not executable"


def test_mit_provenance_recorded(frontmatter: dict) -> None:
    # Upstream ships no LICENSE file, so the header on each script and this
    # frontmatter field are the only MIT notice travelling with the code.
    assert frontmatter["license"] == "MIT"
    for script in (SH, PS1):
        head = script.read_text(encoding="utf-8")[:600]
        assert "github.com/nocodb/agent-skills" in head, (
            f"{script.name} lost its upstream provenance header"
        )


def test_declares_token_env_var(frontmatter: dict) -> None:
    names = {e["name"] for e in frontmatter["required_environment_variables"]}
    assert "NOCODB_TOKEN" in names
    assert frontmatter["prerequisites"]["commands"] == ["curl", "jq"]


def test_platforms_match_shipped_scripts(frontmatter: dict) -> None:
    # Windows is only claimable while the PowerShell port ships alongside.
    assert set(frontmatter["platforms"]) == {"macos", "linux", "windows"}
    assert PS1.is_file()


def test_both_scripts_expose_the_same_commands(sh_commands: set[str]) -> None:
    ps1_commands = _ps1_commands_from(PS1.read_text(encoding="utf-8"))
    assert sh_commands, "no commands parsed out of nocodb.sh"
    assert sh_commands == ps1_commands, (
        "Bash/PowerShell command surfaces diverged — "
        f"sh-only={sorted(sh_commands - ps1_commands)} "
        f"ps1-only={sorted(ps1_commands - sh_commands)}"
    )


def test_documented_commands_all_exist(skill_src: str, sh_commands: set[str]) -> None:
    documented = {
        c for c in _DOC_COMMAND.findall(skill_src) if not c.startswith("app.nocodb")
    }
    assert documented, "no commands found in SKILL.md — regex drift?"
    unknown = documented - sh_commands
    assert not unknown, f"SKILL.md documents commands the scripts lack: {sorted(unknown)}"


def test_documented_command_count_matches(skill_src: str, sh_commands: set[str]) -> None:
    m = re.search(r"identical (\d+)-command surface", skill_src)
    assert m, "SKILL.md no longer states the command-surface size"
    assert int(m.group(1)) == len(sh_commands), (
        f"SKILL.md claims {m.group(1)} commands, scripts implement {len(sh_commands)}"
    )


@pytest.mark.parametrize("script", [SH, PS1], ids=["sh", "ps1"])
def test_scripts_contact_only_nocodb_default_origin(script: Path) -> None:
    # Comment lines are skipped so the provenance header can cite upstream.
    code = "\n".join(
        line
        for line in script.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    hosts = set(re.findall(r"https?://[\w.-]+", code))
    assert hosts == {"https://app.nocodb.com"}, f"unexpected hosts: {sorted(hosts)}"


def test_bash_script_parses() -> None:
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash not available")
    proc = subprocess.run(
        [bash, "-n", str(SH)], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, f"bash -n failed:\n{proc.stderr}"
