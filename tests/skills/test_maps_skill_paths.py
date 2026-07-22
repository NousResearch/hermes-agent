"""Path-safety contracts for the bundled maps skill."""

import os
import subprocess
from pathlib import Path


SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "productivity"
    / "maps"
    / "SKILL.md"
)


def test_maps_helper_uses_hermes_home_under_sandbox_home(tmp_path):
    """Operational commands must not expand ~/.hermes beneath profile HOME."""
    content = SKILL_MD.read_text(encoding="utf-8")
    assignment = next(
        line for line in content.splitlines() if line.startswith("MAPS=")
    )

    assert "~/.hermes/skills/maps" not in content

    hermes_home = tmp_path / ".hermes"
    sandbox_home = hermes_home / "home"
    sandbox_home.mkdir(parents=True)
    env = {
        "HOME": str(sandbox_home),
        "HERMES_HOME": str(hermes_home),
        "PATH": os.environ.get("PATH", ""),
    }
    result = subprocess.run(
        ["bash", "-c", f'{assignment}\nprintf "%s" "$MAPS"'],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout == str(
        hermes_home / "skills" / "maps" / "scripts" / "maps_client.py"
    )
