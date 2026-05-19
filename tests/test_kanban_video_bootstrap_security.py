"""Security tests for the kanban video bootstrap setup generator."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import textwrap
from pathlib import Path


BOOTSTRAP_PATH = (
    Path(__file__).resolve().parents[1]
    / "optional-skills/creative/kanban-video-orchestrator/scripts/bootstrap_pipeline.py"
)


def load_bootstrap_module():
    """Load the bootstrap script as a test module despite hyphenated paths."""
    spec = importlib.util.spec_from_file_location(
        "kanban_video_bootstrap", BOOTSTRAP_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def minimal_plan(**overrides):
    """Return a valid minimal plan for setup generation tests."""
    plan = {
        "title": "Safe demo",
        "slug": "safe-demo",
        "tenant": "safe_demo",
        "duration_s": 30,
        "aspect": "16:9",
        "resolution": "1920x1080",
        "fps": 30,
        "team": [
            {
                "profile": "director",
                "role": "director",
                "toolsets": ["kanban", "terminal", "file"],
                "skills": ["kanban-orchestrator"],
                "responsibilities": "Decompose and route work.",
                "inputs": "brief.md, TEAM.md",
                "outputs": "kanban tasks",
            }
        ],
        "scenes": [
            {"n": 1, "time": "0:00-0:05", "content": "Intro", "tool": "director"}
        ],
        "audio": {"approach": "none", "vo": "n/a", "music": "n/a", "sfx": "n/a"},
        "deliverables": [
            {"format": "mp4", "resolution": "1920x1080", "notes": "primary"}
        ],
        "api_keys_required": [],
        "brief_extra": {},
    }
    plan.update(overrides)
    return plan


def test_validation_rejects_shell_sensitive_identifiers():
    """Reject fields that are later used as shell identifiers or command args."""
    bootstrap = load_bootstrap_module()
    plan = minimal_plan(
        tenant="$(touch /tmp/tenant_pwn)",
        api_keys_required=["OPENROUTER_API_KEY", "BAD;touch /tmp/key_pwn"],
    )
    plan["team"][0]["toolsets"] = ["kanban", "bad;touch"]

    errors = bootstrap.validate_plan(plan)

    assert any("tenant must match" in error for error in errors)
    assert any("api_keys_required[1]" in error for error in errors)
    assert any("team[0].toolsets[1]" in error for error in errors)


def test_generated_setup_does_not_execute_plan_content(tmp_path):
    """Untrusted title and markdown content must be data, not shell syntax."""
    bootstrap = load_bootstrap_module()
    marker = tmp_path / "pwned"
    plan = minimal_plan(
        title=f"Demo $(touch {marker})",
        brief_extra={
            "concept_one_liner": "BRIEF_EOF\n"
            f"touch {marker}\n"
            "TEAM_EOF\nSOUL_EOF\n$(touch /also-not-run)",
        },
    )
    assert bootstrap.validate_plan(plan) == []

    setup_path = tmp_path / "setup.sh"
    setup_path.write_text(
        bootstrap.render_setup_sh(
            plan, bootstrap.render_brief(plan), bootstrap.render_team_md(plan)
        )
    )
    setup_path.chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    hermes_log = tmp_path / "hermes.log"
    fake_hermes = bin_dir / "hermes"
    fake_hermes.write_text(
        textwrap.dedent(
            f"""
            #!/usr/bin/env bash
            set -euo pipefail
            printf '%s\n' "$*" >> {hermes_log!s}
            if [ "${{1:-}} ${{2:-}}" = "profile create" ]; then
                mkdir -p "$HOME/.hermes/profiles/$3"
                printf '{{}}\n' > "$HOME/.hermes/profiles/$3/config.yaml"
            fi
            """
        ).strip()
        + "\n"
    )
    fake_hermes.chmod(0o755)

    # Minimal PyYAML shim for the generated profile patcher subprocess.
    (tmp_path / "yaml.py").write_text(
        "import json\n"
        "def safe_load(f):\n"
        "    data = f.read()\n"
        "    return json.loads(data) if data.strip() else {}\n"
        "def safe_dump(data, f, sort_keys=False):\n"
        "    json.dump(data, f)\n"
    )

    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    subprocess.run(
        ["bash", str(setup_path)], cwd=tmp_path, env=env, check=True, timeout=20
    )

    assert not marker.exists()
    assert f"Direct production of Demo $(touch {marker})" in hermes_log.read_text()
    brief_path = Path(env["HOME"]) / "projects/video-pipeline/safe-demo/brief.md"
    assert "BRIEF_EOF" in brief_path.read_text()
