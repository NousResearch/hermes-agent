"""runtime_readiness.py — UA-P1-003 Runtime Readiness Artifact.

Deterministic runtime/toolchain readiness reporting so UA can distinguish
scan success from blocked build/test verification.

Detection heuristics:
- Go: go.mod or *.go → check `go version`; suggest `go test -short ./...`
- Node: package.json → check `node --version` + package manager; suggest scripts
- Python: pyproject.toml, requirements.txt, or setup.py → check `python --version`;
  suggest `python -m pytest` if tests/ exist
- Rust: Cargo.toml → check `cargo --version`; suggest `cargo test`
- Docker: Dockerfile, docker-compose.yml, or compose.yaml → check `docker --version`

Safety: only runs version/tool-availability commands. Never runs build/test.

Produces:
- runtime-readiness.json (machine-readable)
- runtime-readiness.md (human-readable)
"""
import glob
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


# ── Stack detection ─────────────────────────────────────────────────────

def detect_stacks(target_dir: str) -> list[str]:
    """Detect programming-language/runtime stacks present in *target_dir*.

    Returns an ordered list of stack names (e.g. ``["go", "python"]``).
    """
    stacks: list[str] = []
    td = Path(target_dir)

    # Go
    if (td / "go.mod").exists() or list(td.glob("*.go")):
        stacks.append("go")

    # Node
    if (td / "package.json").exists():
        stacks.append("node")

    # Python
    if ((td / "pyproject.toml").exists()
            or (td / "requirements.txt").exists()
            or (td / "setup.py").exists()):
        stacks.append("python")

    # Rust
    if (td / "Cargo.toml").exists():
        stacks.append("rust")

    # Docker
    if ((td / "Dockerfile").exists()
            or (td / "docker-compose.yml").exists()
            or (td / "compose.yaml").exists()):
        stacks.append("docker")

    return stacks


# ── Command availability —────────────────────────────────────────────────

def check_command(command: str) -> dict:
    """Check whether *command* is available and return its version info.

    Returns:
        {
            "command": <str>,
            "available": <bool>,
            "version": <str or None>,
            "reason": <str>,
        }
    """
    path = shutil.which(command)
    if path is None:
        return {
            "command": command,
            "available": False,
            "version": None,
            "reason": f"{command} command not found",
        }

    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        version_line = (result.stdout or result.stderr or "").splitlines()
        version = version_line[0].strip() if version_line else None
    except (subprocess.TimeoutExpired, OSError):
        version = None

    return {
        "command": command,
        "available": True,
        "version": version,
        "reason": f"{command} available at {path}",
    }


# ── Stack-specific command requirements ──────────────────────────────────

def _go_commands(target_dir: str) -> list[dict]:
    """Required commands for a Go stack."""
    version_reason = "go.mod present" if (Path(target_dir) / "go.mod").exists() else "*.go files present"
    cmd = check_command("go")
    if not cmd["available"]:
        cmd["reason"] = f"command not found ({version_reason})"
    else:
        cmd["reason"] = version_reason
    return [cmd]


def _node_commands(target_dir: str) -> list[dict]:
    """Required commands for a Node stack."""
    cmds = []
    node_cmd = check_command("node")
    if not node_cmd["available"]:
        node_cmd["reason"] = f"command not found (package.json present)"
    else:
        node_cmd["reason"] = "package.json present"
    cmds.append(node_cmd)

    # Check for at least one package manager
    pm_found = False
    for pm in ("npm", "pnpm", "yarn"):
        pm_cmd = check_command(pm)
        if not pm_cmd["available"]:
            pm_cmd["reason"] = f"command not found (package.json present; {pm} package manager)"
        else:
            pm_cmd["reason"] = "package.json present (package manager)"
        cmds.append(pm_cmd)
        if pm_cmd["available"]:
            pm_found = True
    return cmds


def _python_commands(target_dir: str) -> list[dict]:
    """Required commands for a Python stack."""
    if (Path(target_dir) / "pyproject.toml").exists():
        reason = "pyproject.toml present"
    elif (Path(target_dir) / "requirements.txt").exists():
        reason = "requirements.txt present"
    else:
        reason = "setup.py present"

    # Check `python` binary
    cmd = check_command("python")
    if not cmd["available"]:
        reason = f"command not found ({reason})"

    result = {"command": "python", "available": cmd["available"],
              "version": cmd["version"], "reason": reason}
    return [result]


def _rust_commands(target_dir: str) -> list[dict]:
    """Required commands for a Rust stack."""
    cmd = check_command("cargo")
    if not cmd["available"]:
        cmd["reason"] = f"command not found (Cargo.toml present)"
    else:
        cmd["reason"] = "Cargo.toml present"
    return [cmd]


def _docker_commands(target_dir: str) -> list[dict]:
    """Required commands for a Docker stack."""
    cmd = check_command("docker")
    if (Path(target_dir) / "Dockerfile").exists():
        reason = "Dockerfile present"
    elif (Path(target_dir) / "docker-compose.yml").exists():
        reason = "docker-compose.yml present"
    else:
        reason = "compose.yaml present"
    if not cmd["available"]:
        reason = f"command not found ({reason})"
    cmd["reason"] = reason
    return [cmd]


_STACK_COMMAND_MAP = {
    "go": _go_commands,
    "node": _node_commands,
    "python": _python_commands,
    "rust": _rust_commands,
    "docker": _docker_commands,
}


# ── Suggested verification commands ─────────────────────────────────────

def _suggest_verification(stack: str, target_dir: str) -> list[str]:
    """Return suggested verification commands for a detected stack."""
    td = Path(target_dir)

    if stack == "go":
        return ["go test -short ./..."]

    if stack == "node":
        # Read scripts from package.json if available
        suggestions: list[str] = []
        pj_path = td / "package.json"
        if pj_path.exists():
            try:
                import json
                pj = json.loads(pj_path.read_text())
                scripts = pj.get("scripts", {})
                if "test" in scripts:
                    suggestions.append(f"npm test (node; or: {scripts['test']})")
                else:
                    suggestions.append("npm test (node)")
            except Exception:
                suggestions.append("npm test (node)")
        if not suggestions:
            suggestions.append("npm test (node)")
        return suggestions

    if stack == "python":
        suggestions = []
        # Check if a tests/ directory or test files exist
        has_tests = bool(list(td.glob("tests/**/*.py"))) or \
                    bool(list(td.glob("test_*.py"))) or \
                    bool(list(td.glob("*_test.py")))
        if has_tests:
            suggestions.append("python -m pytest")
        else:
            suggestions.append("python -m pytest (if tests exist)")
        return suggestions

    if stack == "rust":
        return ["cargo test"]

    if stack == "docker":
        return ["docker build ."]

    return []


# ── Build completeness artifact ─────────────────────────────────────────

def build_readiness_artifact(target_dir: str) -> dict:
    """Build the runtime-readiness artifact dict for *target_dir*.

    Returns a dict conforming to the minimum JSON shape specified in
    UA-P1-003.
    """
    stacks = detect_stacks(target_dir)

    # Gather all required commands from all detected stacks
    all_commands: list[dict] = []
    for stack in stacks:
        cmd_fn = _STACK_COMMAND_MAP.get(stack)
        if cmd_fn:
            all_commands.extend(cmd_fn(target_dir))

    # Gather suggested verification
    all_suggestions: list[str] = []
    for stack in stacks:
        all_suggestions.extend(_suggest_verification(stack, target_dir))

    # Determine blockers and verification status
    blockers = [c["reason"] for c in all_commands if not c["available"]]
    if not stacks:
        verification_status = "unknown"
    elif blockers and any(not c["available"] for c in all_commands):
        verification_status = "verification_blocked"
    elif all_commands and all(c["available"] for c in all_commands):
        verification_status = "verification_ready"
    else:
        verification_status = "verification_ready"

    return {
        "detected_stacks": stacks,
        "required_commands": all_commands,
        "suggested_verification": all_suggestions,
        "verification_status": verification_status,
        "blockers": blockers,
    }


# ── Markdown rendering ──────────────────────────────────────────────────

def readiness_to_markdown(artifact: dict) -> str:
    """Render a readiness artifact dict as human-readable Markdown."""
    lines = [
        "# Runtime Readiness Report",
        "",
        f"**Verification Status**: `{artifact['verification_status']}`",
        "",
    ]

    lines.append("## Detected Stacks")
    lines.append("")
    if artifact["detected_stacks"]:
        for stack in artifact["detected_stacks"]:
            lines.append(f"- {stack}")
    else:
        lines.append("- *(none detected)*")
    lines.append("")

    lines.append("## Required Commands")
    lines.append("")
    for cmd in artifact["required_commands"]:
        status_icon = "✅" if cmd["available"] else "❌"
        ver = cmd.get("version") or "N/A"
        lines.append(f"- {status_icon} `{cmd['command']}` — {ver} ({cmd['reason']})")
    lines.append("")

    lines.append("## Suggested Verification")
    lines.append("")
    if artifact["suggested_verification"]:
        for s in artifact["suggested_verification"]:
            lines.append(f"- `{s}`")
    else:
        lines.append("- *(none)*")
    lines.append("")

    if artifact["blockers"]:
        lines.append("## Blockers")
        lines.append("")
        for b in artifact["blockers"]:
            lines.append(f"- ⚠️ {b}")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by runtime_readiness.py (UA-P1-003)*")
    return "\n".join(lines) + "\n"


# ── Convenience: write artifacts ─────────────────────────────────────────

def write_readiness_artifacts(target_dir: str, out_dir: str) -> dict:
    """Build and write runtime-readiness.json and runtime-readiness.md.

    Returns the artifact dict.
    """
    import json

    artifact = build_readiness_artifact(target_dir)

    json_path = os.path.join(out_dir, "runtime-readiness.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
        f.write("\n")

    md_path = os.path.join(out_dir, "runtime-readiness.md")
    md_content = readiness_to_markdown(artifact)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return artifact
