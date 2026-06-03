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
            "classification": <str>,  # set to 'required' by default; stack
                                       # functions may override
        }
    """
    path = shutil.which(command)
    if path is None:
        return {
            "command": command,
            "available": False,
            "version": None,
            "reason": f"{command} command not found",
            "classification": "required",
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
        "classification": "required",
    }


# ── Stack-specific command requirements ──────────────────────────────────

def _go_commands(target_dir: str) -> list[dict]:
    """Required commands for a Go stack."""
    version_reason = "go.mod present" if (Path(target_dir) / "go.mod").exists() else "*.go files present"
    cmd = check_command("go")
    cmd["classification"] = "required"
    if not cmd["available"]:
        cmd["reason"] = f"command not found ({version_reason})"
    else:
        cmd["reason"] = version_reason
    return [cmd]


def _infer_pm(target_dir: str) -> Optional[str]:
    """Infer the preferred package manager from lockfiles.

    - package-lock.json → npm
    - pnpm-lock.yaml    → pnpm
    - yarn.lock         → yarn
    - None              → no lockfile found
    """
    td = Path(target_dir)
    if (td / "pnpm-lock.yaml").exists():
        return "pnpm"
    if (td / "yarn.lock").exists():
        return "yarn"
    if (td / "package-lock.json").exists():
        return "npm"
    return None


def _node_commands(target_dir: str) -> list[dict]:
    """Required commands for a Node stack with package-manager classification.

    package-lock.json => npm is 'required'
    pnpm-lock.yaml    => pnpm is 'required'
    yarn.lock         => yarn is 'required'
    no lockfile       => all PMs are 'optional_alternative'

    Only 'required' commands missing become blockers.
    Missing 'optional_alternative' commands are information-only.
    """
    cmds = []
    node_cmd = check_command("node")
    node_cmd["classification"] = "required"
    if not node_cmd["available"]:
        node_cmd["reason"] = "node command not found (package.json present)"
    else:
        node_cmd["reason"] = "package.json present"
    cmds.append(node_cmd)

    # Detect lockfile to infer preferred PM
    preferred_pm = _infer_pm(target_dir)
    all_pms = ("npm", "pnpm", "yarn")

    for pm in all_pms:
        pm_cmd = check_command(pm)
        if preferred_pm and pm == preferred_pm:
            pm_cmd["classification"] = "required"
            pm_cmd["reason"] = f"package.json present (preferred: {preferred_pm})"
            if not pm_cmd["available"]:
                pm_cmd["reason"] = (
                    f"command not found (preferred {preferred_pm} from lockfile)"
                )
        else:
            pm_cmd["classification"] = "optional_alternative"
            pm_cmd["reason"] = (
                "package.json present (optional alternative package manager)"
            )
            if not pm_cmd["available"]:
                pm_cmd["reason"] = (
                    f"command not found (optional alternative: {pm})"
                )
        cmds.append(pm_cmd)

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
    cmd["classification"] = "required"
    if not cmd["available"]:
        reason = f"command not found ({reason})"

    result = {"command": "python", "available": cmd["available"],
              "version": cmd["version"], "reason": reason,
              "classification": cmd["classification"]}
    return [result]


def _rust_commands(target_dir: str) -> list[dict]:
    """Required commands for a Rust stack."""
    cmd = check_command("cargo")
    cmd["classification"] = "required"
    if not cmd["available"]:
        cmd["reason"] = f"command not found (Cargo.toml present)"
    else:
        cmd["reason"] = "Cargo.toml present"
    return [cmd]


def _docker_commands(target_dir: str) -> list[dict]:
    """Required commands for a Docker stack."""
    cmd = check_command("docker")
    cmd["classification"] = "required"
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
        inferred_pm = _infer_pm(str(td))
        pm_name = inferred_pm or "npm"
        if pj_path.exists():
            try:
                import json as _json
                pj = _json.loads(pj_path.read_text())
                scripts = pj.get("scripts", {})
                if "test" in scripts:
                    suggestions.append(
                        f"{pm_name} test (node; or: {scripts['test']})"
                    )
                else:
                    suggestions.append(f"{pm_name} test (node)")
            except Exception:
                suggestions.append(f"{pm_name} test (node)")
        if not suggestions:
            suggestions.append(f"{pm_name} test (node)")
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


def _build_verification_gates(
    suggestions: list[str], all_commands: list[dict], stacks: list[str]
) -> list[dict]:
    """Map inferred suggested verification commands to explicit verification_gates entries.

    UA-P5-007: All *inferred/suggested* become `status: "suggested_not_run"` by default.
    Use `blocked_missing_tool` only for suggestions whose core required tool is already
    known-missing (from readiness blockers/required_commands meta).
    Never execute the gates/commands here — status is declarative only.
    Empty if no suggestions inferred.
    Each entry has at least: command, status. Optional deterministic context (e.g. stack hints) included
    if clearly inferrable without broadening.
    """
    ALLOWED_STATUSES = {
        "suggested_not_run",
        "executed_passed",
        "executed_failed",
        "blocked_missing_tool",
        "not_inferred",
    }
    gates: list[dict] = []
    for raw in suggestions:
        cmd_str = raw.strip() if isinstance(raw, str) else ""
        if not cmd_str:
            continue
        # UA-P5-007: inferred commands are suggestions only. UA does not execute
        # them and must not infer pass/fail from tool availability.
        status = "suggested_not_run"
        lead = cmd_str.split()[0].lower() if cmd_str else ""
        # enough context: include 'stack' hint from first matching detected, if unambiguous
        gate: dict = {"command": cmd_str, "status": status}
        # attach optional stack if determinable for this suggestion context
        if stacks and len(stacks) == 1:
            gate["stack"] = stacks[0]
        elif stacks:
            # minimal, don't over-infer; if e.g. go test → go
            if "go test" in cmd_str or lead == "go":
                if "go" in stacks:
                    gate["stack"] = "go"
            elif any(x in cmd_str for x in ("pytest", "python -m")):
                if "python" in stacks:
                    gate["stack"] = "python"
            # etc. keep other stacks simple
            else:
                for st in stacks:
                    if st in ("node", "rust", "docker") and st in cmd_str.lower():
                        gate["stack"] = st
                        break
        # name/tool optional would be redundant with command here; omit for minimal
        if status not in ALLOWED_STATUSES:
            status = "suggested_not_run"  # safety
            gate["status"] = status
        gates.append(gate)
    return gates


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
    # UA-P5-002: Only 'required' commands missing become blockers.
    # 'optional_alternative' missing entries are informational only.
    blockers = [
        c["reason"] for c in all_commands
        if c.get("classification") == "required" and not c["available"]
    ]
    if not stacks:
        verification_status = "unknown"
    elif blockers:
        verification_status = "verification_blocked"
    elif all_commands and all(c["available"] for c in all_commands):
        verification_status = "verification_ready"
    else:
        # Some optional_alternative commands missing but all required present
        verification_status = "verification_ready"

    verification_gates = _build_verification_gates(
        all_suggestions, all_commands, stacks
    )

    return {
        "detected_stacks": stacks,
        "required_commands": all_commands,
        "suggested_verification": all_suggestions,
        "verification_status": verification_status,
        "blockers": blockers,
        # UA-P5-007: explicit verification_gates for suggested commands (status contract only)
        "verification_gates": verification_gates,
    }


# ── Markdown rendering ──────────────────────────────────────────────────

def readiness_to_markdown(artifact: dict) -> str:
    """Render a readiness artifact dict as human-readable Markdown.

    UA-P5-002 classification-aware:
    - required commands missing → ❌ blocker
    - optional_alternative commands missing → ⚠️ informational (not a blocker)
    - The Blockers section never lists optional_alternative commands.
    """
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
        classification = cmd.get("classification", "required")
        if cmd["available"]:
            status_icon = "✅"
        elif classification == "optional_alternative":
            # UA-P5-002: missing optional alternatives are informational, not blockers
            status_icon = "⚠️"
        else:
            status_icon = "❌"
        ver = cmd.get("version") or "N/A"
        classification_note = ""
        if classification != "required":
            classification_note = f" [{classification}]"
        lines.append(
            f"- {status_icon} `{cmd['command']}` — {ver}"
            f" ({cmd['reason']}){classification_note}"
        )
    lines.append("")

    lines.append("## Suggested Verification")
    lines.append("")
    if artifact["suggested_verification"]:
        for s in artifact["suggested_verification"]:
            lines.append(f"- `{s}`")
    else:
        lines.append("- *(none)*")
    lines.append("")

    lines.append("## Verification Gates")
    lines.append("")
    gates = artifact.get("verification_gates", [])
    if gates:
        lines.append("UA records these gates as suggested or externally reported status only; it does not execute them.")
        lines.append("")
        lines.append("| Command | Status | Stack |")
        lines.append("|---------|--------|-------|")
        for gate in gates:
            command = gate.get("command", "")
            status = gate.get("status", "not_inferred")
            stack = gate.get("stack", "")
            lines.append(f"| `{command}` | `{status}` | {stack or '—'} |")
    else:
        lines.append("- *(none inferred)*")
    lines.append("")

    # UA-P5-002: Blockers section only lists required-command failures.
    # Missing optional_alternative commands are NOT blockers.
    if artifact["blockers"]:
        lines.append("## Blockers")
        lines.append("")
        for b in artifact["blockers"]:
            lines.append(f"- ⚠️ {b}")
        lines.append("")
    else:
        # Check if any optional_alternative commands are missing
        has_missing_optional = any(
            not c["available"]
            for c in artifact["required_commands"]
            if c.get("classification") == "optional_alternative"
        )
        if has_missing_optional:
            lines.append("## Notes")
            lines.append("")
            missing = [
                c["command"]
                for c in artifact["required_commands"]
                if c.get("classification") == "optional_alternative"
                and not c["available"]
            ]
            lines.append(
                f"- Missing optional alternative package managers "
                f"({', '.join(missing)}) are **not blocking**. "
                f"These are informational only."
            )
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
