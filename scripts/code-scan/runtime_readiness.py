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
import json
import os
import re
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
        gate_type = _classify_gate_type(cmd_str) or "verification"
        gate: dict = {
            "command": cmd_str,
            "source": "suggested_verification",
            "gate_type": gate_type,
            "status": status,
            "execution_semantics": "suggested_not_run",
        }
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


_PACKAGE_SCRIPT_GATE_NAMES = {
    "test": "test",
    "build": "build",
    "lint": "lint",
    "typecheck": "typecheck",
    "type-check": "typecheck",
    "type_check": "typecheck",
    "check-types": "typecheck",
    "audit": "audit",
}


def _classify_gate_type(command: str, script_name: str | None = None) -> Optional[str]:
    """Classify known runtime gate commands without executing them."""
    name = (script_name or "").lower().replace(":", "-")
    for marker, gate_type in _PACKAGE_SCRIPT_GATE_NAMES.items():
        if name == marker or name.startswith(f"{marker}-") or name.endswith(f"-{marker}"):
            return gate_type

    normalized = re.sub(r"\s+", " ", command.strip().lower())
    if not normalized:
        return None
    if re.search(r"\b(npm|pnpm|yarn)\s+(ci|install)\b", normalized):
        return "install"
    if re.search(r"\b(pip|poetry|uv|pipenv)\s+.*\binstall\b", normalized):
        return "install"
    if re.search(r"\b(npm|pnpm|yarn)\s+(run\s+)?test\b", normalized):
        return "test"
    if any(token in normalized for token in ("pytest", "go test", "cargo test")):
        return "test"
    if re.search(r"\b(npm|pnpm|yarn)\s+(run\s+)?build\b", normalized):
        return "build"
    if "docker build" in normalized:
        return "build"
    if re.search(r"\b(npm|pnpm|yarn)\s+(run\s+)?lint\b", normalized):
        return "lint"
    if re.search(r"\b(eslint|ruff|flake8|pylint)\b", normalized):
        return "lint"
    if re.search(r"\b(npm|pnpm|yarn)\s+run\s+(typecheck|type-check|check-types)\b", normalized):
        return "typecheck"
    if re.search(r"\b(tsc|mypy|pyright)\b", normalized):
        return "typecheck"
    if re.search(r"\b(npm|pnpm|yarn)\s+(run\s+)?audit\b", normalized):
        return "audit"
    if re.search(r"\b(pip-audit|cargo audit|cargo-audit)\b", normalized):
        return "audit"
    return None


def _gate_record(command: str, source: str, gate_type: str) -> dict:
    return {
        "command": command,
        "source": source,
        "gate_type": gate_type,
        "status": "suggested_not_run",
        "execution_semantics": "suggested_not_run",
    }


def _package_script_gates(target_dir: str) -> list[dict]:
    """Read package.json scripts and expose recognized gates as inventory only."""
    td = Path(target_dir)
    pj_path = td / "package.json"
    if not pj_path.exists():
        return []

    try:
        package_json = json.loads(pj_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    scripts = package_json.get("scripts", {})
    if not isinstance(scripts, dict):
        return []

    pm_name = _infer_pm(str(td)) or "npm"
    gates: list[dict] = []
    for script_name in sorted(scripts):
        script_command = scripts.get(script_name)
        if not isinstance(script_command, str):
            continue
        gate_type = _classify_gate_type(script_command, script_name)
        if gate_type is None:
            continue
        command = f"{pm_name} test" if script_name == "test" else f"{pm_name} run {script_name}"
        gates.append(_gate_record(
            command=command,
            source=f"package.json:scripts.{script_name}",
            gate_type=gate_type,
        ))
    return gates


def _workflow_run_commands(workflow_path: Path) -> list[tuple[int, str]]:
    """Extract simple GitHub Actions run commands without YAML execution semantics."""
    try:
        lines = workflow_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    commands: list[tuple[int, str]] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        match = re.match(r"^\s*(?:-\s*)?run:\s*(.*)$", line)
        if not match:
            idx += 1
            continue
        value = match.group(1).strip()
        line_no = idx + 1
        if value in {"|", ">", "|-", ">-"}:
            base_indent = len(line) - len(line.lstrip())
            idx += 1
            while idx < len(lines):
                block_line = lines[idx]
                if not block_line.strip():
                    idx += 1
                    continue
                indent = len(block_line) - len(block_line.lstrip())
                if indent <= base_indent:
                    break
                command = block_line.strip()
                if command and not command.startswith("#"):
                    commands.append((idx + 1, command))
                idx += 1
            continue
        command = value.strip('"\'')
        if command and not command.startswith("#"):
            commands.append((line_no, command))
        idx += 1
    return commands


def _ci_workflow_gates(target_dir: str) -> list[dict]:
    """Parse GitHub Actions run commands conservatively as non-executed gates."""
    td = Path(target_dir)
    workflow_dir = td / ".github" / "workflows"
    if not workflow_dir.exists():
        return []

    gates: list[dict] = []
    workflow_files = sorted(
        list(workflow_dir.glob("*.yml")) + list(workflow_dir.glob("*.yaml")),
        key=lambda p: p.name,
    )
    for workflow_path in workflow_files:
        rel_path = workflow_path.relative_to(td).as_posix()
        for line_no, command in _workflow_run_commands(workflow_path):
            gate_type = _classify_gate_type(command)
            if gate_type is None:
                continue
            gates.append(_gate_record(
                command=command,
                source=f"{rel_path}:{line_no}",
                gate_type=gate_type,
            ))
    return gates


def _build_runtime_gate_inventory(target_dir: str, verification_gates: list[dict]) -> list[dict]:
    """Combine inferred, package-script, and CI gates without executing them."""
    gates = list(verification_gates)
    gates.extend(_package_script_gates(target_dir))
    gates.extend(_ci_workflow_gates(target_dir))

    seen: set[tuple[str, str, str]] = set()
    unique: list[dict] = []
    for gate in gates:
        key = (gate["command"], gate["source"], gate["gate_type"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(gate)
    return unique


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
    runtime_gate_inventory = _build_runtime_gate_inventory(target_dir, verification_gates)

    return {
        "detected_stacks": stacks,
        "required_commands": all_commands,
        "suggested_verification": all_suggestions,
        "verification_status": verification_status,
        "blockers": blockers,
        # UA-P5-007: explicit verification_gates for suggested commands (status contract only)
        "verification_gates": verification_gates,
        "runtime_gate_inventory": runtime_gate_inventory,
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
