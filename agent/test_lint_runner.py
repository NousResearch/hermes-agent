"""Test and lint runner detection, execution, and output parsing.

Provides automatic detection of test runners and linters based on project
configuration files, plus parsers for extracting structured error information
from their output.
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Test runner detection
# ---------------------------------------------------------------------------

def detect_test_runner(cwd: str) -> Optional[Dict[str, str]]:
    """Detect the project's test runner based on config files in *cwd*.

    Returns a dict with ``cmd`` and ``type`` keys, or ``None`` if no runner
    is detected.
    """
    cwd = str(cwd)

    # Python: pytest
    if os.path.exists(os.path.join(cwd, "pytest.ini")):
        return {"cmd": "python -m pytest", "type": "pytest"}
    pyproject = os.path.join(cwd, "pyproject.toml")
    if os.path.exists(pyproject):
        try:
            text = Path(pyproject).read_text(encoding="utf-8")
            if "[tool.pytest" in text:
                return {"cmd": "python -m pytest", "type": "pytest"}
        except OSError:
            pass

    # Node: npm test
    pkg_json = os.path.join(cwd, "package.json")
    if os.path.exists(pkg_json):
        try:
            data = json.loads(Path(pkg_json).read_text(encoding="utf-8"))
            if data.get("scripts", {}).get("test"):
                return {"cmd": "npm test", "type": "jest/mocha"}
        except (OSError, json.JSONDecodeError):
            pass

    # Rust: cargo test
    if os.path.exists(os.path.join(cwd, "Cargo.toml")):
        return {"cmd": "cargo test", "type": "cargo"}

    # Go: go test
    if os.path.exists(os.path.join(cwd, "go.mod")):
        return {"cmd": "go test ./...", "type": "go"}

    # Makefile with test target
    makefile = os.path.join(cwd, "Makefile")
    if os.path.exists(makefile):
        try:
            text = Path(makefile).read_text(encoding="utf-8")
            if re.search(r"^test\s*:", text, re.MULTILINE):
                return {"cmd": "make test", "type": "make"}
        except OSError:
            pass

    return None


# ---------------------------------------------------------------------------
# Linter detection
# ---------------------------------------------------------------------------

def detect_linter(cwd: str) -> Optional[Dict[str, str]]:
    """Detect the project's linter based on config files in *cwd*.

    Returns a dict with ``cmd`` and ``type`` keys, or ``None``.
    """
    cwd = str(cwd)

    # Python: ruff
    if os.path.exists(os.path.join(cwd, "ruff.toml")):
        return {"cmd": "ruff check .", "type": "ruff"}
    pyproject = os.path.join(cwd, "pyproject.toml")
    if os.path.exists(pyproject):
        try:
            text = Path(pyproject).read_text(encoding="utf-8")
            if "[tool.ruff" in text:
                return {"cmd": "ruff check .", "type": "ruff"}
        except OSError:
            pass

    # Node: eslint
    for name in os.listdir(cwd):
        if name.startswith(".eslintrc"):
            return {"cmd": "npm run lint", "type": "eslint"}
    pkg_json = os.path.join(cwd, "package.json")
    if os.path.exists(pkg_json):
        try:
            data = json.loads(Path(pkg_json).read_text(encoding="utf-8"))
            if data.get("scripts", {}).get("lint"):
                return {"cmd": "npm run lint", "type": "eslint"}
        except (OSError, json.JSONDecodeError):
            pass

    # Rust: clippy
    if os.path.exists(os.path.join(cwd, "Cargo.toml")):
        return {"cmd": "cargo clippy", "type": "clippy"}

    # Go: golangci-lint
    if os.path.exists(os.path.join(cwd, "go.mod")):
        return {"cmd": "golangci-lint run", "type": "golangci"}

    return None


# ---------------------------------------------------------------------------
# Test output parsing
# ---------------------------------------------------------------------------

def parse_test_errors(output: str, runner_type: str) -> List[Dict[str, Any]]:
    """Parse test runner output into structured error dicts.

    Each dict has keys: ``file``, ``line``, ``error``.
    """
    errors: List[Dict[str, Any]] = []

    if runner_type == "pytest":
        # Match patterns like: FAILED tests/test_foo.py::test_bar - AssertionError: ...
        for m in re.finditer(r"FAILED\s+([\w/\\._-]+)::(\S+)", output):
            errors.append({"file": m.group(1), "line": 0, "error": f"FAILED {m.group(2)}"})
        # Also match tracebacks: file.py:123: AssertionError
        for m in re.finditer(r"([\w/\\._-]+\.py):(\d+):\s+(\w+Error.*)", output):
            errors.append({"file": m.group(1), "line": int(m.group(2)), "error": m.group(3).strip()})

    elif runner_type == "jest/mocha":
        # Match: ● Test Suite > test name
        # and FAIL src/foo.test.js
        for m in re.finditer(r"FAIL\s+([\w/\\._-]+)", output):
            errors.append({"file": m.group(1), "line": 0, "error": "FAIL"})
        for m in re.finditer(r"at\s+.*\(([\w/\\._-]+):(\d+):\d+\)", output):
            errors.append({"file": m.group(1), "line": int(m.group(2)), "error": "test failure"})

    elif runner_type == "cargo":
        # Match: error[E0308]: mismatched types --> src/main.rs:10:5
        for m in re.finditer(r"-->\s+([\w/\\._-]+):(\d+):\d+", output):
            errors.append({"file": m.group(1), "line": int(m.group(2)), "error": "test failure"})
        for m in re.finditer(r"(error\[E\d+\]):\s+(.+)", output):
            if errors:
                errors[-1]["error"] = f"{m.group(1)}: {m.group(2)}"

    elif runner_type == "go":
        # Match: --- FAIL: TestFoo (0.00s)
        # and file_test.go:123: ...
        for m in re.finditer(r"([\w/\\._-]+\.go):(\d+):\s+(.+)", output):
            errors.append({"file": m.group(1), "line": int(m.group(2)), "error": m.group(3).strip()})

    else:
        # Generic: look for file:line patterns
        for m in re.finditer(r"([\w/\\._-]+\.\w+):(\d+):\s+(.+)", output):
            errors.append({"file": m.group(1), "line": int(m.group(2)), "error": m.group(3).strip()})

    return errors


# ---------------------------------------------------------------------------
# Lint output parsing
# ---------------------------------------------------------------------------

def parse_lint_errors(output: str, linter_type: str) -> List[Dict[str, Any]]:
    """Parse linter output into structured error dicts.

    Each dict has keys: ``file``, ``line``, ``error``, ``rule``.
    """
    errors: List[Dict[str, Any]] = []

    if linter_type == "ruff":
        # Match: src/foo.py:10:5: E501 Line too long
        for m in re.finditer(r"([\w/\\._-]+\.py):(\d+):\d+:\s+(\S+)\s+(.+)", output):
            errors.append({
                "file": m.group(1),
                "line": int(m.group(2)),
                "rule": m.group(3),
                "error": m.group(4).strip(),
            })

    elif linter_type == "eslint":
        # Match:  10:5  error  No unused vars  no-unused-vars
        # File headers: /path/to/file.js
        current_file = ""
        for line in output.splitlines():
            stripped = line.strip()
            if stripped and not stripped[0].isdigit() and not stripped.startswith("✖") and os.sep in stripped:
                current_file = stripped
                continue
            m = re.match(r"\s*(\d+):\d+\s+(?:error|warning)\s+(.+?)\s{2,}(\S+)\s*$", line)
            if m and current_file:
                errors.append({
                    "file": current_file,
                    "line": int(m.group(1)),
                    "error": m.group(2).strip(),
                    "rule": m.group(3),
                })

    elif linter_type == "clippy":
        # Match: warning: unused variable --> src/main.rs:5:9
        for m in re.finditer(r"(warning|error)(?:\[(\w+)\])?:\s+(.+?)(?:\n\s*-->\s+([\w/\\._-]+):(\d+):\d+)?", output, re.DOTALL):
            if m.group(4):
                errors.append({
                    "file": m.group(4),
                    "line": int(m.group(5)),
                    "error": m.group(3).strip(),
                    "rule": m.group(2) or m.group(1),
                })

    elif linter_type == "golangci":
        # Match: file.go:10:5: errcheck: ...
        for m in re.finditer(r"([\w/\\._-]+\.go):(\d+):\d+:\s+(\S+):\s+(.+)", output):
            errors.append({
                "file": m.group(1),
                "line": int(m.group(2)),
                "rule": m.group(3),
                "error": m.group(4).strip(),
            })

    else:
        # Generic
        for m in re.finditer(r"([\w/\\._-]+\.\w+):(\d+):\d*:?\s+(.+)", output):
            errors.append({
                "file": m.group(1),
                "line": int(m.group(2)),
                "error": m.group(3).strip(),
                "rule": "",
            })

    return errors


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def run_tests(cwd: str, timeout: int = 120) -> Dict[str, Any]:
    """Detect and run the project test suite.

    Returns a dict with ``success``, ``output``, ``errors``, and ``runner``
    keys.
    """
    runner = detect_test_runner(cwd)
    if not runner:
        return {"success": False, "output": "", "errors": [], "runner": None,
                "message": "No test runner detected in this project."}

    try:
        proc = subprocess.run(
            runner["cmd"],
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout + "\n" + proc.stderr
        errors = parse_test_errors(output, runner["type"])
        return {
            "success": proc.returncode == 0,
            "output": output.strip(),
            "errors": errors,
            "runner": runner,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "output": f"Tests timed out after {timeout}s",
                "errors": [], "runner": runner}
    except Exception as e:
        return {"success": False, "output": str(e), "errors": [], "runner": runner}


def run_linter(cwd: str, timeout: int = 60) -> Dict[str, Any]:
    """Detect and run the project linter.

    Returns a dict with ``success``, ``output``, ``errors``, and ``linter``
    keys.
    """
    linter = detect_linter(cwd)
    if not linter:
        return {"success": False, "output": "", "errors": [], "linter": None,
                "message": "No linter detected in this project."}

    try:
        proc = subprocess.run(
            linter["cmd"],
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout + "\n" + proc.stderr
        errors = parse_lint_errors(output, linter["type"])
        return {
            "success": proc.returncode == 0,
            "output": output.strip(),
            "errors": errors,
            "linter": linter,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "output": f"Linter timed out after {timeout}s",
                "errors": [], "linter": linter}
    except Exception as e:
        return {"success": False, "output": str(e), "errors": [], "linter": linter}


def format_test_results(result: Dict[str, Any]) -> str:
    """Format test results for display."""
    if not result.get("runner"):
        return result.get("message", "No test runner detected.")

    runner = result["runner"]
    lines = [f"Test runner: {runner['type']} ({runner['cmd']})"]

    if result["success"]:
        lines.append("✅ All tests passed!")
    else:
        lines.append("❌ Tests failed.")

    if result["errors"]:
        lines.append(f"\n{len(result['errors'])} error(s) found:")
        for err in result["errors"][:20]:  # Limit display
            loc = f"{err['file']}:{err['line']}" if err["line"] else err["file"]
            lines.append(f"  • {loc}: {err['error']}")

    # Include truncated raw output
    output = result.get("output", "")
    if output:
        max_lines = 50
        out_lines = output.splitlines()
        if len(out_lines) > max_lines:
            lines.append(f"\nOutput (last {max_lines} of {len(out_lines)} lines):")
            lines.extend(out_lines[-max_lines:])
        else:
            lines.append("\nOutput:")
            lines.extend(out_lines)

    return "\n".join(lines)


def format_lint_results(result: Dict[str, Any]) -> str:
    """Format lint results for display."""
    if not result.get("linter"):
        return result.get("message", "No linter detected.")

    linter = result["linter"]
    lines = [f"Linter: {linter['type']} ({linter['cmd']})"]

    if result["success"]:
        lines.append("✅ No lint issues found!")
    else:
        lines.append("⚠️  Lint issues detected.")

    if result["errors"]:
        lines.append(f"\n{len(result['errors'])} issue(s) found:")
        for err in result["errors"][:30]:  # Limit display
            loc = f"{err['file']}:{err['line']}" if err["line"] else err["file"]
            rule = f" [{err['rule']}]" if err.get("rule") else ""
            lines.append(f"  • {loc}{rule}: {err['error']}")

    return "\n".join(lines)
