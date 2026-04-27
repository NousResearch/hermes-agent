#!/usr/bin/env python3
"""
CodeIntelligenceService — pragmatic diagnostics for Hermes Code Mode.

Provides workspace and file-level diagnostics by running safe, read-only
commands (typecheck, lint, vet) and normalizing output into a unified
Diagnostic format. Does NOT modify files or execute destructive commands.

Supported stacks:
  - TypeScript/Node: npm/pnpm/yarn/bun run typecheck, run lint
  - Go: go vet ./..., go test ./...
"""

import json
import logging
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

DiagnosticSeverity = str  # "error" | "warning" | "info" | "hint"

DIAGNOSTIC_SEVERITIES = ("error", "warning", "info", "hint")


def _make_diagnostic(
    file: str = "",
    line: Optional[int] = None,
    column: Optional[int] = None,
    severity: str = "error",
    source: str = "",
    code: Optional[str] = None,
    message: str = "",
    raw: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "file": file,
        "line": line,
        "column": column,
        "severity": severity,
        "source": source,
        "code": code,
        "message": message,
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

# TypeScript tsc output:
#   src/file.tsx(10,5): error TS2322: Type 'string' is not assignable to type 'number'.
#   path/file.ts(42,1): error TS1005: ';' expected.
_TS_RE = re.compile(
    r"^(?P<file>.+?)\((?P<line>\d+),(?P<col>\d+)\):\s+"
    r"(?P<severity>error|warning|info)\s+"
    r"(?P<code>TS\d+):\s+"
    r"(?P<message>.+)$"
)

# Go compiler / vet output:
#   path/file.go:12:8: undefined: Foo
#   path/file.go:42:2: printf format %s has arg of wrong type int
_GO_RE = re.compile(r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s+(?P<message>.+)$")

# ESLint stylish / compact:
#   /path/file.tsx
#     10:5  error  Some message  rule-name
#   /path/file.tsx: line 10, col 5, Error - msg (rule)
_ESLINT_FILE_RE = re.compile(r"^(?P<file>.+?)$")
_ESLINT_ENTRY_RE = re.compile(
    r"^\s+(?P<line>\d+):(?P<col>\d+)\s+"
    r"(?P<severity>error|warning|info)\s+"
    r"(?P<message>.+?)\s+"
    r"(?P<code>\S+)\s*$"
)
_ESLINT_COMPACT_RE = re.compile(
    r"^(?P<file>.+?):\s+line\s+(?P<line>\d+),\s+col\s+(?P<col>\d+),\s+"
    r"(?P<severity>Error|Warning|Info)\s+-\s+(?P<message>.+?)\s+\((?P<code>\S+)\)\s*$"
)


def parse_typescript_output(output: str) -> List[Dict[str, Any]]:
    """Parse tsc --noEmit output into diagnostics."""
    diagnostics: List[Dict[str, Any]] = []
    for line_text in output.splitlines():
        line_text = line_text.rstrip()
        if not line_text:
            continue
        m = _TS_RE.match(line_text)
        if m:
            diagnostics.append(
                _make_diagnostic(
                    file=m.group("file"),
                    line=int(m.group("line")),
                    column=int(m.group("col")),
                    severity=m.group("severity"),
                    source="typescript",
                    code=m.group("code"),
                    message=m.group("message"),
                )
            )
    return diagnostics


def parse_go_output(output: str) -> List[Dict[str, Any]]:
    """Parse go vet / go test output into diagnostics."""
    diagnostics: List[Dict[str, Any]] = []
    for line_text in output.splitlines():
        line_text = line_text.rstrip()
        if not line_text:
            continue
        # Skip lines like "# package/name" or "FAIL  package/name"
        if (
            line_text.startswith("# ")
            or line_text.startswith("FAIL")
            or line_text.startswith("ok")
        ):
            continue
        # Skip "--- FAIL:" test headers
        if line_text.startswith("--- FAIL:"):
            continue
        m = _GO_RE.match(line_text)
        if m:
            msg = m.group("message")
            # Classify severity: "undefined" or compile errors are errors;
            # vet warnings tend to be warnings
            severity = "error"
            if "vet:" in msg.lower() or "should" in msg.lower():
                severity = "warning"
            diagnostics.append(
                _make_diagnostic(
                    file=m.group("file"),
                    line=int(m.group("line")),
                    column=int(m.group("col")),
                    severity=severity,
                    source="go",
                    code=None,
                    message=msg,
                )
            )
    return diagnostics


def parse_eslint_output(output: str) -> List[Dict[str, Any]]:
    """Parse ESLint output into diagnostics. Falls back to raw if unparseable."""
    diagnostics: List[Dict[str, Any]] = []
    current_file = ""

    for line_text in output.splitlines():
        line_text_stripped = line_text.rstrip()
        if not line_text_stripped:
            continue

        # Try compact format first
        cm = _ESLINT_COMPACT_RE.match(line_text_stripped)
        if cm:
            sev = cm.group("severity").lower()
            if sev not in DIAGNOSTIC_SEVERITIES:
                sev = "error"
            diagnostics.append(
                _make_diagnostic(
                    file=cm.group("file"),
                    line=int(cm.group("line")),
                    column=int(cm.group("col")),
                    severity=sev,
                    source="eslint",
                    code=cm.group("code"),
                    message=cm.group("message"),
                )
            )
            continue

        # Try file header (stylish format — absolute or relative path)
        fm = _ESLINT_FILE_RE.match(line_text_stripped)
        if fm and (
            "/" in line_text_stripped
            or "\\" in line_text_stripped
            or line_text_stripped.endswith(
                (".ts", ".tsx", ".js", ".jsx", ".vue", ".svelte")
            )
        ):
            current_file = line_text_stripped
            continue

        # Try entry line (stylish format)
        em = _ESLINT_ENTRY_RE.match(line_text)
        if em and current_file:
            sev = em.group("severity").lower()
            if sev not in DIAGNOSTIC_SEVERITIES:
                sev = "error"
            diagnostics.append(
                _make_diagnostic(
                    file=current_file,
                    line=int(em.group("line")),
                    column=int(em.group("col")),
                    severity=sev,
                    source="eslint",
                    code=em.group("code"),
                    message=em.group("message").rstrip(),
                )
            )
            continue

    # If nothing was parsed but there was output, store as raw
    if not diagnostics and output.strip():
        diagnostics.append(
            _make_diagnostic(
                source="eslint",
                severity="error",
                message=output.strip()[:500],
                raw=output.strip()[:2000],
            )
        )

    return diagnostics


# ---------------------------------------------------------------------------
# Command detection helpers
# ---------------------------------------------------------------------------


def _has_script(root: Path, script_name: str) -> bool:
    """Check if a package.json has a specific script."""
    pkg_path = root / "package.json"
    if not pkg_path.exists():
        return False
    try:
        pkg = json.loads(pkg_path.read_text(encoding="utf-8", errors="ignore"))
        scripts = pkg.get("scripts") or {}
        return script_name in scripts
    except Exception:
        return False


def _detect_package_manager_cmd(root: Path) -> str:
    """Return the package manager run prefix (e.g. 'npm run', 'pnpm run')."""
    if (root / "pnpm-lock.yaml").exists():
        return "pnpm run"
    if (root / "yarn.lock").exists():
        return "yarn"
    if (root / "bun.lock").exists() or (root / "bun.lockb").exists():
        return "bun run"
    return "npm run"


# ---------------------------------------------------------------------------
# CodeIntelligenceService
# ---------------------------------------------------------------------------


class CodeIntelligenceService:
    """Pragmatic code diagnostics for Hermes Code Mode.

    Runs safe, read-only diagnostic commands and normalizes output into a
    unified Diagnostic format. Does not modify files.

    Uses WorkspaceDB for workspace resolution. Optionally accepts a
    realtime_hub for WebSocket event broadcasting.
    """

    def __init__(self, db_path: Optional[Path] = None, realtime_hub=None):
        self._db_path = db_path
        self._realtime_hub = realtime_hub

    def _workspace_db(self):
        from hermes_state import WorkspaceDB

        return WorkspaceDB(db_path=self._db_path)

    def _diagnostics_db(self):
        from hermes_state import CodeDiagnosticsDB

        return CodeDiagnosticsDB(db_path=self._db_path)

    def _session_db(self):
        from hermes_state import CodeSessionDB

        return CodeSessionDB(db_path=self._db_path)

    async def _broadcast(self, event_type: str, payload: dict):
        if self._realtime_hub:
            try:
                await self._realtime_hub.broadcast(event_type, payload)
            except Exception:
                pass

    def _add_timeline_event(
        self,
        code_session_id: Optional[str],
        event_type: str,
        message: str,
        payload: dict,
    ):
        if not code_session_id:
            return
        db = self._session_db()
        try:
            db.add_event(code_session_id, event_type, message=message, payload=payload)
        except Exception:
            pass
        finally:
            db.close()

    def _get_workspace(self, workspace_id: str) -> dict:
        wdb = self._workspace_db()
        try:
            ws = wdb.get_workspace(workspace_id)
        finally:
            wdb.close()
        if not ws:
            raise ValueError(f"Workspace not found: {workspace_id}")
        return ws

    def _get_workspace_root(self, workspace_id: str) -> Path:
        ws = self._get_workspace(workspace_id)
        root = Path(ws["path"]).resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Workspace path does not exist: {root}")
        return root

    # ── Language detection ──

    def get_supported_languages(self, workspace_id: str) -> List[str]:
        """Detect which languages/tools can produce diagnostics for this workspace."""
        ws = self._get_workspace(workspace_id)
        root = Path(ws["path"])
        stack = ws.get("detected_stack") or []
        languages: List[str] = []

        if "typescript" in stack or "node" in stack:
            if "typescript" in stack:
                languages.append("typescript")
            # ESLint is common in Node projects
            if _has_script(root, "lint"):
                languages.append("eslint")

        if "go" in stack:
            languages.append("go")

        return languages

    # ── Command building ──

    def _build_typecheck_command(self, root: Path) -> Optional[str]:
        """Build the typecheck command for a Node/TS project."""
        pm = _detect_package_manager_cmd(root)
        if _has_script(root, "typecheck"):
            return f"{pm} typecheck"
        # Fallback: tsc --noEmit if tsconfig exists and tsc is available
        if (root / "tsconfig.json").exists():
            return "npx tsc --noEmit"
        return None

    def _build_lint_command(self, root: Path) -> Optional[str]:
        """Build the lint command for a Node project."""
        pm = _detect_package_manager_cmd(root)
        if _has_script(root, "lint"):
            return f"{pm} lint"
        return None

    def _build_go_vet_command(self, root: Path) -> Optional[str]:
        """Build go vet command if workspace is Go."""
        if (root / "go.mod").exists():
            return "go vet ./..."
        return None

    def _build_go_test_command(self, root: Path) -> Optional[str]:
        """Build go test command if workspace is Go."""
        if (root / "go.mod").exists():
            return "go test ./..."
        return None

    # ── Command execution ──

    def _run_diagnostic_command(
        self, command: str, cwd: Path, timeout: int = 120
    ) -> Tuple[int, str, str]:
        """Run a diagnostic command and return (exit_code, stdout, stderr)."""
        logger.info("Running diagnostic: %s (cwd=%s)", command, cwd)
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return -1, "", str(e)

    def _classify_command_safety(self, command: str) -> str:
        """Check if the command is safe to run. Uses CommandRunner classification."""
        from hermes_cli.code.command_runner import classify_command

        return classify_command(command)

    # ── Core diagnostics ──

    def _run_stack_diagnostics(
        self, root: Path, stack: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Run all applicable diagnostic commands for a stack.

        Returns (diagnostics, commands_run).
        """
        all_diagnostics: List[Dict[str, Any]] = []
        commands_run: List[str] = []

        # TypeScript typecheck
        if "typescript" in stack or "node" in stack:
            cmd = self._build_typecheck_command(root)
            if cmd and self._classify_command_safety(cmd) == "safe":
                rc, stdout, stderr = self._run_diagnostic_command(cmd, root)
                combined = stdout + "\n" + stderr
                parsed = parse_typescript_output(combined)
                all_diagnostics.extend(parsed)
                commands_run.append(cmd)

            # ESLint
            lint_cmd = self._build_lint_command(root)
            if lint_cmd and self._classify_command_safety(lint_cmd) == "safe":
                rc, stdout, stderr = self._run_diagnostic_command(lint_cmd, root)
                combined = stdout + "\n" + stderr
                parsed = parse_eslint_output(combined)
                all_diagnostics.extend(parsed)
                commands_run.append(lint_cmd)

        # Go vet
        if "go" in stack:
            vet_cmd = self._build_go_vet_command(root)
            if vet_cmd and self._classify_command_safety(vet_cmd) == "safe":
                rc, stdout, stderr = self._run_diagnostic_command(vet_cmd, root)
                combined = stdout + "\n" + stderr
                parsed = parse_go_output(combined)
                all_diagnostics.extend(parsed)
                commands_run.append(vet_cmd)

        return all_diagnostics, commands_run

    def _build_summary(self, diagnostics: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count diagnostics by severity."""
        summary = {"errors": 0, "warnings": 0, "info": 0, "hints": 0, "total": 0}
        for d in diagnostics:
            sev = d.get("severity", "error")
            if sev == "error":
                summary["errors"] += 1
            elif sev == "warning":
                summary["warnings"] += 1
            elif sev == "info":
                summary["info"] += 1
            elif sev == "hint":
                summary["hints"] += 1
            summary["total"] += 1
        return summary

    # ── Public API ──

    def get_workspace_diagnostics(
        self,
        workspace_id: str,
        code_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run diagnostics for the entire workspace.

        Returns WorkspaceDiagnosticsResult dict.
        """
        ws = self._get_workspace(workspace_id)
        root = Path(ws["path"])
        stack = ws.get("detected_stack") or []

        if not stack:
            result = {
                "workspace_id": workspace_id,
                "status": "unsupported",
                "diagnostics": [],
                "summary": self._build_summary([]),
                "commands_run": [],
                "duration_ms": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._add_timeline_event(
                code_session_id,
                "diagnostics.completed",
                message="No supported stack detected",
                payload={"workspace_id": workspace_id, "status": "unsupported"},
            )
            return result

        start = time.monotonic()

        # Emit started event
        self._add_timeline_event(
            code_session_id,
            "diagnostics.started",
            message=f"Diagnostics started for workspace {workspace_id}",
            payload={"workspace_id": workspace_id, "stack": stack},
        )

        try:
            diagnostics, commands_run = self._run_stack_diagnostics(root, stack)
            duration_ms = int((time.monotonic() - start) * 1000)
            summary = self._build_summary(diagnostics)
            status = "ok" if summary["errors"] == 0 else "partial"

            result = {
                "workspace_id": workspace_id,
                "status": status,
                "diagnostics": diagnostics,
                "summary": summary,
                "commands_run": commands_run,
                "duration_ms": duration_ms,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # Persist if DB available
            try:
                db = self._diagnostics_db()
                try:
                    db.save_diagnostics(
                        workspace_id=workspace_id,
                        code_session_id=code_session_id,
                        source="workspace",
                        status=status,
                        diagnostics=diagnostics,
                        summary=summary,
                        commands=commands_run,
                        duration_ms=duration_ms,
                    )
                finally:
                    db.close()
            except Exception:
                pass

            self._add_timeline_event(
                code_session_id,
                "diagnostics.completed",
                message=f"Diagnostics completed: {summary['errors']} errors, {summary['warnings']} warnings",
                payload={"workspace_id": workspace_id, "summary": summary},
            )

            return result

        except Exception as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            self._add_timeline_event(
                code_session_id,
                "diagnostics.failed",
                message=f"Diagnostics failed: {str(e)}",
                payload={"workspace_id": workspace_id, "error": str(e)},
            )
            return {
                "workspace_id": workspace_id,
                "status": "error",
                "diagnostics": [],
                "summary": self._build_summary([]),
                "commands_run": [],
                "duration_ms": duration_ms,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

    def get_file_diagnostics(
        self,
        workspace_id: str,
        file_path: str,
        code_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run workspace diagnostics and filter to a specific file.

        file_path is relative to the workspace root.
        """
        result = self.get_workspace_diagnostics(
            workspace_id, code_session_id=code_session_id
        )

        # Normalize file_path for matching
        fp = file_path.replace("\\", "/").strip("/")
        filtered = [
            d
            for d in result["diagnostics"]
            if d.get("file", "").replace("\\", "/").strip("/").endswith(fp)
            or fp.endswith(d.get("file", "").replace("\\", "/").strip("/"))
        ]

        return {
            "workspace_id": workspace_id,
            "file_path": file_path,
            "status": result["status"],
            "diagnostics": filtered,
            "summary": self._build_summary(filtered),
            "commands_run": result["commands_run"],
            "duration_ms": result["duration_ms"],
            "created_at": result["created_at"],
        }

    def restart_language_services(self, workspace_id: str) -> Dict[str, Any]:
        """Restart language services (stub — returns noop in this phase).

        In a future phase, this would restart an LSP server process.
        For now it's a safe no-op that confirms the workspace exists.
        """
        ws = self._get_workspace(workspace_id)
        return {
            "workspace_id": workspace_id,
            "action": "restart",
            "status": "noop",
            "message": "No LSP server running. Diagnostics use direct commands.",
        }
