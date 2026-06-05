#!/usr/bin/env python3
"""run_ua.py — UA-005 Explicit UA Mode Router.

CLI entrypoint that wraps canonical run-bundle behavior (UA-001) with explicit
mode selection.  Modes control which deterministic pipeline stages are executed.

Defaults to ``structure`` mode for backward compatibility with existing skill
instructions.  Mode routing never hides validation failures.  Quick modes
avoid unnecessary graph analytics.

Usage:
    python run_ua.py --target <target_dir> --out <bundle_dir> [--mode <mode>]

Modes:
    inventory   — scan + imports only
    structure   — scan + imports + graph + validation   [default]
    review      — structure + analytics + context envelope + report
    delta       — incremental scan + delta summary against prior manifest
    preflight   — structure + entrypoints/hubs + subagent context
    full        — all available deterministic enrichers
"""
import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure scripts/code-scan is on sys.path for sibling imports
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from extract_imports import build_import_map
from assemble_graph import assemble_graph
from graph_schema import validate_graph
from domain_surfaces import scan_domain_surfaces
from static_signals import (
    build_static_signals_artifact,
    extract_edge_function_markers,
    extract_package_config_markers,
    extract_supabase_migration_markers,
    extract_rust_agent_infra_markers,
)

# ── Optional enrichers (may be absent if upstream beads not yet available) ---
try:
    from analyze_graph import analyze_graph
    _HAS_ANALYTICS = True
except ImportError:
    _HAS_ANALYTICS = False

try:
    from build_context_bundle import build_context_envelope
    _HAS_CONTEXT = True
except ImportError:
    _HAS_CONTEXT = False

# ── Optional project-state integration (UA-006) ─────────────────────────
try:
    from project_state_append import append_project_state as _append_project_state  # noqa: F401
    _HAS_PROJECT_STATE = True
except ImportError:
    _HAS_PROJECT_STATE = False
    _append_project_state = None  # type: ignore[misc]

# ── Runtime readiness (UA-P1-003) ────────────────────────────────────────
try:
    from runtime_readiness import build_readiness_artifact, readiness_to_markdown  # noqa: F401
    _HAS_READINESS = True
except ImportError:
    _HAS_READINESS = False
    build_readiness_artifact = None  # type: ignore[misc,assignment]
    readiness_to_markdown = None  # type: ignore[misc,assignment]

# ── Report data/render integration ─────────────────────────────────────
try:
    from report_data import build_report_data
    from render_report import render_report_data
    _HAS_REPORT_RENDER = True
except ImportError:
    _HAS_REPORT_RENDER = False
    build_report_data = None  # type: ignore[misc,assignment]
    render_report_data = None  # type: ignore[misc,assignment]

# ── Valid modes ─────────────────────────────────────────────────────────────

VALID_MODES = frozenset([
    "inventory",
    "structure",
    "review",
    "delta",
    "preflight",
    "security-review",
    "full",
])

DEFAULT_MODE = "structure"


# ── Helper: script versioning ────────────────────────────────────────────────

def _script_hash(path: Path) -> str:
    """SHA-256 hex digest of a script file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _get_script_versions() -> dict:
    """Collect script hashes for manifest reproducibility."""
    versions = {
        "scan_project.py": _script_hash(_SCRIPT_DIR / "scan_project.py"),
        "extract_imports.py": _script_hash(_SCRIPT_DIR / "extract_imports.py"),
        "assemble_graph.py": _script_hash(_SCRIPT_DIR / "assemble_graph.py"),
        "graph_schema.py": _script_hash(_SCRIPT_DIR / "graph_schema.py"),
        "domain_surfaces.py": _script_hash(_SCRIPT_DIR / "domain_surfaces.py"),
        "static_signals.py": _script_hash(_SCRIPT_DIR / "static_signals.py"),
        "run_ua.py": _script_hash(_SCRIPT_DIR / "run_ua.py"),
        "run_bundle.py": _script_hash(_SCRIPT_DIR / "run_bundle.py"),
    }
    return versions


# ── Helper: git HEAD ─────────────────────────────────────────────────────────

def _get_git_head(target_dir: str) -> Optional[str]:
    """Return the current git HEAD SHA for *target_dir*, or None."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _get_git_remote(target_dir: str) -> Optional[str]:
    """Return the origin remote URL for *target_dir*, or None."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _artifact_integrity(artifact_paths: dict[str, str]) -> dict:
    """Compute byte sizes and SHA-256 hashes for each artifact."""
    integrity: dict = {}
    for name, path in artifact_paths.items():
        if name == "manifest.json":
            continue
        try:
            stat = os.stat(path)
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            integrity[name] = {"bytes": stat.st_size, "sha256": h.hexdigest()}
        except OSError:
            integrity[name] = {"bytes": 0, "sha256": "missing"}
    return integrity


def _build_provenance(runner_name: str, command_flags: dict, target_dir: str) -> dict:
    """Build provenance metadata for the manifest."""
    git_head = _get_git_head(target_dir)
    git_remote = _get_git_remote(target_dir)
    non_git_reason = None
    if git_head is None:
        non_git_reason = f"target is not a git repository or git is unavailable: {target_dir}"
    return {
        "ua_runner": runner_name,
        "argv": [sys.argv[0]] + sys.argv[1:],
        "target_git_head": git_head,
        "target_git_remote": git_remote,
        "non_git_reason": non_git_reason,
    }


# ── Helper: target cleanliness (UA-P1-002) ───────────────────────────────────

def _get_git_dirty_files(target_dir: str) -> list[str]:
    """Return list of dirty file paths via `git status --porcelain=v1`.

    Returns an empty list if the target is not a git repo or any error
    occurs during the call.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            dirty = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) >= 2:
                    dirty.append(parts[1].strip('"'))
            return dirty
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return []


# ── RunUA class ──────────────────────────────────────────────────────────────

class RunUA:
    """Mode-aware UA pipeline runner.

    Routes to the appropriate pipeline stages based on *mode*, writes artifacts
    to *out_dir*, and records a manifest with mode metadata.
    """

    def __init__(
        self,
        target_dir: str,
        out_dir: str,
        *,
        mode: str = DEFAULT_MODE,
        in_repo_cache: bool = False,
        external_cache_dir: Optional[str] = None,
        prior_manifest: Optional[str] = None,
        project_root: Optional[str] = None,
    ) -> None:
        self.target_dir = os.path.realpath(target_dir)
        self.out_dir = os.path.realpath(out_dir)
        self.mode = mode if mode in VALID_MODES else DEFAULT_MODE
        self.in_repo_cache = in_repo_cache
        self.external_cache_dir = external_cache_dir
        self.prior_manifest = prior_manifest
        self.project_root = os.path.realpath(project_root) if project_root else None

        self.artifact_paths: dict[str, str] = {}
        self.scan_data: Optional[dict] = None
        self.imports_data: Optional[dict] = None
        self.graph_data: Optional[dict] = None
        self.validation_data: Optional[dict] = None
        self.analytics_data: Optional[dict] = None
        self.context_data: Optional[dict] = None
        self.summary_data: Optional[dict] = None
        self.delta_data: Optional[dict] = None
        self.runtime_readiness_data: Optional[dict] = None
        self.static_signals_data: Optional[dict] = None
        self._missing_artifacts: list[str] = []
        self._project_state_status: dict = {
            "project_state_recorded": False,
            "ledger_path": None,
            "project_state_append_status": "not_attempted",
            "project_state_append_error": None,
        }
        # Target cleanliness tracking (UA-P1-002)
        self._target_dirty_before: Optional[bool] = None
        self._target_dirty_after: Optional[bool] = None
        self._target_dirty_files_before: list[str] = []
        self._target_dirty_files_after: list[str] = []
        self._unexpected_target_changes: list[str] = []
        self._run_id: Optional[str] = None

    # ── pipeline stages ────────────────────────────────────────

    def _scan(self) -> dict:
        """Run scan_project.py and return the scan dict."""
        import subprocess
        cmd = [
            sys.executable,
            str(_SCRIPT_DIR / "scan_project.py"),
            self.target_dir,
        ]
        if self.in_repo_cache:
            cmd.extend(["--incremental", "--in-repo-cache"])
        else:
            cache_dir = os.path.join(self.out_dir, "cache")
            cmd.extend(["--incremental", "--no-repo-cache",
                        "--external-cache-dir", cache_dir])
        if self.external_cache_dir:
            cmd.extend(["--external-cache-dir", self.external_cache_dir])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(
                f"scan_project.py failed (rc={result.returncode}): {result.stderr}"
            )
        return json.loads(result.stdout)

    def _extract_imports(self) -> dict:
        """Run import extraction on the scan data."""
        return build_import_map(self.scan_data, self.target_dir)

    def _assemble_graph(self) -> tuple[dict, dict]:
        """Assemble graph and validate. Returns (graph, validation)."""
        graph = assemble_graph(
            scans=[self.scan_data],
            imports_list=[self.imports_data] if self.imports_data else [],
        )
        validation = validate_graph(graph)
        return graph, validation

    def _run_analytics(self) -> dict:
        """Run deterministic graph analytics (UA-003)."""
        if not _HAS_ANALYTICS or not self.graph_data:
            return {}
        return analyze_graph(self.graph_data)

    def _build_context(self) -> dict:
        """Build subagent context envelope (UA-004)."""
        if not _HAS_CONTEXT:
            return {}
        context = build_context_envelope(self.out_dir)
        context_target = context.get("target")
        if (
            isinstance(context_target, str)
            and os.path.realpath(context_target) == os.path.realpath(self.out_dir)
        ):
            context["target"] = self.target_dir
            confidence = context.setdefault("confidence", {})
            if isinstance(confidence, dict):
                confidence["target"] = "high"
        return context

    def _build_report_raw(self) -> str:
        """Generate REPORT.md content directly."""
        s = self._build_summary()
        lines = [
            "# UA Run Bundle Report",
            "",
            f"- **Target**: `{self.target_dir}`",
            f"- **Bundle**: `{self.out_dir}`",
            f"- **Mode**: `{self.mode}`",
            f"- **Timestamp**: {s.get('timestamp', 'N/A')}",
            "",
            "## What UA proves / What UA does not prove",
            "",
            "UA proves deterministic inventory, import graph, graph validation, and artifact provenance facts captured in this bundle; it does not prove security, deployment readiness, RLS correctness, or runtime correctness.",
            "",
            "Confidence labels used in this report: deterministic_fact, static_analysis_finding, heuristic_signal, inferred_summary, suggested_verification_not_run, executed_external_gate, outside_ua_scope.",
            "",
            "## Artifacts",
            "",
        ]
        for name, path in sorted(self.artifact_paths.items()):
            lines.append(f"- `{name}` → `{path}`")
        lines.append("")

        if self.scan_data:
            scan = s.get("scan", {})
            lines.extend([
                "## Scan Summary",
                "",
                f"- **Total files**: {scan.get('total_files', 0)}",
                f"- **Total lines**: {scan.get('total_lines', 0)}",
                f"- **Languages**: {', '.join(scan.get('languages', {}).keys()) or 'none'}",
                "",
            ])

        if self.imports_data:
            totals = s.get("imports", {}).get("totals", {})
            lines.extend([
                "## Import Summary",
                "",
                f"- **Files with imports**: {totals.get('files_with_imports', 0)}",
                f"- **Unique modules**: {totals.get('unique_modules', 0)}",
                "",
            ])

        if self.graph_data:
            g = s.get("graph", {})
            lines.extend([
                "## Graph Summary",
                "",
                f"- **Nodes**: {g.get('total_nodes', 0)}",
                f"- **Edges**: {g.get('total_edges', 0)}",
                f"- **Orphan nodes**: {g.get('orphan_nodes', 0)}",
                "",
            ])

        if self.validation_data:
            v = self.validation_data
            lines.extend([
                "## Validation",
                "",
                f"- **Issues**: {len(v.get('issues', []))}",
                f"- **Warnings**: {len(v.get('warnings', []))}",
                "",
            ])
            if v.get("issues"):
                lines.append("### Issues")
                lines.append("")
                for issue in v["issues"]:
                    lines.append(f"- {issue}")
                lines.append("")
            if v.get("warnings"):
                lines.append("### Warnings")
                lines.append("")
                for warning in v["warnings"]:
                    lines.append(f"- {warning}")
                lines.append("")

        if self.static_signals_data:
            static_summary = self.static_signals_data.get("summary", {})
            lines.extend([
                "## Static Signals",
                "",
                "These are heuristic content markers only. They do not prove security, RLS correctness, auth correctness, runtime behavior, deployment readiness, CI success, or policy semantics.",
                "",
                f"- **Total signals**: {static_summary.get('total_signals', 0)}",
                f"- **Claim type**: `{self.static_signals_data.get('claim_type', 'heuristic_signal')}`",
                f"- **Semantic status**: `{self.static_signals_data.get('semantic_status', 'not_validated')}`",
                "",
            ])

        lines.append("---")
        lines.append("*Generated by run_ua.py (UA-005 mode router)*")
        return "\n".join(lines) + "\n"

    def _build_orphan_summary_for_security(self) -> dict:
        """Build a compact orphan summary from graph/validation artifacts."""
        warnings = []
        if self.validation_data:
            warnings = self.validation_data.get("warnings", []) or []
        orphan_warnings = [w for w in warnings if "Orphan node" in str(w)]
        return {
            "totals": {
                "total_orphans": self.graph_data.get("summary", {}).get("orphan_nodes", 0) if self.graph_data else len(orphan_warnings),
                "suspicious": len(orphan_warnings),
                "expected": 0,
            },
            "summary": {
                "category_counts": {"validation_orphan_warning": len(orphan_warnings)} if orphan_warnings else {},
                "representative_examples": {"validation_orphan_warning": orphan_warnings[:3]} if orphan_warnings else {},
                "top_suspicious_examples": [
                    {"node_id": warning, "category": "validation_orphan_warning", "reason": warning}
                    for warning in orphan_warnings[:5]
                ],
            },
        }

    def _build_security_report_data(self) -> dict:
        """Build report-data for security-review mode without executing target gates."""
        if not _HAS_REPORT_RENDER:
            raise RuntimeError("report_data/render_report integration unavailable")
        report = build_report_data(  # type: ignore[misc]
            scan=self.scan_data,
            graph=self.graph_data,
            orphan_triage=self._build_orphan_summary_for_security(),
            domain_surfaces=getattr(self, "domain_surfaces_data", None),
            static_signals=self.static_signals_data,
            readiness=self.runtime_readiness_data,
        )
        critic_packs = {}
        if isinstance(self.context_data, dict):
            critic_packs = self.context_data.get("critic_packs", {}) or {}
        security = report.get("sections", {}).get("security_evidence_gaps", {})
        if isinstance(security, dict):
            security["critic_packs_present"] = sorted(critic_packs.keys())
        return report

    def _build_summary(self) -> dict:
        """Build summary.json data."""
        summary = {
            "target": self.target_dir,
            "bundle_dir": self.out_dir,
            "mode": self.mode,
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "artifact_count": len(self.artifact_paths),
        }
        if self.scan_data:
            summary["scan"] = {
                "total_files": self.scan_data.get("total_files", 0),
                "total_lines": self.scan_data.get("total_lines", 0),
                "languages": self.scan_data.get("languages", {}),
            }
        if self.imports_data:
            summary["imports"] = {
                "schema_version": self.imports_data.get("schema_version"),
                "totals": self.imports_data.get("totals", {}),
            }
        if self.graph_data:
            summary["graph"] = self.graph_data.get("summary", {})
        if self.validation_data:
            summary["validation"] = {
                "issue_count": len(self.validation_data.get("issues", [])),
                "warning_count": len(self.validation_data.get("warnings", [])),
            }
        if getattr(self, "domain_surfaces_data", None):
            summary["domain_surfaces"] = self.domain_surfaces_data.get("summary", {})
        if getattr(self, "static_signals_data", None):
            static_summary = dict(self.static_signals_data.get("summary", {}))
            static_summary["claim_type"] = self.static_signals_data.get(
                "claim_type", "heuristic_signal"
            )
            static_summary["semantic_status"] = self.static_signals_data.get(
                "semantic_status", "not_validated"
            )
            summary["static_signals"] = static_summary
        return summary

    def _build_delta_summary(self) -> dict:
        """Build delta_summary by comparing current scan to prior manifest."""
        delta_summary = {
            "files_scanned": 0,
            "changes": 0,
            "prior_run_id": None,
            "mode": "delta",
        }
        if self.prior_manifest:
            try:
                with open(self.prior_manifest, "r", encoding="utf-8") as f:
                    prior = json.load(f)
                delta_summary["prior_run_id"] = prior.get("run_id", "unknown")
            except (OSError, json.JSONDecodeError):
                delta_summary["changes"] = -1  # could not parse prior
        if self.scan_data:
            delta_summary["files_scanned"] = self.scan_data.get("total_files", 0)
        return delta_summary

    def _record_missing(self, artifact_name: str) -> None:
        """Record an expected artifact that could not be produced."""
        if artifact_name not in self._missing_artifacts:
            self._missing_artifacts.append(artifact_name)

    def _try_record_project_state(self, manifest: dict,
                                   runtime_readiness: Optional[dict] = None,
                                   cleanliness: Optional[dict] = None) -> None:
        """Attempt to append a compact UA section to the project-state ledger.

        Only runs when a project_root was explicitly provided (opt-in) and
        the project_state_append helper is available.  Updates the internal
        project-state status so the manifest can report it.

        Passes runtime_readiness and cleanliness summaries for inclusion
        in the ledger (UA-P1-004).
        """
        if not self.project_root or not _HAS_PROJECT_STATE:
            return

        results = {
            "manifest": manifest,
            "scan": self.scan_data or {},
            "graph": self.graph_data or {},
            "validation": self.validation_data or {},
            "context": self.context_data or {},
        }
        if runtime_readiness:
            results["runtime_readiness"] = runtime_readiness
        if cleanliness:
            results["cleanliness"] = cleanliness

        try:
            status = _append_project_state(results, self.project_root)  # type: ignore[misc]
            self._project_state_status.update(status)
        except Exception:  # noqa: BLE001
            # Project-state errors are never fatal to the UA run.
            self._project_state_status = {
                "project_state_recorded": False,
                "ledger_path": None,
                "project_state_append_status": "failed",
                "project_state_append_error": "unexpected error during append",
            }

    def _build_manifest(self, run_id: str, *, status: str = "complete",
                        failure_stage: Optional[str] = None,
                        error_message: Optional[str] = None) -> dict:
        """Build manifest.json with all required metadata."""
        # Capture post-cleanliness on the first call during a successful run
        if status == "complete":
            self._record_post_cleanliness()

        # Determine cleanliness status
        if self._target_dirty_before is None or self._target_dirty_after is None:
            cleanliness_status = "unknown"
        elif self._target_dirty_files_after != self._target_dirty_files_before:
            cleanliness_status = "mutated"
        elif self._target_dirty_before:
            cleanliness_status = "preexisting_dirty"
        else:
            cleanliness_status = "clean"

        # Compute unexpected changes
        before_set = set(self._target_dirty_files_before)
        after_set = set(self._target_dirty_files_after)
        unexpected = sorted(after_set - before_set)

        manifest = {
            "run_id": run_id,
            "status": status,
            "mode": self.mode,
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "target_path": self.target_dir,
            "target_git_head": _get_git_head(self.target_dir),
            "bundle_dir": self.out_dir,
            "command_flags": {
                "mode": self.mode,
                "in_repo_cache": self.in_repo_cache,
                "external_cache_dir": self.external_cache_dir,
            },
            "provenance": _build_provenance(
                "scripts/code-scan/run_ua.py",
                {
                    "mode": self.mode,
                    "in_repo_cache": self.in_repo_cache,
                    "external_cache_dir": self.external_cache_dir,
                },
                self.target_dir,
            ),
            "artifact_paths": dict(self.artifact_paths),
            "artifact_integrity": _artifact_integrity(self.artifact_paths),
            "script_versions": _get_script_versions(),
            "target_mutation_allowed": self.in_repo_cache,
            "target_dirty_before": self._target_dirty_before if self._target_dirty_before is not None else False,
            "target_dirty_after": self._target_dirty_after if self._target_dirty_after is not None else False,
            "target_dirty_files_before": list(self._target_dirty_files_before),
            "target_dirty_files_after": list(self._target_dirty_files_after),
            "unexpected_target_changes": unexpected,
            "target_cleanliness_status": cleanliness_status,
        }
        if self.delta_data:
            manifest["delta_summary"] = self.delta_data
        manifest["artifacts_missing"] = sorted(self._missing_artifacts) if self._missing_artifacts else []
        # Project-state integration (UA-006) — always present, default false
        manifest["project_state_recorded"] = self._project_state_status["project_state_recorded"]
        manifest["ledger_path"] = self._project_state_status["ledger_path"]
        manifest["project_state_append_status"] = self._project_state_status.get(
            "project_state_append_status", "not_attempted"
        )
        manifest["project_state_append_error"] = self._project_state_status.get(
            "project_state_append_error"
        )
        if status == "failed":
            manifest["failure_stage"] = failure_stage or "unknown"
            manifest["error_message"] = error_message or ""
        return manifest

    def _write_failure_manifest(self, failure_stage: str,
                                error_message: str) -> None:
        """Write a partial-failure manifest when a pipeline stage fails."""
        path = os.path.join(self.out_dir, "manifest.json")
        manifest = self._build_manifest(
            self._run_id or "unknown",
            status="failed",
            failure_stage=failure_stage,
            error_message=error_message,
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")

    def _build_manifest_into_existing(self, manifest: dict) -> str:
        """Re-write manifest.json with updated project-state fields.

        Called after _try_record_project_state to persist the final status.
        """
        path = os.path.join(self.out_dir, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        return path

    def _reserve_artifact_path(self, filename: str) -> str:
        """Register a final artifact path before the file is rewritten."""
        path = os.path.join(self.out_dir, filename)
        self.artifact_paths[filename] = path
        return path

    def _write_manifest_file(self, manifest: dict) -> str:
        """Write manifest.json after its final path has been registered."""
        path = self._reserve_artifact_path("manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        return path

    def _assert_final_bundle_consistency(
        self,
        manifest: dict,
        *,
        include_context: bool,
    ) -> None:
        """Fail before success if final manifest/context artifact facts disagree."""
        artifact_paths = manifest.get("artifact_paths", {})
        if not isinstance(artifact_paths, dict):
            raise RuntimeError("manifest artifact_paths is not a dict")

        present_artifacts: set[str] = set()
        for artifact, raw_path in artifact_paths.items():
            if not isinstance(raw_path, str) or not raw_path:
                raise RuntimeError(f"manifest has invalid artifact path for {artifact}")
            path = raw_path if os.path.isabs(raw_path) else os.path.join(self.out_dir, raw_path)
            if not os.path.isfile(path):
                raise RuntimeError(f"manifest-listed artifact is absent: {artifact}")
            present_artifacts.add(artifact)

        for item in manifest.get("artifacts_missing", []) or []:
            artifact = item.get("artifact") if isinstance(item, dict) else item
            if artifact in present_artifacts:
                raise RuntimeError(
                    f"manifest marks present artifact as missing: {artifact}"
                )

        for artifact, entry in (manifest.get("artifact_integrity", {}) or {}).items():
            raw_path = artifact_paths.get(artifact)
            if not raw_path:
                raise RuntimeError(
                    f"artifact_integrity references unknown artifact: {artifact}"
                )
            path = raw_path if os.path.isabs(raw_path) else os.path.join(self.out_dir, raw_path)
            expected = _artifact_integrity({artifact: path}).get(artifact)
            if entry != expected:
                raise RuntimeError(f"artifact_integrity mismatch for {artifact}")

        if not include_context:
            return

        context_path = artifact_paths.get("subagent-context.json")
        if not context_path:
            raise RuntimeError("manifest omits subagent-context.json")
        if not os.path.isabs(context_path):
            context_path = os.path.join(self.out_dir, context_path)
        try:
            with open(context_path, "r", encoding="utf-8") as f:
                context = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"subagent-context.json is unreadable: {exc}") from exc

        context_missing = {
            item.get("artifact")
            for item in context.get("artifacts_missing", []) or []
            if isinstance(item, dict)
        }
        contradictory = sorted(context_missing & present_artifacts)
        if contradictory:
            raise RuntimeError(
                "subagent context marks present artifacts as missing: "
                + ", ".join(contradictory)
            )

        trust = (
            context.get("critic_packs", {})
            .get("reviewer_critic", {})
            .get("trust_anchor_summary", "")
        )
        if "absent or unreadable" in str(trust):
            raise RuntimeError("reviewer critic pack says manifest is absent")

    def _finalize_bundle(self, run_id: str, *, include_context: bool) -> dict:
        """Write final context/manifest artifacts in dependency order."""
        if include_context:
            if _HAS_CONTEXT:
                self._write_json("subagent-context.json", {"status": "provisional"})
            else:
                self._record_missing("subagent-context.json")

        self._reserve_artifact_path("manifest.json")
        manifest = self._build_manifest(run_id)
        self._write_manifest_file(manifest)

        if include_context and _HAS_CONTEXT:
            self.context_data = self._build_context()
            self._write_json("subagent-context.json", self.context_data)
            manifest = self._build_manifest(run_id)
            self._write_manifest_file(manifest)

        self._assert_final_bundle_consistency(
            manifest,
            include_context=include_context and _HAS_CONTEXT,
        )
        return manifest

    # ── artifact writers ──────────────────────────────────────

    def _write_json(self, filename: str, data: dict) -> str:
        """Write data to a JSON artifact and register it."""
        path = os.path.join(self.out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        self.artifact_paths[filename] = path
        return path

    def _write_text(self, filename: str, text: str) -> str:
        """Write text to a file artifact and register it."""
        path = os.path.join(self.out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self.artifact_paths[filename] = path
        return path

    def _run_domain_surfaces(self) -> None:
        """Run deterministic domain surface inventory and write artifact."""
        if not self.scan_data:
            return
        self.domain_surfaces_data = scan_domain_surfaces(self.scan_data)
        self._write_json("domain-surfaces.json", self.domain_surfaces_data)

    def _run_static_signals(self) -> None:
        """Run Tier 1 heuristic static-signal extraction and write artifact."""
        signals: list[dict] = []
        if self.scan_data:
            for item in self.scan_data.get("files", []) or []:
                if not isinstance(item, dict):
                    continue
                rel_path = str(item.get("relative_path") or item.get("path") or "")
                rel_path = rel_path.replace("\\", "/").lstrip("./")
                if not rel_path:
                    continue
                path = Path(self.target_dir) / rel_path
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                signals.extend(extract_supabase_migration_markers(rel_path, content))
                signals.extend(extract_edge_function_markers(rel_path, content))
                signals.extend(extract_package_config_markers(rel_path, content))
                signals.extend(extract_rust_agent_infra_markers(rel_path, content))

        self.static_signals_data = build_static_signals_artifact(signals)
        self._write_json("static-signals.json", self.static_signals_data)

    def _run_runtime_readiness(self) -> None:
        """Run runtime readiness detection and write artifacts (UA-P1-003).

        Writes runtime-readiness.json and runtime-readiness.md to the
        output bundle and registers them in artifact_paths.
        """
        if not _HAS_READINESS:
            return
        artifact = build_readiness_artifact(self.target_dir)  # type: ignore[misc]
        self.runtime_readiness_data = artifact
        # Write JSON
        json_path = os.path.join(self.out_dir, "runtime-readiness.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)
            f.write("\n")
        self.artifact_paths["runtime-readiness.json"] = json_path
        # Write Markdown
        md_path = os.path.join(self.out_dir, "runtime-readiness.md")
        md_content = readiness_to_markdown(artifact)  # type: ignore[misc]
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        self.artifact_paths["runtime-readiness.md"] = md_path

    def _readiness_summary(self) -> Optional[dict]:
        """Build a compact readiness summary for the project-state ledger.

        Returns None if readiness data is unavailable.
        """
        if not _HAS_READINESS:
            return None
        try:
            artifact = build_readiness_artifact(self.target_dir)  # type: ignore[misc]
            return {
                "verification_status": artifact.get("verification_status", "unknown"),
                "blockers": artifact.get("blockers", []),
            }
        except Exception:  # noqa: BLE001
            return None

    def _cleanliness_summary(self) -> dict:
        """Build a compact cleanliness summary for the project-state ledger."""
        before_set = set(self._target_dirty_files_before)
        after_set = set(self._target_dirty_files_after)
        unexpected = sorted(after_set - before_set)
        return {
            "target_cleanliness_status": self._cleanliness_status(),
            "unexpected_changes_count": len(unexpected),
        }

    def _cleanliness_status(self) -> str:
        """Derive the cleanliness status string."""
        if self._target_dirty_before is None or self._target_dirty_after is None:
            return "unknown"
        if self._target_dirty_files_after != self._target_dirty_files_before:
            return "mutated"
        if self._target_dirty_before:
            return "preexisting_dirty"
        return "clean"

    # ── mode routing ──────────────────────────────────────────

    def run(self) -> dict:
        """Execute the mode-selected pipeline and write artifacts.

        Returns the manifest dict.
        """
        os.makedirs(self.out_dir, exist_ok=True)

        # ── Pre-scan cleanliness snapshot ────────────────────────
        self._run_id = uuid.uuid4().hex
        self._target_dirty_files_before = _get_git_dirty_files(self.target_dir)
        self._target_dirty_before = len(self._target_dirty_files_before) > 0

        try:
            if self.mode == "inventory":
                manifest = self._mode_inventory(self._run_id)
            elif self.mode == "structure":
                manifest = self._mode_structure(self._run_id)
            elif self.mode == "review":
                manifest = self._mode_review(self._run_id)
            elif self.mode == "delta":
                manifest = self._mode_delta(self._run_id)
            elif self.mode == "preflight":
                manifest = self._mode_preflight(self._run_id)
            elif self.mode == "security-review":
                manifest = self._mode_security_review(self._run_id)
            elif self.mode == "full":
                manifest = self._mode_full(self._run_id)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
        except Exception as exc:  # noqa: BLE001
            # Partial failure: record post-cleanliness and write failure manifest
            self._record_post_cleanliness()
            self._write_failure_manifest(
                failure_stage=self._current_stage,
                error_message=str(exc),
            )
            raise

        # Opt-in project-state recording (UA-006) — after all artifacts exist
        readiness = self._readiness_summary()
        cleanliness = self._cleanliness_summary()
        self._try_record_project_state(manifest,
                                       runtime_readiness=readiness,
                                       cleanliness=cleanliness)
        if self.project_root and _HAS_PROJECT_STATE:
            # Update manifest in-place with project-state results, then persist
            manifest["project_state_recorded"] = self._project_state_status["project_state_recorded"]
            manifest["ledger_path"] = self._project_state_status["ledger_path"]
            manifest["project_state_append_status"] = self._project_state_status.get(
                "project_state_append_status", "not_attempted"
            )
            manifest["project_state_append_error"] = self._project_state_status.get(
                "project_state_append_error"
            )
            self._build_manifest_into_existing(manifest)

        self._assert_final_bundle_consistency(
            manifest,
            include_context=(
                self.mode in ("review", "preflight", "security-review", "full")
                and _HAS_CONTEXT
            ),
        )
        return manifest

    def _mode_inventory(self, run_id: str) -> dict:
        """inventory: scan + imports only."""
        self.scan_data = self._scan()
        self._write_json("scan.json", self.scan_data)
        self._run_domain_surfaces()
        self._run_static_signals()

        self.imports_data = self._extract_imports()
        self._write_json("imports.json", self.imports_data)

        self.summary_data = self._build_summary()
        self._write_json("summary.json", self.summary_data)

        return self._finalize_bundle(run_id, include_context=False)

    def _mode_structure(self, run_id: str) -> dict:
        """structure: scan + imports + graph + validation."""
        self.scan_data = self._scan()
        self._write_json("scan.json", self.scan_data)
        self._run_domain_surfaces()
        self._run_static_signals()

        self.imports_data = self._extract_imports()
        self._write_json("imports.json", self.imports_data)

        # Graph + validation — never hidden
        self.graph_data, self.validation_data = self._assemble_graph()
        self._write_json("graph.json", self.graph_data)
        self._write_json("validation.json", self.validation_data)

        # Runtime readiness (UA-P1-003)
        self._run_runtime_readiness()

        self.summary_data = self._build_summary()
        self._write_json("summary.json", self.summary_data)

        return self._finalize_bundle(run_id, include_context=False)

    def _mode_review(self, run_id: str) -> dict:
        """review: structure + analytics + context envelope + report."""
        # Structure pipeline
        self.scan_data = self._scan()
        self._write_json("scan.json", self.scan_data)
        self._run_domain_surfaces()
        self._run_static_signals()

        self.imports_data = self._extract_imports()
        self._write_json("imports.json", self.imports_data)

        # Graph + validation — never hidden
        self.graph_data, self.validation_data = self._assemble_graph()
        self._write_json("graph.json", self.graph_data)
        self._write_json("validation.json", self.validation_data)

        # Analytics (UA-003) — optional enricher
        if _HAS_ANALYTICS:
            self.analytics_data = self._run_analytics()
            self._write_json("analytics.json", self.analytics_data)
        else:
            self._record_missing("analytics.json")

        # Report
        self._write_text("REPORT.md", self._build_report_raw())

        # Runtime readiness (UA-P1-003)
        self._run_runtime_readiness()

        self.summary_data = self._build_summary()
        self._write_json("summary.json", self.summary_data)

        return self._finalize_bundle(run_id, include_context=True)

    def _mode_delta(self, run_id: str) -> dict:
        """delta: incremental scan + delta summary against prior manifest."""
        self.scan_data = self._scan()
        self._write_json("scan.json", self.scan_data)
        self._run_domain_surfaces()
        self._run_static_signals()

        self.imports_data = self._extract_imports()
        self._write_json("imports.json", self.imports_data)

        # Build delta summary
        self.delta_data = self._build_delta_summary()

        return self._finalize_bundle(run_id, include_context=False)

    def _mode_preflight(self, run_id: str) -> dict:
        """preflight: structure + entrypoints/hubs + subagent context."""
        # Structure pipeline
        self.scan_data = self._scan()
        self._write_json("scan.json", self.scan_data)
        self._run_domain_surfaces()
        self._run_static_signals()

        self.imports_data = self._extract_imports()
        self._write_json("imports.json", self.imports_data)

        # Graph + validation — always present for preflight
        self.graph_data, self.validation_data = self._assemble_graph()
        self._write_json("graph.json", self.graph_data)
        self._write_json("validation.json", self.validation_data)

        # Runtime readiness (UA-P1-003)
        self._run_runtime_readiness()

        self.summary_data = self._build_summary()
        self._write_json("summary.json", self.summary_data)

        return self._finalize_bundle(run_id, include_context=True)

    def _mode_security_review(self, run_id: str) -> dict:
        """security-review: planning/preflight evidence gaps; no target gates."""
        self.scan_data = self._scan()
        self._write_json("scan.json", self.scan_data)
        self._run_domain_surfaces()
        self._run_static_signals()

        self.imports_data = self._extract_imports()
        self._write_json("imports.json", self.imports_data)

        self.graph_data, self.validation_data = self._assemble_graph()
        self._write_json("graph.json", self.graph_data)
        self._write_json("validation.json", self.validation_data)

        self._run_runtime_readiness()

        security_report_data = self._build_security_report_data()
        self._write_json("security-report-data.json", security_report_data)
        report_md = render_report_data(security_report_data)  # type: ignore[misc]
        self._write_text("REPORT.md", report_md)

        self.summary_data = self._build_summary()
        self._write_json("summary.json", self.summary_data)

        self._finalize_bundle(run_id, include_context=True)

        security_report_data = self._build_security_report_data()
        self._write_json("security-report-data.json", security_report_data)
        report_md = render_report_data(security_report_data)  # type: ignore[misc]
        self._write_text("REPORT.md", report_md)

        return self._finalize_bundle(run_id, include_context=True)

    def _mode_full(self, run_id: str) -> dict:
        """full: all available deterministic enrichers."""
        # Everything in review (structure + analytics + context + report)
        self.scan_data = self._scan()
        self._write_json("scan.json", self.scan_data)
        self._run_domain_surfaces()
        self._run_static_signals()

        self.imports_data = self._extract_imports()
        self._write_json("imports.json", self.imports_data)

        # Graph + validation — never hidden
        self.graph_data, self.validation_data = self._assemble_graph()
        self._write_json("graph.json", self.graph_data)
        self._write_json("validation.json", self.validation_data)

        # Analytics (UA-003) — optional enricher
        if _HAS_ANALYTICS:
            self.analytics_data = self._run_analytics()
            self._write_json("analytics.json", self.analytics_data)
        else:
            self._record_missing("analytics.json")

        # Report
        self._write_text("REPORT.md", self._build_report_raw())

        # Runtime readiness (UA-P1-003)
        self._run_runtime_readiness()

        self.summary_data = self._build_summary()
        self._write_json("summary.json", self.summary_data)

        return self._finalize_bundle(run_id, include_context=True)

    # ── cleanliness helpers (UA-P1-002) ────────────────────────────

    @property
    def _current_stage(self) -> str:
        """Return the current pipeline stage name for failure reporting."""
        if self.scan_data is None:
            return "scan"
        if self.imports_data is None:
            return "extract_imports"
        if self.graph_data is None and self.mode in ("structure", "review", "preflight", "security-review", "full"):
            return "graph_assembly"
        if self.summary_data is None:
            return "summary"
        return self.mode

    def _record_post_cleanliness(self) -> None:
        """Snapshot post-pipeline target cleanliness."""
        if self._target_dirty_after is not None:
            return  # Already recorded
        self._target_dirty_files_after = _get_git_dirty_files(self.target_dir)
        self._target_dirty_after = len(self._target_dirty_files_after) > 0


# ── Module-level convenience function ────────────────────────────────────

def run_ua_pipeline(
    target_dir: str,
    out_dir: str,
    *,
    mode: str = DEFAULT_MODE,
    in_repo_cache: bool = False,
    external_cache_dir: Optional[str] = None,
    prior_manifest: Optional[str] = None,
    project_root: Optional[str] = None,
) -> dict:
    """Convenience: run the mode-selected UA pipeline and return manifest."""
    ua = RunUA(
        target_dir,
        out_dir,
        mode=mode,
        in_repo_cache=in_repo_cache,
        external_cache_dir=external_cache_dir,
        prior_manifest=prior_manifest,
        project_root=project_root,
    )
    return ua.run()


# ── CLI entry point ────────────────────────────────────────────────────

def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="UA-005 Mode Router — explicit mode selection for UA pipeline.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to the project directory to scan",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to the output bundle directory",
    )
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=sorted(VALID_MODES),
        help=f"Pipeline mode (default: {DEFAULT_MODE})",
    )
    parser.add_argument(
        "--in-repo-cache",
        action="store_true",
        default=False,
        help="Allow writing fingerprints inside the target repo",
    )
    parser.add_argument(
        "--read-only-target",
        action="store_true",
        default=False,
        help="Ensure no cache or artifacts are written inside the target repo "
             "(forces in_repo_cache=False)",
    )
    parser.add_argument(
        "--external-cache-dir",
        default=None,
        help="Directory for external fingerprint cache",
    )
    parser.add_argument(
        "--prior-manifest",
        default=None,
        help="Path to prior manifest.json (for delta mode comparison)",
    )
    parser.add_argument(
        "--record-project-state",
        action="store_true",
        default=False,
        help="Opt-in: append a compact UA section to .hermes/PROJECT_STATE.md",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root directory for project-state ledger (defaults to --target if --record-project-state is set)",
    )
    args = parser.parse_args()

    target = os.path.realpath(args.target)
    if not os.path.isdir(target):
        print(f"Error: '{args.target}' is not a valid directory",
              file=sys.stderr)
        return 1

    # Resolve project_root: explicit value, fallback to target if opt-in set
    project_root = None
    if args.project_root:
        project_root = os.path.realpath(args.project_root)
    elif args.record_project_state:
        project_root = target

    # --read-only-target overrides in_repo_cache to ensure no target mutation
    effective_in_repo_cache = args.in_repo_cache
    if args.read_only_target:
        effective_in_repo_cache = False

    try:
        manifest = run_ua_pipeline(
            target,
            args.out,
            mode=args.mode,
            in_repo_cache=effective_in_repo_cache,
            external_cache_dir=args.external_cache_dir,
            prior_manifest=args.prior_manifest,
            project_root=project_root,
        )
        print(f"Bundle written to: {args.out}")
        print(f"Mode: {manifest['mode']}")
        print(f"Run ID: {manifest['run_id']}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
