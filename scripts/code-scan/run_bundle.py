#!/usr/bin/env python3
"""run_bundle.py — UA-001 canonical run bundle orchestrator.

Runs the full UA pipeline (scan → imports → graph → validation) and writes
a canonical bundle of reproducible artifacts.

Default behaviour is **non-mutating**: fingerprints and caches are written
to an external directory outside the target repo.  Pass ``--in-repo-cache``
to opt into the legacy behaviour that writes ``.hermes/code-state/``
inside the target.

Usage:
    python run_bundle.py <target_dir> <bundle_dir> [--no-graph] [--in-repo-cache]

Artifacts produced in <bundle_dir>:
    scan.json, imports.json, graph.json, validation.json,
    summary.json, manifest.json, REPORT.md
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure scripts/code-scan is on sys.path for sibling imports
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from scan_project import main as scan_main
from extract_imports import build_import_map, load_scan_output
from assemble_graph import assemble_graph
from graph_schema import validate_graph

# ── Runtime readiness (UA-P1-003) ────────────────────────────────────────
try:
    from runtime_readiness import build_readiness_artifact, readiness_to_markdown  # noqa: F401
    _HAS_READINESS = True
except ImportError:
    _HAS_READINESS = False
    build_readiness_artifact = None  # type: ignore[misc]
    readiness_to_markdown = None  # type: ignore[misc]


# ── Helper: script versioning ────────────────────────────────────────────────

def _script_hash(path: Path) -> str:
    """SHA-256 hex digest of a script file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ── Helper: git HEAD ─────────────────────────────────────────────────────────

def _get_git_head(target_dir: str) -> Optional[str]:
    """Return the current git HEAD SHA for *target_dir*, or None."""
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


# ── Helper: target cleanliness ───────────────────────────────────────────────

def _get_git_dirty_files(target_dir: str) -> list[str]:
    """Return list of dirty file paths via `git status --porcelain=v1`.

    Returns an empty list if the target is not a git repo or any error
    occurs during the call.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Each line: "XY filename" or "XY \"filename\""
            lines = result.stdout.strip().splitlines()
            dirty = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # porcelains v1 format: first two chars are index+worktree status
                # then two spaces, then the path
                parts = line.split(None, 1)
                if len(parts) >= 2:
                    dirty.append(parts[1].strip('"'))
            return dirty
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return []


# ── RunBundle class ──────────────────────────────────────────────────────────

class RunBundle:
    """Orchestrates a full UA pipeline run into a canonical artifact bundle."""

    def __init__(
        self,
        target_dir: str,
        bundle_dir: str,
        *,
        no_graph: bool = False,
        in_repo_cache: bool = False,
        external_cache_dir: Optional[str] = None,
    ) -> None:
        self.target_dir = os.path.realpath(target_dir)
        self.bundle_dir = os.path.realpath(bundle_dir)
        self.no_graph = no_graph
        self.in_repo_cache = in_repo_cache
        self.external_cache_dir = external_cache_dir

        self.artifact_paths: dict[str, str] = {}
        self.scan_data: Optional[dict] = None
        self.imports_data: Optional[dict] = None
        self.graph_data: Optional[dict] = None
        self.validation_data: Optional[dict] = None
        self.summary_data: Optional[dict] = None

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
        # We run the scanner programmatically by capturing stdout
        cmd = [
            sys.executable,
            str(_SCRIPT_DIR / "scan_project.py"),
            self.target_dir,
        ]
        # Default: external (non-mutating) cache placed inside the bundle dir.
        # --in-repo-cache means fingerprints go inside the target repo.
        if self.in_repo_cache:
            cmd.extend(["--incremental", "--in-repo-cache"])
        else:
            # Use the bundle dir as external cache root
            cache_dir = os.path.join(self.bundle_dir, "cache")
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
        # Write scan to a temp file for extract_imports
        scan_path = self.artifact_paths["scan.json"]
        return build_import_map(self.scan_data, self.target_dir)

    def _assemble_graph(self) -> tuple[dict, dict]:
        """Assemble graph and validate it. Returns (graph, validation)."""
        graph = assemble_graph(
            scans=[self.scan_data],
            imports_list=[self.imports_data] if self.imports_data else [],
        )
        validation = validate_graph(graph)
        return graph, validation

    # ── reporting ────────────────────────────────────────────

    def _build_summary(self) -> dict:
        """Build summary.json data."""
        summary = {
            "target": self.target_dir,
            "bundle_dir": self.bundle_dir,
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "no_graph": self.no_graph,
            "in_repo_cache": self.in_repo_cache,
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
        return summary

    def _build_report(self) -> str:
        """Generate REPORT.md."""
        s = self._build_summary()
        lines = [
            "# UA Run Bundle Report",
            "",
            f"- **Target**: `{self.target_dir}`",
            f"- **Bundle**: `{self.bundle_dir}`",
            f"- **Timestamp**: {s.get('timestamp', 'N/A')}",
            f"- **No-graph**: {self.no_graph}",
            f"- **In-repo cache**: {self.in_repo_cache}",
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

        lines.append("---")
        lines.append("*Generated by run_bundle.py (UA-001)*")
        return "\n".join(lines) + "\n"

    def _run_runtime_readiness(self) -> None:
        """Run runtime readiness detection and write artifacts (UA-P1-003).

        Writes runtime-readiness.json and runtime-readiness.md to the
        bundle directory and registers them in artifact_paths.
        """
        if not _HAS_READINESS:
            return
        artifact = build_readiness_artifact(self.target_dir)  # type: ignore[misc]
        # Write JSON
        json_path = os.path.join(self.bundle_dir, "runtime-readiness.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)
            f.write("\n")
        self.artifact_paths["runtime-readiness.json"] = json_path
        # Write Markdown
        md_path = os.path.join(self.bundle_dir, "runtime-readiness.md")
        md_content = readiness_to_markdown(artifact)  # type: ignore[misc]
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        self.artifact_paths["runtime-readiness.md"] = md_path

    # ── manifest ────────────────────────────────────────────

    def _build_manifest(self, run_id: str, *, status: str = "complete",
                        failure_stage: Optional[str] = None,
                        error_message: Optional[str] = None) -> dict:
        """Build manifest.json with all required metadata.

        *status* is "complete" for successful runs or "failed" for partial
        failures.  When status is "failed", *failure_stage* and *error_message*
        should also be provided.
        """
        # Determine cleanliness status
        if self._target_dirty_before is None or self._target_dirty_after is None:
            cleanliness_status = "unknown"
        elif self._target_dirty_files_after != self._target_dirty_files_before:
            cleanliness_status = "mutated"
        elif self._target_dirty_before:
            cleanliness_status = "preexisting_dirty"
        else:
            cleanliness_status = "clean"

        # Compute unexpected changes (files that became dirty post-scan that
        # were not dirty pre-scan)
        before_set = set(self._target_dirty_files_before)
        after_set = set(self._target_dirty_files_after)
        unexpected = sorted(after_set - before_set)

        manifest = {
            "run_id": run_id,
            "status": status,
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "target_path": self.target_dir,
            "target_git_head": _get_git_head(self.target_dir),
            "bundle_dir": self.bundle_dir,
            "command_flags": {
                "no_graph": self.no_graph,
                "in_repo_cache": self.in_repo_cache,
                "external_cache_dir": self.external_cache_dir,
            },
            "artifact_paths": dict(self.artifact_paths),
            "script_versions": {
                "scan_project.py": _script_hash(
                    _SCRIPT_DIR / "scan_project.py"
                ),
                "extract_imports.py": _script_hash(
                    _SCRIPT_DIR / "extract_imports.py"
                ),
                "assemble_graph.py": _script_hash(
                    _SCRIPT_DIR / "assemble_graph.py"
                ),
                "graph_schema.py": _script_hash(
                    _SCRIPT_DIR / "graph_schema.py"
                ),
                "run_bundle.py": _script_hash(
                    _SCRIPT_DIR / "run_bundle.py"
                ),
            },
            "target_mutation_allowed": self.in_repo_cache,
            "target_dirty_before": self._target_dirty_before if self._target_dirty_before is not None else False,
            "target_dirty_after": self._target_dirty_after if self._target_dirty_after is not None else False,
            "target_dirty_files_before": list(self._target_dirty_files_before),
            "target_dirty_files_after": list(self._target_dirty_files_after),
            "unexpected_target_changes": unexpected,
            "target_cleanliness_status": cleanliness_status,
        }
        if status == "failed":
            manifest["failure_stage"] = failure_stage or "unknown"
            manifest["error_message"] = error_message or ""
        return manifest

    def _write_failure_manifest(self, failure_stage: str,
                                error_message: str) -> None:
        """Write a partial-failure manifest when a pipeline stage fails."""
        manifest_path = os.path.join(self.bundle_dir, "manifest.json")
        manifest = self._build_manifest(
            self._run_id or "unknown",
            status="failed",
            failure_stage=failure_stage,
            error_message=error_message,
        )
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")

    # ── public API ─────────────────────────────────────────

    def run(self) -> dict:
        """Execute the full pipeline and write all artifacts.

        Returns the manifest dict.
        """
        import uuid

        os.makedirs(self.bundle_dir, exist_ok=True)

        # Generate a unique run ID
        self._run_id = uuid.uuid4().hex

        # ── Pre-scan cleanliness snapshot ────────────────────────
        self._target_dirty_files_before = _get_git_dirty_files(self.target_dir)
        self._target_dirty_before = len(self._target_dirty_files_before) > 0

        try:
            # Stage 1: Scan
            self.scan_data = self._scan()
            scan_path = os.path.join(self.bundle_dir, "scan.json")
            with open(scan_path, "w", encoding="utf-8") as f:
                json.dump(self.scan_data, f, indent=2)
                f.write("\n")
            self.artifact_paths["scan.json"] = scan_path

            # Stage 2: Extract imports
            self.imports_data = self._extract_imports()
            imports_path = os.path.join(self.bundle_dir, "imports.json")
            with open(imports_path, "w", encoding="utf-8") as f:
                json.dump(self.imports_data, f, indent=2)
                f.write("\n")
            self.artifact_paths["imports.json"] = imports_path

            # Stage 3: Graph + validation (optional)
            if not self.no_graph:
                self.graph_data, self.validation_data = self._assemble_graph()

                graph_path = os.path.join(self.bundle_dir, "graph.json")
                with open(graph_path, "w", encoding="utf-8") as f:
                    json.dump(self.graph_data, f, indent=2)
                    f.write("\n")
                self.artifact_paths["graph.json"] = graph_path

                validation_path = os.path.join(self.bundle_dir, "validation.json")
                with open(validation_path, "w", encoding="utf-8") as f:
                    json.dump(self.validation_data, f, indent=2)
                    f.write("\n")
                self.artifact_paths["validation.json"] = validation_path
            else:
                # No graph mode: still write an empty validation to keep shape
                self.graph_data = None
                self.validation_data = {"issues": [], "warnings": []}

            # Stage 4: Summary
            self.summary_data = self._build_summary()
            summary_path = os.path.join(self.bundle_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(self.summary_data, f, indent=2)
                f.write("\n")
            self.artifact_paths["summary.json"] = summary_path

            # Stage 4b: Runtime readiness (UA-P1-003)
            self._run_runtime_readiness()
        except Exception as exc:  # noqa: BLE001
            # Partial failure: write manifest with failure info
            self._record_post_cleanliness()
            self._write_failure_manifest(
                failure_stage=self._current_stage,
                error_message=str(exc),
            )
            raise

        # ── Post-scan cleanliness snapshot ───────────────────────
        self._record_post_cleanliness()

        # Stage 5: Manifest
        # Pre-register artifact paths so the manifest includes itself and REPORT.md
        manifest_path = os.path.join(self.bundle_dir, "manifest.json")
        self.artifact_paths["manifest.json"] = manifest_path

        report_path = os.path.join(self.bundle_dir, "REPORT.md")
        self.artifact_paths["REPORT.md"] = report_path

        manifest = self._build_manifest(self._run_id)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")

        # Stage 6: Report
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self._build_report())

        return manifest

    @property
    def _current_stage(self) -> str:
        """Return the current pipeline stage name for failure reporting."""
        if self.scan_data is None:
            return "scan"
        if self.imports_data is None:
            return "extract_imports"
        if self.graph_data is None and not self.no_graph:
            return "graph_assembly"
        if self.summary_data is None:
            return "summary"
        return "manifest"

    def _record_post_cleanliness(self) -> None:
        """Snapshot post-pipeline target cleanliness."""
        self._target_dirty_files_after = _get_git_dirty_files(self.target_dir)
        self._target_dirty_after = len(self._target_dirty_files_after) > 0


# ── Module-level convenience function ────────────────────────────────────

def run_bundle_pipeline(
    target_dir: str,
    bundle_dir: str,
    *,
    no_graph: bool = False,
    in_repo_cache: bool = False,
    external_cache_dir: Optional[str] = None,
) -> dict:
    """Convenience: run a full bundle pipeline and return the manifest."""
    bundle = RunBundle(
        target_dir,
        bundle_dir,
        no_graph=no_graph,
        in_repo_cache=in_repo_cache,
        external_cache_dir=external_cache_dir,
    )
    return bundle.run()


# ── CLI entry point ────────────────────────────────────────────────────

def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run the full UA pipeline into a canonical artifact bundle.",
    )
    parser.add_argument(
        "target_dir",
        help="Path to the project directory to scan",
    )
    parser.add_argument(
        "bundle_dir",
        help="Path to the output bundle directory",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip graph assembly and validation",
    )
    parser.add_argument(
        "--in-repo-cache",
        action="store_true",
        default=False,
        help="Allow writing fingerprints inside the target repo "
             "(default: non-mutating, external cache)",
    )
    parser.add_argument(
        "--external-cache-dir",
        default=None,
        help="Directory for external fingerprint cache "
             "(default: CWD/.hermes-cache/<target>)",
    )
    args = parser.parse_args()

    target = os.path.realpath(args.target_dir)
    if not os.path.isdir(target):
        print(f"Error: '{args.target_dir}' is not a valid directory",
              file=sys.stderr)
        return 1

    try:
        manifest = run_bundle_pipeline(
            target,
            args.bundle_dir,
            no_graph=args.no_graph,
            in_repo_cache=args.in_repo_cache,
            external_cache_dir=args.external_cache_dir,
        )
        print(f"Bundle written to: {args.bundle_dir}")
        print(f"Run ID: {manifest['run_id']}")
        print(
            f"Target mutation allowed: {manifest['target_mutation_allowed']}",
        )
        return 0
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
