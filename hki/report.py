"""Markdown HKI reports."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from hki.inventory import utc_now_iso
from hki.manifest import Manifest, load_manifest
from hki.paths import HkiScope, as_workspace_relative


def build_sources_report(manifest: Manifest, *, manifest_path: Path) -> str:
    """Return a concise Markdown report for a source manifest."""

    root = Path(manifest.root)
    kind_counts = Counter(source.kind for source in manifest.sources)
    suffix_counts = Counter(Path(source.relative_path).suffix.lower() or "(none)" for source in manifest.sources)
    dir_counts = Counter(_top_level_dir(source.relative_path) for source in manifest.sources)
    largest = sorted(manifest.sources, key=lambda source: (-source.size, source.relative_path))[:10]
    total_size = sum(source.size for source in manifest.sources)
    manifest_ref = as_workspace_relative(manifest_path, root)

    lines = [
        "# HKI Source Inventory Report",
        "",
        f"Generated: {utc_now_iso()}",
        f"Workspace root: `{manifest.root}`",
        f"Manifest: `{manifest_ref}`",
        "",
        "This is an HKI source inventory report, not a semantic dossier yet.",
        "",
        "## Summary",
        "",
        f"- Total included files: {len(manifest.sources)}",
        f"- Total skipped/excluded entries: {manifest.inventory_skipped_count}",
        f"- Total included bytes: {total_size}",
        "",
        "## Counts by Kind",
        "",
    ]
    lines.extend(_count_lines(kind_counts))
    lines.extend(
        [
            "",
            "## Largest Files",
            "",
        ]
    )
    if largest:
        for source in largest:
            lines.append(f"- `{source.relative_path}` ({source.size} bytes, {source.kind})")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Notable Source Groups",
            "",
            "### By Directory",
            "",
        ]
    )
    lines.extend(_count_lines(dir_counts, limit=10))
    lines.extend(
        [
            "",
            "### By Extension",
            "",
        ]
    )
    lines.extend(_count_lines(suffix_counts, limit=10))
    lines.append("")
    return "\n".join(lines)


def write_sources_report(manifest: Manifest, scope: HkiScope) -> Path:
    """Write the source report to ``.hermes/hki/reports/sources.md``."""

    scope.reports_dir.mkdir(parents=True, exist_ok=True)
    content = build_sources_report(manifest, manifest_path=scope.manifest_path)
    path = scope.sources_report_path
    path.write_text(content, encoding="utf-8")
    return path


def load_manifest_for_scope(scope: HkiScope) -> Manifest:
    return load_manifest(scope.manifest_path)


def _count_lines(counter: Counter[str], *, limit: int | None = None) -> list[str]:
    if not counter:
        return ["- None"]
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        items = items[:limit]
    return [f"- `{name}`: {count}" for name, count in items]


def _top_level_dir(relative_path: str) -> str:
    parts = Path(relative_path).parts
    return parts[0] if len(parts) > 1 else "."
