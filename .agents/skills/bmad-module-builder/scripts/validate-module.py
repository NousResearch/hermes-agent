#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Validate a BMad module's structure and help CSV integrity.

Supports two module types:
- Multi-skill modules with a dedicated setup skill (*-setup directory)
- Standalone single-skill modules with self-registration (assets/module-setup.md)

Performs deterministic structural checks:
- Required files exist (setup skill or standalone structure)
- All skill folders have at least one capability entry in the CSV
- No orphan CSV entries pointing to nonexistent skills
- Menu codes are unique
- preceded-by/followed-by references point to real capability entries
- Required module.yaml fields are present
- CSV column count is consistent
"""

import argparse
import csv
import json
import sys
from io import StringIO
from pathlib import Path

REQUIRED_YAML_FIELDS = {"code", "name", "description"}
CSV_HEADER = [
    "module", "skill", "display-name", "menu-code", "description",
    "action", "args", "phase", "preceded-by", "followed-by", "required",
    "output-location", "outputs",
]


def find_setup_skill(module_dir: Path) -> Path | None:
    """Find the setup skill folder (*-setup)."""
    for d in module_dir.iterdir():
        if d.is_dir() and d.name.endswith("-setup"):
            return d
    return None


def find_skill_folders(module_dir: Path, exclude_name: str = "") -> list[str]:
    """Find all skill folders (directories with SKILL.md), optionally excluding one."""
    skills = []
    for d in module_dir.iterdir():
        if d.is_dir() and d.name != exclude_name and (d / "SKILL.md").is_file():
            skills.append(d.name)
    return sorted(skills)


def detect_standalone_module(module_dir: Path) -> Path | None:
    """Detect a standalone module: a single skill folder with assets/module.yaml.

    Works whether ``module_dir`` is the parent folder that contains the skill, or
    the standalone skill directory itself (the path a user is most likely to hand
    over for a single-skill module).
    """
    # Given the skill directory directly.
    if (module_dir / "SKILL.md").is_file() and (module_dir / "assets" / "module.yaml").is_file():
        return module_dir
    # Given the parent folder containing exactly one skill.
    skill_dirs = [
        d for d in module_dir.iterdir()
        if d.is_dir() and (d / "SKILL.md").is_file()
    ]
    if len(skill_dirs) == 1:
        candidate = skill_dirs[0]
        if (candidate / "assets" / "module.yaml").is_file():
            return candidate
    return None


def parse_yaml_minimal(text: str) -> dict[str, str]:
    """Parse top-level YAML key-value pairs (no nested structures)."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if ":" in line and not line.startswith("#") and not line.startswith("-"):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value and not value.startswith(">"):
                result[key] = value
    return result


def parse_csv_rows(csv_text: str) -> tuple[list[str], list[dict[str, str]], list[int]]:
    """Parse CSV text into (header, row dicts, raw column count per data row).

    ``restval=""`` fills missing trailing fields in a short row with empty strings
    instead of ``None``, so downstream ``.strip()`` calls stay safe on malformed
    rows. DictReader pads short rows to the header width, so ``len(row)`` cannot
    reveal a field shortfall; the raw per-row column counts from ``csv.reader``
    (blank lines skipped, to stay aligned with DictReader) are returned separately
    for the column-count consistency check.
    """
    reader = csv.DictReader(StringIO(csv_text), restval="")
    header = reader.fieldnames or []
    rows = list(reader)
    raw_rows = list(csv.reader(StringIO(csv_text)))
    col_counts = [len(r) for r in raw_rows[1:] if r != []]
    return header, rows, col_counts


def validate(module_dir: Path, verbose: bool = False) -> dict:
    """Run all structural validations. Returns JSON-serializable result."""
    findings: list[dict] = []
    info: dict = {}

    def finding(severity: str, category: str, message: str, detail: str = ""):
        findings.append({
            "severity": severity,
            "category": category,
            "message": message,
            "detail": detail,
        })

    # 1. Find setup skill or detect standalone module
    setup_dir = find_setup_skill(module_dir)
    standalone_dir = None

    if not setup_dir:
        standalone_dir = detect_standalone_module(module_dir)
        if not standalone_dir:
            finding("critical", "structure",
                    "No setup skill found (*-setup directory) and no standalone module detected")
            return {"status": "fail", "findings": findings, "info": info}

    # Branch: standalone vs multi-skill
    if standalone_dir:
        info["standalone"] = True
        info["skill_dir"] = standalone_dir.name
        skill_dir = standalone_dir

        # 2s. Check required files for standalone module
        required_files = {
            "assets/module.yaml": skill_dir / "assets" / "module.yaml",
            "assets/module-help.csv": skill_dir / "assets" / "module-help.csv",
            "assets/module-setup.md": skill_dir / "assets" / "module-setup.md",
        }
        # Merge scripts: accept either the dash form the scaffolder emits
        # (merge-config.py) or the importable underscore form (merge_config.py).
        # Both are valid — a module may rename them to be importable from
        # module-setup.md without that being a structural defect.
        required_any = {
            "scripts/merge-config.py (or merge_config.py)": [
                skill_dir / "scripts" / "merge-config.py",
                skill_dir / "scripts" / "merge_config.py",
            ],
            "scripts/merge-help-csv.py (or merge_help_csv.py)": [
                skill_dir / "scripts" / "merge-help-csv.py",
                skill_dir / "scripts" / "merge_help_csv.py",
            ],
        }
        ok = True
        for label, path in required_files.items():
            if not path.is_file():
                finding("critical", "structure", f"Missing required file: {label}")
                ok = False
        for label, candidates in required_any.items():
            if not any(p.is_file() for p in candidates):
                finding("critical", "structure", f"Missing required file: {label}")
                ok = False

        if not ok:
            return {"status": "fail", "findings": findings, "info": info}

        yaml_dir = skill_dir
        csv_dir = skill_dir
    else:
        info["setup_skill"] = setup_dir.name

        # 2. Check required files in setup skill
        required_files = {
            "SKILL.md": setup_dir / "SKILL.md",
            "assets/module.yaml": setup_dir / "assets" / "module.yaml",
            "assets/module-help.csv": setup_dir / "assets" / "module-help.csv",
        }
        for label, path in required_files.items():
            if not path.is_file():
                finding("critical", "structure", f"Missing required file: {label}")

        if not all(p.is_file() for p in required_files.values()):
            return {"status": "fail", "findings": findings, "info": info}

        yaml_dir = setup_dir
        csv_dir = setup_dir

    # 3. Validate module.yaml
    yaml_text = (yaml_dir / "assets" / "module.yaml").read_text(encoding="utf-8")
    yaml_data = parse_yaml_minimal(yaml_text)
    info["module_code"] = yaml_data.get("code", "")
    info["module_name"] = yaml_data.get("name", "")

    for field in REQUIRED_YAML_FIELDS:
        if not yaml_data.get(field):
            finding("high", "yaml", f"module.yaml missing or empty required field: {field}")

    # 4. Parse and validate CSV
    csv_text = (csv_dir / "assets" / "module-help.csv").read_text(encoding="utf-8")
    header, rows, col_counts = parse_csv_rows(csv_text)

    # Check header
    if header != CSV_HEADER:
        missing = set(CSV_HEADER) - set(header)
        extra = set(header) - set(CSV_HEADER)
        detail_parts = []
        if missing:
            detail_parts.append(f"missing: {', '.join(sorted(missing))}")
        if extra:
            detail_parts.append(f"extra: {', '.join(sorted(extra))}")
        finding("high", "csv-header", f"CSV header mismatch: {'; '.join(detail_parts)}")

    if not rows:
        finding("high", "csv-empty", "module-help.csv has no capability entries")
        return {"status": "fail", "findings": findings, "info": info}

    info["csv_entries"] = len(rows)

    # 5. Check column count consistency (using raw field counts: DictReader pads
    #    short rows to the header width, so len(row) alone can't detect a shortfall)
    expected_cols = len(CSV_HEADER)
    for i, (row, n_cols) in enumerate(zip(rows, col_counts)):
        if n_cols != expected_cols:
            finding("medium", "csv-columns", f"Row {i + 2} has {n_cols} columns, expected {expected_cols}",
                    f"skill={row.get('skill', '?')}")

    # 6. Collect skills from CSV and filesystem
    csv_skills = {row.get("skill", "") for row in rows}
    if standalone_dir:
        # The only valid skill is the standalone skill itself, whether we were
        # handed the module's parent folder or the skill directory directly.
        skill_folders = [standalone_dir.name]
    else:
        skill_folders = find_skill_folders(module_dir, setup_dir.name)
    info["skill_folders"] = skill_folders
    info["csv_skills"] = sorted(csv_skills)

    # 7. Skills without CSV entries
    for skill in skill_folders:
        if skill not in csv_skills:
            finding("high", "missing-entry", f"Skill '{skill}' has no capability entries in the CSV")

    # 8. Orphan CSV entries
    setup_name = setup_dir.name if setup_dir else ""
    for skill in csv_skills:
        if skill in skill_folders or skill == setup_name:
            continue
        # For a standalone module, skill_folders already enumerates every valid
        # skill, so any other CSV skill is an orphan — never look at the parent
        # folder (which may hold unrelated sibling skills when validating a skill
        # dir directly). For a multi-skill module, re-check the filesystem: the
        # setup skill lives alongside the others and is excluded from skill_folders.
        if standalone_dir or not (module_dir / skill / "SKILL.md").is_file():
            finding("high", "orphan-entry", f"CSV references skill '{skill}' which does not exist in the module folder")

    # 9. Unique menu codes
    menu_codes: dict[str, list[str]] = {}
    for row in rows:
        code = row.get("menu-code", "").strip()
        if code:
            menu_codes.setdefault(code, []).append(row.get("display-name", "?"))

    for code, names in menu_codes.items():
        if len(names) > 1:
            finding("high", "duplicate-menu-code", f"Menu code '{code}' used by multiple entries: {', '.join(names)}")

    # 10. preceded-by/followed-by reference validation
    # Build set of valid capability references (skill:action)
    valid_refs = set()
    for row in rows:
        skill = row.get("skill", "").strip()
        action = row.get("action", "").strip()
        if skill and action:
            valid_refs.add(f"{skill}:{action}")

    for row in rows:
        display = row.get("display-name", "?")
        for field in ("preceded-by", "followed-by"):
            value = row.get(field, "").strip()
            if not value:
                continue
            # Can be comma-separated
            for ref in value.split(","):
                ref = ref.strip()
                if not ref:
                    continue
                # A colon-less ref is a cross-module positional reference (a bare
                # sibling-module skill name, e.g. "bmad-sprint-planning"). Other
                # installed modules aren't visible here, so it can't be resolved
                # and isn't a defect — only validate intra-module skill:action refs.
                if ":" not in ref:
                    continue
                if ref not in valid_refs:
                    finding("medium", "invalid-ref",
                            f"'{display}' {field} references '{ref}' which is not a valid capability",
                            "Expected format: skill-name:action-name")

    # 11. Required fields in each row
    for row in rows:
        display = row.get("display-name", "?")
        for field in ("skill", "display-name", "menu-code", "description"):
            if not row.get(field, "").strip():
                finding("high", "missing-field", f"Entry '{display}' is missing required field: {field}")

    # Summary
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for f in findings:
        severity_counts[f["severity"]] = severity_counts.get(f["severity"], 0) + 1

    status = "pass" if severity_counts["critical"] == 0 and severity_counts["high"] == 0 else "fail"

    return {
        "status": status,
        "info": info,
        "findings": findings,
        "summary": {
            "total_findings": len(findings),
            "by_severity": severity_counts,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a BMad module's setup skill structure and help CSV integrity"
    )
    parser.add_argument(
        "module_dir",
        help="Path to the module's skills folder (containing the setup skill and other skills)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress to stderr")
    args = parser.parse_args()

    module_path = Path(args.module_dir)
    if not module_path.is_dir():
        print(json.dumps({"status": "error", "message": f"Not a directory: {module_path}"}))
        return 2

    result = validate(module_path, verbose=args.verbose)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
