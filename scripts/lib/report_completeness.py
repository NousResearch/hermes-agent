#!/usr/bin/env python3
"""Relational RUN-to-report completeness check for the North Forge handoff bundle.

Replaces the old date-only heuristic in ``scripts/collect-logs.{ps1,sh}`` (which
proved only "a report exists dated >= the newest ledger id" — one same-day report
made every run that day look covered; ERR-2026-09-07-004 / Codex F-02 / R-01).

This is *relational*: every ``RUN-YYYY-MM-DD-NNN`` id that appears anywhere in the
ledger must have a row in ``logs/ledger/reports/REPORT-MANIFEST.md``, and that row
must either name a session report that actually exists in the handoff out-dir and
mentions that run id, or be an explicit ``ledger-only`` disposition. A run with no
row, or a row pointing at a missing / mismatched report, is a **FAIL** (not WARN).

Invoked by both collectors; it does the heavy parsing once so the PowerShell and
POSIX paths stay byte-identical in behaviour. Output is one ``LEVEL<TAB>message``
line per finding on stdout (LEVEL in OK / WARN / FAIL). Exit 0 = no FAIL.

    python scripts/lib/report_completeness.py \
        --ledger-dir <out>/ledger  --reports-dir <out>  [--manifest <path>]

``--manifest`` defaults to ``<ledger-dir>/reports/REPORT-MANIFEST.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

RUN_RE = re.compile(r"RUN-20\d\d-\d\d-\d\d-\d{3}")
ID_RE = re.compile(r"(?:CHG|ERR|DECISION|AUDIT)-20\d\d-\d\d-\d\d-\d{3}")
WIKILINK_RE = re.compile(r"\[\[([A-Za-z0-9._/-]+)\]\]")
# strip ```fenced``` and `inline` code before scanning for [[links]] — the ledger
# discusses the `[[name]]` syntax (and TOML `[[bin]]`) inside code spans in prose.
CODE_SPAN_RE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)

# The ledger README sanctions `[[name]]` as a link to a memory doc that lives
# outside the repo. These are known-good; not broken file links.
KNOWN_MEMORY_REFS = {"north-forge-architecture", "north-forge-agent-ledger",
                     "north-forge-installer-gui", "collect-logs-standing-practice"}

# A manifest data row starts with `| RUN-<id> |` — the doc tables above it do not.
MANIFEST_ROW_START = re.compile(r"^\|\s*RUN-20\d\d-\d\d-\d\d-\d{3}\s*\|")
# Full row:  | RUN-... | <report or —> | <covers or —> | <source-of-truth> | <sha256 or —> |
ROW_RE = re.compile(
    r"^\|\s*(RUN-20\d\d-\d\d-\d\d-\d{3})\s*"
    r"\|\s*(.*?)\s*"
    r"\|\s*(.*?)\s*"
    r"\|\s*(.*?)\s*"
    r"\|\s*(.*?)\s*\|?\s*$"
)

LEDGER_ONLY_TOKENS = {"ledger-only", "ledgeronly", "—", "-", "n/a", "none"}


class Findings:
    def __init__(self, quiet: bool = False) -> None:
        self.rows: list[tuple[str, str, bool]] = []  # (level, msg, routine)
        self.quiet = quiet

    def ok(self, msg: str, *, routine: bool = False) -> None:
        self.rows.append(("OK", msg, routine))

    def warn(self, msg: str) -> None:
        self.rows.append(("WARN", msg, False))

    def fail(self, msg: str) -> None:
        self.rows.append(("FAIL", msg, False))

    def emit(self) -> int:
        for level, msg, routine in self.rows:
            if self.quiet and routine:
                continue
            print(f"{level}\t{msg}")
        return 1 if any(lvl == "FAIL" for lvl, _, _ in self.rows) else 0


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def parse_manifest(path: Path) -> tuple[dict[str, dict], list[str]]:
    """Return ({run_id: {report, covers, source, sha}}, [parse warnings])."""
    rows: dict[str, dict] = {}
    warns: list[str] = []
    if not path.is_file():
        return rows, [f"REPORT-MANIFEST not found at {path}"]
    for raw in _read(path).splitlines():
        line = raw.rstrip()
        if not MANIFEST_ROW_START.match(line):
            continue
        m = ROW_RE.match(line)
        if not m:
            warns.append(f"unparseable manifest row: {line.strip()[:120]}")
            continue
        run, report, covers, source, sha = (g.strip() for g in m.groups())
        report = "" if report in ("—", "-", "") else report
        rows[run] = {
            "report": report,
            "covers": covers,
            "source": source,
            "sha": sha.lower().replace("…", "").replace("...", "").strip(),
        }
    return rows, warns


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger-dir", required=True, type=Path)
    ap.add_argument("--reports-dir", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the per-run 'covered by' OK lines; keep WARN/FAIL + the summary")
    args = ap.parse_args(argv)

    ledger_dir: Path = args.ledger_dir
    reports_dir: Path = args.reports_dir
    manifest_path: Path = args.manifest or (ledger_dir / "reports" / "REPORT-MANIFEST.md")

    f = Findings(quiet=args.quiet)

    if not ledger_dir.is_dir():
        f.fail(f"ledger dir {ledger_dir} not found — cannot check run coverage")
        return f.emit()

    # 1. every RUN- id used anywhere in the ledger
    ledger_runs: set[str] = set()
    ledger_blob_parts: list[str] = []
    for p in sorted(ledger_dir.rglob("*.md")):
        txt = _read(p)
        ledger_blob_parts.append(txt)
        ledger_runs.update(RUN_RE.findall(txt))
    ledger_blob = "\n".join(ledger_blob_parts)

    if not ledger_runs:
        f.warn("no RUN- ids found in the ledger — nothing to check")
        return f.emit()

    # 2. the manifest
    manifest, mwarns = parse_manifest(manifest_path)
    for w in mwarns:
        f.warn(w)

    # 3. reports present in the out-dir (exclude generated files)
    present: dict[str, Path] = {}
    for p in sorted(reports_dir.glob("*.md")):
        if p.name in {"HANDOFF-INDEX.md"}:
            continue
        present[p.name] = p

    covered = 0
    for run in sorted(ledger_runs):
        row = manifest.get(run)
        if row is None:
            f.fail(
                f"{run} is in the ledger but has no row in REPORT-MANIFEST.md "
                f"— add one (a report, or an explicit ledger-only disposition)"
            )
            continue

        report = row["report"]
        source = (row["source"] or "").lower()
        if not report:
            if source in LEDGER_ONLY_TOKENS or "ledger-only" in source:
                f.ok(f"{run}: ledger-only (no session report) — {row['source'] or 'declared'}", routine=True)
                covered += 1
            else:
                f.fail(
                    f"{run}: manifest row has no report file and is not marked "
                    f"ledger-only (source='{row['source']}')"
                )
            continue

        rp = present.get(report)
        if rp is None:
            f.fail(f"{run}: manifest names report '{report}' but it is not in {reports_dir}")
            continue

        rtext = _read(rp)
        if run not in rtext:
            f.fail(f"{run}: report '{report}' exists but does not mention {run} — wrong file?")
            continue

        # sha check is advisory: a mismatch means the report was edited since the
        # manifest was last refreshed, not that a run is uncovered.
        want = row["sha"]
        if want and want not in ("—", "-", "n/a", ""):
            got = hashlib.sha256(rp.read_bytes()).hexdigest()
            if not (got == want or got.startswith(want)):
                f.warn(
                    f"{run}: report '{report}' sha256 {got[:16]}… != manifest {want[:16]}… "
                    f"(report changed — refresh the manifest row)"
                )
        covered += 1
        f.ok(f"{run}: covered by {report}", routine=True)

    # 4. manifest rows for runs that are not (yet) in the ledger — informational
    for run in sorted(set(manifest) - ledger_runs):
        f.warn(f"REPORT-MANIFEST lists {run} but no ledger entry uses it (stale row?)")

    # 5. unresolved [[wiki-links]] in the ledger (R-01 step 4). Strip code spans
    #    first — the ledger discusses the `[[name]]` / TOML `[[bin]]` syntax in
    #    prose. A remaining link that resolves to a tracked file / manifest run is
    #    fine; a known external memory ref is fine (OK, not WARN — the ledger
    #    README sanctions the convention); anything else is a WARN.
    prose = CODE_SPAN_RE.sub(" ", ledger_blob)
    tracked_stems = {p.stem for p in ledger_dir.rglob("*")} | {p.name for p in ledger_dir.rglob("*")}
    links = set(WIKILINK_RE.findall(prose))
    memory_refs = sorted(n for n in links if n in KNOWN_MEMORY_REFS)
    unresolved = sorted(
        n for n in links
        if n not in tracked_stems and n not in manifest and n not in KNOWN_MEMORY_REFS
    )
    if memory_refs:
        f.ok(f"ledger [[memory refs]] (external, sanctioned by README): {', '.join(memory_refs)}")
    if unresolved:
        f.warn(
            "ledger [[wiki-links]] resolving to no tracked file, manifest run, or "
            "known memory ref: " + ", ".join(unresolved) + " — fix or allowlist"
        )

    total = len(ledger_runs)
    if covered == total:
        f.ok(f"run-to-report: all {total} ledger RUN- id(s) accounted for")
    else:
        f.fail(f"run-to-report: {total - covered} of {total} ledger RUN- id(s) NOT covered")

    return f.emit()


if __name__ == "__main__":
    raise SystemExit(main())
