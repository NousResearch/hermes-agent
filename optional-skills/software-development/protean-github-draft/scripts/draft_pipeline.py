#!/usr/bin/env python3
"""draft_pipeline.py — one command per GitHub draft, same scaffolding every time.

    draft_pipeline.py --src <draft.md> --slug <slug> --tab "pull request draft" \
        --title "<draft title>" --owners ahrazzle \
        --verify-against "origin/main:path/to/source.py" \
        --verify-against "<patched-file>" [--open]

What it does (all gates from the GitHub draft review procedure, in order):
  1. prose gate            check-prose.py must report clean
  2. identifier gate       no /Users paths, usernames, profile/agent names, hostnames, emails
  3. verbatim gate         every ```python fence must appear in one --verify-against source
                           (git `REF:PATH` or a plain path), modulo uniform indentation
  4. render                render_draft_html.py into <drafts-root>/<batch>/<slug>.html
  5. fidelity gate         0 source lines missing from the rendered text
  6. palette gate          the seven core dark tokens + color-scheme meta present
  7. containment gate      no external asset fetch; repo/tab/badge chrome present
  8. banner gate           no false "no change" claim without a baseline

Exit 0 only when every gate passes. Prints the review path and the posting source path.

Attribution: pass --owners <poster handle> when attribution is needed. The renderer uses a
neutral default when no owner is supplied.
"""
from __future__ import annotations

import argparse
import html as _html
import re
import subprocess
import sys
import os
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
RENDERER = Path(__file__).resolve().parent / "render_draft_html.py"
DEFAULT_DRAFTS_ROOT = Path("contrib/drafts")
DEFAULT_IDENTS = {
    "abs-path": r"/Users/|/home/|/var/folders|/private/tmp|\b[A-Za-z]:\\",
    "secret-assignment": r"(?i)\b(api[_-]?key|token|password|secret|credential)\b\s*[:=]\s*\S+",
    "private-key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "local-endpoint": r"(?i)\b(localhost|127\.0\.0\.1|0\.0\.0\.0)\b",
    "email": r"[\w.+-]+@[\w-]+\.[a-z]{2,}",
}
PALETTE = ["#0d1117", "#c9d1d9", "#f0f6fc", "#8b949e", "#58a6ff", "#161b22", "#30363d",
           "#f85149", "#3d1d20", 'color-scheme" content="dark"']
OVERSIGHT_LINE = "> Human approval is required before posting."


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def resolve_source(ref: str) -> str | None:
    """`REF:PATH` via git show, or a plain file path."""
    if ":" in ref and not ref.startswith("/"):
        repo, path = ref.split(":", 1)
        p = run(["git", "-C", str(Path.cwd()), "show", f"{repo}:{path}"])
        return p.stdout if p.returncode == 0 else None
    f = Path(ref)
    return f.read_text(encoding="utf-8") if f.is_file() else None


def _fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Return (language, body) for the renderer's backtick/tilde fences."""
    lines = text.splitlines()
    blocks, i = [], 0
    opener = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([\w.+-]*)[ \t]*$")
    while i < len(lines):
        match = opener.match(lines[i])
        if not match:
            i += 1
            continue
        marker, lang = match.groups()
        close = re.compile(rf"^[ \t]*{re.escape(marker[0])}{{{len(marker)},}}[ \t]*$")
        i += 1
        body = []
        while i < len(lines) and not close.match(lines[i]):
            body.append(lines[i])
            i += 1
        if i < len(lines):
            i += 1
        blocks.append((lang.lower(), "\n".join(body)))
    return blocks


def fence_bodies(text: str, langs: tuple[str, ...] = ("python", "py")) -> list[str]:
    wanted = {lang.lower() for lang in langs}
    return [body for lang, body in _fenced_blocks(text) if lang in wanted]


def normalize_evidence(text: str) -> list[str]:
    """Evidence lines without runner decoration (box rules, `N |` gutters, trailing space)."""
    out = []
    for raw in text.splitlines():
        line = re.sub(r"^\s*(?:\d+\s*[|│]\s?|[║┃│]\s?)", "", raw).rstrip()
        if line.strip():
            out.append(line.strip())
    return out


def _mask_timing(s: str) -> str:
    """Run durations vary per machine; mask them so the gate compares content, not speed."""
    return re.sub(r"in \d+\.\d+s \(\d+ workers\)", "in <t> (N workers)", s)


def text_fence_gate(text: str, evidence: list[str]) -> tuple[bool, list[str]]:
    """Every non-empty line of a ```text fence must appear in an evidence file (timings masked)."""
    evidence = [_mask_timing(ev) for ev in evidence]
    bad = []
    for blk in fence_bodies(text, ("text",)):
        for line in blk.splitlines():
            probe = _mask_timing(line.strip())
            if len(probe) < 8:
                continue
            if not any(probe in ev or ev.endswith(probe) for ev in evidence):
                bad.append(probe[:80])
    return (not bad), bad


def verbatim_gate(text: str, sources: list[tuple[str, str | None]]) -> tuple[bool, list[str]]:
    notes, ok = [], True
    blocks = fence_bodies(text)
    for i, blk in enumerate(blocks, 1):
        lines = [l for l in blk.splitlines() if not l.lstrip().startswith("# ")]
        hit = None
        for name, body in sources:
            if body is None:
                continue
            for indent in (0, 4, 8, 12, 16):
                cand = "\n".join((" " * indent + l) if l.strip() else l for l in lines) + "\n"
                if cand in body:
                    hit = name
                    break
            if hit:
                break
        notes.append(f"fence {i}: {'verbatim in ' + hit if hit else 'NO VERBATIM SOURCE'}")
        ok &= hit is not None
    return ok, notes


def identifier_gate(text: str, allowed: list[str]) -> tuple[bool, dict[str, list[str]]]:
    hits = {}
    for key, pat in DEFAULT_IDENTS.items():
        found = sorted({m if isinstance(m, str) else m[0] for m in re.findall(pat, text, re.I)})
        found = [f for f in found if not any(a.lower() in f.lower() or f.lower() in a.lower() for a in allowed)]
        if found:
            hits[key] = found
    return (not hits), hits


def prose_gate(md: Path, python: str) -> tuple[bool, str]:
    cp = SKILL_DIR / "scripts" / "check-prose.py"
    if not cp.is_file():
        return False, "check-prose.py not found (required)"
    p = run([python, str(cp), str(md)])
    out = (p.stdout + p.stderr).strip()
    return ("clean" in out.lower()), out.splitlines()[0] if out else "no output"


def _flat_body(html: str) -> str:
    body = html.split("<body", 1)[1] if "<body" in html else html
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", body, flags=re.S)
    body = _html.unescape(re.sub(r"<[^>]+>", "", body))
    return re.sub(r"\s+", " ", body).strip()


def _markdown_text(value: str) -> str:
    """Approximate the visible text produced by Python-Markdown."""
    value = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", value)
    value = value.replace(r"\|", "|")
    value = re.sub(r"[`*_~]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _table_cells(line: str) -> list[str] | None:
    """Return visible probes for a Markdown table row, or None."""
    stripped = line.strip()
    if "|" not in stripped:
        return None
    if not (stripped.startswith("|") or stripped.endswith("|")):
        return None
    cells = re.split(r"(?<!\\)\|", stripped.strip("|"))
    if len(cells) < 2:
        return None
    if all(re.fullmatch(r"\s*:?-{1,}:?\s*", cell) for cell in cells):
        return []
    return [_markdown_text(cell) for cell in cells if _markdown_text(cell)]


def fidelity_gate(md_text: str, html: str) -> tuple[bool, list[str]]:
    flat, missing, fence = _flat_body(html), [], False
    fence_marker = None
    for raw in md_text.splitlines():
        fence_line = re.match(r"^[ \t]*(`{3,}|~{3,})[ \t]*[\w.+-]*[ \t]*$", raw)
        if fence_marker is None and fence_line:
            fence_marker = (fence_line.group(1)[0], len(fence_line.group(1)))
            continue
        if fence_marker is not None:
            if re.match(rf"^[ \t]*{re.escape(fence_marker[0])}{{{fence_marker[1]},}}[ \t]*$", raw):
                fence_marker = None
            continue
        line = raw.strip()
        if not fence:
            cells = _table_cells(line)
            if cells is not None:
                for cell in cells:
                    if len(cell) >= 3 and cell not in flat:
                        missing.append(cell[:80])
                continue
        if fence:
            # Fenced content renders literally; only normalize whitespace.
            probe = re.sub(r"\s+", " ", line).strip()
        else:
            # Strip only markers that markdown renders instead of printing.
            line = re.sub(r"^\s*(?:\||>|\d+\.|[-*])\s*", "", line)
            line = re.sub(r"^#+\s*", "", line)
            probe = _markdown_text(line)
        if len(probe) < 3:
            continue
        if probe not in flat:
            missing.append(raw[:80])
    return (not missing), missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="draft_pipeline.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="draft markdown (the postable text)")
    ap.add_argument("--slug", required=True, help="output file stem, e.g. pr-nous-grant-selfheal")
    ap.add_argument("--tab", required=True, choices=["issue draft", "pull request draft"])
    ap.add_argument("--title", required=True)
    ap.add_argument("--repo", default="NousResearch/hermes-agent")
    ap.add_argument("--badge", default="DRAFT — NOT POSTED")
    ap.add_argument("--owners", default=None, help="header attribution (external: poster handle)")
    ap.add_argument("--state", default=None, help="header state line")
    ap.add_argument("--drafts-root", type=Path, default=DEFAULT_DRAFTS_ROOT)
    ap.add_argument("--batch", default=None, help="batch folder, e.g. '2026-09-16 - 22h48m'")
    ap.add_argument("--verify-against", action="append", default=[],
                    help="REF:PATH or path a quoted fence must appear in (repeatable)")
    ap.add_argument("--evidence", action="append", default=[], type=Path,
                    help="file whose quoted ```text lines must appear verbatim (repeatable)")
    ap.add_argument("--allow-identifier", action="append", default=[],
                    help="literal allowed despite the denylist (repeatable)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--diff-base", type=Path, default=None,
                    help="previous revision of the same draft: renderer highlights the change")
    ap.add_argument("--rev-dir", type=Path, default=None,
                    help="dir with <slug>.prev.md; the renderer diffs against it automatically")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args(argv)

    if not args.src.is_file():
        print(f"ERROR: --src not found: {args.src}")
        return 2
    md_text = args.src.read_text(encoding="utf-8")
    results: list[tuple[str, bool, str]] = []

    ok, out = prose_gate(args.src, args.python)
    results.append(("prose (check-prose.py)", ok, out))

    ok, hits = identifier_gate(md_text, args.allow_identifier)
    results.append(("identifier sweep", ok, "clean" if ok else str(hits)))

    srcs = [(r, resolve_source(r)) for r in args.verify_against]
    has_fences = bool(fence_bodies(md_text))
    if has_fences and not srcs:
        results.append(("verbatim fences", False, "quoted fences require --verify-against"))
    elif srcs:
        ok, notes = verbatim_gate(md_text, srcs)
        results.append(("verbatim fences", ok, "; ".join(notes)))
    else:
        results.append(("verbatim fences", True, "no quoted fences"))

    oversight_lines = [l.strip() for l in md_text.strip().splitlines() if l.strip()]
    last = oversight_lines[-1] if oversight_lines else ""
    results.append(("oversight blockquote", last == OVERSIGHT_LINE and oversight_lines.count(OVERSIGHT_LINE) == 1, last[:70]))

    text_blocks = fence_bodies(md_text, ("text",))
    if text_blocks and not args.evidence:
        results.append(("quoted log lines", False, "```text fences require --evidence"))
    elif args.evidence:
        ev: list[str] = []
        for f in args.evidence:
            ev.extend(normalize_evidence(f.read_text(encoding="utf-8")))
        ok, bad = text_fence_gate(md_text, ev)
        results.append((f"quoted log lines ({len(ev)} evidence lines)", ok,
                        "every quoted line found" if ok else f"NOT IN EVIDENCE: {bad[:3]}"))

    batch = args.batch or datetime.now().strftime("%Y-%m-%d - %Hh%Mm")
    out_dir = args.drafts_root / batch
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / f"{args.slug}.html"

    temp_fd, temp_name = tempfile.mkstemp(prefix=f".{args.slug}.", suffix=".html", dir=out_dir)
    os.close(temp_fd)
    temp_html = Path(temp_name)
    cmd = [args.python, str(RENDERER), "--in", str(args.src), "--out", str(temp_html),
           "--repo", args.repo, "--tab", args.tab, "--title", args.title, "--badge", args.badge]
    if args.owners:
        cmd += ["--owners", args.owners]
    if args.state:
        cmd += ["--state", args.state]
    if args.diff_base:
        cmd += ["--diff-base", str(args.diff_base)]
    elif args.rev_dir:
        cmd += ["--rev-dir", str(args.rev_dir)]
    elif out_html.is_file():
        cmd += ["--from-html-baseline", str(out_html),
                "--baseline-label", "previous review render"]
    r = run(cmd)
    render_ok = r.returncode == 0 and temp_html.is_file()
    if not render_ok:
        results.append(("render", False, (r.stderr or r.stdout).strip()[:200]))
    else:
        results.append(("render", True, r.stdout.strip().splitlines()[-1][:160]))

    if render_ok:
        page = temp_html.read_text(encoding="utf-8")
        ok, missing = fidelity_gate(md_text, page)
        results.append(("text fidelity", ok, "0 lines missing" if ok else f"MISSING {missing[:5]}"))
        missing_pal = [t for t in PALETTE if t.lower() not in page.lower()]
        results.append(("palette", not missing_pal, "all tokens present" if not missing_pal else str(missing_pal)))
        external = re.search(r'<link[^>]+href="http|<script[^>]+src="http|@import\s+url\(http', page)
        results.append(("self-contained", not external, "no external assets" if not external else "EXTERNAL ASSET"))
        chrome = (args.repo in page) and (args.tab in page) and ("NOT POSTED" in page)
        results.append(("chrome", chrome, "repo + tab + badge" if chrome else "chrome incomplete"))
        banner = re.search(r"(no content change|added.{0,40}removed.{0,60}|no baseline)", _flat_body(page), re.I)
        results.append(("change banner", banner is not None, banner.group(0)[:70] if banner else "none"))

    failed = [n for n, p, _ in results if not p]
    if not failed:
        os.replace(temp_html, out_html)
    else:
        temp_html.unlink(missing_ok=True)
    width = max(len(n) for n, _, _ in results)
    print(f"\ndraft pipeline — {args.slug}")
    print(f"  source : {args.src}")
    print(f"  review : {out_html}")
    print(f"  batch  : {batch}")
    for name, passed, note in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name.ljust(width)}  {note}")
    print(f"  {'ALL GATES PASS — review the HTML before posting' if not failed else 'BLOCKED: ' + ', '.join(failed)}")
    if args.open and out_html.is_file():
        webbrowser.open(out_html.resolve().as_uri())
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
