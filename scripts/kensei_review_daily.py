#!/usr/bin/env python3
"""
Kensei Review Daily — no_agent aggregator.

Reads the three upstream research artifacts (research-digest brief,
research-paper-synthesis HTML, github-radar txt) plus the wiki paper-mashups
log, and assembles a dark-mode HTML review. The synthesis here is mechanical:
extract headlines/sections from already-synthesised sources and present them.
The actual research judgement lives upstream (Remii's digest, the paper
synthesis LLM cron, the radar scorer). This script only aggregates + formats.

Output: writes review.html under runbooks/kensei-review-daily/YYYY-MM-DD/
and prints a short summary + MEDIA: tag to stdout for cron delivery.
"""
import html
import os
import re
import sys
from datetime import datetime
from pathlib import Path

def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))

HERMES = _hermes_home()
DIGEST_DIR = HERMES / "runbooks" / "research-digest"
SYNTH_DIR = HERMES / "runbooks" / "research-paper-synthesis"
# Radar lives under the Kensei repo root; allow override via env or CLI.
RADAR_DIR = Path(os.environ.get(
    "KENSEI_RADAR_ROOT",
    str(Path(os.environ.get("KENSEI_REPO_ROOT", "/home/kensei/repos/KenseiAgent"))
        / "runbooks" / "github-radar"),
))
MASHUPS_FILE = Path(os.environ.get(
    "KENSEI_WIKI_ROOT", os.path.expanduser("~/docs/wiki"),
)) / "_meta" / "paper-mashups.md"
OUT_DIR = HERMES / "runbooks" / "kensei-review-daily"
TODAY = datetime.now().strftime("%Y-%m-%d")
TODAY_UK = datetime.now().strftime("%d/%m/%Y")

BG = "#11100f"
CARD = "#1c1a18"
ACCENT = "#fbbf24"
TEXT = "#f5f5f4"
MUTED = "#a8a29e"


def latest(pattern_dir, glob):
    files = sorted(pattern_dir.glob(glob), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        if f.stat().st_size > 200:
            return f
    return None


def strip_tags(s):
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip()


def digest_headlines(path):
    """Extract h2 section titles + first paragraph from the digest brief."""
    text = path.read_text(errors="replace")
    # grab <h2>...</h2> and following text until next tag
    blocks = re.findall(r"<h2[^>]*>(.*?)</h2>\s*(.*?)(?=<h[12]|$)", text, re.S)
    out = []
    for title, body in blocks[:6]:
        title = strip_tags(title)
        # first sentence-ish of body
        body_txt = strip_tags(body)
        body_txt = body_txt[:160].replace("\n", " ").strip()
        if title:
            out.append((title, body_txt))
    return out


def synth_sections(path):
    """Extract h2 section headers + first item from paper-synthesis HTML."""
    text = path.read_text(errors="replace")
    # Top picks = repos/items under 'Full Treatment Papers' + 'Mashup Ideas'
    sections = {}
    for m in re.finditer(r"<h2[^>]*>(.*?)</h2>\s*(.*?)(?=<h2|$)", text, re.S):
        title = strip_tags(m.group(1))
        body = strip_tags(m.group(2))
        sections[title] = body[:400].replace("\n", " ").strip()
    picks = []
    for key in ("Full Treatment Papers", "Mashup Ideas", "Stack Improvement Opportunities"):
        if key in sections and sections[key]:
            picks.append((key, sections[key]))
    return picks


def radar_repos(path, classes=("ADOPT", "EXTRACT", "PLUGIN/SKILL"), limit=6):
    """Parse [REPO] blocks from the radar txt, filter by classification."""
    text = path.read_text(errors="replace")
    repos = []
    for m in re.finditer(r"\[REPO\]\s+(\S+)\s*\|\s*stars:(\d+)\s*\|\s*lang:(\S+)\s*\|\s*classification:(\S+)\s*\|\s*score:([\d.]+)[^\n]*\n\s*Description:\s*(.*?)\n\s*Why it matters:\s*(.*?)(?=\n\[REPO\]|\n---|\Z)", text, re.S):
        name, stars, lang, cls, score, desc, why = m.groups()
        if cls.upper() in classes:
            repos.append({
                "name": name, "stars": stars, "lang": lang,
                "cls": cls, "score": float(score),
                "desc": desc.strip()[:160], "why": why.strip()[:160],
            })
    repos.sort(key=lambda r: r["score"], reverse=True)
    return repos[:limit]


def mashup_recent(path, lines=60):
    """Tail the wiki mashups log for recent entries."""
    if not path.exists():
        return []
    content = path.read_text(errors="replace").splitlines()
    tail = [l.strip() for l in content[-lines:] if l.strip() and not l.startswith("#")]
    return tail[:8]


def build_html(digest, synth, radar, mashups):
    sections = []

    if radar:
        rows = "\n".join(
            f'<li><b>{html.escape(r["name"])}</b> <span class="meta">({r["cls"]} · ★{r["stars"]} · {r["lang"]} · score {r["score"]})</span><br>{html.escape(r["desc"])}</li>'
            for r in radar
        )
        sections.append(("⚡ Top Picks (GitHub Radar)", f"<ul>{rows}</ul>"))

    if synth:
        items = "\n".join(
            f'<li><b>{html.escape(t)}</b><br>{html.escape(b)}</li>' for t, b in synth
        )
        sections.append(("📄 Paper Synthesis Highlights", f"<ul>{items}</ul>"))

    if digest:
        items = "\n".join(
            f'<li><b>{html.escape(t)}</b><br>{html.escape(b)}</li>' for t, b in digest
        )
        sections.append(("📡 Research Digest Sections", f"<ul>{items}</ul>"))

    if mashups:
        items = "\n".join(f"<li>{html.escape(m)}</li>" for m in mashups)
        sections.append(("🔀 Recent Mashup Log", f"<ul>{items}</ul>"))

    body = "\n".join(
        f'<section><h2>{html.escape(t)}</h2>{c}</section>' for t, c in sections
    ) or "<section><p>No research signals surfaced today.</p></section>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kensei Review · {TODAY_UK}</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; background: {BG}; color: {TEXT}; max-width: 820px; margin: 2rem auto; padding: 1rem; }}
h1 {{ color: {ACCENT}; border-bottom: 1px solid #34302c; padding-bottom: 0.5rem; }}
h2 {{ color: {ACCENT}; margin-top: 1.5rem; }}
section {{ background: {CARD}; border-radius: 8px; padding: 1.25rem; margin: 1rem 0; border-left: 3px solid {ACCENT}; }}
.meta {{ color: {MUTED}; font-size: 0.85rem; }}
ul {{ margin-left: 1.25rem; }}
li {{ margin-bottom: 0.6rem; }}
</style>
</head>
<body>
<h1>🧠 Kensei Review · {TODAY_UK}</h1>
<p class="meta">Sources: digest · papers · radar · wiki-mashups</p>
{body}
</body>
</html>
"""


def main():
    digest_path = latest(DIGEST_DIR, "research-brief-*.html")
    synth_path = latest(SYNTH_DIR, "research-paper-synthesis-*.html")
    radar_path = None
    if RADAR_DIR.exists():
        radar_sub = sorted(
            [d for d in RADAR_DIR.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime, reverse=True,
        )
        for d in radar_sub:
            cand = d / "github-radar-repos.txt"
            if cand.exists() and cand.stat().st_size > 200:
                radar_path = cand
                break

    if not (digest_path or synth_path or radar_path):
        print("[SILENT]")
        return

    digest = digest_headlines(digest_path) if digest_path else []
    synth = synth_sections(synth_path) if synth_path else []
    radar = radar_repos(radar_path) if radar_path else []
    mashups = mashup_recent(MASHUPS_FILE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / TODAY / "review.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(digest, synth, radar, mashups))

    # Compact chat summary
    n_picks = len(radar) + len(synth) + len(digest)
    headline = radar[0]["name"] if radar else (synth[0][0] if synth else (digest[0][0] if digest else "no signals"))
    print(f"🧠 Kensei Review · {TODAY_UK} — {n_picks} signals aggregated (radar {len(radar)}, papers {len(synth)}, digest {len(digest)}). Headline: {headline}.")
    print(f"MEDIA:{out_path}")


if __name__ == "__main__":
    main()
