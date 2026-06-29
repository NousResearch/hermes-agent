#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Serve the brainstorming technique library without loading it all into context.

The library is a CSV (category, technique_name, description, detail). `description`
is a short gist — enough to propose and run most techniques. `detail` is optional:
a path (relative to the CSV's directory) to a fuller instruction file for a technique
complex enough to warrant one. Only `show` resolves detail files, and only for the
technique asked for — so the heavy material never enters context until it is run.

Commands:
  categories                  list category names + counts (the cheap entry point)
  list --category C [...]      the index (name + gist) for those categories
  list --all                  the whole index at once — deliberate; large, avoid interactively
  show NAME [NAME ...]         full gist for each, inlining its detail file if it has one
  random [--category C] [-n N]  pick N at random (optionally within categories)
  html --out PATH             write the offline 'browse all' selection page to a file

`list` refuses to run with neither --category nor --all, and `html` writes to a file
rather than stdout: dumping the full catalog into context is a footgun, so reaching the
whole library at once must always be an explicit, deliberate choice.

`--extra PATH` merges a JSON overlay of additional techniques (customize.toml's
`additional_techniques`) into every command, so custom techniques and whole new
categories are first-class everywhere — including the browse page and category draws.

Default output is lean text for an LLM to read; pass --json for structured output.
"""
import argparse
import csv
import hashlib
import html
import json
import random
import sys
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parent.parent / "assets" / "brain-methods.csv"
FIELDS = ("category", "technique_name", "description", "detail", "provenance", "good_for", "audience")
# Optional columns beyond the original four — absent in older CSVs and in --extra
# overlays, so always read through .get/setdefault. `provenance` (classic|signature|
# playful) drives the "Proven & Professional" lead group; `good_for` (a |-separated
# list of goal tags) drives the browse page's goal filter; `audience` (solo|group|either)
# is advisory.
OPTIONAL_FIELDS = ("detail", "provenance", "good_for", "audience")


def load(file: Path) -> list[dict]:
    with open(file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in OPTIONAL_FIELDS:
            r.setdefault(k, "")
            r[k] = (r.get(k) or "").strip()
    return rows


def load_extra(file: Path) -> list[dict]:
    """Merge-in techniques from a JSON overlay — a list of
    {category, technique_name, description[, detail]} objects. This is how
    customize.toml's `additional_techniques` become first-class across *every*
    subcommand (categories/list/random/show/html), so the browse page and
    category draws include them too, not just the in-chat flows."""
    data = json.loads(file.read_text(encoding="utf-8"))
    rows = []
    for item in data:
        rows.append({
            "category": str(item.get("category", "")).strip(),
            "technique_name": str(item.get("technique_name", "")).strip(),
            "description": str(item.get("description", "")).strip(),
            "detail": str(item.get("detail") or "").strip(),
            "provenance": str(item.get("provenance") or "").strip(),
            "good_for": str(item.get("good_for") or "").strip(),
            "audience": str(item.get("audience") or "").strip(),
        })
    return rows


def categories(rows: list[dict]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    return sorted(counts.items())


def filter_cats(rows: list[dict], cats: list[str] | None) -> list[dict]:
    if not cats:
        return rows
    wanted = {c.lower() for c in cats}
    return [r for r in rows if r["category"].lower() in wanted]


def find(rows: list[dict], names: list[str]) -> tuple[list[dict], list[str]]:
    by_name = {r["technique_name"].lower(): r for r in rows}
    found, missing = [], []
    for n in names:
        r = by_name.get(n.strip().lower())
        (found if r else missing).append(r if r else n)
    return found, missing


def resolve_detail(row: dict, csv_dir: Path) -> str | None:
    """Return the contents of a row's detail file, or None if there is no detail
    (or the file is missing — a missing file is reported to stderr, not fatal)."""
    if not row.get("detail"):
        return None
    path = (csv_dir / row["detail"]).resolve()
    if not path.is_file():
        print(f"# detail file not found for {row['technique_name']}: {row['detail']}", file=sys.stderr)
        return None
    return path.read_text(encoding="utf-8").strip()


def fmt_categories(cats: list[tuple[str, int]], as_json: bool) -> str:
    if as_json:
        return json.dumps([{"category": c, "count": n} for c, n in cats])
    return "\n".join(f"{c}\t{n}" for c, n in cats)


def fmt_list(rows: list[dict], as_json: bool) -> str:
    if as_json:
        return json.dumps([{k: r[k] for k in ("category", "technique_name", "description")} for r in rows])
    return "\n".join(f"{r['category']}\t{r['technique_name']}\t{r['description']}" for r in rows)


def fmt_show(rows: list[dict], csv_dir: Path, as_json: bool) -> str:
    if as_json:
        out = []
        for r in rows:
            d = resolve_detail(r, csv_dir)
            entry = {k: r[k] for k in ("category", "technique_name", "description")}
            if d:
                entry["detail"] = d
            out.append(entry)
        return json.dumps(out)
    blocks = []
    for r in rows:
        block = f"## {r['technique_name']}  [{r['category']}]\n{r['description']}"
        d = resolve_detail(r, csv_dir)
        if d:
            block += f"\n\n{d}"
        blocks.append(block)
    return "\n\n".join(blocks)


def pretty(cat: str) -> str:
    """Turn a category slug (e.g. 'speculative_future') into a display name."""
    return cat.replace("_", " ").replace("-", " ").title()


# --- card visuals: a crafted duotone icon + hue per category, plus a per-technique icon ---
# The hues and SVG glyphs are *data*, not logic: they live in the icon sidecar
# (assets/brain-icons.json) so the catalog's visuals can be edited without touching code.
# It maps category slug -> {hue, glyph} and technique name -> svg (inner markup, drawn in
# `currentColor` which the CSS sets to the category hue; the shared CHIP frame is added by
# the renderer). Anything missing falls back here — an unknown category gets a hash-derived
# hue + generic glyph, an unknown/not-yet-iconed technique a neutral mark — so custom
# catalogs always render.

ICON_FILE = DEFAULT_FILE.parent / "brain-icons.json"

CHIP = '<rect x="1.5" y="1.5" width="41" height="41" rx="12" fill="currentColor" fill-opacity="0.12"/>'

_FALLBACK_GLYPH = (
    '<circle cx="22" cy="22" r="11" fill="currentColor" fill-opacity="0.16"/>'
    '<circle cx="22" cy="22" r="11" stroke="currentColor" stroke-width="1.6" fill="none"/>'
    '<circle cx="22" cy="22" r="3.4" fill="currentColor"/>'
)
_FALLBACK_TECH = (
    '<rect x="15" y="15" width="14" height="14" rx="2.5" transform="rotate(45 22 22)" '
    'fill="none" stroke="currentColor" stroke-width="2"/><circle cx="22" cy="22" r="2.4" fill="currentColor"/>'
)


def _load_icons(file: Path = ICON_FILE) -> tuple[dict, dict]:
    """Read the icon sidecar: (category slug -> {hue, glyph}, technique name -> svg).
    A missing or malformed file is non-fatal — everything then uses the fallbacks below."""
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, {}
    return (data.get("categories") or {}), (data.get("techniques") or {})


_CATEGORY_STYLES, _TECH_ICONS = _load_icons()


def _hsl_hex(deg: int, s: float, lt: float) -> str:
    import colorsys

    r, g, b = colorsys.hls_to_rgb((deg % 360) / 360, lt, s)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def category_style(cat: str) -> tuple[str, str]:
    """(hue, glyph markup) for a category — from the sidecar for the shipped set, derived for extras."""
    style = _CATEGORY_STYLES.get(cat)
    if style and style.get("hue"):
        return style["hue"], style.get("glyph") or _FALLBACK_GLYPH
    deg = int(hashlib.md5(cat.encode("utf-8")).hexdigest(), 16) % 360
    return _hsl_hex(deg, 0.58, 0.52), _FALLBACK_GLYPH


def tech_icon(name: str) -> str:
    """The hand-picked line-icon for a specific technique (neutral mark if unknown)."""
    return _TECH_ICONS.get(name, _FALLBACK_TECH)


SELECTOR_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BMad Method Brainstorming Selection</title>
<script>
/* set the theme before first paint so there's no light-mode flash */
(function(){ try {
  var t = localStorage.getItem('bmad-theme');
  if (!t) { t = (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light'; }
  document.documentElement.setAttribute('data-theme', t);
} catch(e){} })();
</script>
<style>
  :root {
    --bg:#f6f7fb; --surface:#fff; --ink:#1c1e2b; --muted:#6b7080;
    --accent:#5b4bdc; --accent-ink:#5b4bdc; --warn:#c0561f;
    --line:#e6e8f0; --control:#eef0f7; --control2:#f1f2f8; --raised:#fff;
    --cnt:#b9bdce; --foot:#aeb2c4; --shadow:rgba(20,20,50,.06);
  }
  :root[data-theme="dark"] {
    --bg:#0f1117; --surface:#171a23; --ink:#e7e9f2; --muted:#9aa0b4;
    --accent:#6d5cf0; --accent-ink:#a99bff; --warn:#e08a4a;
    --line:#2a2f3e; --control:#222634; --control2:#1d212d; --raised:#2c3242;
    --cnt:#5a6076; --foot:#5a6076; --shadow:rgba(0,0,0,.45);
  }
  /* lift the category hue toward white on dark surfaces so deep hues stay legible */
  :root[data-theme="dark"] section > h2 { color:color-mix(in srgb, var(--c) 62%, #fff); }
  :root[data-theme="dark"] .tech .ico { color:color-mix(in srgb, var(--c) 68%, #fff); }
  :root[data-theme="dark"] label.tech:has(input:checked) { border-color:color-mix(in srgb, var(--c) 60%, #fff); }
  .titlerow { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }
  .themebtn { flex:none; width:36px; height:36px; border-radius:9px; background:var(--control); color:var(--ink); font-size:17px; line-height:1; display:inline-flex; align-items:center; justify-content:center; }
  .themebtn:hover { background:var(--raised); }
  * { box-sizing:border-box; }
  body { margin:0; font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; background:var(--bg); color:var(--ink); }
  header { position:sticky; top:0; z-index:5; background:var(--surface); padding:20px 0 12px; border-bottom:1px solid var(--line); box-shadow:0 2px 12px var(--shadow); }
  .hwrap { max-width:1120px; margin:0 auto; padding:0 24px; }  /* align header content with the card column on wide screens */
  h1 { margin:0 0 4px; font-size:24px; letter-spacing:-.02em; }
  .sub { margin:0 0 12px; color:var(--muted); font-size:14px; max-width:74ch; }
  button { font:inherit; border:0; border-radius:8px; cursor:pointer; }
  .composer { display:flex; flex-direction:column; gap:9px; margin:6px 0 12px; }
  .grp { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .glabel { font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); min-width:74px; }
  .modes { display:inline-flex; background:var(--control); border-radius:9px; padding:3px; gap:2px; }
  .mode { padding:7px 13px; font-size:14px; font-weight:600; color:var(--muted); background:transparent; }
  .mode.on { background:var(--raised); color:var(--accent-ink); box-shadow:0 1px 3px var(--shadow); }
  .modehint { flex:1 1 240px; min-width:0; font-size:13px; color:var(--muted); font-style:italic; }
  .pill { font-size:13px; color:var(--muted); background:var(--control); padding:6px 12px; border-radius:20px; }
  .pill b { color:var(--accent-ink); }
  .step { display:inline-flex; align-items:center; gap:7px; font-size:13px; color:var(--ink); background:var(--control2); padding:4px 6px 4px 12px; border-radius:20px; }
  .step b { min-width:12px; text-align:center; font-size:14px; color:var(--ink); }
  .step button { width:24px; height:24px; border-radius:50%; background:var(--raised); color:var(--muted); font-size:17px; line-height:22px; text-align:center; box-shadow:0 1px 2px var(--shadow); }
  .step button:hover { color:var(--accent-ink); }
  .total { font-size:12px; color:var(--muted); }
  .total.warn { color:var(--warn); font-weight:600; }
  .bar { display:flex; gap:10px 14px; align-items:center; flex-wrap:wrap; }
  #copy { margin-left:auto; padding:9px 22px; background:var(--accent); color:#fff; font-size:14px; font-weight:700; }
  #copy:hover { filter:brightness(1.07); }
  .chips { flex:1 1 320px; min-width:0; display:flex; gap:7px; flex-wrap:wrap; align-items:center; }
  .chip { font-size:12px; padding:4px 11px; border-radius:16px; border:0; color:#fff; background:var(--cc); font-weight:600; cursor:pointer; }
  .chip:hover { filter:brightness(1.08); }
  .banner { max-height:0; overflow:hidden; transition:max-height .25s ease, padding .22s ease, margin .22s ease; background:linear-gradient(90deg,var(--accent),#8275f2); color:#fff; border-radius:10px; font-weight:700; text-align:center; padding:0 14px; }
  .banner.show { max-height:64px; padding:13px 14px; margin-top:10px; }
  .banner.fail { background:linear-gradient(90deg,var(--warn),#e0894a); }
  main { padding:18px 24px 60px; max-width:1120px; margin:0 auto; }
  section { margin:0 0 26px; }
  section > h2 { font-size:13px; text-transform:uppercase; letter-spacing:.08em; color:var(--c); margin:0 0 10px; border-bottom:1px solid color-mix(in srgb, var(--c) 24%, var(--line)); padding-bottom:6px; }
  section > h2 .cnt { color:color-mix(in srgb, var(--c) 45%, var(--cnt)); margin-left:6px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:10px; }
  label.tech { display:flex; gap:12px; align-items:flex-start; background:color-mix(in srgb, var(--c) 5%, var(--surface)); border:1px solid color-mix(in srgb, var(--c) 18%, var(--line)); border-radius:10px; padding:11px 13px; cursor:pointer; transition:border-color .12s, box-shadow .12s, background .12s; }
  label.tech:hover { border-color:color-mix(in srgb, var(--c) 45%, var(--surface)); }
  label.tech input { margin-top:2px; width:17px; height:17px; accent-color:var(--c); flex:none; }
  label.tech:has(input:checked) { border-color:var(--c); background:color-mix(in srgb, var(--c) 12%, var(--surface)); box-shadow:0 0 0 2px color-mix(in srgb, var(--c) 30%, transparent); }
  .tech .ic2 { display:flex; gap:5px; flex:none; }
  .tech .ico { width:40px; height:40px; flex:none; color:var(--c); }
  .tech .n { font-weight:600; display:block; }
  .tech .d { color:var(--muted); font-size:13.5px; display:block; margin-top:2px; }
  .tech .gf { color:var(--accent-ink); font-size:11px; display:block; margin-top:5px; opacity:.85; }
  .grouphdr { margin:30px 0 12px; font-size:12px; text-transform:uppercase; letter-spacing:.14em; font-weight:700; color:var(--c); opacity:.92; border-bottom:1px solid color-mix(in srgb, var(--c) 22%, var(--line)); padding-bottom:7px; }
  main > .grouphdr:first-child { margin-top:2px; }
  :root[data-theme="dark"] .grouphdr { color:color-mix(in srgb, var(--c) 62%, #fff); }
  .goals { display:flex; gap:7px; flex-wrap:wrap; }
  .goal { font-size:12px; padding:5px 12px; border-radius:16px; background:var(--control); color:var(--muted); font-weight:600; }
  .goal:hover { color:var(--ink); }
  .goal.on { background:var(--accent); color:#fff; }
  label.tech.invent { border-style:dashed; background:transparent; }
  label.tech.invent:hover { border-color:var(--c); }
  label.tech.invent .n { color:var(--c); }
  label.tech.hidden { display:none; }
  footer { text-align:center; color:var(--foot); font-size:12px; padding:24px; }
</style>
</head>
<body>
<header>
  <div class="hwrap">
  <div class="titlerow">
    <h1>BMad Method Brainstorming Selection</h1>
    <button id="theme" class="themebtn" type="button" aria-label="Toggle dark mode" title="Toggle dark mode"></button>
  </div>
  <p class="sub">Compose your session, hit <strong>Copy prompt</strong>, and paste it back into the chat to begin. {{TOTAL}}</p>

  <div class="composer">
    <div class="grp">
      <span class="glabel">Facilitation</span>
      <div class="modes" id="modes">
        <button type="button" class="mode on" data-mode="Facilitator">Facilitator</button>
        <button type="button" class="mode" data-mode="Creative Partner">Creative Partner</button>
        <button type="button" class="mode" data-mode="Ideate for me">Ideate for me</button>
      </div>
      <span class="modehint" id="modehint"></span>
    </div>
    <div class="grp">
      <span class="glabel">Techniques</span>
      <span class="pill">Picked <b id="pickN">0</b></span>
      <span class="step">Random <button type="button" data-step="rand" data-d="-1">&minus;</button><b id="randN">0</b><button type="button" data-step="rand" data-d="1">+</button></span>
      <span class="step">Invent <button type="button" data-step="inv" data-d="-1">&minus;</button><b id="invN">0</b><button type="button" data-step="inv" data-d="1">+</button></span>
      <span class="step">AI picks <button type="button" data-step="ai" data-d="-1">&minus;</button><b id="aiN">0</b><button type="button" data-step="ai" data-d="1">+</button></span>
      <span class="total" id="total">Total 0 &middot; 3&ndash;4 is the sweet spot</span>
      <button id="copy" type="button">Copy prompt</button>
    </div>
  </div>

  {{GOALBAR}}
  <div class="bar">
    <span class="glabel">Jump to</span>
    <div class="chips" id="chips">{{CHIPS}}</div>
  </div>

  <div class="banner" id="banner">&#10003; Copied! Now paste it into the chat to start your session.</div>
  </div>
</header>
<main>
{{BODY}}
</main>
<footer>BMad Method &middot; Brainstorming</footer>
<script>
(function(){
  var $ = function(id){ return document.getElementById(id); };
  var all = Array.prototype.slice;
  var boxes = all.call(document.querySelectorAll('input[type=checkbox]'));
  var techBoxes = boxes.filter(function(b){ return b.dataset.name; });      // real technique cards
  var inventBoxes = boxes.filter(function(b){ return b.dataset.invent; });  // per-category "invent in the spirit of" cards
  var header = document.querySelector('header');
  var sections = all.call(document.querySelectorAll('section'));
  var state = { mode: 'Facilitator', rand: 0, inv: 0, ai: 0 };
  var MODE_HINTS = {
    'Facilitator': 'A forcing function for your ideas — I prompt and push, but never supply them.',
    'Creative Partner': 'We riff together — I facilitate and add ideas too, each logged as yours or mine.',
    'Ideate for me': 'I run the whole session myself, then show you the result and offer to keep going.'
  };
  function setHint(){ $('modehint').textContent = MODE_HINTS[state.mode] || ''; }

  var themeBtn = $('theme');
  function setThemeIcon(){ themeBtn.textContent = document.documentElement.getAttribute('data-theme') === 'dark' ? '☀' : '☾'; }
  themeBtn.addEventListener('click', function(){
    var next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('bmad-theme', next); } catch(e){}
    setThemeIcon();
  });

  all.call(document.querySelectorAll('.mode')).forEach(function(b){
    b.addEventListener('click', function(){
      all.call(document.querySelectorAll('.mode')).forEach(function(m){ m.classList.remove('on'); });
      b.classList.add('on');
      state.mode = b.dataset.mode;
      setHint();
    });
  });

  all.call(document.querySelectorAll('[data-step]')).forEach(function(btn){
    btn.addEventListener('click', function(){
      var k = btn.dataset.step, d = parseInt(btn.dataset.d, 10);
      state[k] = Math.max(0, state[k] + d);
      update();
    });
  });

  // Category chips are jump-nav: click one to smooth-scroll its section into view,
  // offsetting by the sticky header's height so the heading isn't hidden beneath it.
  all.call(document.querySelectorAll('.chip')).forEach(function(chip){
    chip.addEventListener('click', function(){
      var sec = null;
      for (var i = 0; i < sections.length; i++){ if (sections[i].dataset.cat === chip.dataset.cat){ sec = sections[i]; break; } }
      if (!sec){ return; }
      var top = sec.getBoundingClientRect().top + window.pageYOffset - header.offsetHeight - 8;
      window.scrollTo({ top: top, behavior: 'smooth' });
    });
  });

  boxes.forEach(function(b){ b.addEventListener('change', update); });

  // A `classic` technique appears twice (lead "Proven & Professional" group + its home
  // category), so de-dupe checked picks by name; the lead copy carries data-lead.
  function checkedTech(){
    var seen = {}, out = [];
    techBoxes.forEach(function(b){
      if (!b.checked || seen[b.dataset.name]) { return; }
      seen[b.dataset.name] = 1;
      out.push(b);
    });
    return out;
  }
  function checkedInvent(){ return inventBoxes.filter(function(b){ return b.checked; }); }

  function update(){
    $('pickN').textContent = checkedTech().length;
    $('randN').textContent = state.rand;
    $('invN').textContent = state.inv;
    $('aiN').textContent = state.ai;
    var total = checkedTech().length + state.rand + state.inv + checkedInvent().length + state.ai;
    var t = $('total');
    t.textContent = 'Total ' + total + ' · 3–4 is the sweet spot';
    t.classList.toggle('warn', total > 5);
  }

  // "Great for" goal filter: clicking a goal narrows visible cards to those tagged with it.
  var goalBtns = all.call(document.querySelectorAll('.goal'));
  function activeGoals(){ return goalBtns.filter(function(b){ return b.classList.contains('on'); }).map(function(b){ return b.dataset.goal; }); }
  function applyFilter(){
    var act = activeGoals();
    all.call(document.querySelectorAll('label.tech')).forEach(function(lab){
      var inp = lab.querySelector('input');
      if (inp.dataset.invent){ return; }  // invent cards aren't goal-tagged — always visible
      var good = (inp.dataset.good || '').split('|');
      var show = !act.length || act.some(function(g){ return good.indexOf(g) >= 0; });
      lab.classList.toggle('hidden', !show);
    });
  }
  goalBtns.forEach(function(b){ b.addEventListener('click', function(){ b.classList.toggle('on'); applyFilter(); }); });

  function randomPool(){
    var picked = {};
    checkedTech().forEach(function(b){ picked[b.dataset.name] = 1; });
    // draw from unchecked, non-lead copies, skipping anything already picked
    return techBoxes.filter(function(b){ return !b.checked && !b.dataset.lead && !picked[b.dataset.name]; });
  }

  function sample(arr, n){
    var a = arr.slice(), out = [];
    while (out.length < n && a.length){ out.push(a.splice(Math.floor(Math.random() * a.length), 1)[0]); }
    return out;
  }

  function compose(){
    var picks = checkedTech().map(function(b){ return { n: b.dataset.name, c: b.dataset.cat, d: b.dataset.desc, r: false }; });
    var rnd = sample(randomPool(), state.rand).map(function(b){ return { n: b.dataset.name, c: b.dataset.cat, d: b.dataset.desc, r: true }; });
    var techs = picks.concat(rnd);
    var L = ["Let's run my brainstorming session.", "", 'Facilitation mode: ' + state.mode + '.'];
    if (techs.length){
      L.push("", 'Techniques to use:');
      techs.forEach(function(t, i){
        L.push((i + 1) + '.' + (t.r ? ' (random pick)' : '') + ' ' + t.n + '  ·  ' + t.c);
        L.push('   ' + t.d);
      });
    }
    var extra = [];
    if (state.inv > 0){ extra.push('invent ' + state.inv + ' brand-new technique' + (state.inv > 1 ? 's' : '') + ' on the fly'); }
    checkedInvent().forEach(function(b){ extra.push('invent 1 new technique in the spirit of ' + b.dataset.invent); });
    if (state.ai > 0){ extra.push('you choose ' + state.ai + ' more technique' + (state.ai > 1 ? 's' : '') + ' that fit my goal'); }
    if (extra.length){ L.push("", 'Then: ' + extra.join('; and ') + '.'); }
    if (!techs.length && !extra.length){
      L.push("", state.mode === 'Ideate for me'
        ? 'Run the whole session yourself — pick the techniques, generate the ideas, then show me the result.'
        : 'Help me choose 3–4 techniques to start.');
    }
    return L.join('\n');
  }

  function fallbackCopy(t){
    var ta = document.createElement('textarea');
    ta.value = t; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.focus(); ta.select();
    var ok = false;
    try { ok = document.execCommand('copy'); } catch(e){ ok = false; }
    document.body.removeChild(ta);
    return ok;
  }

  function flash(ok, text){
    var b = $('banner');
    b.classList.toggle('fail', !ok);
    b.innerHTML = ok
      ? '✓ Copied! Now paste it into the chat to start your session.'
      : '⚠ Couldn’t reach the clipboard — copy the text in the box, then paste it into the chat.';
    b.classList.add('show');
    setTimeout(function(){ b.classList.remove('show'); }, 4500);
    // Last resort on a hard failure: a prefilled, selectable prompt so the text is never lost.
    if (!ok){ window.prompt('Copy this, then paste it into the chat:', text); }
  }

  $('copy').addEventListener('click', function(){
    var text = compose();
    if (navigator.clipboard && navigator.clipboard.writeText){
      navigator.clipboard.writeText(text).then(
        function(){ flash(true, text); },
        function(){ flash(fallbackCopy(text), text); }
      );
    } else { flash(fallbackCopy(text), text); }
  });

  setHint();
  setThemeIcon();
  update();
})();
</script>
</body>
</html>
"""


# --- browse-page layout: a "Proven & Professional" lead group, then super-groups ----------
CLASSIC_GROUP = "Proven & Professional"
LEAD_HUE = "#3d4f73"  # a dignified slate for the professional lead group

# Super-group order for the shipped categories. Categories not listed (e.g. user-added
# via --extra) render last under "More", alphabetically — so custom catalogs always show.
CATEGORY_GROUPS = (
    ("Structured & Analytical", ("structured", "deep")),
    ("Creative & Generative", ("creative", "biomimetic", "cultural", "speculative_future", "quantum")),
    ("Wild & Playful", ("wild", "absurdist", "theatrical", "constraint")),
    ("Introspective & Personal", ("introspective_delight", "collaborative")),
)

# Human labels for the `good_for` goal tags; this dict's order is the filter-bar order.
GOAL_LABELS = {
    "feature": "Build a feature",
    "novel": "Novel concept",
    "strategy": "Strategy",
    "planning": "Planning",
    "diagnosis": "Diagnose",
    "personal": "Personal / life",
    "unstuck": "Get unstuck",
}


def _good_for_label(good: str) -> str:
    parts = [GOAL_LABELS.get(g, g) for g in good.split("|") if g]
    return ("Great for: " + " · ".join(parts)) if parts else ""


def _svg(inner: str) -> str:
    return f'<svg class="ico" viewBox="0 0 44 44" xmlns="http://www.w3.org/2000/svg">{CHIP}{inner}</svg>'


def _card(r: dict, lead: bool = False) -> str:
    """One technique card. `lead=True` cards live in the cross-cutting professional group;
    they carry their own category hue (inline --c) and data-lead so selection can de-dupe."""
    name = html.escape(r["technique_name"])
    desc = html.escape(r["description"])
    hue, glyph = category_style(r["category"])
    disp_cat = html.escape(pretty(r["category"]))
    good = html.escape(r.get("good_for", ""))
    prov = html.escape(r.get("provenance", ""))
    style = f' style="--c:{hue}"' if lead else ""
    lead_attr = ' data-lead="1"' if lead else ""
    gf = _good_for_label(r.get("good_for", ""))
    gf_html = f'<span class="gf">{html.escape(gf)}</span>' if gf else ""
    return (
        f'<label class="tech"{style}><input type="checkbox" '
        f'data-name="{name}" data-cat="{disp_cat}" data-desc="{desc}" data-good="{good}" data-prov="{prov}"{lead_attr}>'
        f'<span class="ic2">{_svg(glyph)}{_svg(tech_icon(r["technique_name"]))}</span>'
        f'<span><span class="n">{name}</span><span class="d">{desc}</span>{gf_html}</span></label>'
    )


def _invent_card(disp_cat: str, glyph: str) -> str:
    """A dashed 'invent on the fly, in this category's spirit' card appended to each section."""
    return (
        f'<label class="tech invent"><input type="checkbox" data-invent="{disp_cat}">'
        f'<span class="ic2">{_svg(glyph)}</span>'
        f'<span><span class="n">✨ Invent a {disp_cat} technique</span>'
        f'<span class="d">Make up a brand-new technique on the fly, in the spirit of {disp_cat}</span></span></label>'
    )


def html_doc(rows: list[dict]) -> str:
    """Render the self-contained 'browse all techniques' selection page from the catalog.

    Deterministic ordering so the shipped asset can be snapshot-tested against the CSV:
    a cross-cutting "Proven & Professional" lead group (every `classic`-tagged row), then
    the categories in fixed super-group order, then any unlisted/custom categories under
    "More" alphabetically. Techniques render in file order within a category. A `classic`
    row appears both in the lead group and its home category; the page de-dupes on select.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["category"], []).append(r)

    body: list[str] = []
    chips: list[str] = []

    def add_section(cat: str) -> None:
        hue, glyph = category_style(cat)
        disp = html.escape(pretty(cat))
        cards = [_card(r) for r in groups[cat]]
        cards.append(_invent_card(disp, glyph))
        chips.append(f'<button type="button" class="chip" data-cat="{disp}" style="--cc:{hue}">{disp}</button>')
        body.append(
            f'<section data-cat="{disp}" style="--c:{hue}"><h2>{disp}<span class="cnt">{len(groups[cat])}</span></h2>'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )

    # 1) lead group — every classic-tagged technique, cross-category (no invent card here)
    classics = [r for r in rows if r.get("provenance", "").lower() == "classic"]
    if classics:
        disp = html.escape(CLASSIC_GROUP)
        lead_cards = "".join(_card(r, lead=True) for r in classics)
        chips.append(f'<button type="button" class="chip" data-cat="{disp}" style="--cc:{LEAD_HUE}">{disp}</button>')
        body.append(
            f'<section data-cat="{disp}" style="--c:{LEAD_HUE}"><h2>{disp}<span class="cnt">{len(classics)}</span></h2>'
            f'<div class="grid">{lead_cards}</div></section>'
        )

    # 2) shipped categories, in super-group order
    placed = set()
    for group_title, cats in CATEGORY_GROUPS:
        present = [c for c in cats if c in groups]
        if not present:
            continue
        hue, _ = category_style(present[0])
        body.append(f'<h2 class="grouphdr" style="--c:{hue}">{html.escape(group_title)}</h2>')
        for c in present:
            add_section(c)
            placed.add(c)

    # 3) leftover (custom / --extra) categories, alphabetically
    leftover = sorted(c for c in groups if c not in placed)
    if leftover:
        body.append('<h2 class="grouphdr" style="--c:#8a8f9e">More</h2>')
        for c in leftover:
            add_section(c)

    # goal-affinity filter bar — only if the catalog actually carries good_for tags
    present_goals: set[str] = set()
    for r in rows:
        for g in (r.get("good_for", "") or "").split("|"):
            if g:
                present_goals.add(g)
    goalbar = ""
    if present_goals:
        ordered = [g for g in GOAL_LABELS if g in present_goals] + sorted(present_goals - set(GOAL_LABELS))
        gchips = "".join(
            f'<button type="button" class="goal" data-goal="{html.escape(g)}">{html.escape(GOAL_LABELS.get(g, g))}</button>'
            for g in ordered
        )
        goalbar = f'<div class="bar"><span class="glabel">Great for</span><div class="goals" id="goals">{gchips}</div></div>'

    total = html.escape(f"{len(rows)} techniques across {len(groups)} categories.")
    return (
        SELECTOR_TEMPLATE.replace("{{BODY}}", "\n".join(body))
        .replace("{{CHIPS}}", "".join(chips))
        .replace("{{GOALBAR}}", goalbar)
        .replace("{{TOTAL}}", total)
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, default=DEFAULT_FILE, help="technique CSV (default: sibling assets/brain-methods.csv)")
    p.add_argument("--extra", type=Path, help="JSON overlay of additional techniques (customize.toml additional_techniques), merged into every command")
    p.add_argument("--json", action="store_true", help="emit structured JSON instead of lean text")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("categories", help="list category names + counts")
    pl = sub.add_parser("list", help="the index: category/name/gist (needs --category or --all)")
    pl.add_argument("--category", action="append", help="filter to a category (repeatable)")
    pl.add_argument("--all", action="store_true", help="dump the entire catalog (deliberate; large)")
    ps = sub.add_parser("show", help="full gist + detail file for named techniques")
    ps.add_argument("names", nargs="+")
    pr = sub.add_parser("random", help="pick techniques at random")
    pr.add_argument("--category", action="append", help="restrict to a category (repeatable)")
    pr.add_argument("-n", type=int, default=1, help="how many (default 1)")
    ph = sub.add_parser("html", help="write the offline 'browse all' selection page")
    ph.add_argument("--out", help="file to write the page to (required; never prints the catalog)")
    args = p.parse_args(argv)

    if not args.file.is_file():
        print(f"error: technique file not found: {args.file}", file=sys.stderr)
        return 2
    rows = load(args.file)
    if args.extra:
        if not args.extra.is_file():
            print(f"error: --extra file not found: {args.extra}", file=sys.stderr)
            return 2
        rows += load_extra(args.extra)
    csv_dir = args.file.resolve().parent

    if args.cmd == "categories":
        print(fmt_categories(categories(rows), args.json))
    elif args.cmd == "list":
        if not args.category and not args.all:
            print(
                "error: `list` needs --category (one or more) — or --all to dump the whole "
                "catalog on purpose. Use `categories` for the cheap map, or `random` to draw blind.",
                file=sys.stderr,
            )
            return 2
        print(fmt_list(filter_cats(rows, args.category), args.json))
    elif args.cmd == "show":
        found, missing = find(rows, args.names)
        for m in missing:
            print(f"# not found: {m}", file=sys.stderr)
        if not found:
            return 1
        print(fmt_show(found, csv_dir, args.json))
    elif args.cmd == "random":
        pool = filter_cats(rows, args.category)
        if not pool:
            print("# no techniques match", file=sys.stderr)
            return 1
        n = max(0, min(args.n, len(pool)))  # clamp: never crash on a negative or oversized -n
        print(fmt_list(random.sample(pool, n), args.json))
    elif args.cmd == "html":
        if not args.out:
            print(
                "error: `html` needs --out PATH — it writes the selection page to a file and "
                "never prints the catalog to stdout (which would defeat the point).",
                file=sys.stderr,
            )
            return 2
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html_doc(rows), encoding="utf-8")
        print(f"wrote {out} ({len(rows)} techniques, {len(categories(rows))} categories)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
