#!/usr/bin/env python3
"""kingmaker: rank the Hermes Index by measured $/task and keep the main model on
the best in-budget pick. One-shot, deterministic, no LLM scoring at runtime.

Fetches the Hermes Index leaderboard (portal.nousresearch.com/bench), parses the
server-rendered leaderboard table, appends to ledger, and then follows the saved
setup: either apply the best affordable model or merely report the change. The
switch itself is the supported `hermes config set model.default` operation
(provider never touched) after a bounded chat/tool/structured-output smoke.

Two-phase usage:
  setup    --sample prints a live budget sample (what today's money buys);
           --setup --cap <n|none> --mode <apply|notify> [--cadence <cron>]
           [--deliver <route>] saves the wizard answers to answers.json.
  steady   a no-agent cron job runs this script; the saved config supplies cap
           and mode so the job itself stays dumb.

Modes (explicit flag > saved config > collector-only):
  --apply-main   switch the main model when the crown changes (smoke + verify)
  --notify       report the change, never write config
  (neither)      use the saved mode; unconfigured runs collect and say so

Selection policy: among nous-catalog models whose Hermes-measured average cost
per task is STRICTLY UNDER the cap (--cap, default $0.30/task), pick the highest
Hermes Index score. NOT the score/$ sort leader. Usable = exact provider model id
resolved from the current nous catalog via the curated benchmark-name mapping
(NOUS_MODEL_IDS); unknown mappings are never guessed. Provisional (*)
leaderboard entries never qualify. Missing/non-finite values cannot qualify a
candidate.

Exit codes:
  0  ok (applied, or unchanged/silent, or collector tick)
  2  page/parse sanity failure (explicit fail; previous snapshot preserved)
  3  no eligible model (explicit fail; last-good main model preserved)
  4  compatibility smoke failed for the candidate (transient: deferred, no
     cooldown; incompatible: candidate cooled down)
  5  apply/readback/verify failed (rolled back safely; candidate cooled down)
  6  concurrent manual config change detected (aborted before any write)

Decided outcomes (2-6) print their notice and exit 0, because the cron job
delivers stdout and treats a non-zero exit as a broken job. Only an undecided
failure (1) rides the failure channel. Empty output means nothing to report.

Usage: python3 kingmaker-job.py [--cap 0.30|none] [--out snapshot.json] [--quiet]
                                [--force] [--apply-main|--notify] [--sample]
                                [--setup --cron-job <id>]
This install's answers (budget, mode, cadence) live in answers.json beside
SKILL.md; the constants below are the shipped defaults. The only Hermes setting
this script writes is the main model, through `hermes config set model.default`.
Runtime state (latest.json, apply_state.json, apply_receipt.json, ledger.jsonl)
goes to $HERMES_HOME/data/kingmaker/, never inside the
skill.

HERMES_HOME is the only environment variable read (the standard profile root);
everything else is discovered at run time — the interpreter that can import
Hermes, and the `hermes` CLI on PATH.
"""
import argparse
import datetime
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser

logger = logging.getLogger("kingmaker")

HERMES_HOME = os.environ.get("HERMES_HOME") or os.path.join(
    os.path.expanduser("~"), ".hermes")
# Runtime state belongs to the PROFILE, never inside the skill: a hub-installed
# skill directory is replaced on update, so anything written beside this script
# would be silently lost.
DATA_DIR = os.path.join(HERMES_HOME, "data", "kingmaker")
LEDGER = os.path.join(DATA_DIR, "ledger.jsonl")
STATE_FILE = os.path.join(DATA_DIR, "apply_state.json")
RECEIPT_FILE = os.path.join(DATA_DIR, "apply_receipt.json")
KEEP_RUNS = 7
UA = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"

HERMES_CLI = shutil.which("hermes") or "hermes"
_HERMES_PY = None

PROBE_TIMEOUT_S = 90
FETCH_TIMEOUT_S = 50
SMOKE_TIMEOUT_S = 240
CONFIG_TIMEOUT_S = 120
FAILED_CANDIDATE_COOLDOWN_S = 86400  # one daily tick: no re-attempt, no flip-flop
COOLDOWN_HOURS = 24

# Selection policy: the sole decision metrics are the Hermes Index score and the
# Hermes-measured average cost per task. A candidate qualifies only when its avg
# cost per task is STRICTLY UNDER the cap; the highest Hermes Index score among
# qualified nous models is the crown.
CAP_PER_TASK_USD = 0.30      # shipped default; answers.json overrides (None = no ceiling)
BENCH_URL = "https://portal.nousresearch.com/bench"
BENCH_SOURCE = ("portal.nousresearch.com/bench (Hermes Index, pass@1, "
                "Hermes Agent harness)")

SMOKE_ATTEMPTS = 3
SMOKE_RETRY_DELAY_S = 15

# The answers are the user's file, kept inside the skill so they travel with it
# and are deleted with it. The constants above are the shipped defaults, in force
# only until the wizard writes answers.json.
SKILL_DIR = os.path.join(HERMES_HOME, "skills", "kingmaker")
ANSWERS_FILE = os.path.join(SKILL_DIR, "answers.json")
MODES = ("apply", "notify")
# Outcomes the collector reached on purpose: they carry a notice for the user and
# must not be reported as a broken job by the scheduler.
DECIDED_CODES = (2, 3, 4, 5, 6)
DEFAULT_MODE = "notify"        # shipped default; answers.json overrides (safe default)
DEFAULT_CADENCE = "0 7 * * *"  # shipped default; informational - the job's schedule rules
# price points used to render the wizard's live budget lines; only those inside
# the observed price range are shown, so the list stays relevant as prices move
BUDGET_LINE_STEPS = (0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00, 10.00)

# sanity anchors: name -> expected Hermes Index within tolerance; drift => page
# shape changed or a legitimate rerun landed. Verify on the leaderboard first.
ANCHORS = {"Claude Opus 5.5": 63.31, "DeepSeek V4.1 Flash": 36.91,
           "Ling 3.0 Flash": 21.56}
ANCHOR_TOLERANCE = 5.0

# Hermes Index display name -> EXACT nous provider model id. Curated from
# identity evidence (same family AND version on both sides), validated against
# the current catalog at run time (exact string membership). Names with no
# unambiguous nous counterpart are deliberately absent and never guessed:
#   "Qwen 3.8 Max"  catalog only carries qwen3.8-max-0902 / qwen3.8-max-prime
#   "Hy4"           catalog only carries tencent/hy4-preview
NOUS_MODEL_IDS = {
    "Claude Opus 5.5": "anthropic/claude-opus-5.5",
    "Claude Sonnet 5.5": "anthropic/claude-sonnet-5.5",
    "GPT 6 Astra": "openai/gpt-6-astra",
    "GPT 6 Sol": "openai/gpt-6-sol",
    "GPT 6 Luna": "openai/gpt-6-luna",
    "Grok 4.7": "x-ai/grok-4.7",
    "GLM 5.3": "z-ai/glm-5.3",
    "GLM 5.3 Flash": "z-ai/glm-5.3-flash",
    "Kimi K3": "moonshotai/kimi-k3",
    "Gemini Flash 3.8": "google/gemini-3.8-flash",
    "DeepSeek V4.1 Flash": "deepseek/deepseek-v4.1-flash",
    "Ling 3.0 Flash": "inclusionai/ling-3.0-flash",
}


def _now():
    return datetime.datetime.now(datetime.timezone.utc).timestamp()


def _iso(ts):
    return datetime.datetime.fromtimestamp(
        ts, datetime.timezone.utc).isoformat(timespec="seconds")


def _configure_logging():
    """Diagnostics go to stderr: a cron tick's stdout stays silent unless there is
    something the user must be told, while the run log keeps the traceback."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO,
                            format="kingmaker: %(levelname)s %(message)s")


def fetch():
    r = subprocess.run(["curl", "-s", "-m", "40", "-A", UA, BENCH_URL],
                       capture_output=True, text=True, timeout=FETCH_TIMEOUT_S)
    if r.returncode != 0 or len(r.stdout) < 20_000:
        raise SystemExit("fetch failed: page too small or curl error")
    return r.stdout


SEP = "\x1f"

# Leaderboard row shapes seen in the wild; both stay supported so a site revert
# or a stale capture cannot blind the collector.
ROW_SHAPES = {
    7: {"index": 2, "cost": 2, "suites": (3, 4, 5, 6)},
    8: {"index": 2, "cost": 3, "suites": (4, 5, 6, 7)},
}


class _TableRows(HTMLParser):
    """Leaf-text chunks per cell of every table row on the page."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.in_table = 0
        self.cur_row = None
        self.cur_cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table += 1
        elif self.in_table and tag == "tr":
            self.cur_row = []
        elif self.in_table and tag in ("td", "th") and self.cur_row is not None:
            self.cur_cell = []
        elif self.cur_cell is not None:
            self.cur_cell.append(SEP)

    def handle_endtag(self, tag):
        if tag == "table":
            self.in_table = max(0, self.in_table - 1)
        elif tag in ("td", "th") and self.cur_cell is not None:
            self.cur_row.append(self.cur_cell)
            self.cur_cell = None
        elif tag == "tr" and self.cur_row is not None:
            if self.cur_row:
                self.rows.append(self.cur_row)
            self.cur_row = None
        elif self.cur_cell is not None:
            self.cur_cell.append(SEP)

    def handle_data(self, data):
        if self.cur_cell is not None:
            self.cur_cell.append(data)


def _cell_chunks(cell):
    return [" ".join(p.split()) for p in "".join(cell).split(SEP) if p.strip()]


def _cell_metric(cell):
    """(score, cost, raw, provisional) from one leaderboard cell's own text.
    Every value is read from its own cell; nothing is borrowed from neighbors."""
    text = " ".join(" ".join(c.split()) for c in _cell_chunks(cell))
    text = " ".join(text.split())
    provisional = "*" in text
    cost = raw = None
    m = re.search(r"\$\s*([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        cost = float(m.group(1))
        text = text[:m.start()] + " " + text[m.end():]
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*raw", text)
    if m:
        raw = float(m.group(1))
        text = text[:m.start()] + " " + text[m.end():]
    text = re.sub(r"[^0-9.\s]", " ", text)
    m = re.search(r"[0-9]+(?:\.[0-9]+)?", text)
    score = float(m.group(0)) if m else None
    return score, cost, raw, provisional


def parse_bench(page):
    """Leaderboard records from the server-rendered table. Rank-numbered rows of
    a known shape only; other tables on the page (head-to-head) are ignored.
    A truncated row is dropped whole: a missing cell is never invented."""
    p = _TableRows()
    p.feed(page)
    recs = []
    for row in p.rows:
        shape = ROW_SHAPES.get(len(row))
        if not shape:
            continue
        rank = _cell_chunks(row[0])
        if not rank or not re.fullmatch(r"\d+", rank[0]):
            continue
        name_chunks = _cell_chunks(row[1])
        if not name_chunks:
            continue
        idx, combined_cost, _raw, idx_prov = _cell_metric(row[shape["index"]])
        if shape["cost"] == shape["index"]:
            cost, provisional = combined_cost, idx_prov
        else:
            _i, cost, _r, cost_prov = _cell_metric(row[shape["cost"]])
            provisional = idx_prov or cost_prov
        hb, tb4, tbs, sk = (_cell_metric(row[i]) for i in shape["suites"])
        recs.append({
            "rank": rank[0], "name": name_chunks[0],
            "vendor": name_chunks[1] if len(name_chunks) > 1 else "",
            "index": idx, "avg_cost_per_task": cost, "provisional": provisional,
            "suites": {
                "hermes_bench": {"score": hb[0], "cost": hb[1], "raw": hb[2]},
                "terminalbench_4": {"score": tb4[0], "cost": tb4[1], "raw": tb4[2]},
                "terminalbench_science": {"score": tbs[0], "cost": tbs[1], "raw": tbs[2]},
                "skillsbench": {"score": sk[0], "cost": sk[1], "raw": sk[2]}}})
    return recs


def finite_number(value):
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def sanity(recs):
    problems = []
    if len(recs) < 8:
        problems.append(f"only {len(recs)} leaderboard rows (expected >=8); "
                        "page shape likely changed")
    for name, expected in ANCHORS.items():
        hit = next((v for v in recs if v.get("name") == name), None)
        if hit is None:
            problems.append(f"anchor row {name!r} missing; page shape likely changed")
        elif (finite_number(hit.get("index")) is None
              or abs(hit["index"] - expected) > ANCHOR_TOLERANCE):
            problems.append(f"anchor drift: {name!r} index {hit.get('index')!r} "
                            f"vs expected {expected}")
    return problems


def build_rows(recs, cap=CAP_PER_TASK_USD):
    """Rows from parsed leaderboard records. Sole decision metrics: the Hermes
    Index score and the Hermes-measured average cost per task. Missing or
    non-finite values cannot qualify a candidate. ``cap=None`` means no budget
    ceiling (every priced row is in-cap)."""
    rows = []
    for v in recs:
        idx = finite_number(v.get("index"))
        cost = finite_number(v.get("avg_cost_per_task"))
        if idx is None or cost is None:
            continue
        rows.append({"name": v.get("name") or "", "vendor": v.get("vendor") or "",
                     "index": idx, "avg_cost_per_task": cost,
                     "provisional": bool(v.get("provisional")),
                     "suites": v.get("suites") or {},
                     "in_cap": (cap is None) or cost < cap,
                     "provider_model_id": None})
    rows.sort(key=lambda r: -r["index"])
    return rows


def attach_provider_identity(rows, catalog):
    """Fill provider_model_id (exact catalog id) per row from the curated
    benchmark-name mapping. Unknown mappings stay null, never guessed."""
    ids = set(catalog.get("ids") or [])
    for r in rows:
        mid = NOUS_MODEL_IDS.get(r.get("name") or "")
        if not mid or mid not in ids:
            r["provider_model_id"] = None
            r["eligibility_issue"] = ("unmapped benchmark identity" if not mid
                                      else "not in provider catalog")
            continue
        r["provider_model_id"] = mid
    return rows


def pick_crown(rows, usable_ids=None, exclude=None):
    """Selection policy: the smartest model under the cost cap (max Hermes Index
    among in-cap, non-provisional rows), restricted to ACTUAL usable provider
    models. That is THE crown. It is NOT the score/$ sort leader (that is just a
    diagnostic). Zero-cost rows never qualify."""
    ban = set(exclude or ())
    pool = [r for r in rows if r.get("in_cap") and not r.get("provisional")
            and finite_number(r.get("index")) is not None
            and (r.get("avg_cost_per_task") or 0) > 0
            and (usable_ids is None or r.get("provider_model_id") in usable_ids)
            and (r.get("provider_model_id") or "") not in ban]
    if not pool:
        return None
    return max(pool, key=lambda r: (r["index"], -(r.get("avg_cost_per_task") or 0),
                                    r.get("provider_model_id") or ""))


def trim(path, keep):
    if not os.path.exists(path):
        return
    with open(path) as handle:
        lines = handle.read().splitlines()
    if len(lines) > keep:
        with open(path, 'w') as handle:
            handle.write('\n'.join(lines[-keep:]) + '\n')


def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ------------------------------------------------- this install's answers + job
def answers_path():
    """Where this install's answers live: inside the skill, beside SKILL.md.
    Running the copy that ships in the skill resolves to the same file."""
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(here) == "references":
        return os.path.abspath(os.path.join(here, os.pardir, "answers.json"))
    return ANSWERS_FILE


def load_answers():
    """The answers the wizard wrote ({} before setup, or once the skill is gone)."""
    try:
        with open(answers_path(), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_answers(updates):
    """Merge answers into the skill's answers.json — the only place they are kept."""
    path = answers_path()
    answers = load_answers()
    answers.update(updates)
    answers["updated_at"] = _iso(_now())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    atomic_write_json(path, answers)
    return answers


def load_config():
    """The answers in force: the skill's answers.json over the shipped defaults.
    Nothing is read from Hermes config."""
    cfg = {"cap_per_task_usd": CAP_PER_TASK_USD,
           "mode": DEFAULT_MODE,
           "cadence": DEFAULT_CADENCE}
    cfg.update(load_answers())
    return cfg


def parse_cap(text):
    """'none'/'off'/'unlimited' -> None (no ceiling); otherwise a positive float."""
    t = str(text).strip().lower()
    if t in ("none", "off", "unlimited", "no-limit", "nolimit", "inf"):
        return None
    try:
        value = float(t)
    except ValueError:
        raise SystemExit("invalid --cap %r: use a number of USD, or 'none'" % text)
    if value <= 0:
        raise SystemExit("invalid --cap %r: must be greater than zero" % text)
    return value


def cap_label(cap):
    return "no cap" if cap is None else "$%s/task" % ("%g" % cap)


def usable_rows(rows):
    """Rows a user could actually run: mapped to a real provider model, priced,
    and not provisional."""
    return [r for r in rows
            if r.get("provider_model_id") and not r.get("provisional")
            and (r.get("avg_cost_per_task") or 0) > 0]


def best_under(usable, cap):
    pool = [r for r in usable if cap is None or r["avg_cost_per_task"] <= cap]
    if not pool:
        return None
    return max(pool, key=lambda r: (r["index"], -r["avg_cost_per_task"]))


def render_sample(snap, rows):
    """Live budget sample for the setup wizard. Every name and price here comes
    from the page fetched during THIS run — nothing is hardcoded, so the sample
    stays honest as models and prices churn."""
    out = ["kingmaker live sample %s (%s)" % (snap["date"], BENCH_SOURCE), "",
           "%-3s %-26s %6s %9s  %s" % ("#", "model", "index", "$/task", "prov")]
    for i, r in enumerate(rows, 1):
        out.append("%-3d %-26s %6.2f %9.4f  %s"
                   % (i, r["name"][:26], r["index"], r["avg_cost_per_task"],
                      "P" if r["provisional"] else ""))
    usable = usable_rows(rows)
    out.append("")
    out.append("Budget lines — best usable nous model at or under each price point:")
    if usable:
        lo = min(r["avg_cost_per_task"] for r in usable)
        hi = max(r["avg_cost_per_task"] for r in usable)
        spans = []                       # [first_step, pick, last_step]
        for step in BUDGET_LINE_STEPS:
            if step < lo or step > hi:
                continue
            pick = best_under(usable, step)
            if spans and spans[-1][1] is pick:
                spans[-1][2] = step      # same answer: extend the span
            else:
                spans.append([step, pick, step])
        for start, pick, end in spans:
            label = ("<= $%s" % ("%g" % start) if start == end
                     else "<= $%s - $%s" % ("%g" % start, "%g" % end))
            out.append("  %-18s %-22s index %5.2f   $%.4f/task"
                       % (label, pick["name"][:22], pick["index"],
                          pick["avg_cost_per_task"]))
        top = best_under(usable, None)
        out.append("  %-18s %-22s index %5.2f   $%.4f/task"
                   % ("no limit", top["name"][:22], top["index"],
                      top["avg_cost_per_task"]))
    else:
        out.append("  (no usable nous model on this leaderboard right now)")
    excluded = [r["name"] for r in rows if r.get("provisional")]
    unmapped = [r["name"] for r in rows
                if not r.get("provisional") and not r.get("provider_model_id")]
    out.append("")
    out.append("Excluded from selection: provisional rows never qualify"
               + (" (%s)" % ", ".join(excluded) if excluded else "")
               + "; rows with no exact provider id cannot be selected"
               + (" (%s)" % ", ".join(unmapped) if unmapped else "") + ".")
    return "\n".join(out)


def build_snapshot(rows, crown, cap):
    ts = _now()
    unresolved = []
    for r in rows:
        if not r.get("in_cap") or (crown and r["index"] <= crown["index"]):
            continue
        if r.get("provisional"):
            unresolved.append({"name": r["name"],
                               "reason": "provisional benchmark data"})
        elif (not r.get("provider_model_id")
              and r.get("eligibility_issue") != "not in provider catalog"):
            # identity we cannot verify blocks; a mapped id the provider does
            # not serve is simply unroutable and falls through
            unresolved.append({"name": r["name"],
                               "reason": "unmapped benchmark identity"})
    return {"date": _iso(ts)[:10], "timestamp": _iso(ts), "cap_per_task_usd": cap,
            "source": BENCH_SOURCE, "rows": rows,
            "unresolved_candidates": unresolved,
            "crown": ({"model_id": crown["provider_model_id"],
                       "bench_name": crown["name"],
                       "index": crown["index"],
                       "avg_cost_per_task": crown["avg_cost_per_task"]}
                      if crown else None)}


# --------------------------------------------------------------- apply state
def load_state():
    """Apply state. A missing file is just a fresh install and stays silent; an
    unusable one starts fresh AND says so, because a silent reset would drop
    candidate cooldowns without anyone noticing."""
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            s = json.load(f)
        if isinstance(s, dict):
            s.setdefault("last_good", None)
            s.setdefault("failed", {})
            return s
        detail = "not a JSON object"
    except FileNotFoundError:
        return {"last_good": None, "failed": {}}
    except (OSError, UnicodeDecodeError, ValueError) as e:
        detail = type(e).__name__
    logger.warning("apply state unusable at %s (%s); starting fresh", STATE_FILE, detail)
    return {"last_good": None, "failed": {}}


def save_state(state):
    os.makedirs(DATA_DIR, exist_ok=True)
    atomic_write_json(STATE_FILE, state)


def mark_candidate_failed(state, model_id, reason, now=None):
    state.setdefault("failed", {})[model_id] = {
        "reason": reason, "at": _now() if now is None else now}


def cooldown_excluded(state, now=None):
    ts = _now() if now is None else now
    return {mid for mid, f in (state.get("failed") or {}).items()
            if ts - float((f or {}).get("at") or 0) < FAILED_CANDIDATE_COOLDOWN_S}


# --------------------------------------------------- hermes-side probe plumbing
def _hermes_checkout():
    """A source checkout, when one exists: probes import from it when Hermes is
    not installed into the interpreter itself."""
    cand = os.path.join(HERMES_HOME, "hermes-agent")
    return cand if os.path.isdir(cand) else ""


def _hermes_env():
    """Environment for a child hermes process: the profile root pinned, nothing
    invented."""
    env = dict(os.environ)
    env["HERMES_HOME"] = os.path.abspath(HERMES_HOME)
    return env


def _imports_hermes(py):
    try:
        r = subprocess.run([py, "-c", "import hermes_cli"], capture_output=True,
                           text=True, timeout=PROBE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def hermes_python():
    """The interpreter that can import Hermes, found once per process.

    The skill runs on someone else's machine, so it looks for the runtime instead
    of being told: its own interpreter first, then the profile's venv."""
    global _HERMES_PY
    if _HERMES_PY:
        return _HERMES_PY
    candidates = [sys.executable,
                  os.path.join(HERMES_HOME, "venv", "bin", "python"),
                  os.path.join(_hermes_checkout(), "venv", "bin", "python")]
    _HERMES_PY = sys.executable
    for py in candidates:
        if py and os.path.isfile(py) and _imports_hermes(py):
            _HERMES_PY = py
            break
    return _HERMES_PY


PROBE_LOADER = (
    "import importlib.util, os, sys\n"
    "root = os.path.join(os.environ.get('HERMES_HOME', ''), 'hermes-agent')\n"
    "if os.path.isdir(root) and root not in sys.path:\n"
    "    sys.path.insert(0, root)\n"
    "spec = importlib.util.spec_from_file_location('kingmaker_probe_mod', sys.argv[1])\n"
    "mod = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(mod)\n"
    "{'catalog': mod.probe_catalog_main, 'smoke': mod.probe_smoke_main,\n"
    " 'readback': mod.probe_readback_main}[sys.argv[2]]()\n")


def run_probe(kind, *args, timeout=None):
    """Run one hermes-side probe under an interpreter that has Hermes. Bounded;
    returns the probe's JSON dict; raises RuntimeError on any failure (fail
    closed)."""
    cmd = [hermes_python(), "-c", PROBE_LOADER, os.path.abspath(__file__), kind]
    cmd += [str(a) for a in args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout or PROBE_TIMEOUT_S, env=_hermes_env())
    except Exception as e:
        raise RuntimeError("probe %s failed to run: %s" % (kind, type(e).__name__)) from e
    lines = [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
    if r.returncode != 0 or not lines:
        tail = ((r.stderr or r.stdout or "").strip().splitlines() or ["no output"])[-1]
        raise RuntimeError("probe %s exited %s: %s" % (kind, r.returncode, tail))
    try:
        return json.loads(lines[-1])
    except Exception as e:
        raise RuntimeError("probe %s returned non-JSON output" % kind) from e


def _ensure_src_on_path():
    src = _hermes_checkout()
    if src and src not in sys.path:
        sys.path.insert(0, src)
    return src


def _nous_catalog_ids():
    from hermes_cli.models import provider_model_ids
    return list(provider_model_ids("nous") or [])


def _nous_catalog_static_ids():
    from hermes_cli.models_catalog_static import _PROVIDER_MODELS
    return list(_PROVIDER_MODELS.get("nous") or [])


def _probe_catalog():
    _ensure_src_on_path()
    ids, source = [], "catalog unavailable"
    try:
        ids = _nous_catalog_ids()
        source = "provider_model_ids(nous)"
    except Exception as e:
        # Deliberate boundary: hermes internals raise whatever they raise, and a
        # stale-but-usable static catalog beats refusing to rank at all.
        logger.exception("provider_model_ids('nous') failed; using the static catalog")
        try:
            ids = _nous_catalog_static_ids()
            source = ("static _PROVIDER_MODELS[nous] (provider_model_ids failed: %s)"
                      % type(e).__name__)
        except Exception as e2:
            logger.exception("static nous catalog unavailable")
            source = "catalog unavailable (%s, %s)" % (type(e).__name__, type(e2).__name__)
    return {"ids": ids, "source": source}


def probe_catalog_main():
    print(json.dumps(_probe_catalog()))


def _resolve_runtime(provider, model):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    return resolve_runtime_provider(requested=provider, target_model=model)


def _http_post_json(url, headers, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _scrub(text, secret):
    text = str(text)
    if secret and len(str(secret)) > 4:
        text = text.replace(str(secret), "[redacted]")
    return text


def _try_parse_json(content):
    text = (content or "").strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except (ValueError, TypeError):
            return None
    return None


TRANSIENT_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}


def _is_transient(exc):
    """Throttles and network failures are transient capacity signals, NOT proof
    of model incompatibility: they must never burn a candidate cooldown."""
    import socket
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in TRANSIENT_HTTP_CODES
    return isinstance(exc, (urllib.error.URLError, socket.timeout,
                            TimeoutError, OSError))


SMOKE_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Ping tool used by a bounded compatibility smoke.",
        "parameters": {"type": "object",
                       "properties": {"msg": {"type": "string"}},
                       "required": ["msg"]},
    },
}


def run_smoke_checks(model, provider, resolve=None, post=None):
    """Bounded authorized chat/tool/structured-output compatibility smoke for a
    candidate model, strictly on the requested provider runtime (no fallback, no
    local substitution). Fails closed on unknown wire modes."""
    resolve = resolve or _resolve_runtime
    post = post or _http_post_json
    checks = {"chat": False, "tool": False, "structured": False}
    try:
        rt = resolve(provider, model) or {}
    except Exception as e:
        # Deliberate boundary: provider resolution touches hermes internals, and an
        # unresolvable runtime is a smoke failure, not a crash of the whole tick.
        logger.exception("compat smoke: runtime resolve failed for %s", model)
        return {"ok": False, "checks": checks,
                "detail": "runtime resolve failed: " + _scrub(str(e), ""),
                "transient": False}
    api_mode = str(rt.get("api_mode") or "")
    if api_mode != "chat_completions":
        return {"ok": False, "checks": checks,
                "detail": ("unsupported api_mode %r for the compatibility smoke "
                           "(fail closed; no local fallback)" % api_mode),
                "transient": False}
    key = str(rt.get("api_key") or "")
    url = str(rt.get("base_url") or "").rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + key}
    detail = ""
    try:
        r1 = post(url, headers, {
            "model": model,
            "messages": [{"role": "user", "content":
                          "Call the ping tool exactly once with msg='ok'."}],
            "tools": [SMOKE_TOOL], "tool_choice": "auto", "max_tokens": 128}, 45)
        msg = ((r1.get("choices") or [{}])[0].get("message") or {})
        calls = msg.get('tool_calls') or []
        fn = calls[0].get('function', {}) if len(calls) == 1 else {}
        checks['tool'] = (fn.get('name') == 'ping'
                          and _try_parse_json(fn.get('arguments')) == {'msg': 'ok'})
        checks['chat'] = bool(r1.get('choices')) and bool(msg)
        if not checks["tool"]:
            detail = "chat answered but returned no tool_calls"
    except (OSError, TimeoutError, ValueError, KeyError, TypeError) as e:
        return {"ok": False, "checks": checks,
                "detail": "chat/tool smoke failed: " + _scrub(str(e), key),
                "transient": _is_transient(e)}
    try:
        r2 = post(url, headers, {
            "model": model,
            "messages": [{"role": "user", "content":
                          'Reply with a JSON object of the form {"ok": true}. Nothing else.'}],
            "response_format": {"type": "json_object"}, "max_tokens": 128}, 45)
        content = str(((r2.get("choices") or [{}])[0].get("message") or {})
                      .get("content") or "")
        parsed = _try_parse_json(content)
        checks["structured"] = isinstance(parsed, dict) and parsed.get('ok') is True
        if not checks["structured"]:
            detail = detail or "structured output did not parse as a JSON object"
    except (OSError, TimeoutError, ValueError, KeyError, TypeError) as e:
        return {"ok": False, "checks": checks,
                "detail": ((detail + "; ") if detail else "")
                + "structured-output smoke failed: " + _scrub(str(e), key),
                "transient": _is_transient(e)}
    return {"ok": all(checks.values()), "checks": checks, "detail": detail or "ok",
            "transient": False}


def probe_smoke_main():
    model = sys.argv[3] if len(sys.argv) > 3 else ""
    provider = sys.argv[4] if len(sys.argv) > 4 else ""
    print(json.dumps(run_smoke_checks(model, provider)))


def _effective_model_block():
    import pathlib
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_constants import get_hermes_home
    from hermes_cli.config import split_model_config_default
    cfg = load_user_config_effective(
        pathlib.Path(os.path.join(str(get_hermes_home()), "config.yaml")))
    block = cfg.get("model") if isinstance(cfg, dict) else None
    if isinstance(block, str):
        block = {"default": block}
    if not isinstance(block, dict):
        block = {}
    main = block.get("default") or block.get("model") or block.get("name")
    model, embedded_provider = split_model_config_default(main)
    return {'main_model': model,
            'main_provider': embedded_provider or str(block.get('provider') or '').strip()}


def _scheduler_job_rows():
    from cron.jobs import load_jobs
    from cron.scheduler import _load_cron_job_config
    rows, errors = [], []
    for job in load_jobs():
        if job.get("no_agent"):
            continue
        try:
            jc = _load_cron_job_config(job, job.get("id"), str(job.get("name") or ""))
            rows.append({"id": job.get("id"), "name": job.get("name"),
                         "pinned_model": bool(job.get("model")), "model": jc.model,
                         "pinned_provider": str(job.get("provider") or ""),
                         "cron_default_provider": jc.cron_default_provider})
        except Exception as e:
            # Deliberate boundary: one unreadable job must not hide the others, and
            # the error is reported to the user with the resolution findings.
            logger.exception("cron job config unreadable for %s", job.get("id"))
            errors.append({"id": job.get("id"),
                           "error": "%s: %s" % (type(e).__name__, e)})
    return rows, errors


def _probe_readback():
    _ensure_src_on_path()
    errors = []
    try:
        effective = _effective_model_block()
    except Exception as e:
        # Deliberate boundary: readback is evidence gathering; a failure here is
        # recorded as evidence ("unknown") rather than crashing the probe.
        logger.exception("effective config readback failed")
        effective = {"model_block": {}, "main_model": "", "main_provider": ""}
        errors.append({"error": "effective config: %s: %s" % (type(e).__name__, e)})
    try:
        jobs, jerrors = _scheduler_job_rows()
        errors.extend(jerrors)
    except Exception as e:
        logger.exception("scheduler resolution readback failed")
        jobs = []
        errors.append({"error": "scheduler resolution: %s: %s" % (type(e).__name__, e)})
    return {"effective": effective, "jobs": jobs, "errors": errors}


def probe_readback_main():
    print(json.dumps(_probe_readback()))


# --------------------------------------------------------------- live plumbing
def load_nous_catalog():
    try:
        return run_probe("catalog")
    except (RuntimeError, OSError, ValueError) as e:
        return {"ids": [], "source": "catalog probe failed: %s" % e}


def compat_smoke(model_id, provider):
    out = {}
    for attempt in range(SMOKE_ATTEMPTS):
        try:
            out = run_probe("smoke", model_id, provider, timeout=SMOKE_TIMEOUT_S)
        except (RuntimeError, OSError, ValueError) as e:
            out = {"ok": False,
                   "checks": {"chat": False, "tool": False, "structured": False},
                   "detail": "smoke probe failed: %s" % _scrub(str(e), ""),
                   "transient": True}
        if out.get("ok"):
            out["transient"] = False
            return out
        if not out.get("transient"):
            return out
        if attempt < SMOKE_ATTEMPTS - 1:
            time.sleep(SMOKE_RETRY_DELAY_S)
    return out


def read_effective_and_resolution():
    try:
        return run_probe('readback')
    except (RuntimeError, OSError, ValueError) as exc:
        return {'effective': {}, 'jobs': [], 'errors': [{'error': type(exc).__name__}]}


def write_main_model(model_id):
    """Supported hermes config operation; writes ONLY the main-model key."""
    try:
        r = subprocess.run([HERMES_CLI, "config", "set", "model.default", str(model_id)],
                           capture_output=True, text=True, timeout=CONFIG_TIMEOUT_S,
                           env=_hermes_env())
    except (OSError, subprocess.SubprocessError) as e:
        return False, "config set failed to run: %s" % type(e).__name__
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    return r.returncode == 0, out


def _tail(text, n=400):
    text = str(text or "")
    return text[-n:]


# ------------------------------------------------------------------- apply flow
def apply_main(snap, catalog):
    """Selector + applier. Returns (exit_code, status, notice_lines)."""
    state = load_state()
    crown = snap.get("crown")
    rec = {"timestamp": _iso(_now()), "mode": "apply-main", "status": None,
           "previous": None, "chosen": None, "reason": None, "source": None,
           "check": None, "apply": {"attempted": False, "ok": None, "output_tail": ""},
           "verify": {"effective_model": None, "scheduler_next_job_model": None,
                      "ok": None, "rolled_back": False}}
    if crown is None:
        last_good = (state.get("last_good") or {}).get("model_id")
        rec["status"] = "failed_no_eligible"
        rec["reason"] = ("no usable nous model under the $%s/task cap (catalog: %s, "
                         "cooldown exclusions: %d)" % (
                             snap.get("cap_per_task_usd"), catalog.get("source"),
                             len(cooldown_excluded(state))))
        rec["notice"] = ("kingmaker failed: no eligible nous model under the cost cap; no config write. "
                         "Last recorded good: %s. %s" % (last_good or '(none)', rec['reason']))
        atomic_write_json(RECEIPT_FILE, rec)
        print(rec["notice"])
        return 3, rec["status"], [rec["notice"]]

    chosen = crown["model_id"]
    baseline = read_effective_and_resolution()
    cur = (baseline.get("effective") or {}).get("main_model") or ""
    prov = (baseline.get("effective") or {}).get("main_provider") or ""
    rec["previous"] = {"model": cur, "provider": prov}
    rec["chosen"] = {"model_id": chosen, "bench_name": crown.get("bench_name"),
                     "index": crown.get("index"),
                     "avg_cost_per_task": crown.get("avg_cost_per_task")}
    rec["reason"] = ("highest Hermes Index (%s) among nous-catalog models with "
                     "Hermes-measured avg cost $%s/task strictly under the "
                     "$%s/task cap (not the score/$ sort leader)"
                     % (crown.get("index"), crown.get("avg_cost_per_task"),
                        snap.get("cap_per_task_usd")))
    rec["source"] = {"bench_source": snap.get("source"),
                     "bench_name": crown.get("bench_name"),
                     "bench_snapshot_date": snap.get("date"),
                     "bench_timestamp": snap.get("timestamp"),
                     "catalog_source": catalog.get("source"),
                     "selector": "max Hermes Index under cap among usable nous models"}

    def finish(status, code, lines):
        rec["status"] = status
        rec["notice"] = "\n".join(lines)
        atomic_write_json(RECEIPT_FILE, rec)
        for ln in lines:
            print(ln)
        return code, status, lines

    if baseline.get('errors') or not cur or prov != 'nous':
        return finish('failed_config', 5, ['kingmaker blocked: configured provider/config readback is not qualified nous; no write.'])

    if snap.get('unresolved_candidates'):
        first = snap['unresolved_candidates'][0]
        return finish('failed_identity', 3, [
            'kingmaker blocked: stronger candidate %s has %s; no config write.' % (first['name'], first['reason'])])

    if chosen == cur:
        unpinned = [j for j in (baseline.get("jobs") or [])
                    if not j.get("pinned_model")]
        rec["verify"]["effective_model"] = cur
        rec["verify"]["scheduler_next_job_model"] = (
            unpinned[0].get("model") if unpinned else None)
        rec["verify"]["ok"] = all(j.get('model') == cur for j in unpinned)
        if not rec['verify']['ok']:
            return finish('failed_verify', 5, ['kingmaker blocked: unchanged main differs from cron resolution; no write.'])
        return finish("unchanged", 0, [])          # silent: nothing changed

    smoke = compat_smoke(chosen, prov or "nous")
    rec["check"] = {"ok": bool(smoke.get("ok")), "checks": smoke.get("checks"),
                    "detail": smoke.get("detail"),
                    "transient": bool(smoke.get("transient"))}
    if not smoke.get("ok"):
        if smoke.get("transient"):
            return finish("failed_smoke_transient", 4, [
                "KINGMAKER MAIN MODEL SWITCH DEFERRED: candidate %s (Hermes Index %s, "
                "$%s/task) hit a transient upstream failure during the "
                "compatibility smoke (%s). Config unchanged (%s); no cooldown; "
                "the next tick retries." % (
                    chosen, crown.get("index"), crown.get("avg_cost_per_task"),
                    smoke.get("detail"), cur)])
        mark_candidate_failed(state, chosen, "smoke failed: %s" % smoke.get("detail"))
        save_state(state)
        return finish("failed_smoke", 4, [
            "KINGMAKER MAIN MODEL SWITCH BLOCKED: candidate %s (Hermes Index %s, "
            "$%s/task) failed the compatibility smoke (%s). Config unchanged "
            "(%s); candidate cooled for %dh and will not be retried before then. "
            "Verify manually with `hermes chat -q 'ping' --model %s --provider %s`." % (
                chosen, crown.get("index"), crown.get("avg_cost_per_task"),
                smoke.get("detail"), cur, COOLDOWN_HOURS, chosen, prov or "nous")])

    fresh = read_effective_and_resolution()
    fmain = (fresh.get("effective") or {}).get("main_model") or ""
    fprov = (fresh.get("effective") or {}).get("main_provider") or ""
    rec["verify"]["effective_model"] = fmain
    if fresh.get('errors') or fmain != cur or fprov != prov:
        return finish("aborted_concurrent_change", 6, [
            "KINGMAKER MAIN MODEL APPLY ABORTED: config changed concurrently (%s -> %s, "
            "provider %s -> %s); no write performed. Re-run to apply against the "
            "new baseline." % (cur, fmain, prov, fprov)])

    ok, out = write_main_model(chosen)
    rec["apply"] = {"attempted": True, "ok": bool(ok), "output_tail": _tail(out)}
    post = read_effective_and_resolution()
    pmain = (post.get("effective") or {}).get("main_model") or ""
    rec["verify"]["effective_model"] = pmain

    def rollback():
        if pmain != chosen:
            rec['verify']['rollback_note'] = 'not our observed write; preserved'
            return False
        current = read_effective_and_resolution()
        effective = current.get('effective') or {}
        rec['verify']['effective_model'] = effective.get('main_model')
        if (current.get('errors') or effective.get('main_model') != chosen
                or effective.get('main_provider') != prov):
            rec['verify']['rollback_note'] = 'concurrent or unknown state; preserved'
            return False
        ok2, out2 = write_main_model(cur)
        chk = read_effective_and_resolution()
        cmain = (chk.get("effective") or {}).get("main_model") or ""
        rec["verify"]["rollback_output_tail"] = _tail(out2)
        rec['verify']['effective_model'] = cmain or None
        return bool(ok2 and not chk.get('errors') and cmain == cur
                    and (chk.get('effective') or {}).get('main_provider') == prov)

    if pmain != chosen:
        rolled = rollback()
        rec["verify"]["rolled_back"] = bool(rolled)
        rec["verify"]["ok"] = False
        mark_candidate_failed(state, chosen,
                              "apply failed: %s" % (rec["apply"]["output_tail"] or pmain))
        save_state(state)
        status = "failed_apply" if not ok else "failed_readback"
        code_note = ("`hermes config set model.default %s` did not take" % chosen
                     if not ok else
                     "readback showed %r instead of %r" % (pmain, chosen))
        return finish(status, 5, [
            "KINGMAKER MAIN MODEL APPLY FAILED: %s. Config left at %s (rollback %s). "
            "Candidate cooled for %dh. Output: %s" % (
                code_note, rec['verify']['effective_model'] or '(unverified)',
                "verified" if rolled else "not needed or unverified",
                COOLDOWN_HOURS, rec["apply"]["output_tail"] or "(none)")])

    unpinned = [j for j in post.get("jobs", []) if not j.get("pinned_model")]
    bad = [j for j in unpinned if j.get("model") != chosen]
    sched_model = unpinned[0].get("model") if unpinned else None
    rec["verify"]["scheduler_next_job_model"] = sched_model
    if bad or post.get("errors") or (post.get('effective') or {}).get('main_provider') != prov:
        rolled = rollback()
        rec["verify"]["rolled_back"] = bool(rolled)
        rec["verify"]["ok"] = False
        detail = ("scheduler resolution disagrees: %s" % (
            ", ".join("%s=%s" % (j.get("name"), j.get("model")) for j in bad[:3]))
            if bad else "scheduler resolution errors: %s" % (post.get("errors") or [])[:2])
        mark_candidate_failed(state, chosen, "verify failed: %s" % detail)
        save_state(state)
        return finish("failed_verify", 5, [
            "KINGMAKER verification failed; rollback %s; observed main %s. %s" % (
                'verified' if rolled else 'not performed or unverified',
                rec['verify']['effective_model'] or '(unknown)', detail)])

    rec["verify"]["ok"] = True
    state["last_good"] = {"model_id": chosen, "at": _now()}
    (state.get("failed") or {}).pop(chosen, None)
    save_state(state)
    return finish("applied", 0, [
        "KINGMAKER MAIN MODEL SWITCH: %s -> %s (Hermes Index %s, $%s/task). Applied "
        "via `hermes config set model.default`; provider %r unchanged. Applies "
        "to future runs only - running chats keep their current model. Verified: "
        "effective config and next-job scheduler resolution both show %s. "
        "Reason: %s." % (cur, chosen, "%.2f" % (crown.get("index") or 0),
                         crown.get("avg_cost_per_task"), prov or "nous",
                         chosen, rec["reason"])])


# ---------------------------------------------------------------- collector mode
def render_table(snap):
    out = [f"kingmaker snapshot {snap['date']} (Hermes Index, cap {cap_label(snap['cap_per_task_usd'])}, "
           f"{len(snap['rows'])} models)"]
    out.append(f"{'model':26} {'index':>6} {'$/task':>7}  cap prov mapped")
    for r in snap["rows"]:
        out.append(f"{r['name']:26} {r['index']:6.2f} {r['avg_cost_per_task']:7.4f}  "
                   f"{'*' if r['in_cap'] else ' '}   {'P' if r['provisional'] else ' '}    "
                   f"{r.get('provider_model_id') or '-'}")
    return "\n".join(out)


def _prev_ledger_snapshot():
    if not os.path.exists(LEDGER):
        return None
    with open(LEDGER) as handle:
        lines = [ln for ln in handle.read().splitlines() if ln.strip()]
    if len(lines) >= 2:
        try:
            return json.loads(lines[-2])
        except (ValueError, TypeError):
            logger.warning("ledger tail line unreadable; treating the previous "
                           "snapshot as unavailable")
            return None
    return None


def _collector_watchdog(snap, rows, a, notify=False):
    prev = _prev_ledger_snapshot()
    if a.quiet:
        return 0
    if a.force or prev is None:
        print(render_table(snap))
        return 0
    cur = snap.get("crown")
    pcap = [r for r in prev.get("rows", []) if r.get("in_cap")]
    prev_leader = max(pcap, key=lambda r: r["index"])["name"] if pcap else None
    if cur and prev_leader and prev_leader != cur["bench_name"]:
        print("KINGMAKER BEST-IN-BUDGET CHANGE (%s -> %s): %s -> %s (index %.2f, "
              "$%s/task)"
              % (prev.get("date", "?"), snap["date"], prev_leader,
                 cur["bench_name"], cur["index"], cur["avg_cost_per_task"]))
        cap_sorted = sorted([r for r in rows if r["in_cap"]],
                            key=lambda r: -r["index"])[:5]
        print("Top 5 in budget by Hermes Index: " + ", ".join(
            "%s (%.2f)" % (r["name"], r["index"]) for r in cap_sorted))
        if notify:
            print("Notify-only mode: nothing was changed. To switch now, run "
                  "`python3 %s --apply-main`." % os.path.abspath(__file__))
    # else: silent tick (empty stdout -> no delivery)
    return 0


def _run_setup(a):
    """The wizard's record step: this install's answers, into answers.json beside
    SKILL.md — the user's file, inside the skill, so deleting the skill takes
    them with it."""
    updates = {}
    if a.cap is not None:
        updates["cap_per_task_usd"] = parse_cap(a.cap)
    if a.mode:
        updates["mode"] = a.mode
    if a.cadence:
        updates["cadence"] = a.cadence
    if a.deliver:
        updates["deliver"] = a.deliver
    if a.cron_job:
        updates["cron_job_id"] = a.cron_job
    if updates:
        save_answers(updates)
    print("kingmaker answers (%s):" % answers_path())
    if not load_answers():
        print("  (none yet - the setup wizard writes them; see SKILL.md)")
    print(json.dumps(load_config(), indent=1, sort_keys=True))
    return 0


def main(argv=None, page_text=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", default=None,
                    help="max Hermes-measured avg $/task, strictly under; a number "
                         "or 'none' for no ceiling (overrides saved config)")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "latest.json"))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="always print the table (ignores #1-changed gate)")
    ap.add_argument("--apply-main", action="store_true",
                    help="force apply mode: switch the main model when the crown changes")
    ap.add_argument("--notify", action="store_true",
                    help="force notify-only mode: report changes, never write config")
    ap.add_argument("--sample", action="store_true",
                    help="print a live budget sample for the setup wizard")
    ap.add_argument("--setup", action="store_true",
                    help="save setup-wizard answers to answers.json")
    ap.add_argument("--mode", choices=MODES, help="mode to save with --setup")
    ap.add_argument("--cadence", help="cron expression to save with --setup")
    ap.add_argument("--deliver", help="delivery route to save with --setup")
    ap.add_argument("--cron-job", help="cron job id to remember")
    a = ap.parse_args(argv)

    _configure_logging()

    if a.setup:
        return _run_setup(a)

    cfg = load_config()
    cap = parse_cap(a.cap) if a.cap is not None else cfg["cap_per_task_usd"]
    mode = ("apply" if a.apply_main
            else "notify" if a.notify
            else cfg.get("mode") or DEFAULT_MODE)

    if not a.sample and not load_answers():
        # The skill is where the answers live, so no file means the skill is gone
        # (or was never set up). Say so and stop: never act on shipped defaults.
        if not a.quiet:
            print("kingmaker: no answers file at %s - the skill looks uninstalled or "
                  "unconfigured. Reinstall or reconfigure it (SKILL.md), or remove "
                  "this cron job with cronjob_manage." % answers_path())
        return 0

    page = fetch() if page_text is None else page_text
    recs = parse_bench(page)
    probs = sanity(recs)
    if probs:
        # Explicit failure (exit 2). The previous latest.json is preserved
        # untouched: a stale snapshot must never drive a model switch.
        print("kingmaker sanity fail (leaderboard page likely changed; ranking withheld):")
        for p in probs:
            print(" -", p)
        return 2

    os.makedirs(DATA_DIR, exist_ok=True)
    rows = build_rows(recs, cap)
    catalog = load_nous_catalog()
    attach_provider_identity(rows, catalog)
    state = load_state()
    crown = pick_crown(rows, usable_ids=set(catalog.get("ids") or []),
                       exclude=cooldown_excluded(state))
    snap = build_snapshot(rows, crown, cap)
    atomic_write_json(a.out, snap)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(snap) + "\n")
    trim(LEDGER, KEEP_RUNS)

    if a.sample:
        print(render_sample(snap, rows))
        return 0

    if mode == "apply":
        code, _status, _lines = apply_main(snap, catalog)
        return code

    if crown is None:
        print("kingmaker: no eligible crown (usable nous models under %s: 0; "
              "catalog: %s). Ranking published without a crown."
              % (cap_label(cap), catalog.get("source")))
        return 3

    return _collector_watchdog(snap, rows, a, notify=(mode == "notify"))


if __name__ == "__main__":
    _rc = main()
    # A decided outcome is a notice, not a crash: print it and exit 0 so the
    # scheduler delivers the message instead of flagging the job as broken.
    sys.exit(0 if _rc in DECIDED_CODES else _rc)
