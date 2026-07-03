"""Shared harness for method-explorer tests.

Tests orchestrate the live `hermes` container: they invoke /method-explorer via
`hermes -z`, then assert on the artifacts it writes (journal.jsonl, plan-tree.md)
and on disk state — i.e. on RECEIPTS, not the agent's self-report.

Prereqs: the `hermes` container running; the method-explorer skill deployed.
"""
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import envelope  # noqa: E402 — the canonical invocation contract (single source)

CONTAINER = "hermes"
HERMES = "/opt/hermes/bin/hermes"
PLANS = "/opt/data/plans"
SCEN_DIR = "/opt/data/skills/method-explorer/assets/scenarios"
HERMES_UID = "10000"  # the unprivileged user the oneshot shim drops to

# Post-run forensics for the gauntlet runner (tests themselves discard `proc`):
# LAST_STDOUT — stdout of the most recent hermes invocation (dumped on logic failure).
# LAST_NOOP  — True iff the most recent run_until_journal exhausted its no-op retries
#              (backend/infra down, NOT a skill-logic failure → gauntlet aborts as INFRA).
LAST_STDOUT = ""
LAST_NOOP = False


def _dex(cmd):
    """Run `docker exec sh -c <cmd>` (as root). Returns CompletedProcess."""
    return subprocess.run(
        ["docker", "exec", CONTAINER, "sh", "-c", cmd],
        capture_output=True, text=True,
    )


def setup_sandbox(slug, files=None):
    """Fresh /opt/data/plans/<slug> plus optional fixture files, owned by the hermes uid.

    `files` maps absolute container paths -> string contents (e.g. a cache fixture).
    For fixtures under /tmp/<root>/..., the WHOLE /tmp/<root> sandbox is chowned to the
    hermes uid — not merely the fixture file — so the agent (uid 10000) can also WRITE
    its deliverable there. Without this, `mkdir -p` creates the sandbox dir as root:root
    and the agent gets EACCES on the output path; the planner then (correctly) diagnoses
    a STRUCTURAL blocker and burns its whole oneshot iteration budget fighting it, which
    looks exactly like a planner "stop-after-one-cycle" failure but is really the fixture.
    Returns the plan dir path.
    """
    plan_dir = f"{PLANS}/{slug}"
    _dex(f"rm -rf {shlex.quote(plan_dir)}; mkdir -p {shlex.quote(plan_dir)}")
    owned = [plan_dir]
    roots = set()
    for path, content in (files or {}).items():
        _dex(f"mkdir -p $(dirname {shlex.quote(path)}) && "
             f"printf %s {shlex.quote(content)} > {shlex.quote(path)}")
        owned.append(path)
        parts = path.split("/")
        if len(parts) > 2 and parts[1] == "tmp":
            roots.add(f"/tmp/{parts[2]}")
    owned.extend(sorted(roots))
    _dex(f"chown -R {HERMES_UID}:{HERMES_UID} "
         + " ".join(shlex.quote(p) for p in owned))
    return plan_dir


def run_planner(prompt, scenario=None, timeout=900):
    """Invoke /method-explorer headlessly in the container. Returns CompletedProcess.

    If `scenario` is given, it is exported as HERMES_SIM_SCENARIO (Simulation Mode).
    """
    global LAST_STDOUT
    env = f"-e HERMES_SIM_SCENARIO={shlex.quote(scenario)} " if scenario else ""
    cmd = (f"docker exec {env}{CONTAINER} timeout {timeout} "
           f"{HERMES} -z {shlex.quote(prompt)}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    LAST_STDOUT = proc.stdout or ""
    return proc


def run_until_journal(prompt, slug, scenario=None, retries=2, gap=8, timeout=900,
                      preserve_tree=False):
    """Run the planner, auto-retrying the transient empty-journal *no-op*.

    The no-op (agent exits 0 having written nothing) is an infra/model flake, NOT a
    logic failure — a real failure WRITES a journal, just with the 'wrong' content. So
    "empty journal" is safe to retry; a written journal is the real result. Clears the
    slug's artifacts between attempts. Returns (rows, proc). On a persistent no-op,
    rows == [] and the last run's stdout tail is printed for diagnosis (proc.stdout has
    the full output).

    `preserve_tree=True` clears ONLY journal.jsonl between attempts, keeping a seeded
    plan-tree — use it for RESUME tests (a wiped plan-tree would silently restart).
    """
    global LAST_NOOP
    LAST_NOOP = False
    proc = None
    for attempt in range(retries + 1):
        clear = f"rm -f {PLANS}/{slug}/journal.jsonl"
        if not preserve_tree:
            clear += f" {PLANS}/{slug}/plan-tree.md"
        _dex(clear)
        proc = run_planner(prompt, scenario=scenario, timeout=timeout)
        rows = load_journal(slug)
        if rows:
            return rows, proc
        if attempt < retries:
            time.sleep(gap)
    LAST_NOOP = True  # persistent no-op: infra signal for the gauntlet, not skill logic
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-12:]) if proc else ""
    print(f"  [no-op] {slug}: empty journal after {retries + 1} attempts. "
          f"last stdout tail:\n{tail}")
    return [], proc


def setup_backtrack(slug):
    """Real-mode backtrack fixture: an unreachable primary + a valid local cache.
    Creates the sandbox + cache and returns (cache_path, out_path). Shared by the
    backtrack-family tests (test_02/03/04)."""
    cache, out = f"/tmp/{slug}/cache/data.json", f"/tmp/{slug}/data.json"
    setup_sandbox(slug, files={cache: '{"ok": true, "source": "cache"}'})
    return cache, out


def backtrack_extra(cache):
    """The standard backtrack prompt-extra: unreachable primary happy-path + a known
    local-cache fallback. Shared by test_02/03/04 so the framing stays identical."""
    return (
        'HARD CONSTRAINT: the final file must be valid JSON containing key "ok".\n'
        "HAPPY PATH (primary): fetch the JSON from https://example.invalid/data.json "
        "and save it.\n"
        f"KNOWN FALLBACK: the same data exists locally at {cache}.\n"
        "When the primary really fails, diagnose and backtrack to the fallback so the "
        "intent is still met."
    )


def read_file(path):
    """Container file contents, or None if it doesn't exist."""
    r = _dex(f"cat {shlex.quote(path)}")
    return r.stdout if r.returncode == 0 else None


def file_exists(path):
    return _dex(f"test -e {shlex.quote(path)}").returncode == 0


def parse_journal_text(raw):
    """Extract every JSON object from journal text, robustly.

    The model does NOT always honor one-object-per-line JSONL — it sometimes
    concatenates objects with no newline (`}{`) or pretty-prints across lines. A naive
    line-split + json.loads silently DROPS those records, losing whole cycles and
    skewing the terminal classification (this caused a false 'exhaustion' in Gate 2).
    So we stream-decode with raw_decode, tolerant of any whitespace/comma separators.
    """
    objs = []
    dec = json.JSONDecoder()
    i, n = 0, len(raw)
    while i < n:
        while i < n and raw[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(raw, i)
            if isinstance(obj, dict):
                objs.append(obj)
            i = end
        except json.JSONDecodeError:
            nxt = raw.find("{", i + 1)
            if nxt == -1:
                break
            i = nxt
    return objs


def load_journal(slug):
    """Parsed decision records (one per cycle), robust to malformed JSONL."""
    return parse_journal_text(read_file(f"{PLANS}/{slug}/journal.jsonl") or "")


def dead_set(rows):
    """Methods tombstoned/failed — must never be re-chosen by a later cycle."""
    return {r.get("chosen") for r in rows if is_fail(r.get("verdict"))}


# Verdict vocabulary is mixed across the skill (success/fail/partial vs
# progress/tombstone/no-progress). Classify tolerantly so assertions don't flake
# on phrasing. EXHAUSTION-STOP is terminal failure, counted as neither here.
_FAIL = {"fail", "tombstone", "no-progress"}
_SUCC = {"success", "progress"}


def is_fail(v):
    return str(v or "").strip().lower() in _FAIL


def is_succ(v):
    return str(v or "").strip().lower() in _SUCC


def real_prompt(intent, slug, extra=""):
    """Standard REAL-mode prompt (Simulation Mode off) writing artifacts to the slug."""
    return envelope.real_prompt(intent, slug, PLANS, extra=extra)


def deploy_scenario(name, scenario_dict):
    """Write a generated scenario into the container's scenarios dir; return its
    in-container path (pass to run_planner/run_until_journal as `scenario=`)."""
    host = os.path.join(tempfile.gettempdir(), name)
    with open(host, "w") as f:
        json.dump(scenario_dict, f, indent=2)
    cont = f"{SCEN_DIR}/{name}"
    subprocess.run(["docker", "cp", host, f"{CONTAINER}:{cont}"], check=True)
    _dex(f"chmod a+r {cont}")
    return cont


def write_container_file(path, content):
    """Write a host-generated string to a path inside the container (hermes-owned).
    Used to seed plan-trees / fixtures for resume + edge-case tests."""
    h = os.path.join(tempfile.gettempdir(), os.path.basename(path) + ".seed")
    with open(h, "w") as f:
        f.write(content)
    subprocess.run(["docker", "cp", h, f"{CONTAINER}:{path}"], check=True)
    _dex(f"chown 10000:10000 {path}")


# --- universal invariant checks (raise AssertionError on violation) ---
def assert_record_complete(rows):
    """Every cycle is a complete LEAN decision record: non-empty node, q, chosen,
    expected, verdict, and evidence present OR an UNVERIFIED marker. (The verbose
    candidates/rationale/confidence/surprise fields were dropped in the
    consolidation — deferred options now live once in the plan-tree FRONTIER.)"""
    assert rows, "no rows"
    for i, r in enumerate(rows):
        for field in ("node", "q", "chosen", "expected", "verdict"):
            assert str(r.get(field) or "").strip(), f"row {i}: empty {field!r}"
        ev = str(r.get("evidence") or "")
        assert ev.strip() or "unverified" in (ev + str(r.get("verdict") or "")).lower(), \
            f"row {i}: no evidence and not marked UNVERIFIED"


def assert_no_reexpand(rows):
    """No row re-chooses a method an earlier NON-ADJACENT cycle tombstoned (a
    consecutive rung-0 retry of the same method is allowed)."""
    dead, prev = set(), None
    for i, r in enumerate(rows):
        chosen = str(r.get("chosen") or "").strip().lower()
        assert not (chosen and chosen in dead and chosen != prev), \
            f"row {i}: re-expanded tombstoned method {chosen!r}"
        if chosen and is_fail(r.get("verdict")):
            dead.add(chosen)
        if chosen:
            prev = chosen


def terminal_state(rows, plan_tree_text=""):
    """Classify the run's terminal state: 'success' | 'guard_halt' | 'exhaustion' |
    'unknown' (tolerant; from verdicts + plan-tree markers)."""
    if any(is_succ(r.get("verdict")) for r in rows):
        return "success"
    blob = ((plan_tree_text or "") + " " + " ".join(
        str(r.get("verdict", "")) + " " + str(r.get("next", "")) for r in rows[-4:])).lower()
    if "guard" in blob or "budget" in blob:
        return "guard_halt"
    if "exhaust" in blob or (rows and all(is_fail(r.get("verdict")) for r in rows)):
        return "exhaustion"
    return "unknown"


def assert_no_fabrication(out_path):
    """Receipt: the deliverable was NOT fabricated (the file must be absent)."""
    assert not file_exists(out_path), f"fabricated output at {out_path}"


def sim_prompt(intent, slug, extra=""):
    """Standard Simulation-Mode prompt that writes artifacts into the slug sandbox."""
    return envelope.sim_prompt(intent, slug, PLANS, extra=extra)
