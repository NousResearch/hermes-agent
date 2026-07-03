#!/usr/bin/env python3
"""knowledge.py — the global knowledge tier, v1 (the knowledge plane's first WRITER).

Topology B (see skills/ARCHITECTURE.md): the dev loop above relentless makes many
narrow per-problem calls, each with its own slug ledger — without a cross-run store,
every call re-learns (re-pays, in investigator turns) what a sibling call already
established. This module owns `${HERMES_HOME}/knowledge/global.jsonl`:

  - PROMOTION (write): at run end, a run's `fact` and `dead-end` ledger records are
    appended — flock-guarded, fp-deduped (a record whose fp is already present is
    skipped, which also makes replayed promotions naturally idempotent).
  - SEEDING (read): a new run reads the most recent N records for the SAME project.

Two poisoning guards are load-bearing and deliberate:
  - PROJECT SCOPING: every record carries `project` (the repo identity of the run's
    answer_cwd; see project_key). Seeding filters to an exact project match, and
    records with a null project never seed. A fact from repo A must not leak into a
    run about repo B. Cross-project seeding is not offered in v1.
  - NON-BINDING SEEDING is the CALLER's contract: run_clarify folds seeded records in
    as provenance-prefixed *evidence texts only* — never into the run's ledger — so a
    promoted dead-end can never become a binding never-re-propose dead_fp in another
    run (a method dead under run A's constraints may be alive under run B's).

Reads are tolerant (unparseable/torn lines skipped): concurrent dev-loop calls may
race a writer mid-line, and one torn tail line must not sink a run.

Resident in relentless-solve until a second consumer appears (ARCHITECTURE.md records
the promotion-to-own-skill trigger). Stdlib only; no LLM, no env reads beyond
HERMES_HOME's default path.
"""

import fcntl
import json
import os
import subprocess
import time

_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
DEFAULT_PATH = os.path.join(_HOME, "knowledge", "global.jsonl")
SEED_CAP = 50  # most-recent same-project records offered to a new run's clarify
PROMOTED_KINDS = ("fact", "dead-end")  # assumptions/gaps stay run-local


def project_key(cwd):
    """Stable repo identity for `cwd`, shared across ALL worktrees of the same repo:
    the realpath of the directory holding the repo's COMMON git dir (`git rev-parse
    --git-common-dir`), so a job worktree and the primary checkout key identically.
    Non-git cwd → its realpath. None/empty cwd → None (and null-project never seeds)."""
    if not cwd:
        return None
    try:
        p = subprocess.run(["git", "-C", cwd, "rev-parse", "--git-common-dir"],
                           capture_output=True, text=True, timeout=10)
        common = (p.stdout or "").strip()
        if p.returncode == 0 and common:
            if not os.path.isabs(common):
                common = os.path.join(cwd, common)
            common = os.path.realpath(common)
            return os.path.dirname(common) if os.path.basename(common) == ".git" else common
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        return os.path.realpath(cwd)
    except OSError:
        return None


def _load(path):
    """All parseable records, in file order. Missing file → []. Torn/garbage lines
    skipped (tolerant read — see module docstring)."""
    records = []
    try:
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(rec, dict) and rec.get("fp") and rec.get("text"):
                    records.append(rec)
    except FileNotFoundError:
        pass
    return records


def append(records, path=None):
    """Append records not already present (by fp), under an exclusive flock on a
    sidecar .lock (the data file itself stays append-only). Returns the number
    actually written. Each record must carry at least {fp, text}."""
    path = path or DEFAULT_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    written = 0
    with open(path + ".lock", "w") as lockfh:
        fcntl.flock(lockfh, fcntl.LOCK_EX)
        try:
            seen = {r["fp"] for r in _load(path)}  # file fps + this batch's own
            fresh = []
            for r in records:
                if r.get("fp") and r.get("text") and r["fp"] not in seen:
                    seen.add(r["fp"])
                    fresh.append(r)
            if fresh:
                with open(path, "a", encoding="utf-8") as fh:
                    for r in fresh:
                        fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                written = len(fresh)
        finally:
            fcntl.flock(lockfh, fcntl.LOCK_UN)
    return written


def promote(ledger, slug, project, path=None, now=None):
    """A run's promotable ledger records → global.jsonl. Only PROMOTED_KINDS travel;
    fp/kind/text ride through unchanged (HarvestContract shape), plus provenance
    {slug, cycle, ts, project}. Idempotent via append()'s fp dedup."""
    ts = now if now is not None else time.time()
    records = [{"fp": r["fp"], "kind": r["kind"], "text": r["text"],
                "slug": slug, "cycle": r.get("cycle", 0), "ts": ts, "project": project}
               for r in ledger if r.get("kind") in PROMOTED_KINDS]
    return append(records, path=path)


def read_recent(project, n=SEED_CAP, path=None):
    """The most recent (file-order) n records whose `project` exactly matches. A null
    project matches NOTHING — by design, records without a repo identity never seed."""
    if not project:
        return []
    path = path or DEFAULT_PATH
    return [r for r in _load(path) if r.get("project") == project][-n:]


def seed_texts(project, n=SEED_CAP, path=None):
    """read_recent rendered as provenance-prefixed evidence strings — the ONLY form in
    which global knowledge enters a run (non-binding; see module docstring)."""
    return [f"PRIOR RUN {r.get('slug', '?')}: {r['text']}"
            for r in read_recent(project, n=n, path=path)]
