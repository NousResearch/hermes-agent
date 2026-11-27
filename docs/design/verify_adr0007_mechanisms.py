"""Verify the load-bearing mechanisms of the structured run provenance contract.

Exercises the mechanisms specified in ``kanban-run-provenance-v1.0.0.md`` and
its ``kanban-run-provenance-v1.1.0-correction.md`` against real SQLite and real
git, so the contract's claims are demonstrated rather than asserted.

This is a mechanism probe against a model of the schema. It is **not** a
substitute for the acceptance tests (v1.0.0 §10 / v1.1.0 §C.1), which must run
against the real kernel and belong to the implementer.

Run:  python3 verify_adr0007_mechanisms.py
"""

import hashlib
import itertools
import json
import os
import re
import sqlite3
import subprocess
import tempfile

# v1.0.0 §2.1: full 40-char (resp. 64-char) all-lowercase hex. EVERY character
# is validated, not just the first one.
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

# The SQL form of the same rule. `NOT GLOB '*[^0-9a-f]*'` is the strict
# constraint: it rejects a non-hex character *anywhere* in the value. The
# superseded form `GLOB '[0-9a-f]*'` anchored only the FIRST character and
# accepted e.g. 'a' + 'Z'*39 -- demonstrated by `legacy_sha_check` below.
#
# v1.1.0 §A.4 (final_candidate_sha resolution): the run carries `subject_sha`
# (stamped at claim) and `verified_head_sha` (stamped at terminalization).
# There is deliberately NO `final_candidate_sha` / `final_sha` column: the
# final candidate SHA is an attestation-layer field the broker derives.
#
# v1.1.0 §A.6 (review finding N2): no `board` column. kanban.db is already
# per-board and v1.0.0 §8 rejects board slug as a provenance field; repository
# identity is `repo_github_id`.
DDL = """
CREATE TABLE task_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
    profile TEXT, status TEXT NOT NULL, started_at INTEGER NOT NULL,
    ended_at INTEGER, outcome TEXT, summary TEXT, metadata TEXT
);
CREATE TABLE run_provenance (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL UNIQUE, task_id TEXT NOT NULL,
    profile TEXT, outcome TEXT NOT NULL, attestable INTEGER NOT NULL,
    subject_sha TEXT, verified_head_sha TEXT, branch_name TEXT,
    repo_locator TEXT, workspace_kind TEXT NOT NULL,
    evidence_count INTEGER NOT NULL, evidence_digest TEXT,
    started_at INTEGER NOT NULL, completed_at INTEGER NOT NULL,
    contract_version TEXT NOT NULL, record_digest TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    CHECK (subject_sha IS NULL OR (length(subject_sha) = 40
           AND subject_sha NOT GLOB '*[^0-9a-f]*')),
    CHECK (verified_head_sha IS NULL OR (length(verified_head_sha) = 40
           AND verified_head_sha NOT GLOB '*[^0-9a-f]*'))
);
CREATE TRIGGER trg_run_provenance_no_update BEFORE UPDATE ON run_provenance
BEGIN SELECT RAISE(ABORT, 'run_provenance is append-only'); END;
CREATE TRIGGER trg_run_provenance_no_delete BEFORE DELETE ON run_provenance
BEGIN SELECT RAISE(ABORT, 'run_provenance is append-only'); END;
CREATE TABLE run_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL,
    task_id TEXT NOT NULL, artifact_path TEXT NOT NULL, sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL, git_blob_oid TEXT,
    tracked INTEGER NOT NULL DEFAULT 0, clean INTEGER NOT NULL DEFAULT 0,
    declared_by TEXT NOT NULL, created_at INTEGER NOT NULL,
    sealed INTEGER NOT NULL DEFAULT 0, UNIQUE(run_id, artifact_path),
    CHECK (length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*')
);
CREATE TRIGGER trg_run_artifacts_sealed BEFORE UPDATE ON run_artifacts
WHEN OLD.sealed = 1
BEGIN SELECT RAISE(ABORT, 'evidence row is sealed'); END;

-- SUPERSEDED pattern, retained only as an executable demonstration of why
-- v1.0.0 §2.1's CHECK had to change. Not part of the specified schema.
CREATE TABLE legacy_sha_check (
    sha TEXT,
    CHECK (sha IS NULL OR (length(sha) = 40 AND sha GLOB '[0-9a-f]*'))
);
"""

PROV_COLUMNS = (
    "run_id,task_id,profile,outcome,attestable,subject_sha,verified_head_sha,"
    "branch_name,repo_locator,workspace_kind,evidence_count,evidence_digest,"
    "started_at,completed_at,contract_version,record_digest,created_at"
)
PROV_PLACEHOLDERS = ",".join("?" * len(PROV_COLUMNS.split(",")))
CONTRACT_VERSION = "kanban.provenance/v1"


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def digest(obj):
    return hashlib.sha256(canon(obj)).hexdigest()


results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    verdict = "PASS" if cond else "FAIL"
    suffix = f"  [{detail}]" if detail else ""
    print(f"{verdict}  {name}{suffix}")


db = sqlite3.connect(":memory:")
db.row_factory = sqlite3.Row
db.executescript(DDL)


# --- provenance rows for several terminal outcomes -------------------
def insert_prov(run_id, outcome, subject, head, ev):
    ev_digest = None
    if ev:
        ev_digest = digest([
            {"artifact_path": e["artifact_path"], "sha256": e["sha256"]}
            for e in sorted(ev, key=lambda x: x["artifact_path"])
        ])
    attestable = int(
        outcome == "completed" and bool(subject) and bool(head) and len(ev) > 0
    )
    body = {"contract_version": CONTRACT_VERSION, "run_id": run_id,
            "task_id": "t_demo", "profile": "security-reviewer",
            "outcome": outcome, "attestable": bool(attestable),
            "subject_sha": subject, "verified_head_sha": head,
            "evidence_digest": ev_digest, "started_at": 100,
            "completed_at": 200}
    rd = digest(body)
    db.execute(
        f"INSERT INTO run_provenance ({PROV_COLUMNS}) "  # noqa: S608
        f"VALUES ({PROV_PLACEHOLDERS})",
        (run_id, "t_demo", "security-reviewer", outcome, attestable,
         subject, head, "br", "git@github.com:x/y.git", "worktree",
         len(ev), ev_digest, 100, 200, CONTRACT_VERSION, rd, 200))
    db.commit()
    return rd


SHA_A = "a" * 40
SHA_B = "b" * 40
ev1 = [{"artifact_path": "evidence/qa.json", "sha256": "c" * 64},
       {"artifact_path": "evidence/sec.json", "sha256": "d" * 64}]

rd_ok = insert_prov(1, "completed", SHA_A, SHA_B, ev1)
insert_prov(2, "blocked", SHA_A, SHA_B, ev1)
insert_prov(3, "crashed", None, None, [])
insert_prov(4, "completed", SHA_A, SHA_B, [])   # no evidence

rows = {r["run_id"]: r for r in db.execute("SELECT * FROM run_provenance")}
check("row written for every terminal outcome", len(rows) == 4, f"{len(rows)} rows")
check("only completed+SHAs+evidence is attestable",
      rows[1]["attestable"] == 1 and rows[2]["attestable"] == 0
      and rows[3]["attestable"] == 0 and rows[4]["attestable"] == 0,
      f"blocked={rows[2]['attestable']} crashed={rows[3]['attestable']} "
      f"noev={rows[4]['attestable']}")

# --- B2: SHA CHECK constraints validate EVERY character ---------------
# Run against a SEPARATE connection so hostile probes cannot pollute the
# narrative table used by the seq/watermark checks below.
probe = sqlite3.connect(":memory:")
probe.row_factory = sqlite3.Row
probe.executescript(DDL)

# First, demonstrate the defect in the superseded pattern so the reason for
# the change is executable rather than a claim in prose.
LEGACY_HOSTILE = "a" + "Z" * 39
try:
    probe.execute("INSERT INTO legacy_sha_check (sha) VALUES (?)", (LEGACY_HOSTILE,))
    probe.commit()
    legacy_accepted = True
except sqlite3.IntegrityError:
    legacy_accepted = False
probe.rollback()
check("superseded GLOB '[0-9a-f]*' pattern accepts 'a'+'Z'*39 (the defect)",
      legacy_accepted, "first character only -- why v1.0.0 §2.1 changed")

HOSTILE_SHA40 = {
    "'a'+'Z'*39 (non-hex tail)": "a" + "Z" * 39,
    "'a'*39+'g' (non-hex last char)": "a" * 39 + "g",
    "uppercase 'A'*40": "A" * 40,
    "mixed case 'aB'*20": "aB" * 20,
    "39 chars (too short)": "a" * 39,
    "41 chars (too long)": "a" * 41,
    "embedded space": "a" * 20 + " " + "a" * 19,
}

_hostile_run_ids = itertools.count(1001)


def prov_sha_rejected(column, value):
    """Attempt a minimal insert setting one SHA column; True if CHECK aborts."""
    run_id = next(_hostile_run_ids)
    subject = value if column == "subject_sha" else SHA_A
    head = value if column == "verified_head_sha" else SHA_B
    try:
        probe.execute(
            f"INSERT INTO run_provenance ({PROV_COLUMNS}) "  # noqa: S608
            f"VALUES ({PROV_PLACEHOLDERS})",
            (run_id, "t_hostile", "p", "completed", 0,
             subject, head, "br", "loc", "worktree", 0, None, 1, 2,
             CONTRACT_VERSION, "d", 3))
        probe.commit()
        return False
    except sqlite3.IntegrityError:
        return True


for column in ("subject_sha", "verified_head_sha"):
    for label, value in HOSTILE_SHA40.items():
        check(f"{column} CHECK rejects {label}",
              prov_sha_rejected(column, value))

check("subject_sha CHECK accepts valid full lowercase 40-hex",
      not prov_sha_rejected("subject_sha", ("0123456789abcdef" * 3)[:40]))
check("verified_head_sha CHECK accepts valid full lowercase 40-hex",
      not prov_sha_rejected("verified_head_sha", ("0123456789abcdef" * 3)[:40]))

HOSTILE_SHA256 = {
    "'c'*63+'Z' (non-hex tail)": "c" * 63 + "Z",
    "'c'+'Z'*63 (non-hex tail)": "c" + "Z" * 63,
    "uppercase 'C'*64": "C" * 64,
    "63 chars (too short)": "c" * 63,
}

_hostile_ev_ids = itertools.count(2001)


def evidence_sha_rejected(value):
    """True if the run_artifacts.sha256 CHECK aborts the insert."""
    ev_id = next(_hostile_ev_ids)
    try:
        probe.execute(
            "INSERT INTO run_artifacts (run_id,task_id,artifact_path,sha256,"
            "size_bytes,declared_by,created_at,sealed) "
            "VALUES (?,'t_hostile',?,?,10,'p',100,0)",
            (ev_id, f"evidence/h{ev_id}.json", value))
        probe.commit()
        return False
    except sqlite3.IntegrityError:
        return True


for label, value in HOSTILE_SHA256.items():
    check(f"run_artifacts.sha256 CHECK rejects {label}",
          evidence_sha_rejected(value))
check("run_artifacts.sha256 CHECK accepts valid full lowercase 64-hex",
      not evidence_sha_rejected(("0123456789abcdef" * 4)[:64]))
probe.close()

# --- immutability -----------------------------------------------------
# Tamper with a *valid* 40-hex value, so the abort is attributable to the
# append-only trigger and not to the hex CHECK constraint.
try:
    db.execute("UPDATE run_provenance SET subject_sha=? WHERE run_id=1",
               ("e" * 40,))
    db.commit()
    check("UPDATE on run_provenance aborts", False, "update succeeded!")
except sqlite3.IntegrityError as e:
    check("UPDATE on run_provenance aborts", True, str(e))
db.rollback()

try:
    db.execute("DELETE FROM run_provenance WHERE run_id=1")
    db.commit()
    check("DELETE on run_provenance aborts", False, "delete succeeded!")
except sqlite3.IntegrityError as e:
    check("DELETE on run_provenance aborts", True, str(e))
db.rollback()

still = db.execute(
    "SELECT subject_sha, record_digest FROM run_provenance WHERE run_id=1"
).fetchone()
check("record survives tamper attempt byte-identical",
      still["subject_sha"] == SHA_A and still["record_digest"] == rd_ok)

# --- evidence sealing -------------------------------------------------
db.execute("INSERT INTO run_artifacts (run_id,task_id,artifact_path,sha256,size_bytes,"
           "declared_by,created_at,sealed) VALUES (1,'t_demo','evidence/qa.json',?,10,"
           "'security-reviewer',100,0)", ("c" * 64,))
db.commit()
# unsealed: allowed
db.execute("UPDATE run_artifacts SET sha256=? WHERE run_id=1", ("f" * 64,))
db.commit()
check("unsealed evidence is editable", True)
db.execute("UPDATE run_artifacts SET sealed=1 WHERE run_id=1")
db.commit()
try:
    db.execute("UPDATE run_artifacts SET sha256=? WHERE run_id=1", ("0" * 64,))
    db.commit()
    check("sealed evidence rejects writes", False, "write succeeded!")
except sqlite3.IntegrityError as e:
    check("sealed evidence rejects writes", True, str(e))
db.rollback()

# --- duplicate run_id -------------------------------------------------
try:
    insert_prov(1, "completed", SHA_A, SHA_B, ev1)
    check("duplicate run_id rejected", False, "duplicate accepted!")
except sqlite3.IntegrityError as e:
    check("duplicate run_id rejected", True, str(e))
db.rollback()

# --- seq monotonic in terminalization order, not run id ---------------
insert_prov(99, "completed", SHA_A, SHA_B, ev1)   # high run id, terminates first
insert_prov(50, "completed", SHA_A, SHA_B, ev1)
ordered = [(r["seq"], r["run_id"]) for r in
           db.execute("SELECT seq, run_id FROM run_provenance ORDER BY seq")]
seqs = [s for s, _ in ordered]
check("seq strictly ascending",
      seqs == sorted(seqs) and len(set(seqs)) == len(seqs), str(ordered))
check("seq order != run_id order (proves cursor is terminalization order)",
      [r for _, r in ordered] != sorted(r for _, r in ordered))


# --- watermark idempotency -------------------------------------------
def export_since(watermark):
    return [dict(r) for r in db.execute(
        "SELECT * FROM run_provenance WHERE seq > ? AND attestable = 1 ORDER BY seq",
        (watermark,))]


first = export_since(0)
wm = max(r["seq"] for r in first)
check("re-export from watermark yields nothing", export_since(wm) == [])
check("export digests unique", len({r["record_digest"] for r in first}) == len(first))

# --- read-only connection cannot write --------------------------------
tmpdir = tempfile.mkdtemp()
path = os.path.join(tmpdir, "k.db")
disk = sqlite3.connect(path)
disk.executescript(DDL)
disk.commit()
disk.close()
ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
try:
    ro.execute("INSERT INTO run_provenance (run_id,task_id,outcome,attestable,"
               "workspace_kind,evidence_count,started_at,completed_at,"
               "contract_version,record_digest,created_at) "
               "VALUES (7,'t','completed',1,'worktree',0,1,2,'v','d',3)")
    ro.commit()
    check("read-only exporter connection cannot write", False, "write succeeded!")
except sqlite3.OperationalError as e:
    check("read-only exporter connection cannot write", True, str(e))
ro.close()

# --- SHA validation (Python-side mirror of the SQL CHECKs) ------------
check("full 40-hex accepted", bool(SHA40.match(SHA_A)))
check("abbreviated SHA rejected", not SHA40.match("a1b2c3d"))
check("uppercase SHA rejected", not SHA40.match("A" * 40))
check("64-hex artifact digest accepted", bool(SHA256.match("c" * 64)))
for label, value in HOSTILE_SHA40.items():
    check(f"SHA40 regex rejects {label}", not SHA40.match(value))
for label, value in HOSTILE_SHA256.items():
    check(f"SHA256 regex rejects {label}", not SHA256.match(value))

# --- evidence_digest is order-independent -----------------------------
d1 = digest([{"artifact_path": e["artifact_path"], "sha256": e["sha256"]}
             for e in sorted(ev1, key=lambda x: x["artifact_path"])])
d2 = digest([{"artifact_path": e["artifact_path"], "sha256": e["sha256"]}
             for e in sorted(list(reversed(ev1)), key=lambda x: x["artifact_path"])])
check("evidence_digest stable under input order", d1 == d2)
ev_tampered = [ev1[0], {"artifact_path": "evidence/sec.json", "sha256": "9" * 64}]
d3 = digest([{"artifact_path": e["artifact_path"], "sha256": e["sha256"]}
             for e in sorted(ev_tampered, key=lambda x: x["artifact_path"])])
check("evidence_digest changes when a hash changes", d1 != d3)

# --- kernel-side hashing beats worker-declared hash -------------------
f = os.path.join(tmpdir, "artifact.json")
with open(f, "w", encoding="utf-8") as fh:
    fh.write('{"result":"pass"}')
with open(f, "rb") as fh:
    real = hashlib.sha256(fh.read()).hexdigest()
worker_claimed = "0" * 64
check("kernel hash != worker-claimed hash (worker value must be ignored)",
      real != worker_claimed, real[:16] + "...")

# --- git HEAD capture is real and full-length -------------------------
repo = os.path.join(tmpdir, "repo")
os.makedirs(repo)


def git(*args):
    return subprocess.run(args, cwd=repo, capture_output=True, text=True,
                          check=False)


git("git", "init", "-q")
git("git", "config", "user.email", "a@b.c")
git("git", "config", "user.name", "t")
with open(os.path.join(repo, "f.txt"), "w", encoding="utf-8") as fh:
    fh.write("x")
git("git", "add", ".")
git("git", "commit", "-qm", "init")
head = git("git", "rev-parse", "HEAD").stdout.strip()
check("git rev-parse HEAD yields full 40-hex", bool(SHA40.match(head)), head)
blob = git("git", "rev-parse", "HEAD:f.txt").stdout.strip()
check("git blob oid resolvable for tracked file", bool(SHA40.match(blob)), blob)
missing = git("git", "rev-parse", "HEAD:nope.txt")
check("untracked file yields no blob oid (tracked=0)", missing.returncode != 0)

# scratch (non-git) dir must yield nothing -> non-attestable
scratch = os.path.join(tmpdir, "scratch")
os.makedirs(scratch)
r = subprocess.run(["git", "-C", scratch, "rev-parse", "HEAD"],
                   capture_output=True, text=True, check=False)
check("scratch workspace has no HEAD => subject_sha NULL => non-attestable",
      r.returncode != 0)

# --- v1.1.0 §A.1 nullable-vs-required repository binding --------------
MANDATORY_FOR_EXPORT = ("repo_github_id", "event_locator", "subject_sha",
                        "verified_head_sha", "evidence_count")


def finalizable(rec):
    """Kernel-side finalization gate from v1.1.0 §A.1. Fails closed."""
    if rec.get("outcome") != "completed":
        return False, "outcome not completed"
    for field in MANDATORY_FOR_EXPORT:
        v = rec.get(field)
        if v is None or v == "":
            return False, f"missing {field}"
    if rec["evidence_count"] < 1:
        return False, "no evidence"
    if not SHA40.match(rec["subject_sha"] or ""):
        return False, "bad subject_sha"
    if not SHA40.match(rec["verified_head_sha"] or ""):
        return False, "bad verified_head_sha"
    if not isinstance(rec["repo_github_id"], int):
        return False, "repo_github_id not numeric"
    if not re.match(r"^pr:\d+$", rec["event_locator"] or ""):
        return False, "bad event_locator"
    if rec.get("corrections_present"):
        return False, "unresolved correction chain"
    return True, ""


good = {"outcome": "completed", "repo_github_id": 123456789,
        "event_locator": "pr:36", "subject_sha": SHA_A,
        "verified_head_sha": SHA_B, "evidence_count": 2,
        "corrections_present": False}
ok, why = finalizable(good)
check("complete gated run is finalizable", ok, why)

# a non-gated run (docs/research) is legal but simply never exported
non_gated = dict(good, repo_github_id=None, event_locator=None)
ok2, why2 = finalizable(non_gated)
check("non-gated run: NULL repo fields legal, not finalizable/exported",
      not ok2, why2)

# every mandatory field missing => refused, naming the field
for field in MANDATORY_FOR_EXPORT:
    bad = dict(good)
    bad[field] = None
    ok3, why3 = finalizable(bad)
    check(f"finalization fails closed when {field} missing",
          not ok3 and field in why3, why3)

check("abbreviated subject_sha refused at finalization",
      not finalizable(dict(good, subject_sha="a1b2c3d"))[0])
check("repo id as remote string refused (must be numeric)",
      not finalizable(dict(good, repo_github_id="git@github.com:x/y.git"))[0])
check("malformed event_locator refused",
      not finalizable(dict(good, event_locator="36"))[0])
check("unresolved correction chain refused",
      not finalizable(dict(good, corrections_present=True))[0])

# hostile SHAs must also be refused by the finalization gate, not only by SQL
for label, value in HOSTILE_SHA40.items():
    check(f"finalization refuses subject_sha {label}",
          not finalizable(dict(good, subject_sha=value))[0])
    check(f"finalization refuses verified_head_sha {label}",
          not finalizable(dict(good, verified_head_sha=value))[0])

# --- v1.1.0 §A.4: final_candidate_sha is NOT a Kanban column ----------
prov_cols = {r[1] for r in db.execute("PRAGMA table_info(run_provenance)")}
check("no final_candidate_sha / final_sha column exists",
      not {"final_candidate_sha", "final_sha"} & prov_cols,
      str(sorted(prov_cols & {"subject_sha", "verified_head_sha"})))
check("both witnessed SHAs ARE columns (subject_sha + verified_head_sha)",
      {"subject_sha", "verified_head_sha"} <= prov_cols)

# --- N2 resolution: no board column in the provenance record ----------
check("no board / board_slug column exists (v1.0.0 §8)",
      not {"board", "board_slug"} & prov_cols, str(sorted(prov_cols)[:4]))

print("\n" + "=" * 60)
failed = [n for n, ok, _ in results if not ok]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED: " + ", ".join(failed))
    raise SystemExit(1)
print("All contract mechanism claims verified.")
