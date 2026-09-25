"""Phase-correctness + idempotence matrix for the kanban-pipeline observer.

Every fixture is local: an isolated temp HERMES_HOME + board DB, and a fake
artifact transport (either an injected callable or a fake ``gh`` executable on
PATH). No real GitHub call, no real board, no product card is ever touched.

Run:
  venv/bin/python -m pytest contrib/den-plugins/kanban-pipeline/test_pipeline_phase.py -q
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import time
import sys
import tempfile
import threading

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
PLUGIN_SRC = os.environ.get(
    "KP_PLUGIN_SRC",
    os.path.join(REPO, "contrib", "den-plugins", "kanban-pipeline", "__init__.py"),
)
# (KP_PLUGIN_SRC is the mutation/negative-control switch: point it at the
#  pre-repair revision and this whole matrix must go RED.)
PLUGIN_YAML = os.path.join(REPO, "contrib", "den-plugins", "kanban-pipeline", "plugin.yaml")

PR_CURRENT = "https://github.com/VibeTechnologies/AgentPod/pull/4878"
PR_HISTORICAL = "https://github.com/VibeTechnologies/AgentPod/pull/4841"
PR_OTHER = "https://github.com/VibeTechnologies/AgentPod/pull/4949"


# ---------------------------------------------------------------------------
# isolated environment
# ---------------------------------------------------------------------------

# The plugin is OFF unless config says otherwise, and it chains nothing for a
# repository that is not on the explicit allow-list. Every test therefore has
# to opt in exactly the way a board owner would.
CONFIG_YAML = """kanban_pipeline:
  enabled: true
  repos:
    VibeTechnologies/AgentPod:
      merge_command: scripts/safe-merge.sh
      deploy_branch: main
      deploy_workflow: deploy.yml
      deploy_job: deploy
"""


def _fresh_home(tmp_path_factory, config_yaml=CONFIG_YAML):
    home = str(tmp_path_factory.mktemp("kp_home"))
    os.environ["HERMES_HOME"] = home
    for v in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME",
              "HERMES_KANBAN_WORKSPACES_ROOT"):
        os.environ.pop(v, None)
    for prof in ("reviewer", "software-engineer", "default"):
        os.makedirs(os.path.join(home, "profiles", prof), exist_ok=True)
    with open(os.path.join(home, "config.yaml"), "w") as fh:
        fh.write(config_yaml)
    return home


@pytest.fixture()
def env(tmp_path_factory):
    """Temp HOME + freshly imported board module + freshly imported plugin."""
    home = _fresh_home(tmp_path_factory)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    for mod in [m for m in list(sys.modules) if m.startswith("hermes_cli")]:
        del sys.modules[mod]
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as _kbc
    assert home in str(kb.kanban_db_path()), kb.kanban_db_path()

    spec = importlib.util.spec_from_file_location("kp_under_test", PLUGIN_SRC)
    kp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kp)
    kp.ARTIFACT_STATE_FN = lambda url: (_ for _ in ()).throw(
        AssertionError("test did not install a fake transport")
    )
    yield kb, kp


def fake_state(**kw):
    base = {"merged": False, "deployed": False, "approved": True,
            "state": "OPEN", "detail": "fake"}
    base.update(kw)
    return lambda url: dict(base)


def titles(kb, conn):
    return [r[0] for r in conn.execute("select title from tasks order by created_at").fetchall()]


def pipeline_titles(kb, conn):
    return [t for t in titles(kb, conn) if t.startswith("pipeline:")]


def comment_bodies(kb, conn, task_id):
    return [c.body for c in kb.list_comments(conn, task_id)]


# ---------------------------------------------------------------------------
# 1 + 2. the real incident: current artifact wins over a historical body URL,
#        and an already-delivered completion creates nothing at all.
# ---------------------------------------------------------------------------

def test_delivered_current_artifact_beats_historical_url_and_creates_zero(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(merged=True, deployed=True, state="MERGED",
                                      detail="merge=6dc8964d8c35 live=36a655165c3b behind_by=0")
    with _kbc.connect() as conn:
        src = kb.create_task(
            conn, title="P1 — Make new tenant configuration valid",
            assignee="software-engineer",
            body=("CURRENT ACCEPTANCE / SCOPE:\nReuse existing PR " + PR_CURRENT + ".\n\n"
                  "--- Historical task context; obsolete instructions do not override scope above ---\n"
                  "Earlier attempt lived at " + PR_HISTORICAL + "\n"),
        )
        # a six-day-old quoted comment link — the exact selector that misfired
        kb.add_comment(conn, src, "reviewer", "see " + PR_HISTORICAL + " for background")
        kb.complete_task(conn, src, result="delivered", summary="done", fire_lifecycle_hook=False)

    kp._on_completed(task_id=src)

    with _kbc.connect() as conn:
        assert pipeline_titles(kb, conn) == [], titles(kb, conn)
        mine = "\n".join(b for b in comment_bodies(kb, conn, src)
                         if b.startswith("[kanban-pipeline:"))
        assert "[kanban-pipeline:already-delivered:" in mine
        assert PR_CURRENT in mine                   # reasoned about the current artifact
        assert "pull/4841" not in mine              # never selected the historical one
        rows = conn.execute("select count(*) from tasks").fetchone()[0]
        assert rows == 1


def test_historical_url_alone_never_selects_a_target(env):
    """Historical region + comments are the ONLY place a PR appears -> no work."""
    kb, kp = env
    calls = []

    def _probe(url):
        calls.append(url)
        return {"merged": False, "deployed": False, "approved": True, "state": "OPEN",
                "detail": "fake"}

    kp.ARTIFACT_STATE_FN = _probe
    with _kbc.connect() as conn:
        src = kb.create_task(
            conn, title="impl thing", assignee="software-engineer",
            body=("Current scope: finish the config fix.\n"
                  "--- Historical task context ---\n" + PR_HISTORICAL + "\n"),
        )
        kb.add_comment(conn, src, "reviewer", "approved " + PR_HISTORICAL)
        kb.complete_task(conn, src, summary="finished the config fix",
                         fire_lifecycle_hook=False)

    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        assert pipeline_titles(kb, conn) == []
    assert calls == [], "a historical link must not even be probed"


# ---------------------------------------------------------------------------
# 3. ambiguity -> zero + observable notice
# ---------------------------------------------------------------------------

def test_ambiguous_current_metadata_creates_zero_and_is_observable(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state()
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="two artifacts", assignee="software-engineer",
                             body="nothing here")
        kb.complete_task(conn, src,
                         summary="landed %s and %s" % (PR_CURRENT, PR_OTHER),
                         fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        assert pipeline_titles(kb, conn) == []
        bodies = "\n".join(comment_bodies(kb, conn, src))
        assert "[kanban-pipeline:ambiguous-artifact:" in bodies
        assert "#4878" in bodies and "#4949" in bodies
        assert "not a clearance" in bodies


# ---------------------------------------------------------------------------
# 4. probe / metadata read failure -> zero + observable notice
# ---------------------------------------------------------------------------

def test_probe_failure_creates_zero_and_is_observable(env):
    kb, kp = env

    def _boom(url):
        raise kp.ProbeError("read-only artifact probe exit 1: gh: connection refused")

    kp.ARTIFACT_STATE_FN = _boom
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="opened " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        assert pipeline_titles(kb, conn) == []
        bodies = "\n".join(comment_bodies(kb, conn, src))
        assert "[kanban-pipeline:artifact-probe-failed:" in bodies
        assert "connection refused" in bodies


def test_run_metadata_read_failure_creates_zero_and_is_observable(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state()
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="opened " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)

    real_latest = kb.latest_run
    kb.latest_run = lambda conn, tid: (_ for _ in ()).throw(RuntimeError("runs table unreadable"))
    try:
        kp._on_completed(task_id=src)
    finally:
        kb.latest_run = real_latest
    with _kbc.connect() as conn:
        assert pipeline_titles(kb, conn) == []
        bodies = "\n".join(comment_bodies(kb, conn, src))
        assert "[kanban-pipeline:metadata-read-failed:" in bodies
        assert "runs table unreadable" in bodies


def test_deployment_state_unknown_creates_zero(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(merged=True, deployed=None, state="MERGED",
                                      detail="no successful push run to compare against")
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="merged " + PR_CURRENT)
        kb.complete_task(conn, src, summary="merged " + PR_CURRENT, fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        assert pipeline_titles(kb, conn) == []
        assert "[kanban-pipeline:deployment-state-unknown:" in "\n".join(
            comment_bodies(kb, conn, src))


# ---------------------------------------------------------------------------
# 5. existing canonical downstream owner wins
# ---------------------------------------------------------------------------

def test_existing_canonical_owner_is_reused_not_duplicated(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(merged=True, deployed=False, state="MERGED",
                                      detail="merge=abc behind_by=3")
    with _kbc.connect() as conn:
        owner = kb.create_task(conn, title="release convergence (canonical)",
                               assignee="software-engineer",
                               body="owns deployment of " + PR_CURRENT)
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="merged " + PR_CURRENT)
        kb.complete_task(conn, src, summary="merged " + PR_CURRENT, fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        assert pipeline_titles(kb, conn) == []
        bodies = "\n".join(comment_bodies(kb, conn, src))
        assert "[kanban-pipeline:existing-owner:" in bodies
        assert owner in bodies


# ---------------------------------------------------------------------------
# 6 + 7. the two legitimate handoffs still work, each with only its own phase
# ---------------------------------------------------------------------------

def test_fresh_approved_open_pr_creates_merge_gate_with_linked_deploy(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN", approved=True,
                                      detail="state=OPEN reviewDecision=APPROVED")
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl feature", assignee="software-engineer",
                             body="PR " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        rows = conn.execute(
            "select id,title,assignee,status from tasks order by created_at").fetchall()
        got = [r[1] for r in rows if r[1].startswith("pipeline:")]
        assert got == ["pipeline: merge-gate PR #4878",
                       "pipeline: deploy+live-check PR #4878"], got
        gate = [r for r in rows if r[1].endswith("merge-gate PR #4878")][0]
        dep = [r for r in rows if r[1].startswith("pipeline: deploy")][0]
        assert gate[2] == "reviewer" and dep[2] == "software-engineer"
        links = [tuple(l) for l in conn.execute(
            "select parent_id,child_id from task_links").fetchall()]
        assert (src, gate[0]) in links and (gate[0], dep[0]) in links
        assert dep[3] != "ready", "deploy hop must wait behind the merge gate"


def test_merged_but_not_deployed_creates_deploy_only(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(merged=True, deployed=False, state="MERGED",
                                      detail="merge=deadbeef1234 live=cafebabe5678 behind_by=4")
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl feature", assignee="software-engineer",
                             body="PR " + PR_CURRENT)
        kb.complete_task(conn, src, summary="merged deadbeef1234 " + PR_CURRENT,
                         fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        got = pipeline_titles(kb, conn)
        assert got == ["pipeline: deploy+live-check PR #4878"], got
        assert not any("merge-gate" in t for t in got), "merged artifact must not spawn a merge card"
        dep = conn.execute(
            "select id,body,status from tasks where title like 'pipeline: deploy%'").fetchone()
        assert "merge gate not owed" in dep[1]
        links = [tuple(l) for l in conn.execute(
            "select parent_id,child_id from task_links").fetchall()]
        assert (src, dep[0]) in links


def test_closed_unmerged_artifact_creates_zero(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="CLOSED", approved=False, detail="state=CLOSED")
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="PR " + PR_CURRENT)
        kb.complete_task(conn, src, summary="abandoned " + PR_CURRENT, fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        assert pipeline_titles(kb, conn) == []
        assert "[kanban-pipeline:artifact-not-open:" in "\n".join(comment_bodies(kb, conn, src))


# ---------------------------------------------------------------------------
# 8 + 9. idempotence under repeated and concurrent completion events
# ---------------------------------------------------------------------------

def test_repeated_completion_events_create_no_duplicates(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN")
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="PR " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)
    for _ in range(4):
        kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        got = pipeline_titles(kb, conn)
        assert got.count("pipeline: merge-gate PR #4878") == 1, got
        assert got.count("pipeline: deploy+live-check PR #4878") == 1, got
        notices = [b for b in comment_bodies(kb, conn, src)
                   if "[kanban-pipeline:existing-owner:" in b]
        assert len(notices) <= 1, "notices must be deduplicated too"


def test_concurrent_completion_events_create_no_duplicates(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN")
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="PR " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)

    barrier = threading.Barrier(6)
    errors = []

    def _fire():
        try:
            barrier.wait(timeout=30)
            kp._on_completed(task_id=src)
        except Exception as exc:  # must never escape the hook, but assert anyway
            errors.append(exc)

    threads = [threading.Thread(target=_fire) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert errors == [], errors
    with _kbc.connect() as conn:
        got = pipeline_titles(kb, conn)
        assert got.count("pipeline: merge-gate PR #4878") == 1, got
        assert got.count("pipeline: deploy+live-check PR #4878") == 1, got


# ---------------------------------------------------------------------------
# 10. no recursive chain from a pipeline card
# ---------------------------------------------------------------------------

def test_pipeline_card_completion_never_chains(env):
    kb, kp = env
    probed = []
    kp.ARTIFACT_STATE_FN = lambda url: probed.append(url) or fake_state(state="OPEN")(url)
    with _kbc.connect() as conn:
        gate = kb.create_task(conn, title="pipeline: merge-gate PR #4878", assignee="reviewer",
                              body="Merge gate for " + PR_CURRENT)
        kb.complete_task(conn, gate, summary="merged abc123 " + PR_CURRENT,
                         fire_lifecycle_hook=False)
    kp._on_completed(task_id=gate)
    with _kbc.connect() as conn:
        assert pipeline_titles(kb, conn) == ["pipeline: merge-gate PR #4878"]
    assert probed == []


# ---------------------------------------------------------------------------
# 11. tier precedence — structured run metadata outranks body text
# ---------------------------------------------------------------------------

def test_structured_run_metadata_outranks_body_text(env):
    kb, kp = env
    seen = []
    kp.ARTIFACT_STATE_FN = lambda url: seen.append(url) or fake_state(state="OPEN")(url)
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="old draft lived at " + PR_HISTORICAL)
        kb.complete_task(conn, src, metadata={"pr_url": PR_CURRENT},
                         summary="done", fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    assert seen == [PR_CURRENT], seen
    with _kbc.connect() as conn:
        assert pipeline_titles(kb, conn) == ["pipeline: merge-gate PR #4878",
                                             "pipeline: deploy+live-check PR #4878"]


# ---------------------------------------------------------------------------
# 12. board mutation fails open, with a specific automation error
# ---------------------------------------------------------------------------

def test_board_mutation_failure_never_raises(env, caplog):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(merged=True, deployed=True, state="MERGED", detail="x")
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="PR " + PR_CURRENT)
        kb.complete_task(conn, src, summary="merged " + PR_CURRENT, fire_lifecycle_hook=False)
    real_add = kb.add_comment
    kb.add_comment = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("board is read-only"))
    try:
        with caplog.at_level("WARNING"):
            kp._on_completed(task_id=src)  # must not raise
    finally:
        kb.add_comment = real_add
    assert any("board is read-only" in r.getMessage() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]


def test_create_task_failure_never_raises(env, caplog):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN")
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="PR " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)
    real_create = kb.create_task
    kb.create_task = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("create refused"))
    try:
        with caplog.at_level("WARNING"):
            kp._on_completed(task_id=src)
    finally:
        kb.create_task = real_create
    assert any("create refused" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 13. generated instructions honour the sanctioned merge gate and stay in scope
# ---------------------------------------------------------------------------

def test_generated_instructions_honour_merge_gate_and_scope(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN")
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="PR " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        gate_body = conn.execute(
            "select body from tasks where title like 'pipeline: merge-gate%'").fetchone()[0]
        dep_body = conn.execute(
            "select body from tasks where title like 'pipeline: deploy%'").fetchone()[0]
    assert "scripts/safe-merge.sh 4878 --squash --delete-branch" in gate_body
    assert "--admin" in gate_body and "forbidden" in gate_body
    assert "gh pr merge 4878" not in gate_body, "must never instruct a raw merge"
    # B3: approval comes from the reviews API; reviewDecision is corroboration
    assert "pulls/4878/reviews" in gate_body
    assert "Require: reviewDecision=APPROVED" not in gate_body
    assert "CORROBORATION ONLY" in gate_body
    assert "headRefOid" in gate_body
    assert "DISMISSED" in gate_body and "superseded" in gate_body
    # M2: the deploy hop is authorised by the CURRENT SCOPE region only
    assert "CURRENT SCOPE region of source card" in dep_body
    assert "VOID" in dep_body
    assert "explicit human approval" in dep_body
    # the template defers to the source card instead of overriding its acceptance
    for body in (gate_body, dep_body):
        assert src in body
        assert "authoritative" in body or "The authority for this work" in body
    assert "Do NOT provision new tenants" in dep_body
    assert "billing/financial" in dep_body
    # no unscoped live probe is baked into the default template
    assert "/v1/responses" not in dep_body and "litellm/auto" not in dep_body


# ---------------------------------------------------------------------------
# 14. the REAL runtime path: real complete_task -> real lifecycle hook ->
#     discovered plugin -> real default gh transport (faked executable).
# ---------------------------------------------------------------------------

FAKE_GH = r"""#!/usr/bin/env python3
import json, sys
a = sys.argv[1:]
if a[:2] == ["pr", "view"]:
    print(json.dumps({"state": "OPEN", "mergedAt": None, "mergeCommit": None,
                      "reviewDecision": "APPROVED"}))
elif a[:1] == ["api"]:
    print(json.dumps({"workflow_runs": []}))
else:
    sys.exit(3)
"""


def test_real_hook_through_complete_task_and_default_transport(tmp_path_factory):
    home = _fresh_home(tmp_path_factory)
    plug = os.path.join(home, "plugins", "kanban-pipeline")
    os.makedirs(plug, exist_ok=True)
    shutil.copy(PLUGIN_SRC, os.path.join(plug, "__init__.py"))
    shutil.copy(PLUGIN_YAML, os.path.join(plug, "plugin.yaml"))
    with open(os.path.join(home, "config.yaml"), "w") as fh:
        fh.write("plugins:\n  enabled: [kanban-pipeline]\n" + CONFIG_YAML)

    bindir = str(tmp_path_factory.mktemp("fakebin"))
    ghp = os.path.join(bindir, "gh")
    with open(ghp, "w") as fh:
        fh.write(FAKE_GH)
    os.chmod(ghp, os.stat(ghp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    script = r'''
import os, sys
sys.path.insert(0, %(repo)r)
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as _kbc
from hermes_cli import plugins
plugins.discover_plugins(force=True)
assert plugins.has_hook("kanban_task_completed"), "plugin hook not discovered"
with _kbc.connect() as conn:
    src = kb.create_task(conn, title="impl real", assignee="software-engineer",
                         body="PR %(pr)s")
    kb.complete_task(conn, src, summary="opened %(pr)s")   # real lifecycle hook fires
with _kbc.connect() as conn:
    rows = [r[0] for r in conn.execute("select title from tasks order by created_at")]
print(repr(rows))
''' % {"repo": REPO, "pr": PR_CURRENT}

    envv = dict(os.environ)
    envv["HERMES_HOME"] = home
    envv["PATH"] = bindir + os.pathsep + envv["PATH"]
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=envv, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout.strip().splitlines()[-1]
    assert "pipeline: merge-gate PR #4878" in out, out
    assert "pipeline: deploy+live-check PR #4878" in out, out
    assert out.count("merge-gate") == 1, out


# ===========================================================================
# Regressions for the independent review of PR #3 (t_cf118fa9): every
# counterexample that went RED against the previous revision is committed
# here. Source counterexamples: /tmp/rev_cf118fa9/cx{1,2,456,7,8}.
# ===========================================================================

FAKE_GH_FIXTURED = r"""#!/usr/bin/env python3
import json, os, sys
fx = json.load(open(os.environ["KP_FAKE_GH_FIXTURE"]))
a = sys.argv[1:]
log = os.environ.get("KP_FAKE_GH_LOG")
if log:
    with open(log, "a") as fh:
        fh.write(json.dumps(a) + "\n")
def out(v):
    print(json.dumps(v))
    raise SystemExit(0)
if a[:2] == ["pr", "view"]:
    out(fx["pr"])
if a[:1] == ["api"]:
    q = a[1]
    if "/statuses" in q:
        out(fx.get("statuses", []))
    if "deployments" in q:
        out(fx.get("deployments", []))
    if "/jobs" in q:
        out(fx.get("jobs", {"jobs": []}))
    if "actions/runs" in q:
        out(fx.get("runs", {"workflow_runs": []}))
    if "compare" in q:
        out(fx.get("compare", {"behind_by": None}))
sys.exit(3)
"""

MERGE_SHA = "6dc8964d8c35aa11bb22cc33dd44ee55ff667788"
OTHER_SHA = "36a655165c3b99aa88bb77cc66dd55ee44332211"


def _install_fake_gh(tmp_path_factory, fixture):
    """Put a fixtured, argv-logging fake `gh` first on PATH. No network."""
    bindir = str(tmp_path_factory.mktemp("fakebin"))
    ghp = os.path.join(bindir, "gh")
    with open(ghp, "w") as fh:
        fh.write(FAKE_GH_FIXTURED)
    os.chmod(ghp, os.stat(ghp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    fxp = os.path.join(bindir, "fixture.json")
    with open(fxp, "w") as fh:
        json.dump(fixture, fh)
    logp = os.path.join(bindir, "argv.jsonl")
    os.environ["KP_FAKE_GH_FIXTURE"] = fxp
    os.environ["KP_FAKE_GH_LOG"] = logp
    os.environ["PATH"] = bindir + os.pathsep + os.environ["PATH"]
    return logp


def _merged_pr(base="main"):
    return {"state": "MERGED", "mergedAt": "2026-09-10T00:00:00Z",
            "mergeCommit": {"oid": MERGE_SHA}, "reviewDecision": "APPROVED",
            "baseRefName": base}


POLICY = {"deploy_branch": "main", "deploy_workflow": "deploy.yml", "deploy_job": "deploy"}


# --- B1: deployment evidence must be qualified ------------------------------

def test_green_push_run_on_another_workflow_is_never_deployment_evidence(env, tmp_path_factory):
    """CX1: newest successful push run is docs-lint on an unrelated branch."""
    kb, kp = env
    log = _install_fake_gh(tmp_path_factory, {
        "pr": _merged_pr(),
        # branch-filtered query still returns only a docs-lint workflow
        "runs": {"workflow_runs": [{"id": 123, "name": "docs-lint",
                                    "path": ".github/workflows/docs-lint.yml",
                                    "head_branch": "docs/typo-fix", "head_sha": OTHER_SHA,
                                    "event": "push", "status": "completed",
                                    "conclusion": "success"}]},
        "compare": {"behind_by": 0, "ahead_by": 2, "status": "ahead"},
    })
    state = kp._gh_artifact_state(PR_CURRENT, policy=POLICY)
    assert state["merged"] is True
    assert state["deployed"] is None, state
    assert "UNKNOWN" in state["detail"]

    calls = [json.loads(l) for l in open(log)]
    api = [c[1] for c in calls if c[:1] == ["api"]]
    assert any("branch=main" in q for q in api), api          # target branch scoped
    assert not any("compare" in q for q in api), api          # ancestry never alone

    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="CURRENT SCOPE: deliver " + PR_CURRENT)
        kb.complete_task(conn, src, result="delivered", summary="merged " + PR_CURRENT,
                         fire_lifecycle_hook=False)
    kp.ARTIFACT_STATE_FN = kp._gh_artifact_state   # the real default transport
    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        bodies = "\n".join(comment_bodies(kb, conn, src))
        assert "[kanban-pipeline:already-delivered:" not in bodies
        assert "[kanban-pipeline:deployment-state-unknown:" in bodies
        assert "not delivered" in bodies


def test_skipped_deploy_job_is_never_deployment_evidence(env, tmp_path_factory):
    kb, kp = env
    _install_fake_gh(tmp_path_factory, {
        "pr": _merged_pr(),
        "runs": {"workflow_runs": [{"id": 9, "name": "deploy.yml", "path":
                                    ".github/workflows/deploy.yml", "head_branch": "main",
                                    "head_sha": MERGE_SHA, "conclusion": "success"}]},
        "jobs": {"jobs": [{"name": "deploy", "conclusion": "skipped"}]},
    })
    state = kp._gh_artifact_state(PR_CURRENT, policy=POLICY)
    assert state["deployed"] is None, state


def test_deploy_workflow_run_on_wrong_workflow_name_is_not_evidence(env, tmp_path_factory):
    kb, kp = env
    _install_fake_gh(tmp_path_factory, {
        "pr": _merged_pr(),
        "runs": {"workflow_runs": [{"id": 9, "name": "CI / Whisper STT Gate",
                                    "path": ".github/workflows/ci.yml",
                                    "head_branch": "main", "head_sha": MERGE_SHA,
                                    "conclusion": "success"}]},
        "jobs": {"jobs": [{"name": "deploy", "conclusion": "success"}]},
    })
    assert kp._gh_artifact_state(PR_CURRENT, policy=POLICY)["deployed"] is None


def test_unconfigured_repo_policy_yields_unknown_not_deployed(env, tmp_path_factory):
    kb, kp = env
    _install_fake_gh(tmp_path_factory, {
        "pr": _merged_pr(),
        "runs": {"workflow_runs": [{"id": 9, "name": "deploy.yml", "head_branch": "main",
                                    "head_sha": MERGE_SHA, "conclusion": "success"}]},
    })
    state = kp._gh_artifact_state(PR_CURRENT, policy={})   # nothing configured
    assert state["deployed"] is None
    assert "deploy_workflow" in state["detail"]


def test_qualified_deploy_job_at_merge_sha_is_evidence_worded_accurately(env, tmp_path_factory):
    kb, kp = env
    _install_fake_gh(tmp_path_factory, {
        "pr": _merged_pr(),
        "runs": {"workflow_runs": [{"id": 77, "name": "deploy.yml", "head_branch": "main",
                                    "head_sha": MERGE_SHA, "conclusion": "success"}]},
        "jobs": {"jobs": [{"name": "deploy", "conclusion": "success"}]},
    })
    state = kp._gh_artifact_state(PR_CURRENT, policy=POLICY)
    assert state["deployed"] is True, state
    assert "deploy job" in state["detail"] and MERGE_SHA[:12] in state["detail"]

    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="CURRENT SCOPE: " + PR_CURRENT)
        kb.complete_task(conn, src, summary="merged " + PR_CURRENT, fire_lifecycle_hook=False)
    kp.ARTIFACT_STATE_FN = kp._gh_artifact_state   # the real default transport
    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        assert pipeline_titles(kb, conn) == []
        bodies = "\n".join(comment_bodies(kb, conn, src))
        assert "[kanban-pipeline:already-delivered:" in bodies
        assert "not a verified healthy customer response" in bodies


def test_ancestry_is_corroboration_on_top_of_a_qualified_run_only(env, tmp_path_factory):
    """A qualified deploy run at a LATER sha delivers the merge; ancestry alone never does."""
    kb, kp = env
    _install_fake_gh(tmp_path_factory, {
        "pr": _merged_pr(),
        "runs": {"workflow_runs": [{"id": 78, "name": "deploy.yml", "head_branch": "main",
                                    "head_sha": OTHER_SHA, "conclusion": "success"}]},
        "jobs": {"jobs": [{"name": "deploy", "conclusion": "success"}]},
        "compare": {"behind_by": 0},
    })
    assert kp._gh_artifact_state(PR_CURRENT, policy=POLICY)["deployed"] is True
    _install_fake_gh(tmp_path_factory, {
        "pr": _merged_pr(),
        "runs": {"workflow_runs": [{"id": 79, "name": "deploy.yml", "head_branch": "main",
                                    "head_sha": OTHER_SHA, "conclusion": "success"}]},
        "jobs": {"jobs": [{"name": "deploy", "conclusion": "success"}]},
        "compare": {"behind_by": 3},          # merge NOT contained in the deploy target
    })
    assert kp._gh_artifact_state(PR_CURRENT, policy=POLICY)["deployed"] is None


def test_deployments_api_success_for_exact_sha_is_recorded_deployment(env, tmp_path_factory):
    kb, kp = env
    policy = dict(POLICY, use_deployments=True, deploy_environment="production")
    _install_fake_gh(tmp_path_factory, {
        "pr": _merged_pr(),
        "deployments": [{"id": 5, "sha": MERGE_SHA, "environment": "production"}],
        "statuses": [{"state": "success"}],
    })
    state = kp._gh_artifact_state(PR_CURRENT, policy=policy)
    assert state["deployed"] is True
    assert "RECORDED" in state["detail"]
    assert "not a verified healthy customer response" in state["detail"]


def test_deployment_recorded_for_a_different_sha_is_not_evidence(env, tmp_path_factory):
    kb, kp = env
    policy = dict(use_deployments=True, deploy_environment="production")
    _install_fake_gh(tmp_path_factory, {
        "pr": _merged_pr(),
        "deployments": [{"id": 5, "sha": OTHER_SHA, "environment": "production"}],
        "statuses": [{"state": "success"}],
    })
    assert kp._gh_artifact_state(PR_CURRENT, policy=policy)["deployed"] is None


def test_probe_transport_error_during_deploy_evidence_is_unknown(env, tmp_path_factory):
    """gh failing on the runs call must degrade to UNKNOWN, not to delivered."""
    kb, kp = env
    _install_fake_gh(tmp_path_factory, {"pr": _merged_pr()})   # api calls exit 3
    state = kp._gh_artifact_state(PR_CURRENT, policy=POLICY)
    assert state["merged"] is True and state["deployed"] is None
    assert "unavailable" in state["detail"] or "UNKNOWN" in state["detail"]


# --- B2: ownership is exact, per-phase, current, and completes partials ------

def test_pr_number_prefix_collision_is_not_ownership(env):
    """CX2a: .../pull/1 is a substring of .../pull/10 — identities are parsed."""
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN")
    base = "https://github.com/VibeTechnologies/AgentPod/pull/"
    with _kbc.connect() as conn:
        kb.create_task(conn, title="pipeline: merge-gate PR #10", assignee="reviewer",
                       body="Merge gate for " + base + "10")
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="CURRENT SCOPE: " + base + "1")
        kb.complete_task(conn, src, summary="opened " + base + "1", fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        got = pipeline_titles(kb, conn)
        assert "pipeline: merge-gate PR #1" in got, got
        assert "pipeline: deploy+live-check PR #1" in got, got


def test_archived_quote_never_suppresses_chaining(env):
    """CX2b: an archived card that merely quotes the URL is not an owner."""
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN")
    with _kbc.connect() as conn:
        stale = kb.create_task(conn, title="stale autogenerated card", assignee="reviewer",
                               body="pipeline: deploy+live-check for " + PR_HISTORICAL)
        kb.archive_task(conn, stale)
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="CURRENT SCOPE: " + PR_HISTORICAL)
        kb.complete_task(conn, src, summary="opened " + PR_HISTORICAL,
                         fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        got = pipeline_titles(kb, conn)
        assert got == ["pipeline: merge-gate PR #4841",
                       "pipeline: deploy+live-check PR #4841"], got


def test_historical_quote_in_a_live_card_is_not_ownership(env):
    """CX2c: a live card whose CURRENT scope is a different artifact owns nothing here."""
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN")
    with _kbc.connect() as conn:
        kb.create_task(conn, title="unrelated live work", assignee="software-engineer",
                       body=("CURRENT SCOPE: deliver " + PR_OTHER + "\n\n"
                             "--- Historical task context; obsolete instructions do not "
                             "override scope above ---\nbackground: see " + PR_HISTORICAL))
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="CURRENT SCOPE: " + PR_HISTORICAL)
        kb.complete_task(conn, src, summary="opened " + PR_HISTORICAL,
                         fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        got = pipeline_titles(kb, conn)
        assert got.count("pipeline: merge-gate PR #4841") == 1, got
        assert got.count("pipeline: deploy+live-check PR #4841") == 1, got


def test_partial_chain_is_adopted_and_completed_without_duplicates(env):
    """CX7: event 1 dies after the merge gate; the retry mints ONLY the missing hop."""
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN")
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="CURRENT SCOPE: " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)

    real_create = kb.create_task
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("process died mid-chain")
        return real_create(*a, **k)

    kb.create_task = flaky
    try:
        kp._on_completed(task_id=src)
    finally:
        kb.create_task = real_create
    with _kbc.connect() as conn:
        after1 = pipeline_titles(kb, conn)
    assert after1 == ["pipeline: merge-gate PR #4878"], after1

    kp._on_completed(task_id=src)            # the natural retry
    with _kbc.connect() as conn:
        after2 = pipeline_titles(kb, conn)
        assert after2.count("pipeline: merge-gate PR #4878") == 1, after2
        assert after2.count("pipeline: deploy+live-check PR #4878") == 1, after2
        assert "adopted existing merge hop" in "\n".join(comment_bodies(kb, conn, src))


def test_done_merge_gate_does_not_suppress_the_owed_deploy_hop(env):
    """A completed, verified phase covers ITSELF only."""
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN")
    with _kbc.connect() as conn:
        gate = kb.create_task(conn, title="pipeline: merge-gate PR #4878", assignee="reviewer",
                              body="Merge gate for " + PR_CURRENT,
                              idempotency_key=kp._phase_key("merge", kp._identity(PR_CURRENT)))
        kb.complete_task(conn, gate, summary="merged abc123", fire_lifecycle_hook=False)
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="CURRENT SCOPE: " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        got = pipeline_titles(kb, conn)
        assert got.count("pipeline: merge-gate PR #4878") == 1, got
        assert got.count("pipeline: deploy+live-check PR #4878") == 1, got


# --- B3 / M1: merge instructions + repository allow-list --------------------

def test_repo_without_sanctioned_helper_states_a_configuration_gap(env, tmp_path_factory):
    home = _fresh_home(tmp_path_factory, config_yaml=(
        "kanban_pipeline:\n  enabled: true\n  repos:\n"
        "    VibeTechnologies/AgentPod: {}\n"))
    for mod in [m for m in list(sys.modules) if m.startswith("hermes_cli")]:
        del sys.modules[mod]
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as _kbc
    spec = importlib.util.spec_from_file_location("kp_gap", PLUGIN_SRC)
    kp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kp)
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN")
    assert home in str(kb.kanban_db_path())
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="CURRENT SCOPE: " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        gate_body = conn.execute(
            "select body from tasks where title like 'pipeline: merge-gate%'").fetchone()[0]
    assert "OWNER CONFIGURATION GAP" in gate_body
    assert "merge_command` is unset" in gate_body
    assert "--admin" in gate_body and "forbidden" in gate_body
    assert "gh pr merge 4878" not in gate_body
    assert "safe-merge.sh" not in gate_body, "must not invent a command for a generic repo"


def test_foreign_repository_is_out_of_scope_and_never_probed(env):
    """CX5: a cited third-party PR mints nothing and is not even probed."""
    kb, kp = env
    probed = []
    kp.ARTIFACT_STATE_FN = lambda url: probed.append(url) or fake_state(state="OPEN")(url)
    foreign = "https://github.com/some-vendor/unrelated-oss/pull/12"
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="investigate upstream", assignee="software-engineer",
                             body="CURRENT SCOPE: read-only investigation, see " + foreign)
        kb.complete_task(conn, src, summary="investigated, see " + foreign,
                         fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    assert probed == []
    with _kbc.connect() as conn:
        assert pipeline_titles(kb, conn) == []
        bodies = "\n".join(comment_bodies(kb, conn, src))
        assert "[kanban-pipeline:repo-out-of-scope:" in bodies
        assert "allow-list" in bodies


# --- M3: bounded chain lock beside the DB -----------------------------------

def test_chain_lock_lives_beside_the_board_db_and_is_used(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN")
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="CURRENT SCOPE: " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    lock = str(kb.kanban_db_path()) + ".kanban-pipeline.chain.lock"
    assert os.path.exists(lock), lock          # the DEFAULT board is protected too
    with _kbc.connect() as conn:
        assert "lock exclusive" in "\n".join(comment_bodies(kb, conn, src))


def test_chain_lock_timeout_defers_bounded_and_mints_nothing(env, tmp_path_factory):
    """CX4: a wedged holder in another OS process must not block or bypass."""
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN")
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="CURRENT SCOPE: " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)
    lock = str(kb.kanban_db_path()) + ".kanban-pipeline.chain.lock"
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl,sys,time;fh=open(sys.argv[1],'a+');"
         "fcntl.flock(fh.fileno(),fcntl.LOCK_EX);print('held',flush=True);time.sleep(120)",
         lock],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        t0 = time.time()
        kp._on_completed(task_id=src)          # cfg lock_timeout default 5.0s
        waited = time.time() - t0
    finally:
        holder.kill()
        holder.wait(timeout=30)
    assert waited < 20, waited                 # bounded, never an indefinite hang
    with _kbc.connect() as conn:
        assert pipeline_titles(kb, conn) == [], "must not mint outside the atomic section"
        assert "[kanban-pipeline:chain-lock-busy:" in "\n".join(comment_bodies(kb, conn, src))


# --- M4 / A1 / A2: notices, board payload, delivery-shaped completions ------

def test_changed_failure_condition_is_not_swallowed_by_the_first_one(env):
    """CX6: dedup is on code + scope + condition digest, not on the code alone."""
    kb, kp = env
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="CURRENT SCOPE: " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)

    kp.ARTIFACT_STATE_FN = lambda u: (_ for _ in ()).throw(RuntimeError("gh auth token expired"))
    kp._on_completed(task_id=src)
    kp._on_completed(task_id=src)              # identical condition -> still one
    with _kbc.connect() as conn:
        first = [b for b in comment_bodies(kb, conn, src) if b.startswith("[kanban-pipeline:")]
    assert len(first) == 1, first

    kp.ARTIFACT_STATE_FN = lambda u: (_ for _ in ()).throw(
        RuntimeError("artifact repository was DELETED - escalate now"))
    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        bodies = [b for b in comment_bodies(kb, conn, src) if b.startswith("[kanban-pipeline:")]
    assert len(bodies) == 2, bodies
    assert any("DELETED" in b for b in bodies)
    assert any("token expired" in b for b in bodies), "the last good state is never erased"


def test_board_payload_is_honoured_from_a_worker_thread(env):
    """CX8 arm 2: ContextVars are not inherited — the explicit board must be used."""
    kb, kp = env
    kb.create_board("alt", name="alt board")
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN")
    with kb.scoped_current_board("alt"):
        with _kbc.connect() as conn:
            src = kb.create_task(conn, title="impl on alt", assignee="software-engineer",
                                 body="CURRENT SCOPE: " + PR_CURRENT)
            kb.complete_task(conn, src, summary="opened " + PR_CURRENT,
                             fire_lifecycle_hook=False)
    t = threading.Thread(target=lambda: kp._on_completed(task_id=src, board="alt"))
    t.start()
    t.join(timeout=60)
    with kb.scoped_current_board("alt"):
        with _kbc.connect() as conn:
            on_alt = pipeline_titles(kb, conn)
    with _kbc.connect() as conn:
        on_default = pipeline_titles(kb, conn)
    assert on_alt == ["pipeline: merge-gate PR #4878",
                      "pipeline: deploy+live-check PR #4878"], on_alt
    assert on_default == [], on_default        # never cross-routed to the default board


def test_invalid_or_unknown_board_payload_is_refused(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN")
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="CURRENT SCOPE: " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)
    kp._on_completed(task_id=src, board="../../etc")       # invalid slug
    kp._on_completed(task_id=src, board="no-such-board")   # valid slug, absent board
    with _kbc.connect() as conn:
        assert pipeline_titles(kb, conn) == []


def test_artifactless_completion_is_quiet_but_delivery_shaped_gets_one_notice(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN")
    with _kbc.connect() as conn:
        quiet = kb.create_task(conn, title="write the doc", assignee="software-engineer",
                               body="no artifact here")
        kb.complete_task(conn, quiet, summary="wrote the doc", fire_lifecycle_hook=False)
        shaped = kb.create_task(conn, title="ship it", assignee="software-engineer",
                                body="no artifact here")
        kb.complete_task(conn, shaped, summary="merged and deployed the pull request",
                         fire_lifecycle_hook=False)
    kp._on_completed(task_id=quiet)
    kp._on_completed(task_id=shaped)
    kp._on_completed(task_id=shaped)
    with _kbc.connect() as conn:
        assert [b for b in comment_bodies(kb, conn, quiet)
                if b.startswith("[kanban-pipeline:")] == []
        notices = [b for b in comment_bodies(kb, conn, shaped)
                   if b.startswith("[kanban-pipeline:")]
        assert len(notices) == 1, notices
        assert "[kanban-pipeline:artifact-unparseable:" in notices[0]


def test_plugin_is_inert_without_explicit_config(tmp_path_factory):
    """Containment: no config -> disabled, zero cards, zero notices."""
    home = _fresh_home(tmp_path_factory, config_yaml="")
    for mod in [m for m in list(sys.modules) if m.startswith("hermes_cli")]:
        del sys.modules[mod]
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as _kbc
    spec = importlib.util.spec_from_file_location("kp_off", PLUGIN_SRC)
    kp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kp)
    assert home in str(kb.kanban_db_path())
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="CURRENT SCOPE: " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with _kbc.connect() as conn:
        assert pipeline_titles(kb, conn) == []
        assert comment_bodies(kb, conn, src) == []


# --- cross-OS-PROCESS concurrency and crash/retry ---------------------------

_CHILD_HEAD = r'''
import importlib.util, os, sys, time
sys.path.insert(0, %(repo)r)
os.environ["HERMES_HOME"] = %(home)r
for v in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
    os.environ.pop(v, None)
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as _kbc
spec = importlib.util.spec_from_file_location("kp_child", %(src)r)
kp = importlib.util.module_from_spec(spec); spec.loader.exec_module(kp)
kp.ARTIFACT_STATE_FN = lambda u: {"merged": False, "deployed": False, "approved": True,
                                  "state": "OPEN", "detail": "fake"}
'''


def _child(home, body):
    return (_CHILD_HEAD % {"repo": REPO, "home": home, "src": PLUGIN_SRC}) + body


def _seed_completed_source(kb, board=None):
    with _kbc.connect(board=board) as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="CURRENT SCOPE: " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)
    return src


@pytest.mark.parametrize("board", [None, "worker"])
def test_cross_process_completion_events_create_no_duplicates(tmp_path_factory, board):
    """CX3: real OS processes (not threads) on the DEFAULT and a named board."""
    home = _fresh_home(tmp_path_factory)
    for mod in [m for m in list(sys.modules) if m.startswith("hermes_cli")]:
        del sys.modules[mod]
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as _kbc
    if board:
        kb.create_board(board, name=board)
    src = _seed_completed_source(kb, board)

    start = time.time() + 3.0
    body = ("t = %r\nwhile time.time() < t: time.sleep(0.005)\n"
            "kp._on_completed(task_id=%r, board=%r)\n" % (start, src, board))
    procs = [subprocess.Popen([sys.executable, "-c", _child(home, body)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for _ in range(8)]
    outs = [p.communicate(timeout=300) for p in procs]
    for p, (o, e) in zip(procs, outs):
        assert p.returncode == 0, e[-800:]
    with _kbc.connect(board=board) as conn:
        got = pipeline_titles(kb, conn)
    assert got.count("pipeline: merge-gate PR #4878") == 1, got
    assert got.count("pipeline: deploy+live-check PR #4878") == 1, got
    lock = str(kb.kanban_db_path(board=board)) + ".kanban-pipeline.chain.lock"
    assert os.path.exists(lock), "the cross-process lock must actually exist: " + lock


def test_cross_process_partial_creation_is_completed_by_a_later_process(tmp_path_factory):
    """Process A is killed between hops; process B completes the chain, once."""
    home = _fresh_home(tmp_path_factory)
    for mod in [m for m in list(sys.modules) if m.startswith("hermes_cli")]:
        del sys.modules[mod]
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as _kbc
    src = _seed_completed_source(kb)

    crash = (
        "real = kb.create_task\n"
        "calls = {'n': 0}\n"
        "def flaky(*a, **k):\n"
        "    calls['n'] += 1\n"
        "    if calls['n'] == 2:\n"
        "        os._exit(9)            # hard kill between the two hops\n"
        "    return real(*a, **k)\n"
        "kb.create_task = flaky\n"
        "kp._on_completed(task_id=%r)\n" % src)
    p = subprocess.run([sys.executable, "-c", _child(home, crash)],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 9, (p.returncode, p.stderr[-500:])
    with _kbc.connect() as conn:
        assert pipeline_titles(kb, conn) == ["pipeline: merge-gate PR #4878"]
    lock = str(kb.kanban_db_path()) + ".kanban-pipeline.chain.lock"
    assert os.path.exists(lock)

    retry = "kp._on_completed(task_id=%r)\n" % src
    p2 = subprocess.run([sys.executable, "-c", _child(home, retry)],
                        capture_output=True, text=True, timeout=300)
    assert p2.returncode == 0, p2.stderr[-800:]
    with _kbc.connect() as conn:
        got = pipeline_titles(kb, conn)
    assert got.count("pipeline: merge-gate PR #4878") == 1, got
    assert got.count("pipeline: deploy+live-check PR #4878") == 1, got


def test_completion_hook_never_hangs_or_breaks_the_transition(env):
    """Fail-open contract: the durable `done` write is committed before us."""
    kb, kp = env

    def _slow_boom(url):
        time.sleep(0.05)
        raise RuntimeError("probe exploded")

    kp.ARTIFACT_STATE_FN = _slow_boom
    with _kbc.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="CURRENT SCOPE: " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=True)
    t0 = time.time()
    kp._on_completed(task_id=src)
    assert time.time() - t0 < 30
    with _kbc.connect() as conn:
        assert kb.get_task(conn, src).status == "done"
