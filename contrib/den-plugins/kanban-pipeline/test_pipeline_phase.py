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

def _fresh_home(tmp_path_factory):
    home = str(tmp_path_factory.mktemp("kp_home"))
    os.environ["HERMES_HOME"] = home
    for v in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME",
              "HERMES_KANBAN_WORKSPACES_ROOT"):
        os.environ.pop(v, None)
    for prof in ("reviewer", "software-engineer", "default"):
        os.makedirs(os.path.join(home, "profiles", prof), exist_ok=True)
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
    with kb.connect() as conn:
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

    with kb.connect() as conn:
        assert pipeline_titles(kb, conn) == [], titles(kb, conn)
        mine = "\n".join(b for b in comment_bodies(kb, conn, src)
                         if b.startswith("[kanban-pipeline:"))
        assert "[kanban-pipeline:already-delivered]" in mine
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
    with kb.connect() as conn:
        src = kb.create_task(
            conn, title="impl thing", assignee="software-engineer",
            body=("Current scope: finish the config fix.\n"
                  "--- Historical task context ---\n" + PR_HISTORICAL + "\n"),
        )
        kb.add_comment(conn, src, "reviewer", "approved " + PR_HISTORICAL)
        kb.complete_task(conn, src, summary="finished the config fix",
                         fire_lifecycle_hook=False)

    kp._on_completed(task_id=src)
    with kb.connect() as conn:
        assert pipeline_titles(kb, conn) == []
    assert calls == [], "a historical link must not even be probed"


# ---------------------------------------------------------------------------
# 3. ambiguity -> zero + observable notice
# ---------------------------------------------------------------------------

def test_ambiguous_current_metadata_creates_zero_and_is_observable(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state()
    with kb.connect() as conn:
        src = kb.create_task(conn, title="two artifacts", assignee="software-engineer",
                             body="nothing here")
        kb.complete_task(conn, src,
                         summary="landed %s and %s" % (PR_CURRENT, PR_OTHER),
                         fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with kb.connect() as conn:
        assert pipeline_titles(kb, conn) == []
        bodies = "\n".join(comment_bodies(kb, conn, src))
        assert "[kanban-pipeline:ambiguous-artifact]" in bodies
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
    with kb.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="opened " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with kb.connect() as conn:
        assert pipeline_titles(kb, conn) == []
        bodies = "\n".join(comment_bodies(kb, conn, src))
        assert "[kanban-pipeline:artifact-probe-failed]" in bodies
        assert "connection refused" in bodies


def test_run_metadata_read_failure_creates_zero_and_is_observable(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state()
    with kb.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="opened " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)

    real_latest = kb.latest_run
    kb.latest_run = lambda conn, tid: (_ for _ in ()).throw(RuntimeError("runs table unreadable"))
    try:
        kp._on_completed(task_id=src)
    finally:
        kb.latest_run = real_latest
    with kb.connect() as conn:
        assert pipeline_titles(kb, conn) == []
        bodies = "\n".join(comment_bodies(kb, conn, src))
        assert "[kanban-pipeline:metadata-read-failed]" in bodies
        assert "runs table unreadable" in bodies


def test_deployment_state_unknown_creates_zero(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(merged=True, deployed=None, state="MERGED",
                                      detail="no successful push run to compare against")
    with kb.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="merged " + PR_CURRENT)
        kb.complete_task(conn, src, summary="merged " + PR_CURRENT, fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with kb.connect() as conn:
        assert pipeline_titles(kb, conn) == []
        assert "[kanban-pipeline:deployment-state-unknown]" in "\n".join(
            comment_bodies(kb, conn, src))


# ---------------------------------------------------------------------------
# 5. existing canonical downstream owner wins
# ---------------------------------------------------------------------------

def test_existing_canonical_owner_is_reused_not_duplicated(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(merged=True, deployed=False, state="MERGED",
                                      detail="merge=abc behind_by=3")
    with kb.connect() as conn:
        owner = kb.create_task(conn, title="release convergence (canonical)",
                               assignee="software-engineer",
                               body="owns deployment of " + PR_CURRENT)
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="merged " + PR_CURRENT)
        kb.complete_task(conn, src, summary="merged " + PR_CURRENT, fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with kb.connect() as conn:
        assert pipeline_titles(kb, conn) == []
        bodies = "\n".join(comment_bodies(kb, conn, src))
        assert "[kanban-pipeline:existing-owner]" in bodies
        assert owner in bodies


# ---------------------------------------------------------------------------
# 6 + 7. the two legitimate handoffs still work, each with only its own phase
# ---------------------------------------------------------------------------

def test_fresh_approved_open_pr_creates_merge_gate_with_linked_deploy(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN", approved=True,
                                      detail="state=OPEN reviewDecision=APPROVED")
    with kb.connect() as conn:
        src = kb.create_task(conn, title="impl feature", assignee="software-engineer",
                             body="PR " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with kb.connect() as conn:
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
    with kb.connect() as conn:
        src = kb.create_task(conn, title="impl feature", assignee="software-engineer",
                             body="PR " + PR_CURRENT)
        kb.complete_task(conn, src, summary="merged deadbeef1234 " + PR_CURRENT,
                         fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with kb.connect() as conn:
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
    with kb.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="PR " + PR_CURRENT)
        kb.complete_task(conn, src, summary="abandoned " + PR_CURRENT, fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with kb.connect() as conn:
        assert pipeline_titles(kb, conn) == []
        assert "[kanban-pipeline:artifact-not-open]" in "\n".join(comment_bodies(kb, conn, src))


# ---------------------------------------------------------------------------
# 8 + 9. idempotence under repeated and concurrent completion events
# ---------------------------------------------------------------------------

def test_repeated_completion_events_create_no_duplicates(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN")
    with kb.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="PR " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)
    for _ in range(4):
        kp._on_completed(task_id=src)
    with kb.connect() as conn:
        got = pipeline_titles(kb, conn)
        assert got.count("pipeline: merge-gate PR #4878") == 1, got
        assert got.count("pipeline: deploy+live-check PR #4878") == 1, got
        notices = [b for b in comment_bodies(kb, conn, src)
                   if "[kanban-pipeline:existing-owner]" in b]
        assert len(notices) <= 1, "notices must be deduplicated too"


def test_concurrent_completion_events_create_no_duplicates(env):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(state="OPEN")
    with kb.connect() as conn:
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
    with kb.connect() as conn:
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
    with kb.connect() as conn:
        gate = kb.create_task(conn, title="pipeline: merge-gate PR #4878", assignee="reviewer",
                              body="Merge gate for " + PR_CURRENT)
        kb.complete_task(conn, gate, summary="merged abc123 " + PR_CURRENT,
                         fire_lifecycle_hook=False)
    kp._on_completed(task_id=gate)
    with kb.connect() as conn:
        assert pipeline_titles(kb, conn) == ["pipeline: merge-gate PR #4878"]
    assert probed == []


# ---------------------------------------------------------------------------
# 11. tier precedence — structured run metadata outranks body text
# ---------------------------------------------------------------------------

def test_structured_run_metadata_outranks_body_text(env):
    kb, kp = env
    seen = []
    kp.ARTIFACT_STATE_FN = lambda url: seen.append(url) or fake_state(state="OPEN")(url)
    with kb.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="old draft lived at " + PR_HISTORICAL)
        kb.complete_task(conn, src, metadata={"pr_url": PR_CURRENT},
                         summary="done", fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    assert seen == [PR_CURRENT], seen
    with kb.connect() as conn:
        assert pipeline_titles(kb, conn) == ["pipeline: merge-gate PR #4878",
                                             "pipeline: deploy+live-check PR #4878"]


# ---------------------------------------------------------------------------
# 12. board mutation fails open, with a specific automation error
# ---------------------------------------------------------------------------

def test_board_mutation_failure_never_raises(env, caplog):
    kb, kp = env
    kp.ARTIFACT_STATE_FN = fake_state(merged=True, deployed=True, state="MERGED", detail="x")
    with kb.connect() as conn:
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
    with kb.connect() as conn:
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
    with kb.connect() as conn:
        src = kb.create_task(conn, title="impl", assignee="software-engineer",
                             body="PR " + PR_CURRENT)
        kb.complete_task(conn, src, summary="opened " + PR_CURRENT, fire_lifecycle_hook=False)
    kp._on_completed(task_id=src)
    with kb.connect() as conn:
        gate_body = conn.execute(
            "select body from tasks where title like 'pipeline: merge-gate%'").fetchone()[0]
        dep_body = conn.execute(
            "select body from tasks where title like 'pipeline: deploy%'").fetchone()[0]
    assert "scripts/safe-merge.sh 4878 --squash --delete-branch" in gate_body
    assert "--admin" in gate_body and "forbidden" in gate_body
    assert "gh pr merge 4878" not in gate_body, "must never instruct a raw merge"
    # the template defers to the source card instead of overriding its acceptance
    for body in (gate_body, dep_body):
        assert src in body and "authoritative" in body
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
        fh.write("plugins:\n  enabled: [kanban-pipeline]\n"
                 "kanban_pipeline:\n  enabled: true\n")

    bindir = str(tmp_path_factory.mktemp("fakebin"))
    ghp = os.path.join(bindir, "gh")
    with open(ghp, "w") as fh:
        fh.write(FAKE_GH)
    os.chmod(ghp, os.stat(ghp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    script = r'''
import os, sys
sys.path.insert(0, %(repo)r)
from hermes_cli import kanban_db as kb
from hermes_cli import plugins
plugins.discover_plugins(force=True)
assert plugins.has_hook("kanban_task_completed"), "plugin hook not discovered"
with kb.connect() as conn:
    src = kb.create_task(conn, title="impl real", assignee="software-engineer",
                         body="PR %(pr)s")
    kb.complete_task(conn, src, summary="opened %(pr)s")   # real lifecycle hook fires
with kb.connect() as conn:
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
