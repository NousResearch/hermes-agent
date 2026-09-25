"""Hermetic test for kanban-pipeline: complete a card with a PR URL -> two chained cards appear once."""
import importlib.util, os, sys, tempfile

home = tempfile.mkdtemp(prefix="kp_test_")
os.environ["HERMES_HOME"] = home
for v in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME", "HERMES_KANBAN_WORKSPACES_ROOT"):
    os.environ.pop(v, None)
for prof in ("reviewer", "software-engineer", "default"):
    os.makedirs(os.path.join(home, "profiles", prof), exist_ok=True)
sys.path.insert(0, "/Users/engineer/.hermes/hermes-agent")
from hermes_cli import kanban_db as kb
assert home in str(kb.kanban_db_path()), kb.kanban_db_path()

spec = importlib.util.spec_from_file_location("kp", "/Users/engineer/.hermes/plugins/kanban-pipeline/__init__.py")
kp = importlib.util.module_from_spec(spec); spec.loader.exec_module(kp)

with kb.connect() as conn:
    src = kb.create_task(conn, title="impl thing", assignee="software-engineer",
                         body="Opened https://github.com/VibeTechnologies/AgentPod/pull/4927")
    # a card whose title starts with pipeline: must NOT chain
    loop = kb.create_task(conn, title="pipeline: merge-gate PR #1", assignee="reviewer",
                          body="https://github.com/x/y/pull/1")
    nopr = kb.create_task(conn, title="no pr here", assignee="reviewer", body="nothing")

kp._on_completed(task_id=src)
kp._on_completed(task_id=src)   # idempotent
kp._on_completed(task_id=loop)  # no chain
kp._on_completed(task_id=nopr)  # no chain

with kb.connect() as conn:
    rows = conn.execute("select id,title,assignee,status,idempotency_key from tasks order by created_at").fetchall()
    for r in rows: print(tuple(r))
    titles = [r[1] for r in rows]
    assert titles.count("pipeline: merge-gate PR #4927") == 1, titles
    assert titles.count("pipeline: deploy+live-check PR #4927") == 1, titles
    assert len(rows) == 5, len(rows)
    gate = [r for r in rows if r[1] == "pipeline: merge-gate PR #4927"][0]
    dep = [r for r in rows if r[1] == "pipeline: deploy+live-check PR #4927"][0]
    assert gate[2] == "reviewer" and dep[2] == "software-engineer"
    links = conn.execute("select parent_id,child_id from task_links").fetchall()
    assert (src, gate[0]) in [tuple(l) for l in links], links
    assert (gate[0], dep[0]) in [tuple(l) for l in links], links
    # deploy must be dependency-blocked (gate not done), gate must be ready/todo
    print("gate status:", gate[3], " deploy status:", dep[3])
    assert dep[3] != "ready", "deploy card must wait for merge-gate"
    comments = kb.list_comments(conn, src)
    assert any("chained" in c.body for c in comments)
print("PASS")
