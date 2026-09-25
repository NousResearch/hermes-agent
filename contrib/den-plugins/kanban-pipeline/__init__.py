"""kanban-pipeline — auto-chain the delivery pipeline after an approved PR card.

When a card that references a GitHub PR is completed (i.e. review approved and
the reviewer/EM called `complete`), this observer creates the next two hops as
dependency-linked cards so nobody has to hand-create them:

  <done card>  ─parent→  "pipeline: merge-gate PR #N"      (assignee: reviewer)
                            └parent→ "pipeline: deploy+live-check PR #N" (assignee: software-engineer)

Idempotent via create_task(idempotency_key=...). Skips cards whose title already
starts with "pipeline:" (no infinite chain) and cards with no PR URL in body or
comments. Fails open — any error is logged, never breaks dispatch.

Config (config.yaml):
  kanban_pipeline.enabled          default true
  kanban_pipeline.merge_assignee   default "reviewer"
  kanban_pipeline.deploy_assignee  default "software-engineer"
  kanban_pipeline.live_check       default text appended to the deploy card
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_PR_RE = re.compile(r"https://github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)")

_MERGE_BODY = """Merge gate for {url} (from card {src}).

1. `gh pr view {n} --json state,reviewDecision,mergeable,statusCheckRollup` — paste output.
2. Require: reviewDecision=APPROVED, mergeable=MERGEABLE, every REQUIRED check green.
   Non-required red checks: list them and say why they do not block, or fix them.
3. If the PR has the `rebootstrap`/`deploy` label, confirm the supervisor applied it (label author is not a worker).
4. `gh pr merge {n} --squash --delete-branch` — paste the merge commit SHA.
5. `hermes kanban complete <this-card> --summary "merged <sha>"`.
If any requirement is false, `hermes kanban block` with the exact failing line — do not merge.
"""

_DEPLOY_BODY = """Deploy + live check for {url} (merge-gate card {gate}).

1. Wait for the main deploy workflow on the merge SHA; paste run URL + conclusion.
2. If the change is tenant-delivered (bootstrap/, openclaw-rc.d/, gateway image), confirm the fleet rollout job ran and paste the rollout summary.
3. Live acceptance (paste raw output, not a summary):
{live_check}
4. `hermes kanban complete <this-card> --summary "<what was proven, with URLs>"`.
Red anywhere → `hermes kanban block` with the raw failing output. Never mark done on a green label alone.
"""

_DEFAULT_LIVE_CHECK = (
    "   - fresh tenant AND an existing tenant return HTTP 200 for the default model id "
    "on /v1/responses (openclaw, openclaw/default, litellm/auto).\n"
    "   - /v1/models ids are all resolvable (canary-models-resolvable-probe green)."
)


def _cfg():
    try:
        from hermes_cli.config import load_config
        return (load_config() or {}).get("kanban_pipeline", {}) or {}
    except Exception:
        return {}


def _find_pr(kb, conn, task):
    texts = [task.title or "", task.body or ""]
    try:
        texts += [c.body or "" for c in kb.list_comments(conn, task.id)]
    except Exception:
        pass
    for t in texts:
        m = _PR_RE.search(t)
        if m:
            return m.group(0), m.group(3)
    return None, None


def _on_completed(task_id=None, **_):
    try:
        cfg = _cfg()
        if cfg.get("enabled", True) is False:
            return
        from hermes_cli import kanban_db as kb
        with kb.connect() as conn:
            task = kb.get_task(conn, task_id)
            if not task or (task.title or "").lower().startswith("pipeline:"):
                return
            url, n = _find_pr(kb, conn, task)
            if not url:
                return
            gate = kb.create_task(
                conn,
                title=f"pipeline: merge-gate PR #{n}",
                body=_MERGE_BODY.format(url=url, n=n, src=task_id),
                assignee=cfg.get("merge_assignee", "reviewer"),
                created_by="kanban-pipeline",
                parents=[task_id],
                idempotency_key=f"pipeline:merge:{url}",
            )
            gate_id = gate.id if hasattr(gate, "id") else gate
            dep = kb.create_task(
                conn,
                title=f"pipeline: deploy+live-check PR #{n}",
                body=_DEPLOY_BODY.format(
                    url=url, gate=gate_id,
                    live_check=cfg.get("live_check", _DEFAULT_LIVE_CHECK),
                ),
                assignee=cfg.get("deploy_assignee", "software-engineer"),
                created_by="kanban-pipeline",
                parents=[gate_id],
                idempotency_key=f"pipeline:deploy:{url}",
            )
            dep_id = dep.id if hasattr(dep, "id") else dep
            kb.add_comment(conn, task_id, "kanban-pipeline", f"kanban-pipeline: chained {gate_id} (merge-gate) -> {dep_id} (deploy+live-check) for {url}")
            logger.info("[kanban-pipeline] %s -> %s -> %s", task_id, gate_id, dep_id)
    except Exception as e:
        logger.warning("[kanban-pipeline] %s", e)


def register(ctx) -> None:
    ctx.register_hook("kanban_task_completed", _on_completed)
