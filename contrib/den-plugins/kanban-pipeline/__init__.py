"""kanban-pipeline — chain the *current* delivery hop after a card completes.

A completed card does NOT mean "needs merge". This observer answers two
questions from supported task/run metadata before it creates anything:

  1. WHICH artifact is this completion about?  (current, never historical)
  2. WHAT delivery phase does that artifact actually still owe?

Artifact selection is tiered and stops at the first tier that yields a
reference, highest authority first:

  1. closing run ``metadata`` (structured handoff)
  2. closing run ``summary`` (this completion's own statement)
  3. ``tasks.result``
  4. the CURRENT-scope region of the card body (text above the historical
     delimiter)

Comments are NOT a selection source. The 2026-09-16 incident (t_4a602990)
was caused by scanning comments and taking the first URL found: a six-day-old
quoted link to PR #4841 selected the target while the card's current artifact,
PR #4878, was already merged AND deployed. Two stale cards were spawned.

Phase is then derived from a bounded, read-only artifact probe
(``ARTIFACT_STATE_FN``; tests inject a fake transport):

  * merged AND deployed        -> nothing owed, zero cards
  * merged, not yet deployed   -> deploy card only (no stale merge gate)
  * open                       -> merge gate, with the deploy hop linked behind it
  * closed-unmerged / unknown  -> zero cards, bounded notice on the owner card

Ambiguity, a metadata read failure, or a probe failure NEVER invents
downstream work; it records one specific, deduplicated notice on the
completed card so the current owner sees it instead of a silent false clear.
An existing canonical downstream owner always wins: if another live card
already references the artifact, nothing parallel is created.

Idempotent via ``create_task(idempotency_key=...)`` and marker-deduplicated
notices. Skips cards whose title starts with "pipeline:" (no recursive chain).
Fails open — any error is logged as a specific automation error and never
breaks the completion transition. Never merges, deploys, or mutates anything
outside the board.

Config (config.yaml):
  kanban_pipeline.enabled          default true
  kanban_pipeline.merge_assignee   default "reviewer"
  kanban_pipeline.deploy_assignee  default "software-engineer"
  kanban_pipeline.merge_command    default "scripts/safe-merge.sh"
  kanban_pipeline.live_check       extra, non-overriding deploy instructions
  kanban_pipeline.probe_timeout    default 45 (seconds, read-only gh calls)
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading

logger = logging.getLogger(__name__)

_PR_RE = re.compile(r"https://github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)")

# Everything at or below one of these lines is historical context on a card
# body and must never select the current artifact.
_HISTORY_MARKERS = (
    "--- Historical task context",
    "--- historical task context",
    "--- Historical context",
    "Historical task context;",
)

# Structured run-metadata keys a worker may use to declare its artifact.
_METADATA_KEYS = ("pr_url", "pr", "artifact_url", "artifact", "primary_artifact")

_MERGE_BODY = """Merge gate for {url} (current artifact of card {src}).

Artifact state at chain time: {state}. Source card {src} and its own acceptance
text are authoritative — this template does not replace the model, tests, or
acceptance that card requires.

1. `gh pr view {n} --json state,reviewDecision,mergeable,statusCheckRollup` — paste output.
2. Require: reviewDecision=APPROVED, mergeable=MERGEABLE, every REQUIRED check green.
   Non-required red checks: list them and say why they do not block, or fix them.
3. If the PR carries a `rebootstrap`/`deploy` label, confirm a supervisor applied it
   (a worker may not label its own PR).
4. Merge ONLY through the repository's sanctioned merge gate:
   `{merge_command} {n} --squash --delete-branch` — paste the merge commit SHA.
   Raw `gh pr merge`, `--admin`, and any other override of the gate are forbidden;
   if the gate refuses, block this card with its verbatim output. Do not route the
   merge through another process to get around it.
5. `hermes kanban complete <this-card> --summary "merged <sha>"`.
If any requirement is false, `hermes kanban block` with the exact failing line — do not merge.
"""

_DEPLOY_BODY = """Deploy + live check for {url} ({gate_line}).

Artifact state at chain time: {state}. Source card {src} and its own acceptance
text are authoritative — this template does not replace the model, tests, or
acceptance that card requires, and it does not authorize work that card did not
ask for.

1. Wait for the deploy workflow that carries the merge SHA; paste run URL + conclusion.
2. If the change is delivered to running instances, confirm the rollout job ran and
   paste its summary.
3. Live acceptance — run ONLY what source card {src} names as its acceptance, with
   the raw output pasted (not a summary). Do NOT provision new tenants, exercise
   billing/financial flows, or probe customer/production surfaces unless {src}'s own
   acceptance text explicitly asks for it; if it does not, say so and block for scope.
{live_check}
4. `hermes kanban complete <this-card> --summary "<what was proven, with URLs>"`.
Red anywhere → `hermes kanban block` with the raw failing output. Never mark done on a
green label alone.
"""

_DEFAULT_LIVE_CHECK = (
    "   - (no extra checks configured; source card acceptance is the whole list)"
)

_NOTICE_PREFIX = "kanban-pipeline"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def _cfg():
    try:
        from hermes_cli.config import load_config
        return (load_config() or {}).get("kanban_pipeline", {}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# artifact selection — current only
# ---------------------------------------------------------------------------

def _current_scope(body):
    """Return only the current-scope region of a card body."""
    text = body or ""
    cut = len(text)
    lowered = text.lower()
    for marker in _HISTORY_MARKERS:
        idx = lowered.find(marker.lower())
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def _refs(text):
    """Ordered, de-duplicated (url, number) PR references in *text*."""
    out = []
    seen = set()
    for m in _PR_RE.finditer(text or ""):
        url = m.group(0)
        if url not in seen:
            seen.add(url)
            out.append((url, m.group(3)))
    return out


def _metadata_text(run):
    """Flatten the declared-artifact keys of a run's structured metadata."""
    meta = getattr(run, "metadata", None)
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = None
    if not isinstance(meta, dict):
        return ""
    parts = []
    for key in _METADATA_KEYS:
        val = meta.get(key)
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, (list, tuple)):
            parts.extend([v for v in val if isinstance(v, str)])
    return "\n".join(parts)


def _select_artifact(task, run, hook_summary):
    """Pick the CURRENT primary artifact.

    Returns ``(url, number, tier, error)``. ``error`` is non-None when the
    tiers are ambiguous; ``url`` is None with no error when the completion
    simply references no artifact.
    """
    tiers = [
        ("run_metadata", _metadata_text(run)),
        ("run_summary", (getattr(run, "summary", None) or hook_summary or "")),
        ("task_result", (getattr(task, "result", None) or "")),
        ("body_current_scope", _current_scope(getattr(task, "body", None))),
    ]
    for tier, text in tiers:
        refs = _refs(text)
        if not refs:
            continue
        numbers = {n for _u, n in refs}
        if len(numbers) > 1:
            listed = ", ".join(sorted("#" + n for n in numbers))
            return None, None, tier, (
                "ambiguous artifact: the current %s names %d different pull requests "
                "(%s). A completion must name exactly one current primary artifact."
                % (tier, len(numbers), listed)
            )
        url, num = refs[0]
        return url, num, tier, None
    return None, None, None, None


# ---------------------------------------------------------------------------
# bounded, read-only artifact probe
# ---------------------------------------------------------------------------

class ProbeError(RuntimeError):
    """The artifact state could not be established. Never invent work."""


def _gh_json(args, timeout):
    if not shutil.which("gh"):
        raise ProbeError("gh CLI not available for read-only artifact validation")
    try:
        proc = subprocess.run(
            ["gh"] + list(args),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise ProbeError("read-only artifact probe timed out: gh %s" % " ".join(args))
    except Exception as exc:
        raise ProbeError("read-only artifact probe failed: %s" % exc)
    if proc.returncode != 0:
        raise ProbeError(
            "read-only artifact probe exit %d: %s"
            % (proc.returncode, (proc.stderr or "").strip()[:300])
        )
    try:
        return json.loads(proc.stdout or "null")
    except Exception as exc:
        raise ProbeError("read-only artifact probe returned non-JSON: %s" % exc)


def _gh_artifact_state(url, timeout=45):
    """Default transport: read-only `gh` reads. No mutation, no token minting.

    Returns ``{"merged", "deployed", "approved", "state", "detail"}``.
    ``deployed`` may be None, which callers must treat as unknown (no work).
    """
    m = _PR_RE.match(url)
    if not m:
        raise ProbeError("unparseable artifact url: %s" % url)
    owner, repo, number = m.group(1), m.group(2), m.group(3)
    pr = _gh_json(
        ["pr", "view", number, "--repo", "%s/%s" % (owner, repo),
         "--json", "state,mergedAt,mergeCommit,reviewDecision"],
        timeout,
    ) or {}
    state = (pr.get("state") or "").upper()
    merged = state == "MERGED"
    approved = (pr.get("reviewDecision") or "").upper() == "APPROVED"
    if not merged:
        return {
            "merged": False, "deployed": False, "approved": approved,
            "state": state or "UNKNOWN",
            "detail": "state=%s reviewDecision=%s" % (state or "UNKNOWN",
                                                      pr.get("reviewDecision")),
        }
    sha = ((pr.get("mergeCommit") or {}) or {}).get("oid")
    if not sha:
        raise ProbeError("merged artifact %s has no merge commit oid" % url)
    runs = _gh_json(
        ["api", "repos/%s/%s/actions/runs?status=success&event=push&per_page=1"
         % (owner, repo)],
        timeout,
    ) or {}
    entries = runs.get("workflow_runs") or []
    if not entries:
        return {"merged": True, "deployed": None, "approved": approved,
                "state": "MERGED",
                "detail": "merge=%s; no successful push run to compare against" % sha[:12]}
    live = entries[0].get("head_sha")
    if not live:
        return {"merged": True, "deployed": None, "approved": approved,
                "state": "MERGED", "detail": "merge=%s; latest run has no head_sha" % sha[:12]}
    cmp_ = _gh_json(
        ["api", "repos/%s/%s/compare/%s...%s" % (owner, repo, sha, live)], timeout,
    ) or {}
    behind = cmp_.get("behind_by")
    deployed = (behind == 0) if behind is not None else None
    return {
        "merged": True, "deployed": deployed, "approved": approved, "state": "MERGED",
        "detail": "merge=%s live=%s behind_by=%s" % (sha[:12], live[:12], behind),
    }


#: Injection point. Tests and alternate deployments replace this with a local
#: fake transport; nothing else in the plugin talks to the network.
ARTIFACT_STATE_FN = _gh_artifact_state


# ---------------------------------------------------------------------------
# board helpers (all mutations fail open)
# ---------------------------------------------------------------------------

_INACTIVE = {"done", "archived"}


def _notice(kb, conn, task_id, code, text):
    """Record one deduplicated, bounded notice on the current owner card."""
    marker = "[%s:%s]" % (_NOTICE_PREFIX, code)
    try:
        for c in kb.list_comments(conn, task_id):
            if marker in (c.body or ""):
                return False
    except Exception as exc:
        logger.warning("[kanban-pipeline] notice dedup read failed for %s: %s", task_id, exc)
    body = (
        "%s %s\nNo downstream card was created. This is an automation notice, not a "
        "clearance: the current owner of %s decides what happens next. "
        "kanban-pipeline never merges, deploys, or closes anything."
        % (marker, text, task_id)
    )
    try:
        kb.add_comment(conn, task_id, _NOTICE_PREFIX, body)
        return True
    except Exception as exc:
        logger.warning(
            "[kanban-pipeline] board mutation failed (notice %s on %s): %s",
            code, task_id, exc,
        )
        return False


def _existing_owner(kb, conn, url, source_id):
    """Return a live card that already owns this artifact's delivery, if any."""
    try:
        tasks = kb.list_tasks(conn, include_archived=True)
    except Exception as exc:
        raise ProbeError("board read failed while checking existing owners: %s" % exc)
    for t in tasks:
        if t.id == source_id:
            continue
        haystack = "%s\n%s" % (t.title or "", t.body or "")
        key = t.idempotency_key or ""
        if url not in haystack and url not in key:
            continue
        if (t.status or "") in _INACTIVE:
            # A finished hop for this artifact still means "already owned" —
            # do not re-mint work someone already carried out.
            return t
        return t
    return None


def _create(kb, conn, **kw):
    task = kb.create_task(conn, **kw)
    return task.id if hasattr(task, "id") else task


_LOCAL_CHAIN_LOCK = threading.Lock()


@contextlib.contextmanager
def _chain_lock(kb):
    """Serialize adopt-before-mint across threads AND processes.

    ``create_task(idempotency_key=...)`` reads before it writes, so two
    completion events racing on the same artifact can both miss and both mint
    (measured: 4 duplicate merge-gate cards from 6 concurrent events). The
    check and the creates must be one critical section. Fails open: if the
    lock cannot be taken we still run, protected only by the key check.
    """
    path = None
    try:
        path = os.path.join(str(kb.board_dir()), ".kanban-pipeline.chain.lock")
    except Exception:
        path = None
    with _LOCAL_CHAIN_LOCK:          # same-process threads
        fh = None
        try:
            if path:
                import fcntl
                fh = open(path, "a+")
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except Exception as exc:     # no fcntl (Windows) or unwritable board dir
            logger.debug("[kanban-pipeline] chain lock unavailable: %s", exc)
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
                fh = None
        try:
            yield
        finally:
            if fh is not None:
                try:
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                finally:
                    try:
                        fh.close()
                    except Exception:
                        pass


# ---------------------------------------------------------------------------
# hook
# ---------------------------------------------------------------------------

def _on_completed(task_id=None, run_id=None, summary=None, **_):
    try:
        cfg = _cfg()
        if cfg.get("enabled", True) is False:
            return
        from hermes_cli import kanban_db as kb
        with kb.connect() as conn:
            try:
                task = kb.get_task(conn, task_id)
            except Exception as exc:
                logger.warning("[kanban-pipeline] task read failed for %s: %s", task_id, exc)
                return
            if not task:
                return
            if (task.title or "").lower().startswith("pipeline:"):
                return  # never chain off our own cards

            run = None
            try:
                if run_id is not None:
                    run = kb.get_run(conn, run_id)
                if run is None:
                    run = kb.latest_run(conn, task_id)
            except Exception as exc:
                _notice(kb, conn, task_id, "metadata-read-failed",
                        "could not read this card's run metadata (%s), so the current "
                        "primary artifact is undetermined." % exc)
                return

            url, num, tier, err = _select_artifact(task, run, summary)
            if err:
                _notice(kb, conn, task_id, "ambiguous-artifact", err)
                return
            if not url:
                return  # nothing claims an artifact — quiet, as before

            try:
                state = ARTIFACT_STATE_FN(url) or {}
            except Exception as exc:
                _notice(kb, conn, task_id, "artifact-probe-failed",
                        "current artifact %s (from %s) could not be validated read-only: "
                        "%s." % (url, tier, exc))
                return

            merged = bool(state.get("merged"))
            deployed = state.get("deployed")
            detail = state.get("detail") or ""
            label = "%s (%s)" % (state.get("state") or "UNKNOWN", detail)

            if merged and deployed is True:
                _notice(kb, conn, task_id, "already-delivered",
                        "current artifact %s (from %s) is already merged AND deployed — "
                        "%s. Nothing is owed downstream." % (url, tier, detail))
                return
            if merged and deployed is None:
                _notice(kb, conn, task_id, "deployment-state-unknown",
                        "current artifact %s (from %s) is merged but its deployment state "
                        "could not be established (%s)." % (url, tier, detail))
                return
            if not merged and (state.get("state") or "").upper() not in ("OPEN", ""):
                _notice(kb, conn, task_id, "artifact-not-open",
                        "current artifact %s (from %s) is %s and was never merged."
                        % (url, tier, state.get("state")))
                return

            merge_command = cfg.get("merge_command", "scripts/safe-merge.sh")
            live_check = cfg.get("live_check", _DEFAULT_LIVE_CHECK)
            created = []

            # Adopt-before-mint, inside one cross-process critical section so
            # duplicate/concurrent completion events cannot both mint.
            with _chain_lock(kb):
                try:
                    owner = _existing_owner(kb, conn, url, task_id)
                except ProbeError as exc:
                    _notice(kb, conn, task_id, "owner-scan-failed", str(exc))
                    return
                if owner is not None:
                    _notice(kb, conn, task_id, "existing-owner",
                            "current artifact %s (from %s) is already owned downstream by %s "
                            "(%s, status=%s)."
                            % (url, tier, owner.id, owner.title, owner.status))
                    return

                gate_id = None
                if not merged:
                    gate_id = _create(
                        kb, conn,
                        title="pipeline: merge-gate PR #%s" % num,
                        body=_MERGE_BODY.format(
                            url=url, n=num, src=task_id, state=label,
                            merge_command=merge_command,
                        ),
                        assignee=cfg.get("merge_assignee", "reviewer"),
                        created_by=_NOTICE_PREFIX,
                        parents=[task_id],
                        idempotency_key="pipeline:merge:%s" % url,
                    )
                    created.append("%s (merge-gate)" % gate_id)

                dep_id = _create(
                    kb, conn,
                    title="pipeline: deploy+live-check PR #%s" % num,
                    body=_DEPLOY_BODY.format(
                        url=url, src=task_id, state=label, live_check=live_check,
                        gate_line=("merge-gate card %s" % gate_id) if gate_id
                        else "already merged; merge gate not owed",
                    ),
                    assignee=cfg.get("deploy_assignee", "software-engineer"),
                    created_by=_NOTICE_PREFIX,
                    parents=[gate_id or task_id],
                    idempotency_key="pipeline:deploy:%s" % url,
                )
                created.append("%s (deploy+live-check)" % dep_id)

            try:
                kb.add_comment(
                    conn, task_id, _NOTICE_PREFIX,
                    "kanban-pipeline: chained %s for current artifact %s "
                    "[selected from %s; state %s]" % (" -> ".join(created), url, tier, label),
                )
            except Exception as exc:
                logger.warning(
                    "[kanban-pipeline] board mutation failed (chain comment on %s): %s",
                    task_id, exc,
                )
            logger.info("[kanban-pipeline] %s -> %s", task_id, ", ".join(created))
    except Exception as exc:
        logger.warning("[kanban-pipeline] automation error: %s", exc)


def register(ctx) -> None:
    ctx.register_hook("kanban_task_completed", _on_completed)
