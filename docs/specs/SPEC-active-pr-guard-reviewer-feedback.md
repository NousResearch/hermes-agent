# active-PR guard: reviewer-feedback release (covers `t_de993dac` case)

## Problem

Two compounding bugs in the kanban dispatch + GH-mirror pipeline currently
silently swallow reviewer feedback on an open PR. Concretely reproduced
2026-08-20 on `t_de993dac` (alias `aliaadil/alerthq#174`, PR `aliaadil/alerthq#178`):

1. **PR-thread comments never reach the kanban task.** The GH mirror
   (`scripts/github_issues_mirror.py`) only watches the GH ref resolved
   from the task body / title / first-matching comment via
   `_pick_ref`. For `t_de993dac`, every reference in body/title is to
   issue `#174`. The PR `#178` was opened and the worker posted its
   PR-URL into comments, but `_pick_ref` never sees those because:
   - `GH_REF_RE` matches only `github.com/.../issues/<N>`, not
     `github.com/.../pull/<N>` — PR full-URLs are silently dropped at
     line 55.
   - Short-form `aliaadil/alerthq#174` in the body is reached first by
     `mirror_pull` and short-circuits all further scanning.

   Result: the user's PR-thread comment ("the logging is not
   sufficient. ALL actions performed by the user and server need to be
   logged properly…") was *never mirrored* into the kanban task. The
   task has zero matching rows in `task_comments`.

2. **The active-PR respawn guard is too broad.** Even if the comment
   *were* mirrored, `check_respawn_guard` in
   `hermes_cli/kanban_db.py:9126-9133` returns `"active_pr"` whenever a
   recent (24h) task comment contains a PR URL. The guard has no
   exception for "reviewer feedback exists after that PR URL". So the
   dispatcher refused to re-spawn the builder. We observed
   **45 consecutive `respawn_guarded` events with `reason: active_pr`**
   between 18:45 and 19:29 UTC, all silent, no work.

This combination makes "leave reviewer feedback on an open PR and expect
the builder to iterate" silently impossible — which is the canonical
reviewer-feedback loop.

The existing spec at `aliaadil/hermes-agent#1` covers bug #2 (the guard
narrowing) but assumes the mirror will surface the feedback. It does not
cover bug #1 (the mirror's PR-thread blindness). This spec extends #1 to
cover both, keeping the bug #2 portion intact (slightly tightened with
the empirical learning from `t_de993dac`).

## Proposed Fix

### Part A — Mirror must surface PR-thread reviewer feedback

In `scripts/github_issues_mirror.py`:

1. **Teach `GH_REF_RE` to match PR URLs too.** Currently:
   ```python
   GH_REF_RE = re.compile(r"https?://(?:[a-z0-9-]+\.)*github\.com/([^/]+)/([^/]+)/issues/(\d+)", re.IGNORECASE)
   ```
   Change to also accept `/pull/<N>`:
   ```python
   GH_REF_RE = re.compile(r"https?://(?:[a-z0-9-]+\.)*github\.com/([^/]+)/([^/]+)/(?:issues|pull)/(\d+)", re.IGNORECASE)
   ```
   The `/issues/N/comments` GH REST endpoint already returns issue-style
   comments for PRs (verified 2026-08-20: `gh api repos/aliaadil/alerthq/issues/178/comments`
   returns the aliadil "the logging is not sufficient" comment), so no
   endpoint switch is needed.

2. **`_pick_ref` must prefer the most recent GH ref across body / title /
   comments, not just the first.** Currently body+title are scanned
   first; if either yields a full URL, comments are never scanned. This
   means the canonical "task was imported from issue #174" always wins
   even when the worker subsequently linked a PR. Fix: scan all three
   sources, then prefer the **latest-in-time** ref by the max
   `created_at` of the comment(s) that contain it. Tie-break by full-URL
   over short-ref. Body+title count as `created_at=0` so a fresh PR
   mention in a recent comment naturally outranks the original issue
   link.

3. **Add a sibling function `_latest_pr_ref_for_task(task_id)`** that
   scans `task_comments` for any `github.com/.../pull/<N>` or
   `owner/repo#N` whose N matches an open PR in the repo
   (`gh pr list --repo ... --state open --json number`) and returns the
   freshest. The mirror pulls *both* the issue ref AND the PR ref into
   the task. Comments from either side get appended. Sidecar
   `last_gh_comment_at` gets a per-ref key so the cutoffs stay
   independent.

4. **Idempotency / dedupe must extend to PR-thread comments.** A single
   GH comment can be referenced from both `issue/178/comments` and
   `pulls/178/comments` endpoints with the same numeric ID; dedupe by
   `(owner, repo, number, comment_id)` instead of just the sidecar
   cutoff. The `_content_key` SHA1 already prevents identical bodies
   looping; extend it to be the primary key so the same comment id can't
   appear twice even if both endpoints return it.

### Part B — `check_respawn_guard` must release on reviewer feedback

In `hermes_cli/kanban_db.py:9126-9133`:

5. **Add a "reviewer-feedback release" exception before the
   `active_pr` return.** When the loop in step 4 is about to return
   `"active_pr"`, instead check whether any comment whose `created_at`
   is *after* the most recent PR-URL comment satisfies **any** of:
   - `author != 'default'` AND `len(body) >= 80` (rules out auto-mirrored
     status pings — they're `default`-authored and short).
   - Distinct `_content_key` from the prior PR-URL comment
     (idempotent re-push of the same comment does NOT release).
   - The comment matches reviewer-feedback patterns: contains any of
     `please update`, `fix in this pr`, `also address`, `needs to`,
     `this pr`, `not sufficient`, `all actions`, `all clicks`,
     `please add`, `please include`, or references the PR number
     directly (`#178` / `PR #178` / `pull/178`).
   - The linked PR has `reviewDecision == 'CHANGES_REQUESTED'`
     (queried via `gh pr view <url> --json reviewDecision`). This is a
     stronger signal than comment text and covers cases where the
     reviewer used the GitHub review UI instead of leaving an inline
     comment.

   If any trigger fires, return `None` (allow the respawn). Otherwise
   keep the existing `"active_pr"` return.

6. **Branch-routing on release.** When the guard releases because of
   reviewer feedback, the dispatched Builder run must use the **same
   branch and head SHA** as the most recent PR-URL comment, not a
   fresh `feat/<task-id>-<new-slug>` branch. Implementation:
   - Look up the latest task comment matching
     `_RESPAWN_GUARD_PR_URL_RE` (already exists).
   - Extract the PR URL; resolve to branch + head SHA via
     `gh pr view <url> --json headRefName,headRefOid`.
   - Pass `(branch, head_sha)` as an override into the Builder worktree
     provisioning so the new commit lands on the same branch tip.
   - Skip the `create_agent_worktree.py` branch-cut logic; instead,
     `git fetch origin <branch> && git checkout -B <branch> origin/<branch>`
     in the existing worktree.

   Mirrors the existing `review`-lane behavior — that lane already
   skips `recent_success` and `active_pr` because "recent PR URL" is
   its precondition. The `ready` lane must do the same when reviewer
   feedback arrives.

## Acceptance Criteria

- [ ] `check_respawn_guard` returns `None` for a task that has a recent
      PR-URL comment AND a new (non-default-authored, dedup-key-distinct)
      comment after that PR URL.
- [ ] `check_respawn_guard` returns `None` for a task whose linked PR has
      `reviewDecision == 'CHANGES_REQUESTED'` even if no new comment
      body matches the pattern list.
- [ ] `check_respawn_guard` still returns `"active_pr"` for a task that
      has a recent PR-URL comment but NO new reviewer feedback AND
      `reviewDecision != 'CHANGES_REQUESTED'`.
- [ ] Auto-mirrored `default`-authored status comments do NOT release
      the guard.
- [ ] Re-mirroring the same comment (same `_content_key`) does NOT
      release the guard.
- [ ] `GH_REF_RE` matches both `github.com/.../issues/<N>` and
      `github.com/.../pull/<N>` URLs.
- [ ] `_pick_ref` prefers the freshest GH ref across body/title/comments
      rather than the first one in body.
- [ ] `mirror_pull` polls BOTH the issue ref AND any PR ref attached to
      the task; comments from either source are appended.
- [ ] When the guard releases, the dispatched Builder run targets the
      same branch as the most recent PR-URL comment (verified via
      `git rev-parse --abbrev-ref HEAD` in the worktree at spawn time).
- [ ] New unit tests in
      `tests/hermes_cli/test_kanban_review_lifecycle.py` covering all
      six trigger conditions and the four non-trigger conditions.
- [ ] New unit tests in
      `tests/scripts/test_github_issues_mirror.py` (new file) covering:
      PR-URL regex matching, `_pick_ref` recency tie-break,
      dual-ref mirror_pull (issue + PR), and dedupe by
      `(owner, repo, number, comment_id)`.
- [ ] No regression: tasks that previously sat in `ready` for the full
      24h PR window still do so when no reviewer feedback arrives AND
      `reviewDecision != 'CHANGES_REQUESTED'`.
- [ ] End-to-end repro: replay the `t_de993dac` timeline against the
      patched system, verify that within 2 ticks of the
      19:10:04 reviewer comment, the task re-dispatches against
      `feat/t_de993dac-add-logging` and produces a new commit on PR
      #178.

## Out of Scope

- Changes to the `_RESPAWN_GUARD_PR_WINDOW` constant (24h stays).
- Changing `_RESPAWN_GUARD_PR_URL_RE` (it correctly matches PR URLs).
- Adding a separate "reviewer-feedback lane" — that's a bigger
  architectural change.
- Changing the debounce on the natural-language unblocker.
- Switching `gh_issue_comments` to also call
  `repos/{owner}/{repo}/pulls/{N}/comments` (review-submission
  comments live there but the issue-style comment surface is what
  users actually use for threaded discussion; if Part B's
  `reviewDecision` check passes, this isn't needed). Documented as
  follow-up if review-submission comments prove to be the dominant
  feedback channel.

## Test Plan

1. **Reproduce `t_de993dac`.** Mark the task `done`, leave the
   19:10:04 comment on PR #178, advance cron ticks, verify the task
   re-dispatches within 2 ticks against `feat/t_de993dac-add-logging`.
2. **Negative (no feedback).** Same setup but only auto-mirrored
   `default`-authored status pings after the PR URL. Verify guard
   still returns `"active_pr"`.
3. **Idempotency.** Same comment posted twice (dedup). Verify guard
   does NOT release on the second post.
4. **Pattern match.** Post a short comment "needs to log all clicks".
   Verify guard releases on body-pattern trigger.
5. **`reviewDecision` trigger.** Approve the PR instead of requesting
   changes; verify guard still holds. Then use the GitHub UI to
   request changes; verify guard releases even with no new comment.
6. **Dual-ref mirror.** A task that links both issue #174 and PR #178.
   Post comments on both. Verify both arrive in `task_comments` with
   the `_↩ from GH comment` marker.
7. **PR-URL regex.** Unit test confirms `GH_REF_RE` matches
   `https://github.com/owner/repo/pull/178`.
8. **Cross-lane test.** Same flow on the `review` lane. Verify the
   existing review-lane behavior is preserved (it already skips both
   `recent_success` and `active_pr`).

## Files Touched (predicted)

- `hermes_cli/kanban_db.py` — `check_respawn_guard` (Part B),
  `_RESPAWN_GUARD_PR_URL_RE` constant unchanged.
- `scripts/github_issues_mirror.py` — `GH_REF_RE`, `_pick_ref`,
  `mirror_pull`, new `_latest_pr_ref_for_task` (Part A).
- `tests/hermes_cli/test_kanban_review_lifecycle.py` — new cases.
- `tests/scripts/test_github_issues_mirror.py` — new file.

## Risk

- **GH rate-limit.** Doubling the mirror's polling (issue + PR) doubles
  `gh api` calls per task per tick. Mitigated by the existing
  `last_gh_comment_at` per-ref sidecar (no calls if neither ref has
  moved) and by `MAX_GH_COMMENTS_PER_TICK` cap.
- **Backward compat.** `_pick_ref` already returns a 3-tuple in legacy
  callers; new logic preserves that shape and only changes ranking.
- **Branch-routing override** interacts with worktree-lifecycle. The
  existing `t_6042789f` case in issue #1 already established the
  pattern — no new lifecycle code, just parameter passthrough.
