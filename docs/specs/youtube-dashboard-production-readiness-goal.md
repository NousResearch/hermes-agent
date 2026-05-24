# Goal: YouTube dashboard production readiness

## Operator context
Henry visually accepted the local YouTube dashboard MVP. The current slice is local-state only and intentionally keeps public publishing/upload disabled until OAuth, quota, audit, and approval gates are explicit.

Current verified behavior:
- Dashboard loads at `/youtube`.
- Publish-plan card shows structural blockers for empty assets:
  - `asset_paths.video is required`
  - `asset_paths.thumbnail is required`
  - `asset_paths.captions is required`
  - `publish disabled`
- Browser console was clean in prior QA.
- QA row was removed from `/home/henry/.hermes/youtube/queue.json`.
- Mads/OpenClaw second review returned OK, no blockers before OAuth.
- One Mads should-fix was already patched: focused item should not fall through to another row if `focusedItemId` is stale.

## Mission
Make the YouTube dashboard production-ready enough for Henry to use as a real local queue for ScriptureDepth and Newslish before OAuth/upload is wired.

Production-ready here means: safe local queue management, trustworthy readiness gates, usable import/export, clear dry-run publish payloads, tests, and no accidental public publishing.

## Hard constraints
- Do not wire real YouTube OAuth.
- Do not upload, schedule, publish, or call YouTube APIs.
- Do not add secrets or credentials.
- Do not deploy externally.
- Do not restart the live Discord gateway.
- Do not touch unrelated Hermes features unless required by the dashboard integration.
- Keep changes surgical and boring.
- Preserve Henry-specific local work; do not reset, force checkout, or discard unrelated changes.

## Required deliverables
1. **Asset path existence checks**
   - Server readiness must verify required `asset_paths.video`, `asset_paths.thumbnail`, and for Shorts `asset_paths.captions` are not only present but point to existing readable files.
   - Missing path and nonexistent file should produce distinct, human-readable blockers.
   - Keep public publish disabled even when assets exist.

2. **Queue import/export reliability**
   - Verify CSV and JSON templates round-trip into queue items.
   - Add/adjust tests for malformed booleans, unknown fields if currently allowed, missing required fields, and duplicate IDs/titles if relevant to current schema.
   - Avoid making import overly clever; reject unsafe ambiguity.

3. **UI production polish for local use**
   - Empty state should clearly explain how to add/import first real items.
   - Publish-plan dry run should remain visibly disabled for public publishing.
   - If asset blockers exist, they should be easy to see in the Review workspace/publish card.
   - Do not over-design; this is operator tooling.

4. **Audit and concurrency sanity**
   - Verify create/patch/archive/import actions write audit entries.
   - Review JSON write path for obvious corruption risk. If cheap and safe, implement atomic write via temp file + replace. If not, document why deferred.
   - Do not introduce a database in this goal.

5. **Tests and verification**
   - Add or update focused backend tests for readiness, import/export, audit, and API endpoints.
   - Run targeted pytest for YouTube queue/dashboard tests.
   - Run web build/typecheck.
   - If practical, run local dashboard and do a browser smoke test of `/youtube`.

6. **Operator summary**
   - At the end, report:
     - files changed
     - tests run and exact result
     - remaining blockers before OAuth
     - whether dashboard is safe for local production queue use

## Optional if low-risk
- Add a small sample manifest under docs/specs or similar, but do not populate Henry's live queue with fake rows unless explicitly approved.
- Add docs for local queue usage if it fits naturally.

## Completion criteria
Goal is DONE only when:
- Required deliverables 1-6 are complete or explicitly marked deferred with a concrete reason.
- No real YouTube publishing path exists.
- Tests/build/smoke verification are run and summarized.
- Any deferred item is listed as pre-OAuth or post-OAuth.

If blocked, stop and report the smallest blocking issue. Do not broaden the task into OAuth or monetization work.
