# M4 Salvage Wiring - Cron and Discord Handler (NOT registered)

This document describes the wiring needed to activate the M4 salvage
fixes. Nothing here is registered or enabled - it is documentation
for a future, explicitly-approved enablement step.

## C019: GitNexus Event-Driven Reindex Hook

The hook script is at scripts/gitnexus-reindex-hook.sh. It triggers a
GitNexus re-index on git push to HEAD branch, with full-SHA verification
against the GitNexus registry and explicit failure reporting.

### Wiring (manual, not automated)

1. Copy the hook into the repo git hooks:
   cp scripts/gitnexus-reindex-hook.sh .git/hooks/post-receive
   chmod +x .git/hooks/post-receive
2. The existing daily cron (scripts/gitnexus-reindex-daily.sh) remains
   as a fallback. The hook is event-driven and supersedes the daily poll
   for HEAD pushes; the daily script catches up if a push was missed
   (e.g. hook not installed yet).

### What the hook does

- Reads old-sha new-sha ref-name from stdin (post-receive format)
- Only acts on refs/heads/HEAD-branch updates (skips tags, other branches)
- Verifies the pushed SHA is a valid 40-char hex SHA
- Compares against the GitNexus registry lastCommit for KenseiAgent
- If stale: detaches a background re-index worker (incremental or full)
- Logs all outcomes to ~/.hermes/logs/gitnexus/reindex-hook-*.log
- Explicit error messages on missing runner/gitnexus binary or invalid SHA

## C026/C027: Postiz Single Authoritative Enqueue Route

### Cron wiring (NOT registered)

The publish_to_postiz.py script is the single authoritative Postiz
enqueue route. It should be wired as a no-agent cron job:

  Example cron entry (NOT registered here):
  every 15 min: /home/kensei/repos/KenseiAgent/scripts/publish_to_postiz.sh

The script polls for approved drafts with enqueue_state IS NULL or pending
and postiz_id IS NULL, atomically claims each draft, enqueues via
postiz_bridge.queue_post, and marks it published. The idempotent claim
prevents duplicate enqueues on concurrent runs or retries.

### Discord approval handler wiring (NOT registered)

The content_engine.py approve command is now state-only - it marks
a draft as approved and generates media, but does NOT enqueue to Postiz.
The Discord approval handler (if wired) should call approve_draft() only;
the publish_to_postiz cron picks up approved drafts and enqueues them.

If a Discord bot handler is added in the future, it should:
1. Listen for approval reactions on draft preview cards
2. Call approve_draft(draft_id) to set state to approved
3. NOT call queue_post directly - that is the cron job

## C028: Atomic Config Writes

The xurl config write in engagement_suggester.configure_xurl_from_postiz()
now uses an atomic write pattern (temp file + fsync + 0o600 permissions +
os.replace). This prevents partial writes from corrupting the existing
config and avoids the TOCTOU window of write-then-chmod.

## C030: GitRadar Shared Runbook Root

The GitRadar producer (scripts/github-radar-discover.py) and consumer
(scripts/archive/gitradar-upstream-monitor.py) are aligned to a shared
runbook root at repo/runbooks/github-radar/. See the script headers
for the resolved paths.
