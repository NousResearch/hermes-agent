---
name: collaboration
description: Use for Slack, Jira, Google Workspace, meetings, Fellow/Meet, Timetastic, attachments, and shared teammate-facing outputs.
version: 1.1.0
author: YallaPlay
license: Proprietary
metadata:
  hermes:
    tags: [collaboration, slack, jira, google-workspace, meetings, timetastic, yallaplay]
    related_skills: [knowledge, liveops]
---

# Collaboration Skill

## Overview

Use this skill for teammate-facing systems and shared artifacts: Slack, Jira, Google Workspace, meetings, Fellow/Meet, Timetastic, attachments, and files intended for teammates. Default to read-only fetches and drafts. Anything that notifies people or changes shared state requires explicit confirmation.

This is the Hermes equivalent of the legacy `personas/collaboration.md` workflow.

## When to Use

- Reading or drafting Slack/Jira/Google Workspace content.
- Looking up teammates, channels, docs, sheets, meeting notes, transcripts, time-off, or attachments.
- Creating shared outputs for review, e.g. CSVs, charts, docs, summaries.
- Posting or updating teammate-visible systems after confirmation.

Load LiveOps too for Helpshift/OneSignal/support-user-visible communication. Load Analytics too for metric-backed shared outputs.

## Required Context

Use these wiki areas for durable context:

- `yallaplay-wiki/tools/team/slack.md`
- `yallaplay-wiki/tools/team/jira.md`
- `yallaplay-wiki/tools/team/google_workspace.md`
- `yallaplay-wiki/tools/team/fellow_meet.md`
- `yallaplay-wiki/tools/team/timetastic.md`
- `yallaplay-wiki/people/index.md`, `person-directory.md`, `teams-and-roles.md`, and `time-off.md`.
- `yallaplay-wiki/journal/` for meeting records if present.

Search with `python3 tools/wiki_search.py <terms> --domain collaboration`.

## Tool Routing

Legacy commands from `yallaplay-analytics-agent-gpt/scripts/` are the reference for unported functionality:

- Slack: `slack_user_lookup.py`, `slack_fetch.py`, `slack_post.py`, `slack_upload.py`, `slack_react.py`, `slack_pin.py`.
- Jira: `jira_client.py`.
- Google Workspace: `google_workspace.py`.
- Meetings: `meet_store.py`, meeting transcript helpers.
- Fellow: `fellow_client.py`, `fellow_sync.py`.
- Timetastic: `timetastic_client.py`.

In this Hermes pilot, use ported wrappers when available. If a required wrapper is not ported here, report that directly and use the wiki or legacy repo only as reference.

## Workflow

1. **Classify visibility.** Determine whether the action is private read, draft, shared-file write, or teammate-visible notification. Completion: risk level is explicit.
2. **Resolve identities.** For named people, use authoritative lookup rather than guessing IDs, emails, or Slack handles. Completion: ID/source is verified.
3. **Fetch before drafting.** Read the source thread/doc/ticket/meeting before summarizing or replying. Completion: response is grounded in actual content.
4. **Draft first.** For posts, comments, docs, sheets, uploads, reactions, pins, or Jira writes, prepare the exact visible content and target. Completion: user can approve or edit.
5. **Confirm writes.** Execute only after explicit confirmation naming the target/action. Completion: side effect matches approved draft.
6. **Save reusable knowledge.** Meeting decisions, recurring processes, and collaboration gotchas belong in `yallaplay-wiki/`, not memory. Completion: durable item is saved or intentionally skipped.

## Files and Attachments

- Read attachments before answering questions about them; do not infer from filenames.
- Put generated artifacts under `outputs/` with unique descriptive names.
- Avoid clobbering shared files; use timestamped filenames unless updating an explicitly named target.
- Do not expose private document contents beyond the user's request and access context.

## Safety

Do not perform these without explicit confirmation:

- Slack posts, uploads, reactions, pins, or bookmarks.
- Jira creates/updates/transitions/comments.
- Google Docs/Sheets/Drive writes or permission changes.
- Meeting invitations, calendar edits, or mass notifications.
- Helpshift replies and OneSignal pushes; those are LiveOps and user-visible.

## Common Pitfalls

1. **Dead prose options.** If asking the user to choose, use actual choice UI when available; otherwise make drafts directly.
2. **Guessing people.** Names are ambiguous; resolve IDs before messaging or attributing.
3. **Posting summaries without source reads.** Fetch thread/doc/meeting first.
4. **Notification surprise.** Jira and Slack writes notify teammates; draft and confirm.
5. **Shared-output clobbering.** Use unique names unless the user explicitly asked to update a specific artifact.
6. **Over-gathering for simple lookups.** For simple read-only counts/status checks where the wrapper and source are already known, use the direct query path first (e.g. Jira JQL via `jira_client.py`) and avoid extra wiki/repo/status reads unless the result is ambiguous or blocked.

## Verification Checklist

- [ ] Source content was read before summary/reply.
- [ ] Named people/artifacts were resolved from authoritative lookup.
- [ ] Teammate-visible writes were confirmed with exact target and content.
- [ ] Generated files are linked by path and uniquely named.
- [ ] Reusable decisions/process knowledge was captured in the wiki when appropriate.
