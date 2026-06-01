---
name: hermes-x-posting-workflows
description: "Draft, post, and remember X content safely."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [x, twitter, social-media, lm-twitterer, xurl, memory]
    related_skills: [xurl, ebbinghaus-memory]
---

# Hermes X Posting Workflows Skill

Use this skill to turn a user's X/Twitter request into a safe draft, posting action, and follow-up memory record. It coordinates the `xurl` skill, the `lm-twitterer` plugin, and memory-aware public-output guardrails without bypassing user confirmation.

This skill does not authorize public posts by itself. It preserves the existing rule that live public writes need explicit user approval for the final text.

## When to Use

- Use when the user asks Hermes to draft, post, reply, quote, or reconcile X/Twitter content.
- Use when the post should align with durable persona or project memory.
- Use when `xurl` or `lm-twitterer` is configured and the user expects Hermes to produce a URL after posting.
- Do not use for unrelated social platforms or private messaging.

## Prerequisites

- The active session has `terminal` access for local CLI checks, or plugin tools such as `lm_twitterer_post` are available.
- For official X API workflows, the `xurl` skill and local `xurl` authentication are already configured.
- For LM-twitterer workflows, `plugins/lm-twitterer` is enabled and its auth check succeeds.
- For memory sync, the Ebbinghaus provider or another explicit memory bridge is configured.
- The user has authorized any live public write after seeing the final draft.

## How to Run

Start with the safest available route:

- Prefer `xurl` for standard X API posting when it is authenticated.
- Use `lm-twitterer` when the user wants Hermes-generated posts or configured reply workflows.
- Keep draft and live paths separate. A dry run is not permission to publish.

Use `terminal` only for status checks that do not expose secrets:

```bash
xurl auth status
hermes lm-twitterer status
hermes lm-twitterer auth-check
```

Never read or print local credential stores such as `.xurl`, browser cookies, `LM_TWITTERER_AUTH_TOKEN`, or `LM_TWITTERER_CT0`.

## Quick Reference

| Step | Action |
| --- | --- |
| Draft | Produce concise text and state that it is not posted yet. |
| Confirm | Ask for explicit approval of the final public text. |
| Post | Use `xurl` or `lm-twitterer` only after approval. |
| Verify | Return the final posted text and URL. |
| Remember | Store public post metadata, never secrets or raw auth state. |

## Procedure

1. Identify whether the request is a draft, a live post, a reply, a quote, or a memory reconciliation task.
2. If the user did not already approve exact public text, produce a draft and stop for confirmation.
3. Before any live action, run a non-secret readiness check for the chosen route.
4. If X rejects length or weighted-character limits, shorten once while preserving the user's stated intent and report that it was shortened.
5. After a successful live post, return the URL and the final posted text.
6. If memory sync is configured, remember only the public artifact: text, URL, topic, source, and date.

## Pitfalls

- Do not treat "try it" or "draft one" as live-post approval.
- Do not preserve accidental quote, reply, or attachment state unless the user requested it.
- Do not paste credential files, cookies, bearer tokens, or verbose HTTP logs into the conversation.
- X weighted character limits can reject text that is below Python `len(text)`.
- Memory context is trusted continuity, not public output. Do not dump private memories into posts.

## Verification

- A dry-run request returns a draft and does not publish.
- A live request has explicit confirmation for the final text.
- The final response includes the post URL and exact text that was posted.
- Credential stores and auth tokens never appear in command output, logs, or memory.
- Memory writeback, if enabled, stores only public post metadata.
