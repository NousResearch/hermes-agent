---
name: torben-operator-maintenance
description: "Maintain Eric's Torben Signal COO operator: scheduled jobs, backend prompt contracts, action-ledger behavior, and user-facing draft/brief style guardrails."
---

# Torben Operator Maintenance

Use this skill when Eric asks to adjust Torben's backend behavior, scheduled jobs, Signal COO briefs, realtime email triage, email drafting, calendar/action-ledger handling, or operating guardrails.

## Core principles

1. Treat Torben as an operating system, not a chat-only assistant.
   - Find the durable backend surface that will affect future runs: cron job prompts, profile scripts, relationship context, action-ledger policy, or profile memory.
   - Do not only fix the current reply if Eric asked for backend behavior to change.

2. Preserve the mutation boundary.
   - Draft/stage/summarize is allowed when requested.
   - Sending email, changing calendars, posting, trading, archiving, labeling, or deleting external records requires explicit approval and verification.
   - When changing behavior, keep prompts explicit about “not sent / draft-only / no mutation” unless the user approved the mutation.
   - Successful background maintenance should be silent. Notify Eric only for actionable decisions, explicit approval requests, failures, blocked sources, dry-run findings, or mutation caps.
   - Gmail hygiene cleanup must be applied only through the guarded action-ledger resolver after Eric approves a specific handle.
   - Relationship learning updates belong in `config/learned_contacts.yaml` through the guarded resolver, then merge into `relationship_context.yaml` at runtime.
   - Morning newsletter/tool findings should use the durable findings ledger and should not repeat already-briefed stories or tools unless there is a materially new angle.
   - Pre-call meeting alerts should be actionable, deduped by event/time bucket, skip synthetic `Busy` blocks, and stay silent when no real meeting is approaching.

3. Prefer class-level updates.
   - If a recurring workflow changed, update the scheduled job prompt and the script-provided contract so future LLM runs see it.
   - If Eric expressed a durable preference, save compact memory and also encode it in the skill or backend prompt that governs the class of task.

## Workflow

1. Identify the live surface.
   - List relevant cron jobs when a scheduled behavior is involved.
   - Inspect profile scripts under the active Hermes profile when they emit LLM context.
   - Inspect job prompts when the LLM behavior is controlled by scheduler text.
   - Inspect relationship context when the change concerns people, source rules, or routing.
   - For Signal-facing identity changes, inspect the active Torben profile's Signal wrapper/state before touching files directly.

2. Patch the backend, not just the current answer.
   - For scheduled LLM jobs, update both:
     - the script output contract, when it supplies structured guardrails; and
     - the cron prompt, when it supplies the natural-language behavior contract.
   - Keep wording concise and operational.

3. For Torben Signal avatar/icon updates.
   - Treat the user-provided image as an external profile mutation request that is approved when Eric explicitly asks to add it as the icon.
   - Prepare a durable, square copy under `/Users/ericfreeman/.hermes/profiles/torben/assets/`, commonly `torben-signal-avatar.jpg` at 512x512.
   - Prefer the running signal-cli daemon JSON-RPC endpoint (`http://127.0.0.1:18080/api/v1/rpc`) for `updateProfile` and `getAvatar`; do not compete for the signal-cli data lock with direct foreground `signal-cli -d ...` commands while the daemon is running.
   - Verify by fetching the profile avatar back and comparing bytes/hash to the staged file.

4. Verify.
   - Run syntax checks for modified scripts.
   - Re-read or search for the new guardrail text in the profile to confirm it landed in the intended places.
   - For avatar/icon updates, verify the provider accepted the mutation and read back the resulting image.
   - Report exactly what changed and what was verified.

## Email draft guardrails for Eric

Eric explicitly prefers email drafts and posts to be tight and direct with no AI-sounding slop. Apply this whenever drafting or revising replies:

- Short sentences.
- Concrete facts.
- No padded empathy.
- No generic polish.
- No em dashes.
- No predictable “not X but Y” contrast templates.
- No beige corporate phrasing.
- No filler openers.
- Sound candid, human, specific, and practical.
- Produce usable copy; avoid generic assistant prefaces.

When Eric says a draft is “AI sounding,” “slop,” too padded, too verbose, or too polished, patch the governing email/social drafting contract immediately.

## Pitfalls

- Do not respond with only a better draft if Eric asked to update the backend.
- Do not assume memory alone is enough for operational behavior; scheduled jobs need their own prompts/contracts updated.
- Do not encode transient setup failures as durable rules.
- Do not weaken safety boundaries while improving tone.

## References

- `references/email-draft-guardrails-2026-06-25.md` — session-specific example of applying X/LinkedIn anti-slop guardrails to Torben email draft engines.
- `references/signal-avatar-update-2026-06-25.md` — working JSON-RPC pattern for setting and verifying Torben's Signal profile avatar without fighting the signal-cli data lock.
