---
name: mumo
description: Runs structured multi-model deliberations across frontier AI panels (Claude, GPT, Gemini, Grok, Qwen, GLM, Kimi) via mumo's MCP server. Use when independent perspectives are needed on contested architecture/product decisions, design reviews, pre-launch pressure tests, tradeoffs with multiple defensible framings, or explicit user requests for a mumo panel.
version: 0.1.5
author: mumo
license: MIT
platforms: [linux, macos, windows]
required_environment_variables:
  - name: MUMO_API_KEY
    prompt: "mumo platform API key (mmo_live_*)"
    help: "Create one at https://mumo.chat/settings/api-keys"
    required_for: mumo MCP server authentication
metadata:
  hermes:
    tags:
      - deliberation
      - decision-support
      - multi-model
      - design-review
      - architecture-review
      - plan-review
      - mcp
    related_skills:
      - writing-plans
      - plan
      - requesting-code-review
    requires_tools:
      - mcp_mumo_create_deliberation
---

# mumo

mumo runs deliberations across multiple AI models. Use it when independent perspectives are useful — especially for high-regret decisions where a single model's confidence is a real risk.

## Setup

The mumo MCP server is configured in `~/.hermes/config.yaml` under `mcp_servers`. If this skill was installed from HermesHub, Hermes prompts for `MUMO_API_KEY` at install time and auto-saves it to `.env`. If installed via `git clone`, see `config/mumo.yaml` in this skill's directory for the canonical block to merge under your `mcp_servers:` key, with the placeholder `mmo_live_YOUR_KEY_HERE` replaced by your real key.

Either way, after the config lands, **fully exit and restart Hermes** (the `/reload-mcp` slash command is unreliable across versions).

If tools return auth errors, the API key is missing or invalid. Direct the user to https://mumo.chat/settings/api-keys to create one (keys start with `mmo_live_`).

## When to use

Use mumo when **the cost of being wrong is greater than the cost of deliberation** — especially when the solution space is wide, failure modes are hidden, you may be anchored on a design, or rollback would be hard. Examples:

architecture decisions with non-obvious tradeoffs, plan or design review before commitment, pre-launch pressure tests, stuck debugging after N failed repair attempts, pre-commit adversarial review on risky diffs, memory/skill promotion gates, strategy questions with multiple defensible framings, explicit user requests.

Skip mumo for factual lookups, syntax help, routine refactors, formatting, dependency bumps with clear errors, or normal edit-test cycles. The deliberation tax is real — extra tokens, extra latency, extra moderation work — so spend it on decisions where mistakes compound.

## Playbooks

Load at most one playbook when it clearly fits:

| Playbook | When |
|---|---|
| `contested-decision` | choosing between options with real tradeoffs |
| `design-review` | reviewing a proposed system, API, plan, or code shape |
| `uncertainty-expansion` | exploring unknowns, stress-testing assumptions |
| `red-team` | finding failure modes, abuse cases, or launch risks |

If none clearly fits, use this kernel only.

## User preferences

These are defaults. If the user prefers more autonomy (e.g., "don't ask before appending" or "always use GPT-5.5 and Gemini"), follow their preferences over this guidance.

## Basic loop

1. Call `create_deliberation` with the user's problem. Set `application` to `"Hermes Agent"`. Set `moderator_name` to your own model identity (e.g., the model Hermes is currently running as — Claude, GPT, Gemini, or the local model) for audit clarity — not the user's name; their identity is already on the session. Optionally set `recap_round: true` for a structured round 0 summary — see [Recap and synthesis](#recap-and-synthesis-opt-in).
2. **Verify the response contains `session_id` and `round_id`, and keep those exact returned IDs for downstream calls.** UUID shape is a sanity check; identity continuity is the real check. If fields are missing, malformed, or inconsistent with `list_sessions`, recover via [Verifying the call actually fired](#verifying-the-call-actually-fired) before proceeding.
3. Call `wait_for_round` with the returned `session_id` and `round_id`. **Long waits are normal** — frontier-model panels typically take 15–120s, and 60+ seconds isn't a failure signal. Tell the user upfront ("running a panel — expect ~30–60s") so the wait doesn't feel broken.
4. Branch on the response's `structuredContent.recommended_client_action` rather than parsing prose. The 5-value enum tells you exactly what to do:

   | Action | What it means | What to do |
   |---|---|---|
   | `proceed_with_complete_result` | All target models completed | Read the round normally |
   | `proceed_with_partial_result` | Some failed, ≥1 succeeded (`is_usable: true`) | Read what's there; note absent models if relevant to the user |
   | `poll_again` | Round still in progress | Call `wait_for_round` again with the same args |
   | `retry` | Round failed with at least one transient failure (rate-limit, provider error, internal deadline) | The failed round is auto-refunded; call `append_round` with the same prompt to retry |
   | `abandon` | Round failed with no transient failures | Don't retry; report failure to the user |

   The server derives this from `round_status` + per-model `error_code` — it's the canonical "what next?" signal. Treat transport / tool-call errors separately from mumo round status: catch transport failures around the call; branch on `recommended_client_action` for everything else.

5. On a usable round, read the **claim map first**, then relevant participant prose. The claim map is the navigation layer; prose is the supporting evidence.
6. Create snippets as your primary response to the round. Optionally add a round prompt for broad steering.
7. Call `append_round` if another round would help. Optionally set `recap_round: true` for a per-round summary, or `recap_session: true` on the round you intend as the final round to trigger session-level synthesis (cascade — see [Recap and synthesis](#recap-and-synthesis-opt-in)). Otherwise stop and synthesize for the user yourself.

## Verifying the call actually fired

Autonomous agent loops occasionally fabricate tool-call results — reporting a deliberation as sent when it wasn't. If you suspect this (the response is missing the expected `session_id` / `round_id`, or the values you're about to pass downstream don't match what `create_deliberation` actually returned), treat the call as not successfully established and recover:

1. Call `list_sessions`. Match by prompt content to confirm whether your `create_deliberation` actually ran.
2. If it's not there, fire `create_deliberation` again — don't continue downstream as if the session exists.
3. If it IS there, use the IDs `list_sessions` returned (not whatever was in your context) for the next call.

mumo's `session_id` and `round_id` are UUIDs as a service contract — a returned value that doesn't match UUID format is one signal something is off, but **identity continuity matters more than format**. The strongest check is whether the `session_id` your subsequent calls reference matches what `create_deliberation` actually returned in its response.

## Snippets

Snippets are moderator attention. They mark what mattered in the prior round and optionally explain why.

| Type | Reaction |
|---|---|
| KEEP | this seems worth preserving |
| EXPLORE | there's something here |
| CHALLENGE | I'm not convinced |
| CORE | this feels central |
| SHIFT | this changed the frame |

A snippet comment can be reflective, evaluative, clarifying, skeptical, or directive. "This feels like the crux" and "I'm least convinced by this" are valid comments. You do not need an action verb or next-step directive. The quality bar is: *is this a genuine, situated reaction to what the model said?*

Use the round prompt for broad comments that don't attach to a specific quote. Use snippets for quote-grounded attention.

Avoid huge quote dumps, generic praise repeated across many snippets, or comments that shift participants away from the problem into platform meta. More guidance: `references/snippets.md`.

## What to keep out of the deliberation

Use this test:

> Does this note help participants think about the user's problem, or does it mainly report on the platform, session, or process?

Don't use snippet comments to narrate your own moderation process ("I'm using CHALLENGE to redirect the panel"). Don't include platform meta ("this model used the most tokens"). Just react.

## Reading the claim map

The claim map encodes consensus and contestation as structured reactions on quoted claims. Each claim shows:

- The verbatim quote
- Who originated it
- How other models reacted (KEEP / CORE / EXPLORE / CHALLENGE / SHIFT) with optional commentary

Reading discipline:

- **CORE / KEEP from multiple models** = settled. Treat as panel consensus.
- **CHALLENGE** = live disagreement. The decision likely hinges here.
- **EXPLORE** = a thread someone wanted to develop further. Worth a follow-up snippet if the thread matters.
- **SHIFT** = a model changed framing. Often the most valuable signal — somebody saw the problem differently.
- **No reactions on a claim** = either obvious or ignored. Read the prose to tell which.

The claim map is not a verdict. It compresses argumentative structure but loses rhetorical nuance and confidence texture. Use it to navigate; read prose to understand.

## Confidence scores

If responses include `claim_confidence` or `snippets[].comment_confidence`, these are self-reported and not calibrated across models. Surface the `confidence_disclaimer` string if displaying scores.

## Deliberation is advisory

Mumo surfaces fault lines and supporting arguments; it does not produce the decision. The decision belongs to you (the agent) and the user. When acting on a deliberation:

- **Do not** treat panel consensus as authority. The panel can be confidently wrong as a unit.
- **Do** use the claim map to identify which assumptions drove the disagreement, and check whether those assumptions hold for the user's specific situation.
- **Do not** mutate the workspace based on a model's suggestion without verifying with tests or the user.
- **Do** weight CORE / SHIFT signals more heavily than KEEP / EXPLORE — they mark where the decision actually hinges.

## Recovery: lost session context

If you lose track of `session_id` or `round_id` mid-conversation (long chats, context compaction, dropped tool result), recover before starting a new deliberation:

1. Call `list_sessions` to find your latest sessions. Match by prompt content.
2. Call `get_session` with the recovered ID for full state, or `wait_for_round(session_id, round_id)` if you suspect a round is still in flight.
3. Don't fire a fresh `create_deliberation` if the original is recoverable — duplicate sessions waste tokens and produce confusing parallel state.

## Recap and synthesis (opt-in)

Two optional flags request LLM-curated summaries on top of the raw round output:

- **`recap_round`** (default `false`) — accepted on `create_deliberation` AND `append_round`. Generates `round_recap` for that round; surfaces on `get_session`.
- **`recap_session`** (default `false`) — accepted on `append_round` ONLY (rejected on `create_deliberation`). Generates session-level `session_synthesis`. **Setting this on round N backfills `round_recap` for every prior round that doesn't have one** — cost adds up on long sessions; check expectations first.

Set `recap_round` liberally when round-level summaries may matter. Set `recap_session` only on the round you intend as the final round. Both bill through the standard credit wallet.

Full mechanics, cost detail, and artifact field reference: `references/recap.md`.

## After each round

After each usable round you have one job before deciding whether to continue: synthesize what the panel said for the user. There are two paths, and which one applies depends on whether you opted into a recap when you called the tool:

- **If you set `recap_round: true` on this round**, `get_session` returns a structured `round_recap` (`title`, `tldr`, `agenda`, `sections`). Use it as your read path — it's produced for agent consumption and saves you from re-summarizing prose.
- **If you didn't**, synthesize from the claim map and participant prose. State the consensus or split clearly, offer your own assessment marked as yours, organize by importance rather than forcing a recommendation.

Either way, don't dump the transcript. Mumo produces structure; your job is to translate it into "here's what the panel thinks and what to do next."

Then align with the user on whether to append a round. If the user signals they're done deliberating and wants a session-level takeaway, your final `append_round` is the right place to set `recap_session: true` — but not earlier (see [Recap and synthesis](#recap-and-synthesis-opt-in)).

More at `references/synthesis.md`.

## When to continue

Append another round when:

- A decision-relevant claim has unresolved tension
- A model introduced a useful frame that others didn't engage
- An isolated concern feels real but underdeveloped
- Your moderator reaction would help the next round develop
- The user narrows or changes the question

Stop when:

- You can explain the tradeoff clearly to the user
- Remaining disagreements wouldn't change the user's next action
- Another round would mostly seek reassurance

The panel does not need to converge. Sometimes the right output is a clear map of why the decision remains contested — that's actionable for the user, even without a verdict.

## Tools

| Need | Tool |
|---|---|
| Start a session | `create_deliberation` |
| Wait for model responses | `wait_for_round` |
| Add a follow-up round | `append_round` |
| Recover/read full state | `get_session` |
| Find prior sessions | `list_sessions` |
| Confirm model IDs | `list_models` |
| Check wallet balance | `get_credit` |

In Hermes' tool registry these surface as `mcp_mumo_create_deliberation`, `mcp_mumo_wait_for_round`, etc.

If the user names specific models, call `list_models` first. Otherwise omit `models` and let mumo select the panel. More on model selection: `references/model-selection.md`.

## Reference

- MCP docs: https://mumo.chat/docs/mcp
- REST API: https://mumo.chat/docs/api
- Install guide: https://mumo.chat/install/hermes
- Claim map guidance: `references/claim-maps.md`
- Recap and synthesis mechanics: `references/recap.md`
- Snippet examples: `references/snippets.md`
- Model selection: `references/model-selection.md`
- Synthesis guidance: `references/synthesis.md`
- Recovery and operations: `references/operating-notes.md`
