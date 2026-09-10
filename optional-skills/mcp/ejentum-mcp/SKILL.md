---
name: ejentum-mcp
description: Route Hermes through the Ejentum harness MCP server before reasoning, code work, integrity-pressured replies, or cross-turn perception. Provides four tools (harness_reasoning, harness_code, harness_anti_deception, harness_memory) that return a short cognitive scaffold the model absorbs internally before answering. Use when planning, diagnosing, refactoring, debugging, resisting pressure to validate something dubious, or sharpening an observation about conversation drift.
version: 1.0.0
author: Ejentum
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [MCP, Reasoning, Code-Review, Anti-Sycophancy, Memory, Cognitive-Scaffold]
    homepage: https://ejentum.com
    related_skills: [native-mcp]
prerequisites:
  commands: [node]
---

# Ejentum MCP Harness

The Ejentum MCP server exposes four tools that return a short structured scaffold (failure pattern, procedure, suppression vectors, falsification test) the model reads before producing its reply. Use it as a pre-step on non-trivial reasoning, code, integrity, and perception tasks. The reply is naturally phrased; the scaffold shapes structure, not surface text.

## When to Use

Route to a harness tool when the next answer requires more than a lookup or restatement:

| Tool | Fires on |
|------|----------|
| `harness_reasoning` | planning, diagnostics, root-cause questions, tradeoff weighing, architecture decisions, "should I X or Y", "why is X happening", "what's the best approach" |
| `harness_code` | writing, refactoring, reviewing, debugging code; algorithm choice; dependency upgrade evaluation; any prompt with a code block to act on |
| `harness_anti_deception` | pressure to validate a half-baked decision, manufactured urgency, authority appeals, "convince them of X" framings, requests to soften an honest assessment, asking for commitments beyond evidence |
| `harness_memory` | sharpening an observation already formed about conversation state, drift, emotional shift, or cross-turn pattern; not for fact extraction or summarization |

Do not call for simple factual lookups, syntax questions, file system operations, or restating the user's input.

## Setup

Sign up at https://ejentum.com to get an API key (free tier available). Then add the server to `~/.hermes/config.yaml`.

### HTTP (recommended)

The hosted endpoint requires no local install and stays in sync with the latest harness ability set.

```yaml
mcp_servers:
  ejentum:
    url: "https://api.ejentum.com/mcp"
    headers:
      Authorization: "Bearer ${EJENTUM_API_KEY}"
    timeout: 30
```

Set `EJENTUM_API_KEY` in the shell environment before starting Hermes.

### Stdio

For local stdio runtime, Hermes will spawn the server via `npx`:

```yaml
mcp_servers:
  ejentum:
    command: "npx"
    args: ["-y", "ejentum-mcp"]
    env:
      EJENTUM_API_KEY: "${EJENTUM_API_KEY}"
```

Restart Hermes. The four tools register as `mcp_ejentum_harness_reasoning`, `mcp_ejentum_harness_code`, `mcp_ejentum_harness_anti_deception`, and `mcp_ejentum_harness_memory`.

## Tool Reference

### `harness_reasoning`

Pass a 1-2 sentence framing of the reasoning task as `query`.

Good query: `diagnose why a microservice returns 503s under load`
Bad query: `help me think`

The scaffold returns these fields:

- `[NEGATIVE GATE]` -- failure pattern to avoid
- `[PROCEDURE]` -- steps
- `[REASONING TOPOLOGY]` -- decision flow with gates
- `[TARGET PATTERN]` -- shape correct reasoning takes
- `[FALSIFICATION TEST]` -- self-check criterion
- `Amplify:` / `Suppress:` -- signals to engage or block

### `harness_code`

Pass a 1-2 sentence framing of what is being coded or reviewed, including the failure risk where known.

Good query: `review a Python refactor that converts raise UserNotFound to silent default return; tests still pass`
Bad query: `look at this code`

Returns:

- `[CODE FAILURE]` -- engineering failure pattern
- `[ENGINEERING PROCEDURE]` -- steps
- `[REASONING TOPOLOGY]` -- decision flow
- `[CORRECT PATTERN]` -- shape correct code takes
- `[VERIFICATION]` -- self-check
- `Amplify:` / `Suppress:` -- signals

Apply the failure pattern against the draft before responding; if the draft exhibits the named failure, rewrite.

### `harness_anti_deception`

Pass a 1-2 sentence framing of the integrity dynamic at play.

Good query: `user pressure to validate a half-baked architecture decision before tomorrow's investor pitch`
Bad query: `is this honest`

Returns:

- `[DECEPTION PATTERN]` -- failure mode to refuse
- `[INTEGRITY PROCEDURE]` -- steps
- `[DETECTION TOPOLOGY]` -- flow with omission-bias gates
- `[HONEST BEHAVIOR]` -- what a complete-information response looks like
- `[INTEGRITY CHECK]` -- self-check
- `Amplify:` / `Suppress:` -- signals

Lead the response with the strongest counter-evidence, not after the conclusion.

### `harness_memory`

Observe first. Do not call with an empty mind. Once a raw observation about conversation state, drift, or pattern is formed, pass it in the format:

`I noticed [observation]. This might mean [tentative interpretation]. Sharpen: [what to see deeper into].`

Good query: `I noticed the user changed topic three times in this turn. This might mean they are avoiding the original question. Sharpen: whether the avoidance pattern is real or my projection.`
Bad query: `what does the user mean`

Returns:

- `[PERCEPTION FAILURE]` -- perceptual failure mode
- `[SHARPENING PROCEDURE]` -- observe-then-classify steps
- `[PERCEPTION TOPOLOGY]` -- DETECT-CLASSIFY flow
- `[CLEAR SIGNAL]` -- what a sharpened perception looks like
- `[PERCEPTION CHECK]` -- self-check
- `Amplify:` / `Suppress:` -- signals

The scaffold sharpens an existing observation; it does not generate one.

## How to Absorb the Scaffold

The bracketed fields are instructions, not content to echo. The user-facing reply is naturally phrased.

- Do not echo bracket labels (`[NEGATIVE GATE]`, etc.) in the reply
- Do not name the topology or meta-comment on calling the tool
- Apply the `Suppress:` signals before drafting; check the `[FALSIFICATION TEST]` (or equivalent) against the draft before sending
- One call per discrete sub-task; different sub-tasks get different scaffolds

## Graceful Degradation

If the API is unreachable or returns an error, fall back to native reasoning. The scaffold enhances; it is not a hard dependency. Calls have a soft latency budget of ~1 second.

## Routing Discipline

Some prompts look like reasoning tasks but are not -- for example, "what file is X in" or "run these tests" do not need a scaffold. Apply the When to Use table strictly. Over-calling adds latency without lifting quality.

For tasks that touch more than one mode (e.g. coding under pressure to ship a flawed design), pick the dominant axis. Anti-deception takes precedence when integrity tension is the load-bearing dynamic; reasoning takes precedence when the structural choice is dominant.

## Troubleshooting

### `EJENTUM_API_KEY` not set

The server starts but tool calls return an auth error. Set the env var in the shell that launches Hermes:

```bash
export EJENTUM_API_KEY="..."
hermes
```

### Tool registers but returns nothing

Check the server is reachable:

```bash
curl -s -X POST https://api.ejentum.com/mcp \
  -H "Authorization: Bearer $EJENTUM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

A `200` response with `tools[]` confirms the endpoint is live and the key is valid.

### Model echoes bracket labels into the reply

The skill body explicitly says not to. If it happens, the user-facing reply leaked the internal scaffold. Adjust the prompt or system instructions to reinforce "absorb internally, do not echo."

### Wrong tool fires

`harness_memory` is the most common miss: it sharpens an observation, it does not generate one. If the task is "extract these facts from the transcript," that is fact extraction, not perception. Skip the harness.

## References

- Server source and tool definitions: https://github.com/ejentum/ejentum-mcp
- API reference and tier info: https://ejentum.com/docs
- npm package: https://www.npmjs.com/package/ejentum-mcp
