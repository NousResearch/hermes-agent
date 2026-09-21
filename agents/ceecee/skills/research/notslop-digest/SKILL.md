---
name: notslop-digest
description: Pull a multi-source social digest (Reddit, HN, blogs, optional X) on a topic using the notslop CLI, reranked by ZeroEntropy. Feeds current real signal into content drafting. Backbone for grounded content pipeline.
version: 1.0.0
author: Sahil Saghir
license: MIT
metadata:
  hermes:
    tags: [research, social-digest, content-pipeline, grounding, real-time-signal, notslop]
    related_skills: [notslop-content]
---

# notslop-digest

Fresh social context for content drafting. Uses the `notslop` CLI to fetch current posts from Reddit, Hacker News, RSS blogs, and X, reranked by ZeroEntropy. The output grounds your drafts in what people are actually saying right now, not your training data.

## Setup (one-time)

1. **Install notslop globally** (already done on this VPS via `npm install -g notslop@0.8.0`)

2. **Get a ZeroEntropy API key** (free):
   - Sign up at https://dashboard.zeroentropy.dev
   - Generate an API key (format: `ze_xxxxxxxxxxxx`)
   - Set it: `notslop init` or `export ZEROENTROPY_API_KEY=ze_xxxx`

3. **Optional — X/Twitter** (only if you need X signal):
   - Sign up at https://orthogonal.com/sign-up ($10 free credits)
   - Set key: `export ORTHOGONAL_API_KEY=orth_live_xxxx`

4. **Verify it works**:
   ```bash
   npx notslop@latest digest "test" --since 6h --top 3
   ```
   Should return reranked results (not just an error).

## When to use this skill

Activate when:
- "What's everyone saying about <topic>?"
- "Give me a digest on <topic> for content drafting"
- "Pull fresh social signal on <topic>"
- "What's the buzz around <release/company/product>?"
- "Ground this post in current conversations about <topic>"
- Any time you need to check what's actually being discussed today before drafting

## Commands available

| Command | Purpose |
|---|---|
| `notslop digest "<topic>" --since 24h --format md` | Multi-source digest, curated summary |
| `notslop digest "<topic>" --since 7d --top 15 --for-content` | Token-efficient output for feeding into content drafts |
| `notslop trending "<niche>" --since 6h --format md` | What's blowing up right now |
| `notslop pulse "<topic>" --since 7d --format md` | Mention tracker, clustered into themes |
| `notslop voices "<topic>" --format md` | Influential authors on a topic |
| `notslop find-related "<URL_or_text>" --since 14d --top 10` | Semantic similarity to a draft or URL |
| `notslop sources` | Status of every configured source |

## Steps

1. **Extract the topic** from the user's prompt. Keep it short and quoted.

2. **Pick a window:**
   - `6h` — very fresh, breaking-news level
   - `24h` — default for most content drafting
   - `7d` — LinkedIn posts, weekly recaps
   - `30d` — slow-moving topics, research

3. **Run the digest**:
   ```bash
   npx notslop@latest digest "<TOPIC>" --since <WINDOW> --top 10 --format md
   ```

4. **Parse the output.** It includes source tags (`reddit`, `hn`, `blogs`, `x`) and URLs. Preserve these — they're the grounding for your draft.

5. **Identify 2-3 specific data points** from the output to ground your draft in:
   - A concrete number or stat
   - A recent discussion trend
   - A specific quote or observation

6. **Feed into content drafting.** Pass the digest output + selected data points as context. Reference the matched brand voice skill for tone.

## Example

> User: "Draft me a LinkedIn post about AI in product management for 2026"
>
> → Load `notslop-digest` skill
> → Run: `npx notslop@latest digest "AI product management 2026" --since 7d --top 10 --for-content`
> → Parse output: identify themes (context engineering, Datadog report, RAG adoption)
> → Load `sahil-linkedin-voice` skill
> → Draft grounded in the real data points with Sahil's LinkedIn voice
> → Show draft + cite 2-3 source URLs below it

## Fallback behaviour

If ZeroEntropy key is not configured, the CLI returns an actionable error message. Surface it to the user and stop — don't guess.

If X data is unavailable (no Orthogonal key), the digest still works with Reddit + HN + blogs. That's usually enough for content grounding. Only request Orthogonal setup if the topic specifically demands Twitter signal.

## Pitfalls

- The CLI caches results to avoid re-fetching. Use `--since` to control freshness.
- The `--for-content` flag gives token-efficient output, best for feeding agentic drafting.
- The `--top N` flag controls how many results come back. 10 is usually enough.
- British English users: the digest output contains whatever people wrote. Don't copy phrasing — use the data points as grounding, not templates.
- If `npx notslop@latest digest` hangs, it's likely waiting for network. Check connectivity.

## Verification

After setup, run once to confirm:
```bash
npx notslop@latest digest "Claude Code" --since 6h --top 5
```
Should return reranked posts with source tags. If it errors, the ZeroEntropy key isn't set.
