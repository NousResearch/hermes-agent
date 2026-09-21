# hermes-multichannel-prompt-optimizer

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that rewrites your prompts before they reach the LLM. Works across every surface Hermes runs on: CLI, TUI, Discord, Telegram, and any other gateway adapter.

Same agent, sharper prompts, lower token bills, better answers — without you having to think about prompt craft.

---

## Why

A senior PM types `"hey could you maybe explain to me very kindly what python generators are please when you get a chance"` — 18 words, mostly filler. The model burns context on politeness and noise. With this plugin, the agent sees `"Explain Python generators."` — clearer, cheaper, and the answer comes back sharper.

This pattern repeats across every conversation. Over a month, the savings are meaningful: in dogfooding so far the optimiser averages **+55 quality points** and **20–80% token reduction** per rewrite. You also build up a private dataset of your own prompt patterns and improvements, viewable via `/prompt-insights`.

---

## What it does

- **Intercepts** every user message before it reaches the agent — on CLI, TUI, and any gateway platform.
- **Tailors** the rewrite to the target model along two axes: **vendor family** (claude, openai, deepseek, google, nvidia, kimi, qwen, mistral) and **capability** (reasoning vs general). Same prompt headed to o3-mini gets front-loaded constraints; same prompt headed to Claude Sonnet gets XML-tagged structure.
- **Preserves non-English prompts** — automatically detects the language and keeps the rewrite in the same language, no translation.
- **Scores** before/after across 5 dimensions: clarity, specificity, terminology, actionability, structure.
- **Records** every rewrite into a local SQLite database for analytics and longitudinal coaching.
- **Surfaces** insights via slash commands: comparisons, reusable suggestions, analytics by day/week/month.
- **Renders** an arrow-key approval overlay on CLI (via `ctx.ask_user`) and a full before/after panel in the TUI.

---

## Surfaces

| Surface | Auto mode | Interactive mode |
|---|---|---|
| `hermes chat` (CLI) | Silent rewrite | Arrow-key overlay (accept / reject) |
| `hermes chat --tui` | Silent rewrite | Before/after panel with quality scores |
| Discord | Silent rewrite | Diff sent as a message; reply `y` / `n` |
| Telegram, Slack, IRC, etc. | Silent rewrite | Same as Discord |

### Multi-language support

The plugin **automatically detects non-English prompts** and preserves the original language during rewriting. If you type in Arabic, French, Chinese, Spanish, German, Russian, or any other language:

- The rewriter is instructed to **keep the same language** — no translation.
- The heuristic quality scorer **skips English-only checks** (action verbs, context words) so non-English prompts aren't unfairly penalised.
- Detection uses `langid` (97 languages) with a Unicode-range fallback for CJK, Arabic, Cyrillic, Hebrew, Devanagari, etc.

The detection runs at the engine level, so it applies to **all surfaces equally** — CLI, TUI, Discord, Telegram.

To install the language detection dependency:

```bash
pip install langid
```

Without `langid`, the plugin falls back to a Unicode-range heuristic that catches CJK, Arabic, Cyrillic, Hebrew, and other non-Latin scripts, but won't distinguish French/Spanish/German etc. from English for all-ASCII text.

---

## Requirements

- **Hermes Agent** with the `pre_user_message` plugin hook. This hook is needed for CLI/TUI rewrites. If your Hermes build is missing it, the gateway path (Discord/Telegram/Slack/...) still works via `pre_gateway_dispatch` — only `hermes chat` and `hermes chat --tui` are affected.
  - Upstream PR adding the hook: [NousResearch/hermes-agent#29526](https://github.com/NousResearch/hermes-agent/pull/29526). Once merged, every Hermes install will support all surfaces out of the box.
- Python 3.11+
- **Optional:** `pip install langid` for accurate language detection across 97 languages. Without it, only non-Latin scripts (CJK, Arabic, Cyrillic, etc.) are detected via Unicode-range heuristic.
- An LLM provider configured in Hermes for the optimiser model (the plugin uses Hermes's `ctx.llm` facade, so it inherits your active provider/auth — no separate keys needed by default).

---

## Install

```bash
hermes plugins install Sahil-SS9/hermes-multichannel-prompt-optimizer
```

Then enable it in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - prompt-optimizer
```

Restart your Hermes session. Confirm it's loaded:

```bash
hermes plugins list | grep prompt-optimizer
```

You should see it as `enabled`. Then in `hermes chat`:

```
/prompt-optimizer status
```

If the status block prints, you're set.

---

## Modes

| Mode | Behaviour |
|---|---|
| `auto` *(default)* | Silent rewrite — agent sees the optimised version, you don't see the diff. |
| `interactive` | Show the diff first, ask for approval before sending. |
| `off` | Pass everything through untouched. |

Toggle mid-session:

```
/prompt-optimizer auto
/prompt-optimizer interactive
/prompt-optimizer off
```

---

## Slash commands

| Command | Description |
|---|---|
| `/prompt-optimizer [auto\|interactive\|off\|status]` | Set mode or print status. |
| `/prompt-insights` | Full report: overview, insights, suggestions, comparisons, analytics. |
| `/prompt-insights --html` | Same report plus a styled HTML file under `reports/`. |
| `/prompt-compare --limit 5` | Latest before/after comparisons. |
| `/prompt-suggestions --limit 8` | Reusable prompt-replacement patterns mined from your history. |
| `/prompt-analytics [daily\|weekly\|monthly\|all]` | Period analytics. |
| `/prompt-stats --raw` | JSON summary for today, week, month. Useful for cron / dashboards. |

---

## Configuration

The plugin needs no `config.yaml` entries to run with sensible defaults. To pin the optimiser to a specific cheap-and-fast model, override under `plugins.entries`:

```yaml
plugins:
  enabled:
    - prompt-optimizer
  entries:
    prompt-optimizer:
      llm:
        allow_model_override: true
        allowed_models:
          - deepseek-v4-flash
        allow_provider_override: true
        allowed_providers:
          - nous
```

This isolates the optimiser's LLM cost from your main session model — you can run Claude Opus for the agent while a £0.05/M token model handles rewrites.

### Model profiles — how the rewrite gets tailored

Every model resolves along two orthogonal axes:

1. **Family** — one of `claude`, `openai`, `deepseek`, `google`, `nvidia`, `kimi`, `qwen`, `mistral`, `llama`, `nousresearch`, `xai`, `amazon`, `cohere`, `microsoft`, `perplexity`, `zhipu`, `liquid`, `minimax`, `ibm`, `inflection`, `xiaomi`, or `None` (unknown vendor). Coverage spans ~21 vendor families derived from the OpenRouter catalogue.
2. **Capability** — `reasoning` (o-series, r-series, `*-thinking`, magistral, glm-z, deepresearch, nemotron-3-super, qwq, allenai olmo-think, etc.) or `general` (everything else where a family was detected).

Examples:

| Model string | Resolves to |
|---|---|
| `claude-opus-4-7` | `(claude, general)` |
| `openai/gpt-4o` | `(openai, general)` |
| `openai/o3-mini` | `(openai, reasoning)` |
| `deepseek/deepseek-r1` | `(deepseek, reasoning)` |
| `gemini-2.0-flash-thinking` | `(google, reasoning)` |
| `nvidia/nemotron-3-super-120b-a12b` | `(nvidia, reasoning)` |
| `mistralai/magistral-medium-2509` | `(mistral, reasoning)` |
| `meta-llama/llama-4-scout` | `(llama, general)` |
| `nousresearch/hermes-4-70b` | `(nousresearch, general)` |
| `x-ai/grok-4` | `(xai, general)` |
| `cohere/command-a` | `(cohere, general)` |
| `perplexity/sonar-pro-search` | `(perplexity, general)` |
| `z-ai/glm-z1-reasoning` | `(zhipu, reasoning)` |
| `minimax/minimax-m2.7` | `(minimax, general)` |
| `allenai/olmo-3-32b-think` | `(None, reasoning)` — capability without known family |
| `unknown-vendor/foo` | `(None, None)` — base template, no injection |

The rewriter system prompt is built by **composing** whichever axes resolved. `gpt-4o` gets the openai-family tactics. `o3-mini` gets the openai-family tactics PLUS the reasoning capability tactics. Unknown models fall through to the base template — no fake guidance injected.

### Editing `model-profiles.yaml`

The shipped YAML has full coverage for the 8 families above plus the two capability profiles. You can edit it in place at `~/.hermes/plugins/prompt-optimizer/model-profiles.yaml`:

```yaml
families:
  claude:
    prompt_tactics:
      - "Use XML tags (<thinking>, <answer>, <example>) to mark structure"
      - "State constraints and boundaries explicitly"
      # … add or override any rule …
    token_efficiency_rules:
      - "Replace 'Could you please' with imperative verbs"

capabilities:
  reasoning:
    prompt_tactics:
      - "Front-load ALL constraints — no incremental hints"
      # …

family_aliases:
  claude: ["claude-", "anthropic/"]
  # …

reasoning_indicators:
  - "o1"
  - "o3"
  - "thinking"
  # …
```

The tactics are sourced from each vendor's published prompt-engineering guidance:

- Claude — https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering
- OpenAI (general) — https://platform.openai.com/docs/guides/prompt-engineering
- OpenAI (reasoning) — https://platform.openai.com/docs/guides/reasoning
- DeepSeek — https://api-docs.deepseek.com/guides/reasoning_model
- Gemini — https://ai.google.dev/gemini-api/docs/prompting-intro

If your local YAML is missing or malformed, the plugin falls back to baked-in defaults so nothing breaks.

---

## Privacy

- All metrics live in a **local SQLite database** at `~/.hermes/plugins/prompt-optimizer/metrics.db`. Nothing is uploaded.
- The optimiser does call your configured LLM provider for the rewrite step — that's a third-party API call subject to your provider's privacy policy. If you don't want any external calls, set `/prompt-optimizer off`.
- The local database keeps 90 days of rewrites by default before pruning. Delete `metrics.db` any time to reset.

---

## Hooks used

| Hook | Purpose |
|---|---|
| `pre_user_message` | Rewrite messages from CLI / TUI before they reach the agent. |
| `pre_gateway_dispatch` | Rewrite messages from Discord / Telegram / Slack / etc. |
| `transform_llm_output` | Append an inline quality-badge to the assistant's reply when a rewrite happened. |

---

## Bypass prefixes

The plugin used to support `/quick`, `*simple`, `#basic` as one-off bypasses. In practice the slash-command dispatcher in `hermes chat` claims anything starting with `/`, so only gateway surfaces honour the prefixes reliably. **Recommended**: use mode flips (`/prompt-optimizer off` then `/prompt-optimizer auto`) instead.

---

## Development

Clone, edit, link into Hermes:

```bash
git clone https://github.com/Sahil-SS9/hermes-multichannel-prompt-optimizer ~/.hermes/plugins/prompt-optimizer
hermes plugins enable prompt-optimizer
```

Run the test suite:

```bash
cd /path/to/hermes-agent
venv/bin/pytest tests/plugins/test_prompt_optimizer_plugin.py -v
```

PRs welcome. Please include tests for any new hook semantics or scoring changes.

---

## Roadmap

- [x] Family + capability composition for model-tailored rewrites.
- [x] Fix CLI `model=""` plumbing so the target model reaches the optimiser.
- [x] Multi-language support — auto-detect non-English prompts, preserve original language during rewrite.
- [ ] LLM-judged second-pass scoring (ask the target model to rate the rewrite). Adds latency; pending data on whether composition alone is enough.
- [ ] Per-user model-profile overrides scoped per session.
- [ ] Optional GitHub Actions example for cron-driven weekly digests posted to Discord/Slack.

---

## Credits

Built by [Sahil Saghir](https://github.com/Sahil-SS9) for the KENSEI / Octacon personal-agent stack. Released under MIT in case it's useful to anyone else running Hermes Agent in production.

---

## License

MIT — see [LICENSE](./LICENSE).
