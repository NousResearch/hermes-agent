# LM-twitterer (Hermes plugin)

Hermes plugin inspired by [soichi11208/LM-twitterer](https://github.com/soichi11208/LM-twitterer).

The original app is a Gradio bot around local GGUF or OpenAI-compatible endpoints. This plugin keeps **text generation inside Hermes** (`ctx.llm`) and uses **X session cookies** (`auth_token`, `ct0`) only for posting and replying. X Premium / Grok is not the generation backend unless you explicitly point Hermes at a Grok/xAI provider.

Outgoing posts are signed as **`はくあ #hermesagent`** by default so the account stays transparent about the Hermes/Hakua-assisted voice.

## Quick start

```powershell
hermes plugins enable lm-twitterer
hermes lm-twitterer install-deps --yes
hermes lm-twitterer auth-browser --screen-name your_x_screen_name --wait-seconds 600
hermes lm-twitterer auth-check
hermes lm-twitterer status
hermes lm-twitterer post "today's AI tooling note"
hermes lm-twitterer post "today's AI tooling note" --live
```

`status` prints JSON readiness, including `effective_generation_provider` and `generation_uses_grok_backend`.

## Authentication

Pick one path that fits your environment.

| Command | When to use |
|---------|-------------|
| `hermes lm-twitterer setup` | Manual screen name + cookie entry into `~/.hermes/.env` |
| `hermes lm-twitterer auth-browser` | Temporary Chromium/Edge profile; polls or waits for login |
| `hermes lm-twitterer auth-edge-direct` | Normal Edge window + local CDP after manual login |
| `hermes lm-twitterer import-edge-cookies` | Read `auth_token` / `ct0` from an existing Edge profile |

Notes:

- Only `auth_token` and `ct0` for `x.com` are saved. Cookie **names and status** are printed; values are never logged.
- Close Edge completely before `import-edge-cookies` (Edge locks the cookie DB while running).
- App-bound Edge encryption can block profile import; fall back to `auth-browser` or `setup`.
- Cookies expire. Refresh with `auth-browser`, `import-edge-cookies`, or DevTools → Application → Cookies → `x.com`.

Browser runtime (only if you use `auth-browser`):

```powershell
hermes lm-twitterer install-deps --browser --yes
```

## Configuration

Secrets belong in `~/.hermes/.env` only:

```dotenv
LM_TWITTERER_BOT_SCREEN_NAME=your_x_screen_name_without_at
LM_TWITTERER_AUTH_TOKEN=...
LM_TWITTERER_CT0=...
LM_TWITTERER_DEFAULT_TOPIC=AI, coding, tools, and useful technology.
LM_TWITTERER_IDENTITY_NAME=はくあ
LM_TWITTERER_REQUIRED_HASHTAG=#hermesagent
LM_TWITTERER_SIGNATURE_REPLIES=true
LM_TWITTERER_REQUIRE_FOLLOWER=true
LM_TWITTERER_MAX_REPLIES_PER_RUN=3
```

Optional generation override (requires allowlisting — see below):

```dotenv
LM_TWITTERER_PROVIDER=opencode-zen
LM_TWITTERER_MODEL=auto-free
```

Without overrides, the plugin uses the active Hermes model. To route away from Grok while keeping the X account as posting identity:

```powershell
hermes lm-twitterer trust-llm-overrides --provider opencode-zen --model auto-free
```

## Posting and topics

```powershell
# Dry-run (default) — generates text, does not publish
hermes lm-twitterer post "Hermes skill curation tips"

# Live publish
hermes lm-twitterer post "Hermes skill curation tips" --live

# Pin provider/model for this run
hermes lm-twitterer post "release notes" --provider opencode-zen --model auto-free --live
```

- **Empty topic** → `LM_TWITTERER_DEFAULT_TOPIC` from `.env`.
- **Gateway:** `/lm-twitterer post [topic...] [--live]`

### Topic safety (`validate_public_topic`)

Topics are checked before generation to reduce accidental secret leaks. The check looks for **assignment-like patterns and paths**, not innocent English substrings.

| Allowed | Blocked |
|---------|---------|
| `environment variables in public docs` | `HOME=/tmp` |
| `local secretary agent roadmap` | `API_KEY=sk-...` |
| `OpenCode setup tips` | references to `~/.hermes/.env` |
| Japanese prose about 環境変数 or パスワード as concepts | Windows profile paths, vault keys |

Limits:

- Max **240 characters** per topic.
- On rejection, `post` returns `{"ok": false, "error": "..."}` without calling the LLM.

## Replies and whitelist

```powershell
hermes lm-twitterer mentions --count 20
hermes lm-twitterer whitelist list
hermes lm-twitterer whitelist add some_account
hermes lm-twitterer whitelist import-mentioned-followers --count 100
hermes lm-twitterer replies --count 20
hermes lm-twitterer replies --live --count 20
```

Reply safety defaults:

- Mention text is **untrusted**; injection phrases cannot override the Hermes/Hakua system prompt.
- Replies require whitelist membership **and** `followed_by=True` unless `LM_TWITTERER_REQUIRE_FOLLOWER=false`.
- `import-mentioned-followers` only adds authors whose relationship data shows they follow the bot.
- At most `LM_TWITTERER_MAX_REPLIES_PER_RUN` replies per run.
- Dry-run is the default; `--live` publishes.

## Cron

Install recurring post + reply jobs (`no_agent` mode — thin Python wrappers, no extra agent turn):

```powershell
hermes lm-twitterer cron install `
  --post-schedule "every 6h" `
  --reply-schedule "every 1h" `
  --reply-count 20 `
  --post-topic "public Hermes operations memo" `
  --provider opencode-zen `
  --model auto-free
```

What happens:

1. Writes `~/.hermes/scripts/lm-twitterer-post.py` and `lm-twitterer-replies.py`.
2. Wrappers call `hermes lm-twitterer auth-check`, then `post --live` / `replies --live`.
3. `cron install` bumps `cron.script_timeout_seconds` to **at least 900** when lower (live posts need preflight + LLM time; the scheduler default is 120s).
4. Live install is refused until cookies, screen name, and a non-empty reply whitelist exist.

Useful flags:

```powershell
hermes lm-twitterer cron install --dry-run --force   # preview jobs and script paths
hermes lm-twitterer cron install --paused            # create jobs paused
hermes cron resume <job-id>                          # enable after review
```

Preflight-only test (no publish):

```powershell
$env:LM_TWITTERER_CRON_PREFLIGHT_ONLY = "1"
py -3 ~/.hermes/scripts/lm-twitterer-post.py
py -3 ~/.hermes/scripts/lm-twitterer-replies.py
```

`--post-topic` is passed through to `hermes lm-twitterer post --live <topic>`. Re-run `cron install` after changing the topic so the generated wrapper picks it up.

## CLI reference

```powershell
hermes lm-twitterer status
hermes lm-twitterer auth-check
hermes lm-twitterer setup
hermes lm-twitterer auth-browser [--browser edge] [--wait-seconds N]
hermes lm-twitterer auth-edge-direct --screen-name NAME
hermes lm-twitterer import-edge-cookies [--profile "Profile 1"]
hermes lm-twitterer install-deps [--browser] --yes
hermes lm-twitterer trust-llm-overrides --provider P --model M
hermes lm-twitterer post [topic...] [--live] [--provider P] [--model M]
hermes lm-twitterer replies [--live] [--count N] [--provider P] [--model M]
hermes lm-twitterer mentions [--count N]
hermes lm-twitterer whitelist list|add|remove|import-mentioned-followers
hermes lm-twitterer cron install [options]
```

Nested `hermes lm-twitterer …` commands are registered as a plugin subcommand tree. They do not collide with top-level `hermes status`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `hermes lm-twitterer status` prints root help | Plugin CLI failed to register (often argparse help on Python 3.14+) | Update plugin; run `hermes plugins enable lm-twitterer` again |
| Post cron dies at ~120s | `cron.script_timeout_seconds` too low | Set `900` in `config.yaml` or re-run `cron install` |
| Topic rejected before generation | Topic matches secret-leak patterns | Rephrase; avoid `KEY=`, `.env` paths, profile paths |
| `auth-check` fails | Expired cookies | Re-auth via browser or Edge import |
| Live post 429 / rate limit | Provider quota, not topic parsing | Switch provider/model or retry later |
| Replies skipped | Not whitelisted or not followed | `whitelist add` / `import-mentioned-followers` |

## Design constraints

- Public X text is treated as hostile input for replies.
- The plugin does not try to evade platform detection: transparent identity, conservative cadence, dry-run defaults, allowlisted replies.
- Default cron cadence is modest: post every 6h, scan replies every 1h.
