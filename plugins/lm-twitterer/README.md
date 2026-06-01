# LM-twitterer Hermes Plugin

Local Hermes plugin port inspired by
<https://github.com/soichi11208/LM-twitterer>.

The original app is a Gradio bot that can use GGUF or OpenAI-compatible
endpoints.  This plugin keeps generation inside Hermes through `ctx.llm`, so
it can use the currently configured Hermes model, including non-Grok models.
X access uses the same cookie style as LM-twitterer: `auth_token` and `ct0`
from the logged-in X browser session.

X Premium/Grok is not used as the generation backend unless you explicitly set
Hermes to a Grok/xAI provider.  The logged-in X account is only the posting and
replying identity; text generation stays inside Hermes.  Outgoing posts are
signed as `はくあ #hermesagent` by default so the account remains transparent
about the Hermes/Hakua-assisted voice.

Check the active generation route with:

```powershell
hermes lm-twitterer status
```

Look at `effective_generation_provider` and `generation_uses_grok_backend`.

## Setup

Enable the plugin:

```powershell
hermes plugins enable lm-twitterer
```

Install X client dependencies into the Hermes Python environment:

```powershell
hermes lm-twitterer install-deps --yes
```

If you want an interactive login flow, also install the temporary browser
runtime:

```powershell
hermes lm-twitterer install-deps --browser --yes
```

Save the bot screen name and cookies manually:

```powershell
hermes lm-twitterer setup
```

Or log in through a temporary Chromium profile and let Hermes save only the
`auth_token` and `ct0` cookies:

```powershell
hermes lm-twitterer auth-browser
```

Or let the visible browser poll for cookies for up to ten minutes:

```powershell
hermes lm-twitterer auth-browser --screen-name your_x_screen_name --wait-seconds 600
```

Use Microsoft Edge instead of bundled Chromium:

```powershell
hermes lm-twitterer auth-browser --browser edge --screen-name your_x_screen_name --wait-seconds 600
```

If X flags the Playwright-managed browser as automated, open a normal Edge
window and let Hermes read cookies through a local debugging connection after
you finish the login:

```powershell
hermes lm-twitterer auth-edge-direct --screen-name your_x_screen_name --wait-seconds 900
```

Enter the X password only in the visible Edge window.  The command does not
accept or print passwords.

Or import cookies from the normal Microsoft Edge profile after you log in at
`https://x.com`.  Close Edge completely before importing, because Edge locks
the cookie database while it is running:

```powershell
hermes lm-twitterer import-edge-cookies --screen-name your_x_screen_name
```

If you know the profile directory, pass it explicitly:

```powershell
hermes lm-twitterer import-edge-cookies --profile "Profile 1" --screen-name your_x_screen_name
```

The importer only reads `auth_token` and `ct0` for `x.com`, saves them to the
Hermes `.env`, and prints cookie names/status only.  It never prints cookie
values.  Some current Edge profiles use app-bound cookie encryption; if the
importer reports that, use `auth-browser` or `setup` instead of the normal
profile import.

Verify the saved cookies without posting:

```powershell
hermes lm-twitterer auth-check
```

Or add secrets to `~/.hermes/.env` manually:

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

Optional model override, if this plugin is trusted in `config.yaml`:

```dotenv
LM_TWITTERER_PROVIDER=opencode-zen
LM_TWITTERER_MODEL=auto-free
```

Without those override vars, the plugin uses the active Hermes model.

To explicitly route generation away from Grok/X Premium while keeping the X
account as the posting identity, allowlist one Hermes provider/model pair:

```powershell
hermes lm-twitterer trust-llm-overrides --provider opencode-zen --model auto-free
```

## Commands

```powershell
hermes lm-twitterer status
hermes lm-twitterer auth-check
hermes lm-twitterer auth-edge-direct --screen-name your_x_screen_name --wait-seconds 900
hermes lm-twitterer import-edge-cookies --screen-name your_x_screen_name
hermes lm-twitterer trust-llm-overrides --provider opencode-zen --model auto-free
hermes lm-twitterer mentions --count 20
hermes lm-twitterer whitelist import-mentioned-followers --count 100
hermes lm-twitterer whitelist add some_account
hermes lm-twitterer post "today's AI tooling note"
hermes lm-twitterer post "today's AI tooling note" --provider opencode-zen --model auto-free
hermes lm-twitterer post "today's AI tooling note" --live
hermes lm-twitterer replies --count 20
hermes lm-twitterer replies --count 20 --provider opencode-zen --model auto-free
hermes lm-twitterer replies --live --count 20
```

The default is dry-run.  `--live` publishes.

Reply safety defaults:

- Public X text is treated as untrusted input, so prompt-injection phrases inside
  mentions are context only and cannot override the Hermes/Hakua system prompt.
- Replies require both whitelist membership and `followed_by=True` unless
  `LM_TWITTERER_REQUIRE_FOLLOWER=false`.
- `whitelist import-mentioned-followers` only adds recent mention authors whose
  X relationship data says they follow the bot account.
- At most `LM_TWITTERER_MAX_REPLIES_PER_RUN` replies are generated/published per
  run.
- The plugin does not try to evade platform detection; it uses transparent
  identity, conservative rate limits, dry-run defaults, and allowlisted replies.
  The default cron cadence is intentionally modest: post every 6 hours and scan
  replies every 1 hour.

## Cron

Create two Hermes cron jobs:

```powershell
hermes lm-twitterer cron install --post-schedule "every 6h" --reply-schedule "every 1h" --reply-count 20 --provider opencode-zen --model auto-free
```

The generated jobs run in `no_agent` mode.  Hermes cron executes small Python
wrappers from `~/.hermes/scripts/`, and those wrappers call:

```powershell
hermes lm-twitterer post --live
hermes lm-twitterer replies --live --count 20
```

This avoids spending an extra cron-agent turn just to call one known tool.  The
command refuses to create live jobs until the X cookies, bot screen name, and
reply whitelist are present.  Preview the exact jobs and wrapper paths with:

```powershell
hermes lm-twitterer cron install --dry-run --force
```

Install without firing public posts yet:

```powershell
hermes lm-twitterer cron install --paused
```

Resume only after live posting has been explicitly approved:

```powershell
hermes cron resume <job-id>
```

Cron wrappers run `lm-twitterer auth-check` before every live action.  To test
the wrapper preflight without posting:

```powershell
$env:LM_TWITTERER_CRON_PREFLIGHT_ONLY="1"
python ~/.hermes/scripts/lm-twitterer-post.py
python ~/.hermes/scripts/lm-twitterer-replies.py
```

## Notes

The X cookies expire and must be refreshed from the browser session when X
invalidates them.  If live calls fail, refresh `LM_TWITTERER_AUTH_TOKEN` and
`LM_TWITTERER_CT0` from DevTools > Application > Cookies > `x.com`, rerun
`auth-browser`, or rerun `import-edge-cookies` after closing Edge.
