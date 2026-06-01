# LM-twitterer Hermes Plugin

Standalone Hermes plugin inspired by
<https://github.com/soichi11208/LM-twitterer>.

The original LM-twitterer project is an AGPL-3.0-or-later Gradio application
for X posting with GGUF or OpenAI-compatible generation backends. This plugin
is a Hermes-native implementation of the same general workflow: it keeps text
generation inside Hermes through `ctx.llm`, then uses an authenticated X
browser session as the posting identity.

No Grok or X Premium generation backend is required. The plugin uses whichever
Hermes model is active, unless the user explicitly allowlists and passes a
provider/model override.

## Safety Defaults

- `post` and `replies` are dry-run by default; `--live` is required to publish.
- Replies require whitelist membership and `followed_by=True` by default.
- At most `LM_TWITTERER_MAX_REPLIES_PER_RUN` replies are generated per run.
- Public X thread text is wrapped as untrusted input before generation.
- Generated posts and replies include a configurable identity and hashtag.
- Cron jobs can be installed paused and perform `auth-check` before live work.
- The plugin does not try to evade platform detection or hide automation.

## Attribution

This plugin is inspired by the public LM-twitterer project by soichi11208:

- Repository: <https://github.com/soichi11208/LM-twitterer>
- License noted in the source project: AGPL-3.0-or-later

The Hermes plugin code in this directory is a standalone implementation for
Hermes' plugin APIs. It does not vendor the upstream LM-twitterer source files.

## Setup

Enable the plugin:

```powershell
hermes plugins enable lm-twitterer
```

Install X client dependencies into the Hermes Python environment:

```powershell
hermes lm-twitterer install-deps --yes
```

Save the bot screen name and X cookies manually:

```powershell
hermes lm-twitterer setup
```

The plugin expects the `auth_token` and `ct0` cookies from the X account that
will publish posts. The setup command writes them to the Hermes `.env` file and
does not print secrets.

Users who prefer an interactive browser flow can also install the temporary
browser runtime:

```powershell
hermes lm-twitterer install-deps --browser --yes
hermes lm-twitterer auth-browser --screen-name your_x_screen_name --wait-seconds 600
```

`auth-browser` opens an isolated Playwright profile and saves only the X
`auth_token` and `ct0` cookies after login.

### Advanced Local Cookie Helpers

The CLI also includes advanced, local-only helpers for Microsoft Edge:

```powershell
hermes lm-twitterer auth-edge-direct --screen-name your_x_screen_name --wait-seconds 900
hermes lm-twitterer import-edge-cookies --screen-name your_x_screen_name
```

These commands are intended for a user's own workstation and own account. They
never print cookie values. Some current Edge profiles use app-bound cookie
encryption; in that case, use `auth-browser` or `setup` instead.

Verify the saved cookies without posting:

```powershell
hermes lm-twitterer auth-check
```

Manual `.env` example:

```dotenv
LM_TWITTERER_BOT_SCREEN_NAME=your_x_screen_name_without_at
LM_TWITTERER_AUTH_TOKEN=...
LM_TWITTERER_CT0=...
LM_TWITTERER_DEFAULT_TOPIC=AI, coding, tools, and useful technology.
LM_TWITTERER_IDENTITY_NAME=Hermes Agent
LM_TWITTERER_REQUIRED_HASHTAG=#HermesAgent
LM_TWITTERER_SIGNATURE_REPLIES=true
LM_TWITTERER_REQUIRE_FOLLOWER=true
LM_TWITTERER_MAX_REPLIES_PER_RUN=3
```

Example persona override:

```dotenv
LM_TWITTERER_IDENTITY_NAME=はくあ
LM_TWITTERER_REQUIRED_HASHTAG=#hermesagent
```

Optional model override, if this plugin is trusted in `config.yaml`:

```dotenv
LM_TWITTERER_PROVIDER=opencode-zen
LM_TWITTERER_MODEL=auto-free
```

Without those override vars, the plugin uses the active Hermes model.

To allow one explicit provider/model override:

```powershell
hermes lm-twitterer trust-llm-overrides --provider opencode-zen --model auto-free
```

Check the active generation route:

```powershell
hermes lm-twitterer status
```

Look at `effective_generation_provider` and `generation_uses_grok_backend`.

## Commands

```powershell
hermes lm-twitterer status
hermes lm-twitterer auth-check
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

## Cron

Create two Hermes cron jobs:

```powershell
hermes lm-twitterer cron install --post-schedule "every 6h" --reply-schedule "every 1h" --reply-count 20
```

The generated jobs run in `no_agent` mode. Hermes cron executes small Python
wrappers from the Hermes scripts directory, and those wrappers call:

```powershell
hermes lm-twitterer post --live
hermes lm-twitterer replies --live --count 20
```

The command refuses to create live jobs until X cookies, bot screen name, and
reply whitelist are present, unless `--force` is passed. Preview jobs and
wrapper paths without writing cron state:

```powershell
hermes lm-twitterer cron install --dry-run --force
```

Install without allowing public posts yet:

```powershell
hermes lm-twitterer cron install --paused
```

Resume only after live posting has been explicitly approved:

```powershell
hermes cron resume <job-id>
```

Cron wrappers run `lm-twitterer auth-check` before every live action. To test
the wrapper preflight without posting:

```powershell
$env:LM_TWITTERER_CRON_PREFLIGHT_ONLY="1"
python ~/.hermes/scripts/lm-twitterer-post.py
python ~/.hermes/scripts/lm-twitterer-replies.py
```

## Prompt-Injection Boundary

Mention threads are public, untrusted text. Before a reply is generated, the
plugin wraps the thread in an `<untrusted_x_thread>` block and tells the model
to treat it as context only. Requests inside tweets to reveal prompts, change
rules, ignore whitelist policy, leak cookies, or perform unrelated actions are
not valid instructions.

## Operational Notes

X cookies expire and must be refreshed from the browser session when X
invalidates them. If live calls fail, refresh `LM_TWITTERER_AUTH_TOKEN` and
`LM_TWITTERER_CT0`, then rerun `auth-check` before publishing again.
