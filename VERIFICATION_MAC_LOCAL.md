# Verification on steven's Mac mini

Date: 2026-04-24
Repo commit tested: `6051fba9`

## Environment

- macOS Darwin 25.3.0 (arm64)
- Node `v25.8.1`
- npm `11.11.0`
- uv `0.10.4`
- project venv Python `3.11.14`

## Install result

Succeeded with:

```bash
uv venv venv --python 3.11
uv pip install --python venv/bin/python -e '.[messaging,cli,web,mcp,pty,honcho,acp,voice]'
npm install
```

`uv sync --all-extras --locked` did **not** succeed on Python 3.11 because the
`all` extra pulls `dev`, and `dev` includes `yc-bench`, which is currently
Python-3.12+ only.

## Runtime verification

Using isolated home:

```bash
HERMES_HOME="$PWD/.hermes-home" scripts/run-local-hermes.sh doctor
HERMES_HOME="$PWD/.hermes-home" scripts/run-local-hermes.sh status
HERMES_HOME="$PWD/.hermes-home" scripts/run-local-hermes.sh gateway --help
HERMES_HOME="$PWD/.hermes-home" scripts/run-local-hermes.sh chat -q 'hello from isolated local setup test'
```

Observed:

- CLI loads successfully
- doctor/status run successfully
- gateway command loads successfully
- chat starts successfully and fails cleanly only because no inference provider is configured yet

## Auth conclusion

- Hermes **does support OpenAI OAuth directly** for `openai-codex`
- This is a **device code flow**, not a silent noninteractive login
- Standard OpenAI API usage can still use API keys instead
- On the tested local build, the old `hermes login` flow has been replaced by `hermes auth add`

Working interactive step:

```bash
HERMES_HOME="$PWD/.hermes-home" ./scripts/run-local-hermes.sh auth add openai-codex --type oauth --no-browser
```

Then open `https://auth.openai.com/codex/device`, enter the displayed code, and
finish approval in the browser.

After approval, set the active provider/model:

```bash
./scripts/run-local-hermes.sh config set model.provider openai-codex
./scripts/run-local-hermes.sh config set model.default openai-codex/gpt-5.4
```

Verified result:
- `hermes auth list` shows `openai-codex` device-code credentials
- `hermes status` shows `Provider: OpenAI Codex`
- `hermes chat -q 'Reply with exactly OK'` succeeds under isolated `.hermes-home/`
- Telegram bot token for `@spak47moltbot` is configured in repo-local `.hermes-home/.env`
- Telegram access is restricted to Steven (`8459630899`)
- `hermes gateway start/status` now shows a matching loaded launchd service
- Steven confirmed in live testing that Hermes received Telegram messages successfully
