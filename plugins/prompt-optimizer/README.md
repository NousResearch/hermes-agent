# Prompt Optimizer (Hermes Desktop Plugin)

One-click prompt optimization for the Hermes desktop app: a ⚡ button next to
the composer rewrites your draft prompt with the **current session model**,
auto-fills the result back into the input, and supports one-click undo.

- **Stateless** — the optimization runs outside the conversation; session
  history and prompt caching are untouched.
- **Session model inheritance** — the request rides the same
  provider/model/credentials as the session you are looking at (falls back to
  the configured auxiliary backend when no live agent is resolvable).
- **Session-bound writes** — the result is written back to the composer of the
  session that initiated the click, even if you switch sessions while the
  model is still generating. It never lands in another session's input.

## Layout

```
plugins/prompt-optimizer/
├── dashboard/
│   ├── manifest.json      # backend declaration (hidden tab, API only)
│   └── plugin_api.py      # FastAPI routes: POST /api/plugins/prompt-optimizer/optimize
├── plugin.js              # frontend runtime plugin (desktop UI, see install below)
├── install.ps1 / install.sh
├── LICENSE
└── tests/                 # regression / diagnosis scripts (not wired into CI)
```

## Backend (bundled)

As a bundled dashboard plugin, the backend is discovered and mounted
automatically by the web server (including the `hermes serve` headless backend
used by the desktop app) at startup:

```
POST /api/plugins/prompt-optimizer/optimize
{ input, instructions, session_id, max_tokens, temperature, timeout } → { text }
```

The handler runs `agent.oneshot.run_oneshot` synchronously with a configurable
deadline (default 300 s) and resolves the live session agent through
`tui_gateway.methods_session._sessions` to inherit its model. No session
history is mutated.

## Frontend (user-installed runtime plugin)

The desktop UI plugin is a runtime plugin by design — the desktop app loads
plugins from `<hermes home>/desktop-plugins/<name>/plugin.js` on disk and hot-
reloads them on change (no build step, no bundled distribution).

```bash
# from a checkout of this repo
mkdir -p "${HERMES_HOME:-$HOME/.hermes}/desktop-plugins/prompt-optimizer"
cp plugins/prompt-optimizer/plugin.js "${HERMES_HOME:-$HOME/.hermes}/desktop-plugins/prompt-optimizer/plugin.js"
```

Windows users can run `install.ps1` (PowerShell) or `bash install.sh` instead
— both are idempotent. If the ⚡ button does not appear within a few seconds,
run `Ctrl/Cmd+K → Reload desktop plugins`.

> **Why the frontend does not call `host.request('llm.oneshot')`**: the desktop
> renderer's JSON-RPC client enforces a hard 30 s request timeout
> (`DEFAULT_GATEWAY_REQUEST_TIMEOUT_MS` in `apps/desktop/src/hermes.ts`) and
> the SDK's `host.request` does not expose a per-call timeout override. Model
> responses for prompt optimization routinely take 28–52 s, so every call
> failed. The frontend therefore uses `ctx.rest` (which passes `timeoutMs`
> through to the backend HTTP request) against the plugin's own backend
> namespace, with a 300 s deadline end to end.

## Tests

```bash
# session-binding regression (6 scenarios, 18 assertions; pure Node, no deps)
node plugins/prompt-optimizer/tests/test_session_binding.js

# backend mount / discovery simulation (needs the repo venv: uv pip install -e ".[all,dev]")
python plugins/prompt-optimizer/tests/test_mount.py

# end-to-end handler call (real model call — slow, hits your configured backend)
python plugins/prompt-optimizer/tests/test_api.py
```

`test_oneshot.py` / `test_oneshot_session.py` measure one-shot latency on the
auxiliary backend vs. a session-model override and are useful for diagnosing
slow-provider timeouts.

## Compatibility

- Hermes Desktop ≥ v0.20 (SDK exports `COMPOSER_AREAS` and `PluginContext.rest`)
- Frontend: single-file ESM (~14 KB), only depends on `@hermes/plugin-sdk` + `react`
- No core source changes; the backend rides the existing dashboard plugin
  discovery/mount machinery

## License

MIT — see [LICENSE](./LICENSE).
