# RCA: SSL CA cert bundle corruption after `hermes update`

**Status:** resolved by `fix(agent,gateway): add SSL CA cert bundle fail-fast guard`
**Severity:** P2 — degrades the agent into a crash-loop until the user re-installs deps.

## Summary

A `git pull` (or `hermes update`) that lands new code without finishing `uv pip install -e .` leaves the certifi CA bundle stale or missing on disk. The first outbound HTTPS call (OpenAI, Telegram, Discord, etc.) then crashes with a raw `ssl.SSLCertVerificationError` and Hermes enters a crash-loop, surfacing only a traceback to the user.

## Root cause

`certifi.where()` returns the path to the CA bundle shipped by the `certifi` package inside the active venv. When the venv is partially refreshed (new `certifi` files copied but old certs in the wheel cache, or a half-deleted install), the bundle can be:

- **missing** (file removed but Python still imports the package),
- **empty / truncated** (partial write),
- **unloadable** (cert format mismatch on a Python upgrade).

Hermes used to let those failures bubble up uncaught, so the gateway would log a stacktrace and the agent would retry the same broken network call on the next turn.

## Fix

`agent/ssl_guard.py` runs a `verify_ca_bundle()` pre-flight right after the `hermes_bootstrap` import in both `run_agent.py` and `gateway/run.py`. It:

1. Resolves the certifi bundle path,
2. Asserts the file exists and is at least 1 KB,
3. Builds an `ssl.SSLContext` from it,
4. Falls back to the system trust store on macOS when the bundle is empty but the system store works (covers corporate proxies / MDM setups),
5. Raises a typed `SSLConfigurationError` with a clear remediation hint otherwise.

`run_agent.py` and `gateway/run.py` import the guard in a guarded `try/except` so a bug in the guard itself cannot prevent startup — we log a warning and continue.

`hermes_cli doctor` now exposes a `SSL / CA Certificates` section so users can detect the failure with a single command.

## Recovery

When the guard fires, the user sees:

```
⚠️ SSL certificate bundle issue detected.
   Run: pip install -e .
```

`pip install -e .` (or the equivalent `uv pip install -e .`) reinstalls certifi and restores the bundle.

## Environment escape hatch

Set `HERMES_SKIP_SSL_GUARD=1` to bypass the check. Intended for sandboxed environments that ship their own trust store.
