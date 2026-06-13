# RCA: SSL CA cert bundle corruption after `hermes update`

**Status:** resolved by `fix(ssl): surface broken CA bundles before provider calls`
**Severity:** P2 — degrades the agent into opaque provider/client failures until the user repairs deps or CA configuration.

## Summary

A partial `hermes update`, interrupted venv repair, or stale CA-bundle environment variable can leave Python TLS configuration pointing at a missing, empty, or unloadable CA bundle. The first outbound HTTPS client creation or request can then fail with a raw `FileNotFoundError: [Errno 2] No such file or directory` or a low-level SSL error that does not name the broken CA path.

## Root cause

Hermes uses OpenAI/httpx and requests-based clients for provider calls, model metadata, gateway delivery, and web tools. Those clients inherit CA bundle settings from:

- `HERMES_CA_BUNDLE`
- `SSL_CERT_FILE`
- `REQUESTS_CA_BUNDLE`
- `CURL_CA_BUNDLE`
- the bundled `certifi` package's `cacert.pem`

When the venv is partially refreshed, or when one of those env vars points at a file that no longer exists, provider client construction can fail before Hermes has enough context to produce a useful message.

## Fix

`agent/ssl_guard.py` validates CA bundle configuration before the OpenAI-compatible provider client is created in `agent/agent_init.py`. It:

1. Checks explicit CA bundle env vars and reports the exact broken variable/path,
2. Verifies `certifi` is importable,
3. Verifies `certifi.where()` points at an existing file of plausible size,
4. Builds an `ssl.SSLContext` from each checked bundle,
5. Raises a typed `SSLConfigurationError` with a repair hint before httpx/OpenAI can raise a raw low-level error.

`hermes_cli doctor` exposes the same check under `SSL / CA Certificates`, so users can diagnose the problem without starting a model session.

## Recovery

When the guard fires during agent init, the user sees a message like:

```text
Failed to initialize OpenAI client: SSL_CERT_FILE points to a missing CA bundle: C:\path\to\missing\cacert.pem
Repair: python -m pip install --force-reinstall certifi openai httpx
If you configured a custom corporate CA bundle, fix or unset the broken CA bundle environment variable.
```

For a normal corrupted Hermes venv, reinstall the affected client dependencies:

```bash
python -m pip install --force-reinstall certifi openai httpx
```

For a custom/corporate CA setup, fix the env var so it points at a real PEM bundle, or unset it if Hermes should use the bundled `certifi` store.

## Environment escape hatch

Set `HERMES_SKIP_SSL_GUARD=1` to bypass the preflight check. This is intended only for sandboxed or managed-trust environments where the Python CA path looks unusual but downstream clients are known to work.
