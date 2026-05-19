# Root-Cause Analysis: `ssl.create_default_context` failure after `git pull`

## Symptom

After pulling upstream changes via ``git pull``
([nousresearch/hermes-agent](https://github.com/nousresearch/hermes-agent))
and restarting, every incoming chat message causes a
``RuntimeError: ('Invalid automatic root certificates path',)`` from
``ssl.create_default_context`` inside the OpenAI SDK.

The agent never replies; the Telegram/Gateway responses only show the
raw traceback.

## Root Cause

Large source updates (such as those that add vendor directories or large
image assets) shift the ``cwd ⇒ import certifi ⇒ data file`` path mapping
inside the virtual environment, or the ``certifi`` package itself is
updated by the pull but its bundled :file:`cacert.pem` is stale/missing.

When the OpenAI SDK (or Anthropic / HTTPX / any TLS-based HTTP client)
builds its ``SSLContext`` via ``ssl.create_default_context``, it asks
``certifi.where()`` for the CA-bundle path.  If that file is missing or
unparseable, ``create_default_context`` raises a cryptic ``RuntimeError``
instead of a user-actionable message.

## Reproduction

1. Install from source in a venv: ``pip install -e .``
2. ``git pull`` a large update (≥ thousands of changed files)
3. Restart the gateway or open a new chat **without** reinstalling
4. Send any message → agent crash with the above ``RuntimeError``

| Check | Expected |
|---|---|
| ``python -c "import certifi, ssl; ssl.create_default_context(cafile=certifi.where())"`` | ✅ silently passes |
| After reproduction step 2 | ❌ raises ``RuntimeError: ('Invalid automatic root certificates path',)`` |

## Remediation

### Immediate fix

```bash
pip install -e .
```

A reinstall rebuilds the ``certifi`` metadata and restores the CA bundle.

### Long-term fix

A pre-flight ``check_ssl_ca_bundle()`` guard was added in
``agent/ssl_guard.py`` (see :pr:`#<pr_number>`):

* **Fail-fast** — validates ``HERMES_CA_BUNDLE``, ``REQUESTS_CA_BUNDLE``,
  ``SSL_CERT_FILE``, or the default ``certifi`` bundle before an OpenAI
  client is constructed.
* **Catch-fallback** — if the guard is bypassed, ``gateway/run.py`` catches
  ``SSLConfigurationError`` and returns a user-facing message instead of a
  traceback.
* **Observability** — ``hermes doctor`` will gain a dedicated SSL CA section
  (planned as follow-up).

## Prevention Checklist

- [x] Agent-level ``check_ssl_ca_bundle()`` fails fast
- [x] Gateway catches ``SSLConfigurationError`` gracefully
- [ ] ``hermes doctor`` SSL CA section (follow-up)
