# openrouter-tool-use-404

Classifies OpenRouter's `No endpoints found that support tool use` 404 as
`model_not_found` (`retryable=False`, `should_fallback=True`) so the agent
fails over to a configured fallback model immediately instead of retrying a
deterministic rejection 3–5 times.

Also the reference implementation for the `classify_api_error` plugin hook:
provider plugins can own their provider's error quirks without patching
`agent/error_classifier.py`. The callback contract is documented on the
`classify_api_error` entry in `hermes_cli.plugins.VALID_HOOKS`.

## Enable

Like all bundled standalone plugins, this ships opt-in:

```bash
hermes plugins enable openrouter-tool-use-404
```

## Origin

Classification logic from PR #58451 by @webtecnica, re-implemented as a
plugin to demonstrate the hook.
