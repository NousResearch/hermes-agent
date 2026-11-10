# security-guidance

Pattern-matched security warnings for code the agent writes. When the agent
calls `write_file`, `patch`, or `skill_manage` with content that matches a
known-dangerous code pattern (eval, pickle.load, yaml.load, os.system,
subprocess with `shell=True`, `dangerouslySetInnerHTML`, `verify=False`, ECB
mode, GitHub Actions `${{ github.event.* }}` injection, `torch.load` without
`weights_only=True`, ...), the plugin appends a warning to the tool's result.
The file is still written; the model sees the warning in the next turn and
can fix the code or briefly document why the construct is safe.

This is layer 1 of Anthropic's `security-guidance` plugin design — a fast
first-pass that runs locally with zero LLM tokens spent. Layers 2 and 3 (LLM
diff review on turn end, agentic commit review) are not ported; the agent
can already run those kinds of reviews on demand via `delegate_task`.

## Coverage (25 rules)

The pattern set is forked verbatim from Anthropic's `claude-plugins-official`
under Apache-2.0. Categories:

| Category | Rules |
|---|---|
| Unsafe deserialization | `pickle.load`, `cPickle/cloudpickle/dill.load`, `marshal.loads`, `shelve.open`, `yaml.load`, `yaml.unsafe_load`, `torch.load` (without `weights_only=True`), `joblib.load`, `pandas.read_pickle`, `numpy.load(allow_pickle=True)` |
| Command injection | `os.system`, `subprocess(...,  shell=True)`, JS `child_process.exec`, Go `exec.Command("sh"...)` |
| Code injection | `eval(`, JS `new Function(...)` |
| XSS sinks | `.innerHTML =`, `.outerHTML =`, `.insertAdjacentHTML(`, `document.write`, React `dangerouslySetInnerHTML` |
| Crypto footguns | AES ECB mode, Node `crypto.createCipher` (no IV), TLS verification disabled (`verify=False`, `rejectUnauthorized: false`, `InsecureSkipVerify: true`, ...) |
| XXE | `xml.etree`, `minidom`, `xml.sax` without `defusedxml` |
| Supply chain | `<script src="https://..."` without `integrity=` SRI hash |
| CI/CD injection | GitHub Actions workflow files using `${{ github.event.* }}` in `run:` |

The pattern data uses Python regex + literal-substring matching. Each rule
carries a per-extension `path_filter` lambda — Python-only rules skip `.js`,
JS rules skip `.py`, all rules skip `.md/.txt/.rst/.json/.yaml`. Lookbehind
assertions exclude method calls (so `model.eval()` and `redis.eval()` don't
trip the `eval(` rule). False-positive rate is mediocre but tolerable; the
plugin is warn-by-default precisely because of that.

## Enabling

Plugins are opt-in. Add it to your allow-list:

```bash
hermes plugins enable security-guidance
# or edit ~/.hermes/config.yaml manually:
plugins:
  enabled:
    - security-guidance
```

## Modes

| Env var | Default | Effect |
|---|---|---|
| (none) | warn | Appends a `⚠️ Security guidance` block to the tool result. The file is written. |
| `SECURITY_GUIDANCE_BLOCK=1` | unset | Refuses the write entirely with the warning as the block reason. Use for stricter environments. |
| `SECURITY_GUIDANCE_DISABLE=1` | unset | Kill switch — plugin loads but does nothing. |

## What it does **not** do (yet)

* **No LLM diff review.** Anthropic's layer 2 spawns an auxiliary LLM call
  on every agent turn that touched files. On hermes that would route
  through the main model by default (`auxiliary_client._resolve_auto()` is
  main-model-first), which is real money on reasoning models. A separate
  PR can wire layer 2 to a cheap auxiliary model with explicit opt-in.
* **No agentic commit review.** Anthropic's layer 3 spawns an SDK subagent
  with `Read`/`Grep`/`Glob` to trace data flow on `git commit`. That's a
  follow-up that would build on `delegate_task`.
* **No project-local rules file.** Anthropic's `.claude/claude-security-guidance.md`
  is read by their layer 2/3 LLM prompts, not the pattern scanner. We can
  add an analogous `.hermes/security-guidance.md` once layer 2 lands.

## Limitations

This is a best-effort assistive tool. Pattern matching can miss
vulnerabilities and produce false positives. Treat warnings as suggestions,
not a substitute for code review, SAST, dependency scanning, or pen testing.

## Attribution and licensing

* `patterns.py` is a verbatim fork from
  [`anthropics/claude-plugins-official`](https://github.com/anthropics/claude-plugins-official/tree/main/plugins/security-guidance/hooks)
  (commit `0bde168`, 2026-05-26), licensed under the
  [Apache License 2.0](./LICENSE). See [NOTICE](./NOTICE) for the full
  attribution.
* `__init__.py`, `plugin.yaml`, `README.md`, and tests are original work by
  NousResearch, MIT-licensed alongside the rest of hermes-agent.
