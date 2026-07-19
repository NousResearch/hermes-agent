# self-verification

Built-in verification framework for Hermes Agent that audits AI output for
factual accuracy, logic consistency, and output completeness. Uses confidence
scoring (0-100 continuous), adversarial self-refutation, and auto-fix retry
loops.

This is layer 1 of a multi-layer verification architecture, inspired by
Claude Code's `code-review` and `security-guidance` plugins, and the
LLM-as-a-Verifier research (arXiv 2607.05391).

## Features (v0.2.0)

| Feature | Description |
|---|---|
| **Confidence scoring** | 0-100 continuous score, not binary pass/fail. Levels: 0=uncertain, 25=suspicious, 50=probably real, 75=highly confident, 100=absolutely certain |
| **Self-refute** | After finding issues, adversarially tries to DISPROVE each one. Default = SURVIVES (keep unless refuted). |
| **Auto-fix loop** | Max 3 retries. If verification fails, issues are injected as context for the next LLM call. |
| **Language auto-detection** | Reads `config.yaml` → `display.language`. Defaults to `zh` (Chinese). `en` supported. |
| **Output completeness** | Checks all user sub-questions are answered (VMAO pattern). |
| **Non-blocking by default** | Warns via appended footnote. Strict blocking mode available via config. |

## Enabling

Plugins are opt-in. Add it to your allow-list:

```bash
hermes plugins enable self-verification
# or edit ~/.hermes/config.yaml manually:
plugins:
  enabled:
    - self-verification
```

## Configuration

All settings are optional and go under `plugins.self-verification` in
`config.yaml`:

```yaml
plugins:
  self-verification:
    enabled: true                # Default: true
    confidence_threshold: 50     # 0-100, only report below this (default: 50)
    max_retries: 3               # Auto-fix max retries (default: 3)
    strict: false                # Blocking mode (default: false)
```

### Environment variables

| Variable | Effect |
|---|---|
| `HERMES_SELF_VERIFICATION=0` | Disable the plugin entirely |
| `SELF_VERIFICATION_BLOCK=1` | Enable strict (blocking) mode |
| `SELF_VERIFICATION_THRESHOLD=75` | Override confidence threshold |
| `SELF_VERIFICATION_MAX_RETRIES=5` | Override max retries |
| `SELF_VERIFICATION_DISABLE=1` | Kill switch (plugin loads but does nothing) |

### Verifier model

The verification model is configurable via `agent.self_verification`:

```yaml
agent:
  self_verification:
    verifier_model: deepseek-v4-flash    # Primary model
    verifier_provider: deepseek          # Primary provider
    fallback_model: qwen3.6-plus         # Fallback model
    fallback_provider: zai               # Fallback provider
    verifier_timeout: 20                 # Timeout in seconds
```

Or via environment:
- `HERMES_VERIFIER_MODEL=deepseek-v4-flash`
- `HERMES_VERIFIER_PROVIDER=deepseek`

## How it works

```
User message → LLM generates response
                  ↓
         transform_llm_output hook fires
                  ↓
    ┌─────────────────────────────┐
    │ Layer 1: Verification       │
    │  - Factual accuracy         │
    │  - Logic consistency        │
    │  - Output completeness      │
    └─────────────┬───────────────┘
                  ↓
    ┌─────────────────────────────┐
    │ Layer 2: Confidence scoring │
    │  0-100 continuous score     │
    │  Threshold filtering        │
    └─────────────┬───────────────┘
                  ↓
    ┌─────────────────────────────┐
    │ Layer 3: Self-refute        │
    │  Adversarial disprove       │
    │  Filter false positives     │
    └─────────────┬───────────────┘
                  ↓
    ┌─────────────────────────────┐
    │ Layer 4: Auto-fix (optional)│
    │  Max 3 retries with context │
    └─────────────┬───────────────┘
                  ↓
         Response + footnote (or retry)
```

## Architecture

### Files

| File | Purpose |
|---|---|
| `plugin.yaml` | Plugin metadata and hook declaration |
| `__init__.py` | Hook registration + transform_llm_output handler |
| `verifier.py` | Verification engine (factual, logic, completeness, scoring, self-refute) |
| `conf.py` | Confidence scoring config, threshold, config reading |
| `README.md` | This documentation |
| `tests/` | Test suite |

### Hooks

- **`transform_llm_output`**: Fires once per turn after the tool-calling loop
  completes. Runs all verification layers and appends a footnote for warn/fail
  verdicts. In strict mode, returns a retry instruction instead.

### Relationship to other Hermes components

| Component | Relationship |
|---|---|
| `security-guidance` plugin | Complementary: security-guidance checks code safety, self-verification checks output quality |
| File-Mutation Verifier | Complementary: we verify content correctness, it verifies file writes landed |
| Turn Completion Explainer | Complementary: we verify intermediate steps, it verifies turn-end state |

## Scoring reference

| Score | Level | Meaning |
|---|---|---|
| 0 | Uncertain | Likely a false positive |
| 25 | Suspicious | Needs further verification |
| 50 | Probable | May be real, impact manageable |
| 75 | Confident | Real and important |
| 100 | Certain | Definitely exists |

Default threshold: **50**. Only issues with confidence below this threshold
are reported as warnings.

## What it does NOT do (yet)

- **No tool-call verification.** Layer 2 (pre_tool_call hook) for checking
  tool results mid-execution is planned but not yet implemented.
- **No LLM diff review on file changes.** That's a separate concern already
  partially covered by `security-guidance`.
- **No project-local rules file.** A `.hermes/self-verification.md` for
  per-project verification rules is planned for v0.3.0.

## Limitations

This is a best-effort assistive tool. The Verifier model can produce false
positives and false negatives. Self-refutation reduces but does not eliminate
false positives. Always verify critical claims independently.

## References

- [LLM-as-a-Verifier (arXiv 2607.05391)](https://arxiv.org/abs/2607.05391)
- [VMAO: Verify Model Answers with Objectives (ICLR 2026)](https://arxiv.org/pdf/2603.11445)
- [Claude Code code-review plugin](https://github.com/anthropics/claude-code/blob/main/plugins/code-review/README.md)
- [Claude Code security-guidance plugin](https://github.com/anthropics/claude-code/blob/main/plugins/security-guidance/hooks/review_api.py)

## Attribution

- Original `self-verification` standalone plugin: [WeilaiSun/hermes-self-verification](https://github.com/WeilaiSun/hermes-self-verification)
- Built-in plugin port: NousResearch
- License: MIT (alongside the rest of hermes-agent)
