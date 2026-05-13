# Spec — Xiaomi MiMo Anthropic-compatible thinking-mode support

- Issue: [#24884](https://github.com/NousResearch/hermes-agent/issues/24884)
- Branch: `fix/24884-xiaomi-mimo-thinking`
- Status: draft (pre-implementation)

## Problem statement

Xiaomi exposes the `mimo-v2.5-pro` model behind an Anthropic Messages-compatible
endpoint at `https://token-plan-cn.xiaomimimo.com/anthropic`. When thinking mode
is enabled and the conversation has at least one prior assistant turn, the
upstream rejects replays with:

```
HTTP 400
The reasoning_content in the thinking mode must be passed back to the API.
```

Root cause: `agent/anthropic_adapter.py::convert_messages_to_anthropic` treats
every non-Anthropic host as a generic third-party endpoint and strips *all*
`thinking` / `redacted_thinking` blocks from assistant messages — including the
unsigned blocks Hermes synthesises from OpenAI-style `reasoning_content`. Kimi
(`/coding`) and DeepSeek (`/anthropic`) already have carve-outs that preserve
unsigned thinking while still stripping Anthropic-signed blocks (third-party
proxies cannot validate Anthropic signatures). Xiaomi MiMo's `/anthropic` route
follows the same contract and needs the same carve-out.

## Goals

1. Detect Xiaomi MiMo's Anthropic-compatible endpoint.
2. Route it through the `_preserve_unsigned_thinking` branch alongside Kimi and
   DeepSeek so unsigned `thinking` blocks survive the conversion.
3. Continue stripping Anthropic-*signed* thinking blocks (Xiaomi cannot validate
   them) and continue stripping `cache_control` from thinking blocks.
4. Do not regress Kimi, DeepSeek, MiniMax, native Anthropic, or any other
   provider.

## Non-goals

- Re-architecting the thinking-block pipeline.
- Generic "any `/anthropic` path preserves unsigned thinking" rule — kept as
  per-host carve-outs to fail closed.
- Provider plugin registration, billing config, or auth changes — Xiaomi already
  works for non-thinking requests; only the message-conversion bug is in scope.
- Live integration testing against Xiaomi's servers.

## Acceptance criteria

- AC1: `_is_xiaomi_mimo_anthropic_endpoint("https://token-plan-cn.xiaomimimo.com/anthropic")`
  returns `True`. Variants tested: trailing slash, `/anthropic/v1`, uppercase
  host. Bare `https://xiaomimimo.com` (no `/anthropic` path) returns `False` so
  any future OpenAI-compat base on the same domain is not misclassified.
- AC2: Calling `convert_messages_to_anthropic(..., base_url=<Xiaomi /anthropic
  URL>)` on a conversation with a replayed assistant tool-call message carrying
  `reasoning_content="planning the tool call"` produces an assistant message
  whose `content` list contains exactly one `{"type": "thinking", "thinking":
  "planning the tool call"}` block (no `signature` field).
- AC3: Across multiple prior assistant turns (not just the last), every one of
  them retains its unsigned `thinking` block — Xiaomi validates history, not
  just the latest turn.
- AC4: A pre-existing Anthropic-signed thinking block (with `signature` or
  `data`) on an assistant message is stripped — Xiaomi cannot validate
  Anthropic-proprietary signatures.
- AC5: `cache_control` keys are stripped from any `thinking` /
  `redacted_thinking` block in the converted output.
- AC6: Regression: an unrelated third-party `/anthropic` endpoint (use MiniMax
  as the canary) still strips all thinking blocks. DeepSeek and Kimi tests
  continue to pass.
- AC7: `python -m pytest tests/agent/test_deepseek_anthropic_thinking.py
  tests/agent/test_xiaomi_mimo_anthropic_thinking.py` passes locally and in CI.

## Affected files

| File | Change |
| --- | --- |
| `agent/anthropic_adapter.py` | Add `_is_xiaomi_mimo_anthropic_endpoint` helper; include in `_preserve_unsigned_thinking` predicate. |
| `tests/agent/test_xiaomi_mimo_anthropic_thinking.py` | NEW. Mirror DeepSeek test file structure. |
| `docs/specs/24884-xiaomi-mimo-thinking.md` | This document. |

Nothing else (no provider registration, no model-metadata changes, no doctor /
auth wiring) — the issue is purely a message-conversion bug for users who have
already configured Xiaomi as a custom Anthropic-compatible provider.

## Design

### Detector

Add next to `_is_deepseek_anthropic_endpoint`:

```python
def _is_xiaomi_mimo_anthropic_endpoint(base_url: str | None) -> bool:
    """Return True for Xiaomi MiMo's Anthropic-compatible /anthropic route.

    Xiaomi's ``token-plan-cn.xiaomimimo.com/anthropic`` endpoint speaks the
    Anthropic Messages protocol but, when thinking mode is enabled, requires
    unsigned thinking blocks (synthesised from ``reasoning_content``) to
    round-trip on every replayed assistant turn — otherwise upstream returns::

        The reasoning_content in the thinking mode must be passed back to the API.

    Same strip-signed / keep-unsigned policy used for Kimi /coding and
    DeepSeek /anthropic.  Match pinned to the ``/anthropic`` path so any future
    OpenAI-compat base on the same domain is not misclassified.
    See hermes-agent#24884.
    """
    if not base_url_host_matches(base_url or "", "xiaomimimo.com"):
        return False
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    return "/anthropic" in normalized.rstrip("/").lower()
```

`base_url_host_matches` accepts subdomains (`token-plan-cn.xiaomimimo.com`
matches `xiaomimimo.com`), which is what we want — Xiaomi documents the
regional host today but may add others.

### Predicate wire-up

In `convert_messages_to_anthropic`:

```python
_preserve_unsigned_thinking = (
    _is_kimi_family_endpoint(base_url, model)
    or _is_deepseek_anthropic_endpoint(base_url)
    or _is_xiaomi_mimo_anthropic_endpoint(base_url)
)
```

Update the inline comment to add `#24884 (Xiaomi MiMo)` after the existing
`#13848 (Kimi) and #16748 (DeepSeek)` references so the carve-out is
self-documenting.

No other branches need to change: the rest of the thinking-block handling
(`_THINKING_TYPES`, `cache_control` strip, image eviction) is provider-agnostic.

## TDD plan

Write the test file first; it must fail against `main` and pass after the
production change.

`tests/agent/test_xiaomi_mimo_anthropic_thinking.py` — mirror the DeepSeek
test layout, replacing host and issue references:

1. `test_unsigned_thinking_block_survives_replay` — parametrised over
   `https://token-plan-cn.xiaomimimo.com/anthropic`,
   `https://token-plan-cn.xiaomimimo.com/anthropic/`,
   `https://token-plan-cn.xiaomimimo.com/anthropic/v1`,
   `https://Token-Plan-CN.XiaomiMiMo.com/anthropic`. Asserts a single unsigned
   `thinking` block with the original `reasoning_content` text and no
   `signature` key.
2. `test_unsigned_thinking_preserved_on_non_latest_assistant_turn` — two prior
   tool-call turns, both must retain their unsigned thinking blocks.
3. `test_signed_anthropic_thinking_block_is_stripped` — assistant message with
   `signature` set; converted output must contain zero `thinking` blocks.
4. `test_cache_control_stripped_from_thinking_block` — invariant check on
   converted output.
5. `test_openai_compat_xiaomi_base_is_not_matched` — direct unit test of
   `_is_xiaomi_mimo_anthropic_endpoint`:
   - `https://token-plan-cn.xiaomimimo.com` → `False`
   - `https://token-plan-cn.xiaomimimo.com/v1` → `False`
   - `https://token-plan-cn.xiaomimimo.com/anthropic` → `True`
   - `https://token-plan-cn.xiaomimimo.com/anthropic/v1` → `True`
   - `https://other.xiaomimimo.com/anthropic` → `True` (subdomain coverage)
6. `test_non_xiaomi_third_party_still_strips_all_thinking` — MiniMax canary,
   asserts zero thinking blocks remain.

Use class `TestXiaomiMiMoAnthropicPreservesThinking`. Reuse the exact message
fixtures from the DeepSeek file — the contract is identical.

## Docker / containerization considerations

- No new runtime dependencies, no environment variables, no schema changes.
- Tests run inside the existing pytest container without additional setup.
- `Dockerfile` and `docker-compose*.yml` need no edits; do not touch them.
- Verify the change works under the project's existing Docker workflow by
  running the pytest target in the same container the CI uses (see
  Verification §1).

## Risks

- **R1 — Hostname drift.** Xiaomi could add additional regional hosts
  (`token-plan-na.xiaomimimo.com`, `mimo.xiaomi.com`, etc.). `base_url_host_matches`
  on `xiaomimimo.com` covers any `*.xiaomimimo.com` subdomain today; if Xiaomi
  later serves the same API under `xiaomi.com`, a follow-up will be needed. Out
  of scope for this issue.
- **R2 — Future OpenAI-compat base on the same domain.** Pinning to
  `/anthropic` in the path keeps a future `xiaomimimo.com/v1` (OpenAI wire)
  from being misrouted. Matches the DeepSeek precedent.
- **R3 — Signed-block regression.** If Xiaomi ever begins issuing
  Anthropic-compatible signatures we would over-strip — but the same risk
  applies to Kimi and DeepSeek and has not materialised; the strip-signed
  branch matches the documented compatibility behaviour.
- **R4 — Live verification.** Without Xiaomi API credentials we rely on the
  unit-level contract. The fix mirrors a pattern proven against two other
  providers, which materially lowers this risk.

## Senior Engineer Review

Reviewing my own plan with a critical eye.

**What's right**

- Smallest viable change: one detector + one predicate slot. No new abstraction,
  no premature "third-party Anthropic registry" refactor. Three carve-outs is
  still under the rule-of-three threshold — converting to a table now would be
  speculative.
- Path-pinned detector (matches DeepSeek precedent) avoids the failure mode
  where a same-domain OpenAI-compat base gets misclassified.
- Test coverage mirrors an existing, proven file, so reviewers can diff the two
  to spot omissions.
- AC list pins both the new behaviour and the non-regression behaviour (MiniMax
  canary, DeepSeek/Kimi suites green).

**Concerns and adjustments**

1. *Substring `/anthropic` could match `/anthropic-legacy` or `/anthropic2`.*
   The DeepSeek detector has the same property and it has not bitten in
   production. Tightening to a path-segment check (`startswith("/anthropic")`
   after stripping the host, plus boundary check) would be safer but
   inconsistent with DeepSeek. **Adjustment:** keep parity with DeepSeek for
   now; if we tighten, do it for all three carve-outs in a follow-up. Noted as
   R2 above.
2. *Subdomain-matching surface area.* `base_url_host_matches("xiaomimimo.com")`
   will accept *any* subdomain. That is desired for regional hosts but means a
   hypothetical `staging.xiaomimimo.com` proxy with different semantics would
   also be routed through this branch. **Adjustment:** acceptable — `/anthropic`
   path pin narrows the surface, and Xiaomi-controlled subdomains are
   presumed-honest by definition.
3. *Should this be a generic config flag instead?* A `preserve_unsigned_thinking:
   true` per-provider config would scale better. **Adjustment:** out of scope.
   Users on `fix/24884` need the bug gone today; the config refactor is a
   separate ticket and would require schema, docs, and migration. Three
   carve-outs is still cheap.
4. *No live integration test.* We can't hit Xiaomi from CI. The unit-test
   contract is what the existing two providers ship with — accept the same
   bar. The user reporting the issue will validate end-to-end against their
   credentials before close.
5. *Model name field — should `_preserve_unsigned_thinking` also detect Xiaomi
   by model name (`mimo-*`) the way Kimi does?* Kimi has that branch because
   private gateways front Kimi with custom hostnames. Xiaomi MiMo has no
   evidence of that deployment pattern today. **Adjustment:** host-only match
   for now; revisit if a private-gateway report comes in.
6. *Thinking kwarg suppression.* Kimi has a second carve-out in `_build_kwargs`
   that drops the `thinking` parameter entirely (Kimi drives thinking
   server-side). Xiaomi's bug message asks us to *pass thinking content back*
   — the kwarg itself is fine; the issue is in message replay. **Adjustment:**
   no change to the `thinking` kwarg branch.

**Verdict:** plan is minimal, mirrors precedent, has clear acceptance criteria
and a non-regression canary. Proceed.

## Verification commands

Run from the repo root after implementation:

```bash
# 1. Targeted unit tests — must pass.
python -m pytest -q \
  tests/agent/test_xiaomi_mimo_anthropic_thinking.py \
  tests/agent/test_deepseek_anthropic_thinking.py

# 2. Broader adapter regression net.
python -m pytest -q tests/agent/

# 3. Lint / type-check parity with CI (only if the repo's lint target exists).
ruff check agent/anthropic_adapter.py tests/agent/test_xiaomi_mimo_anthropic_thinking.py

# 4. Confirm the new test file fails on main (TDD red).
git stash && python -m pytest -q tests/agent/test_xiaomi_mimo_anthropic_thinking.py ; git stash pop
```

(Step 4 is informational — once the production patch is applied the test must
go green.)

## Out of scope (follow-ups, not this PR)

- Live e2e test against `mimo-v2.5-pro`.
- Consolidating Kimi/DeepSeek/Xiaomi into a config-driven registry.
- Tightening `/anthropic` path detection to a strict segment check across all
  three providers.
- Adding Xiaomi to the provider picker / `model_metadata.py` if/when Hermes
  ships it as a first-class provider.
