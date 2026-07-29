# Review range: `feat/clarify-rich-options` (issue #2 / chore #12)

This file pins the **fixed point** for reviewing the issue-#2 follow-up
(Discord/Telegram corrections + approved remediations) and records why the
review range must be computed against that fixed point — not against the
stale upstream merge-base.

## The fixed point

- **Tag:** `clarify-rich-options-baseline`
- **Commit:** `5aba0bfda` — `fix(clarify): harden discord import guard
  against test-module mock pollution`
- **What it is:** the rich-options feature as it existed *before* the
  issue-#2 follow-up. The four baseline commits (`9facaa54c`, `65d644c27`,
  `b73e0d560`, `5aba0bfda`) — schema, interactive views, validation tests,
  mock-pollution guard — are the pre-existing feature and must **not** be
  presented as new #2 scope.

## The review range

The correction surface is pinned by **two tags** so it is stable and does
not drift as chore/doc commits land above it:

- **`clarify-rich-options-baseline`** → `5aba0bfda` (fixed point)
- **`clarify-rich-options-tip`** → `8723b93d5` (last correction commit)

```bash
# Spec-scoped correction range — what reviewers and /code-review diff:
git diff clarify-rich-options-baseline...clarify-rich-options-tip   # 15 files, 9 commits
git log  clarify-rich-options-baseline..clarify-rich-options-tip     # the commit list
```

**9 commits, 15 files**, every one of them a requested correction or an
approved remediation ticket. `...HEAD` additionally contains only this
chore note (commit `788de87df`), which touches no reviewed code.

## Why not the merge-base three-dot / two-dot

The branch diverged from `origin/main` long ago (merge-base `a7f65e3bc`,
`fix(gateway): tolerate scalar gateway config block`), so the natural
PR review ranges are wrong here:

| Range                                   | Files | Includes                                                        |
| --------------------------------------- | ----: | -------------------------------------------------------------- |
| `origin/main..HEAD` (two-dot)           | 3970  | staleness explosion — every delta between stale base and HEAD  |
| `origin/main...HEAD` (merge-base 3-dot) | 21    | the whole baseline feature folded in as "new scope"            |
| `clarify-rich-options-baseline...tip`   | 15    | **only** the issue-#2 corrections + remediations (correct)     |

The 6 files the spec-scoped range **excludes** (folded in by the merge-base
three-dot) are exactly the categories #12 says must not appear as new scope:

- `toolsets.py` — toolset wiring (baseline feature plumbing)
- `website/docs/reference/tools-reference.md` — documentation
- `gateway/platforms/whatsapp_cloud.py`, `gateway/platforms/base.py` — adapters unrelated to the #2 corrections
- `agent/agent_runtime_helpers.py`, `agent/tool_executor.py` — core plumbing

## Commit → issue → category

The history separates behavior fixes, security hardening, diagnostics, and
integration verification into distinct conventional-commit commits:

| Commit     | Issue | Subject                                                          | Category                |
| ---------- | ----: | ---------------------------------------------------------------- | ----------------------- |
| `d44017271` | #2   | feat(clarify): capture session_owner_user_id in clarify primitive | behavior fix (primitive) |
| `71dde0086` | #2/#9 | feat(telegram): render rich clarify options and resolve taps to value | behavior fix (adapter) |
| `8bfcbdf2f` | #2   | feat(clarify): coerce typed rich-option replies to value in gateway | behavior fix (primitive) |
| `ca2279a12` | #2   | feat(discord): admit session initiator via origin_user_id in rich clarify | behavior fix (auth) |
| `7e1e237ee` | #7   | fix(clarify): restore one canonical handler, flatten structured choices | behavior fix (primitive) |
| `a7c7aee8c` | #10  | fix(discord): bound modal upload resources                        | **security hardening** |
| `fd08bc945` | #11  | fix(clarify): retain traceback context on resolution failures     | **diagnostics**        |
| `1a3de64e7` | #9   | fix(telegram): preserve bare-string rich clarify contract         | behavior fix (contract) |
| `8723b93d5` | #8   | test(discord): verify clarify owner authorization end to end      | **integration verification** |

## Focus audit

The 15 files in `clarify-rich-options-baseline...clarify-rich-options-tip`:

- **Primitive:** `tools/clarify_gateway.py`, `tools/clarify_tool.py`
- **Discord surface:** `plugins/platforms/discord/adapter.py`,
  `tools/discord_interactive_views.py`, `tools/discord_auth_helpers.py`
- **Telegram surface:** `plugins/platforms/telegram/adapter.py`
- **Gateway wiring:** `gateway/run.py`
- **Config backing for #10:** `hermes_cli/config.py` — the two
  `modal_upload_max_*_bytes` defaults under the top-level `gateway` section
  are the backing for the #10 modal-bound caps (non-secret behavioural
  setting in `config.yaml`, per project policy; new keys in an existing
  section need no version bump).
- **Tests (7 files in range):** `tests/tools/test_clarify_rich_options.py`,
  `tests/tools/test_discord_interactive_views.py`,
  `tests/tools/test_discord_auth_helpers.py`,
  `tests/tools/test_discord_clarify_owner_e2e.py`,
  `tests/tools/test_discord_modal_upload_bounds.py`,
  `tests/gateway/test_discord_clarify_buttons.py`,
  `tests/gateway/test_telegram_clarify_rich.py`.

Verified invariants:

- No `toolsets.py` change (the only toolset delta is in the excluded
  baseline feature, not in this range).
- No `CLARIFY_SCHEMA` / tool-description change. The `tools/clarify_tool.py`
  delta is the #7 canonical-handler consolidation — it removes a duplicate
  legacy `clarify_tool` definition and routes structured choices through
  `_flatten_choice`; the OpenAI function-calling schema is byte-stable.
- No documentation or unrelated-adapter changes.

## Verification (criterion 5)

`scripts/run_tests.sh` over the focused clarify / Discord / Telegram suites —
**315 tests, 0 failed**:

```
tests/tools/test_clarify_rich_options.py        (86✓)
tests/tools/test_discord_interactive_views.py   (50✓)
tests/gateway/test_discord_component_auth.py    (41✓)
tests/gateway/test_discord_clarify_buttons.py   (27✓)
tests/tools/test_clarify_tool.py                (28✓)
tests/tools/test_discord_modal_upload_bounds.py (20✓)
tests/tools/test_clarify_gateway.py             (18✓)
tests/tools/test_discord_auth_helpers.py        (16✓)
tests/gateway/test_telegram_clarify_buttons.py  (14✓)
tests/gateway/test_telegram_clarify_rich.py      (9✓)
tests/tools/test_discord_clarify_owner_e2e.py    (5✓)
tests/gateway/test_clarify_active_session_bypass.py (1✓)
```

The final three-dot diff (`clarify-rich-options-baseline...clarify-rich-options-tip`)
and commit list are non-empty, focused, and ready for a fresh two-axis review.

## Follow-ups from the two-axis review

The fresh `/code-review` (Standards + Spec) over
`clarify-rich-options-baseline...clarify-rich-options-tip` found **no documented-standard
violations** and **all six tickets (#2/#7/#8/#9/#10/#11) correctly
implemented**. It surfaced only minor Standards-axis judgement calls in
#10's modal-upload code. They are recorded here as **optional pre-merge
polish** — deliberately *not* folded into this range, because #12 criterion 2
requires the range to contain only the requested corrections + remediation
tickets (these are style refinements to closed-ticket code, not a ticket):

- `tools/discord_interactive_views.py` emits the same "could not determine
  the size of an attached file…" string in the `size is None` and the
  `int()`-failure branches of `_validate_file_uploads` — lift to a module
  constant (the file already defines `_UPLOAD_READ_FAILURE_MESSAGE` for the
  read path; same pattern).
- The per-file-too-large message is duplicated across Phase 2 (reported
  size) and Phase 3 (`len(data)` defense-in-depth) — extract a
  `_per_file_too_large_msg(field_key, size, limit)` helper so the two
  sites cannot drift.
- (Minor) `_get_modal_upload_limits()` returns an unlabelled
  `(per_file, aggregate)` tuple and is read twice per submission; a
  `NamedTuple` resolved once in `on_submit` would tidy both.
