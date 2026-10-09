# Phase 5.8.7.4 — Platform consumer hard-cut closeout

Base: `583851b17a` (5.8.7.3).
Branch: `refactor/phase5-8-runtime-consumers`.

## Ownership and implementation

- **Provider display labels:** Discord, Telegram, Slack and Matrix now request
  declared provider display labels from `providers.identity.get_provider_label`
  instead of `hermes_cli.providers.get_label`. Unknown identities retain the
  canonical id fallback, and each platform owns its presentation formatting.
  The supplementary Slack and Matrix cases were found in this source audit.
- **Selection confirmations:** Discord and Telegram use the already-existing
  `application_model_selection_guards.combined_selection_warning`, preserving
  async offloading so a possible catalogue/pricing cache miss cannot block
  the adapter's event loop. Confirm/cancel callback semantics remain
  platform-owned. The application aggregator retains cost, training-data
  and measured-session context-cache warning semantics.
- **Provider display grouping:** The original display-only
  `hermes_cli/provider_groups.py` implementation was moved unchanged into
  `application_provider_groups.py`, and the old CLI-owned implementation
  deleted. Telegram's top-level and group-member pickers plus the CLI
  provider picker all use the one application-owned fold and group table.
  Telegram's exception fallback that silently disabled grouping was
  removed, so a missing group module can no longer alter picker behaviour.
  The existing `hermes_cli.models` external lazy compatibility export
  destinations were updated to the new owner; they do not maintain
  duplicate grouping rules.
- **Feishu:** Verified `feishu_comment.py` already uses
  `gateway.model_runtime_facts.provider_default_model` for its fallback.
  No new Feishu model-resolution path was added.
- **Boundaries:** No credential acquisition, persistence or provider/model
  routing logic was moved into platforms; no new registry or cache created.

## Verification

- Group folding, hidden-provider filtering, Discord/Telegram picker and
  callback confirmation, Slack picker and Feishu comment regressions:
  **61 passed** in the combined focused suite.
- Platform ownership architecture tests cover canonical label imports,
  both confirmation entry points, single grouping owner, CLI picker,
  Feishu fallback, context-warning gates and warning composition ordering.
- Focused Ruff, Python compilation, and Git diff checks verified.
- No full CBM scan/Arcana dependency or repository-wide suite is claimed.

Existing `hermes_cli.model_selection_guards` is still used by CLI selection
entry points, and is a separate pre-existing application-policy
implementation. The final 5.8.7.6 ownership audit should reconcile
remaining duplication; this phase cuts the **platform** consumers, not
unrelated CLI warning entry points.

Next: **5.8.7.5** — environment-policy and inventory/TUI carryovers.
