# Phase 5.8.6.7 — Web/desktop residual model consumer audit

Base: `e66ec19e92` (5.8.6.6 dashboard assignment).
Branch: `refactor/phase5-8-runtime-consumers`.

## Inventory of active surfaces

| Surface | Catalogue/default | Mutation | Ownership |
| --- | --- | --- | --- |
| Browser chat picker | Gateway `model.options` | Gateway `config.set` / slash command | Transport/UI only; gateway model transaction owns selection |
| Browser standalone/Models page | Profile-scoped `/api/model/options` | `/api/model/set` | Dashboard application (5.8.6.6) |
| Desktop global picker and settings | Profile-scoped REST and gateway `model.options` | `/api/model/set` | Dashboard/gateway applications, not local TypeScript policy |
| Desktop per-bot picker | Owner-routed `requestForBot(...,'model.options')` | Profile/bot RPC | Owner gateway; never borrow ambient REST on owner-routed failure |
| Desktop onboarding | `getGlobalModelOptions` + `getRecommendedDefaultModel` | Shared `setMainModelAssignment` | Dashboard application; not a separate onboarding selector |
| Dashboard recommended default | Profile-specific account facts and shared selection | None directly | Application-level default-model policy over `models.selection` |

The browser and desktop frontends send the selected provider/model without
re-deriving native providers, credential eligibility, transport protocols, or
persisted shape. Presentation helpers retain only display/catalog matching
and a narrow legacy bare-vs-durable custom-provider identity correction for
already-saved desktop picks. These are not runtime route authorities.

## Hard cut completed in 5.8.6.7

- Moved `hermes_cli/model_selection_defaults.py` to
  `application_model_selection_defaults.py`, rewriting its CLI, model-fact,
  account-setup and dashboard consumers and the relevant tests. No forwarding
  compatibility copy or synchronized second selection owner was created.
  The application helper gathers account/cache facts and delegates actual
  default-model policy to `models.selection.select_default_model` and
  `select_nous_default_model`; `models/` has no application imports.
- Dashboard `/api/model/recommended-default?provider=nous&profile=...`
  now executes its Nous tier/account lookup inside the **requested profile**
  context, matching the generic-provider path. It can no longer silently use
  the process's launch-profile entitlements for a different profile.
- New Python ownership and behavior tests cover the browser/desktop dispatch
  boundaries, scoped recommendation reads, absence of CLI semantic selection
  imports in dashboard handlers, and the explicit deferred inventory seam.

## Remaining dependencies: distinct future owners

1. **5.8.7 provider/plugin discovery**: `hermes_cli.inventory` is still the
   single shared catalogue fact acquisition pipeline for dashboard, TUI,
   gateway and ACP. Its `build_models_payload` reaches
   `hermes_cli.model_switch.list_authenticated_providers`, and its
   read-like recommendation path may still lazily persist discovered
   custom-provider rows. Do not build a dashboard-specific second inventory
   or describe that pipeline as read-only. Move actual discovery/refresh and
   catalogue ownership once, for all consumers, in 5.8.7.
2. **5.8.7/5.9**: application defaults still acquire cached model, Nous
   entitlement, pricing and authorization facts through legacy CLI-hosted
   catalogue/auth readers. Those functions are acquisition leaves, *not*
   fallback model-selection algorithms. Preserve their account semantics as
   their owners move.
3. **5.8.6.8 boundary closeout**: lock the consumer import/ownership gates
   and run final comprehensive integration checks without inventing
   compatibility shims. The previous 5.8.6.6 session/database caveat remains.
4. **Phase 6**: credential/OAuth/pool ownership; never copy auth or secret
   resolution into web/desktop renderers.

## Verification

- Final disjoint focused runs: **12 passed** (new web/desktop audit +
  application defaults), **4 passed** (REST model options/context),
  and **35 passed** (model selection/runtime boundaries), plus **2 passed**
  (Nous recommendation endpoint): **53 passed total**.
- Ruff, targeted Python compilation and the unstaged Git diff
  whitespace check pass; the staged gate runs before commit.
- The independent frontend TypeScript suite cannot run in this worktree
  without installing dependencies: root, web and desktop
  `node_modules` directories are absent. Existing frontend contracts were
  inspected and guarded by the new static tests; no frontend source files
  were modified.
- The unrelated `tests/hermes_cli/test_nous_policy_surfaces.py`
  auxiliary case still imports the previously removed
  `hermes_cli.model_selection_auxiliary`; it is not evidence that the
  web/desktop default-model move failed, and this phase does not restore
  that obsolete module.

Next: 5.8.6.8 — consumer ownership and regression closeout.
