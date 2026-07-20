# Dashboard Kit Adoption

This document tracks which dashboards are using the Hermes dashboard design system and how far each one is from the target state.

Run:

```bash
npm run dashboard:design-system:status
```

Use `-- --strict` in CI when all listed dashboards are expected to be synced. Use `-- --sync` only when intentionally updating copied static adapters from the canonical CSS source.

Install local pre-commit hooks across every registered dashboard repo:

```bash
npm run dashboard:design-system:hooks:install
```

Those hooks run the strict checker before commits in this repo and in registered sibling dashboard repos.

If drift is detected during a local commit, the hook will:

1. Run the controlled sync command.
2. Verify that all registered adapters now match the canonical CSS.
3. Stop the commit once so the healed files can be reviewed and staged.

CI remains check-only. It does not auto-heal files inside GitHub Actions.

## Adoption States

| State | Meaning |
| --- | --- |
| `package-native` | Dashboard imports `@hermes/dashboard-kit` components directly. |
| `static-adapter` | Dashboard uses copied or served `hermes-dashboard-kit.css` classes. |
| `needs-sync` | Dashboard has a copied adapter but it does not match the canonical CSS. |
| `missing` | Dashboard is registered but the target adapter file is absent. |
| `unknown` | Dashboard exists but has not been registered or audited yet. |

## Registered Dashboards

The canonical registry is `docs/design/dashboard-kit-adoption.json`.

Current known dashboard adapter targets:

- `khashi-vc`: `../khashi-vc/public/roc/hermes-dashboard-kit.css`
- `media-engine`: `../media-engine/core/operations/hermes-dashboard-kit.css`
- `media-business-operations`: `../media-business-operations/public/dashboard/hermes-dashboard-kit.css`
- `business-mapper`: `../business-mapper/business_mapper/static/hermes-dashboard-kit.css`
- `Meal-assistant`: `../Meal-assistant/src/hermes-dashboard-kit.css`

## Migration Rule

Static adapters are bridge infrastructure. They are acceptable when a dashboard is not React package-native yet, but they must be tracked and checked for drift.

## Drift Handling

Drift is not fixed by editing copied dashboard CSS manually. The repair path is:

```bash
npm run dashboard:design-system:status -- --sync
npm run dashboard:design-system:status -- --strict
```

The GitHub Actions workflow validates the canonical registry and source package. Local hooks validate the full multi-project workspace because sibling dashboard projects are outside this repository's GitHub checkout.
