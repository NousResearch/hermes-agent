# REBRAND_NOTES — "Hermes Agent" → "IX Agency"

The product is now branded **IX Agency** (Intelliverse X, https://intelli-verse-x.ai) in all
user-visible surfaces: README/docs, CLI banner/help, web dashboard, desktop app UI/dialogs,
and cross-repo admin-portal copy. The identifiers below intentionally keep the internal
`hermes` name. Do **not** rename them without coordinating the listed dependents.

## Kept internal identifiers (and why)

### CLI / Python runtime (`hermes-agent` repo)
| Identifier | Why it stays |
|---|---|
| `hermes` CLI binary + all subcommands (`hermes chat`, `hermes kanban create`, `hermes desktop`, `hermes update`, `hermes console`, …) | Other agents and scripts invoke them by name; the kube card-intake sidecar shells out to `hermes kanban create`. |
| `hermes_cli/` Python package and module paths | Import paths; renaming breaks every `from hermes_cli import …`. |
| `~/.hermes/` config directory, `HERMES_HOME`, `HERMES_*` env vars | Existing installs and scripts read these paths/vars. |
| `pyproject.toml` package name / PyPI + git install refs | Distribution identity; changing it orphans existing installs. |
| "Hermes 3 / Hermes 4", "Hosted Hermes & Nous-trained models" | Actual LLM model-family names from Nous Research, not product branding. |
| Upstream URLs (`hermes-agent.nousresearch.com`, `github.com/NousResearch/…`) | Live endpoints/repos; must keep working. |
| "HermesClaw" | Third-party community project name mentioned in READMEs. |

### Web dashboard (`web/`)
| Identifier | Why it stays |
|---|---|
| Theme preset ids (`hermesTeal`, `hermesTealLarge`) and i18n keys containing `hermes` | Persisted in user settings; labels were rebranded, ids must stay stable. |
| `gatewayClient` internals, `X-Hermes-Session-Token` header | Wire-protocol compatibility with the Python gateway. |

### Desktop app (`apps/desktop/`)
| Identifier | Why it stays |
|---|---|
| `package.json` `productName: "Hermes"`, `executableName: "Hermes"`, `CFBundleExecutable/CFBundleName: "Hermes"` | Auto-updater, install paths (`Hermes.app`, `Hermes.exe`, `Hermes-Setup`), and OS keychain entries are keyed to these; changing them breaks in-place updates for existing installs. Display name is overridden to "IX Agency" via `APP_NAME` in `electron/main.ts`. |
| `hermes:*` IPC channel names, code identifiers (`startHermes`, `resolveHermesBackend`, `HERMES_HOME`, `ACTIVE_HERMES_ROOT`, `hermesApi`, …) | Internal API surface shared between main/preload/renderer. |
| `@hermes/shared` workspace package alias | Build-time module resolution. |
| Install locations (`%LOCALAPPDATA%\hermes\…`, `~/Library/Application Support/Hermes`, venv paths) | Existing user installs live there. |
| `User-Agent: Hermes-Desktop` | Server-side analytics/allowlists may key on it. |
| Internal code comments/log labels mentioning Hermes | Not user-visible; refer to the runtime by its internal name. |

### Cross-repo
| Identifier | Repo | Why it stays |
|---|---|---|
| `kind: 'hermes-task'` dispatch payload | Intelliverse-X-AI | Contract with the kanban worker; renaming breaks dispatch. |
| `hermes-admin-worker` deployment/service/secret/configmap names | intelli-verse-kube-infra | Live k8s resource names. |
| `hermes kanban daemon` / `hermes kanban create` invocations in worker scripts + configmap | intelli-verse-kube-infra | The CLI binary name (see above). |
| Git repo/dir names (`hermes-agent`, `hermes-admin-worker`) | all | Out of scope per rebrand rules. |

## Brand assets
New logo family lives in `assets/brand/`:
`ix-agency-mark.svg` (logomark), `ix-agency-lockup.svg` / `ix-agency-lockup-dark.svg`
(horizontal lockups), `ix-agency-appicon.svg` (+512/1024 PNGs), preview PNGs.
Desktop icons (`icon.icns`, `icon.ico`, `icon_1024.png`) in `apps/desktop/assets/` are
generated from `ix-agency-appicon.svg`. Palette: cyan `#0D96EA` / `#38BDF8` → navy
`#1E3E8F` / `#2563EB`, sampled from intelli-verse-x.ai.
