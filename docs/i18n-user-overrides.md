# User locale overrides (update-proof translations)

## Problem

Every translation in Hermes is **baked into the bundle that ships**:

- Desktop UI strings are compiled in (`apps/desktop/src/i18n/<lang>.ts`, imported
  statically by `catalog.ts`).
- The TUI renders hard-coded strings in `ui-tui`.
- The Python agent/CLI strings live in `locales/<lang>.yaml` inside the install.

An app update replaces that bundle wholesale, so any **local** edit to a shipped
catalog is reverted on the next update. The only durable home for a translation
is therefore *outside* the bundle — a user-writable directory updates don't
touch.

## The convention

A per-language override file:

```
<hermes-home>/locale-overrides/<lang>.yaml
```

- `<hermes-home>` is the usual Hermes home (`~/.hermes`, or `%LOCALAPPDATA%\hermes`
  on Windows; `HERMES_HOME` wins if set). Override the directory directly with
  `HERMES_LOCALE_OVERRIDES`.
- The file uses the **same nested key structure** as the bundled catalog. It is
  **deep-merged on top** of the bundled catalog, key by key:
  - a key present in the override **replaces** the bundled value;
  - a key absent from the override keeps the bundled value (or, ultimately, the
    English fallback).
- Missing or malformed files are ignored (more English fallback, never a crash).

Example — pin two strings without forking:

```yaml
# ~/.hermes/locale-overrides/ja.yaml
approval:
  denied: "却下しました"
gateway:
  goal_cleared: "✓ 目標をクリアしました。"
```

These survive every update because they live in your home directory, not in the
app bundle.

## Status

| Surface | Layer | Override support |
|---|---|---|
| Python agent / CLI (`locales/*.yaml`) | `agent/i18n.py` | ✅ implemented (this change) |
| Desktop (Electron, `src/i18n/*.ts`) | renderer `TRANSLATIONS` | 🔜 proposed (below) |
| TUI (`ui-tui`, Ink) | n/a (no i18n layer yet) | 🔜 proposed (below) |

### Python (implemented)

`agent.i18n._load_catalog()` loads the bundled `locales/<lang>.yaml`, then merges
`<hermes-home>/locale-overrides/<lang>.yaml` over it. Caches per process; call
`reset_language_cache()` to pick up edits without a restart.

### Desktop (proposed)

`apps/desktop/src/i18n/define-locale.ts` already has the exact deep-merge needed
(`mergeTranslations`). To make desktop overrides update-proof, at startup the
Electron **main** process would read `<hermes-home>/locale-overrides/<lang>.json`
and hand it to the renderer (config is already plumbed this way via
`getHermesConfigRecord`). The `I18nProvider` then merges it onto
`TRANSLATIONS[locale]` with the existing `mergeTranslations`, mirroring this
file's semantics. JSON keeps it parseable in the renderer without a YAML dep.

### TUI (proposed)

`ui-tui` has no i18n layer today — strings are hard-coded. Adoption is a larger,
separate effort: extract user-facing strings into an `en` base + a `t()` helper
(mirroring the desktop pattern), then read the same
`<hermes-home>/locale-overrides/<lang>.{yaml,json}` on top. Much of what the user
already sees in the terminal (approval prompts, slash-command replies) is emitted
by Python and is **already** localized + overridable via the layer above.

## Relationship to upstreaming

Overrides are the durable home for translations that haven't landed upstream yet,
and for personal wording. Translations that *do* land upstream ship in the bundle
and become the default for everyone — at which point the matching override keys
are redundant and can be dropped.
