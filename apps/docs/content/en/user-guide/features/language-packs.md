---
sidebar_position: 11
title: "Language Packs"
description: "Add or override UI languages for the CLI, gateway, TUI and Desktop with a plugin or a personal overlay"
---

# Language Packs

Hermes ships 17 UI languages (`display.language`). The list is **pluggable**: a language pack is a
plugin (or a folder in your Hermes home) that adds a new language or overrides the wording of an
existing one — for the Python side (CLI approval prompts, gateway replies, tool verbs, tips), the
`hermes --tui` interface and the Desktop app, all from one pack.

What a pack translates is the **static UI text**. Agent responses, tool output, logs and slash-command
*names* stay as they are; to make the agent itself answer in another language, say so in your prompt.

## Where languages come from

When Hermes looks up a string it walks these layers, top first, and takes the first hit:

1. **Plugin language packs** — every installed plugin that declares `provides_locales` (the last one
   loaded wins when two packs cover the same key).
2. **Your overlay** — `<HERMES_HOME>/locales/<lang>.yaml` (per profile: each profile home has its own
   `locales/` folder).
3. **Bundled** — the `locales/<lang>.yaml` files shipped with Hermes.
4. The same three layers for English, then the raw key.

Every layer may be **partial**: a pack or overlay only needs the keys it changes.

`hermes config set display.language <id>` accepts any id one of these layers supplies. Unknown ids
are refused with the list of available languages:

```
$ hermes config set display.language pl
✗ Unknown language 'pl' for display.language. Available: en, af, ar, de, es, ...
  Install a language pack plugin (hermes plugins install <pack>) or drop <HERMES_HOME>/locales/<id>.yaml to add one.
```

## Using a pack

```bash
hermes plugins install https://github.com/teknium1/hermes-lang-pl   # or a catalog name
hermes config set display.language pl
```

Restart running gateways/TUIs so they pick up the pack. The TUI and Desktop language switchers list
bundled languages, your overlay languages and every installed pack by their native name (endonym).
The `HERMES_LANGUAGE` entry in `.env` still overrides `display.language`.

## Writing a pack

A pack needs **no Python code**. Minimal layout:

```
hermes-lang-pl/
  plugin.yaml
  locales/
    pl.yaml            # core: Python-side strings (approval.*, gateway.*, cli.*, display.*, slash.*, tips.*, ...)
    pl.tui.yaml        # optional: hermes --tui strings
    pl.desktop.yaml    # optional: Hermes Desktop strings
```

`plugin.yaml`:

```yaml
name: hermes-lang-pl
version: 1.0.0
description: Polish language pack for Hermes
provides_locales:
  - id: pl            # lowercase BCP-47-style tag: pl, pt-br, zh-hant
    endonym: Polski   # shown in language switchers
    rtl: false        # true for right-to-left scripts (flips the Desktop layout)
```

`provides_locales` entries can also be plain ids (`- pl`); the endonym then defaults to the id.
When the manifest declares `provides_locales`, the plugin loader registers every
`locales/<lang>[.tui|.desktop].yaml` automatically.

### Files and keys

- **`pl.yaml` (core)** mirrors the structure of the bundled `locales/en.yaml`. Nested YAML flattens to
  dotted keys (`approval.denied`, `gateway.goal_cleared`). Copy `en.yaml`, translate the values, delete
  what you do not want to override.
- **`pl.tui.yaml` / `pl.desktop.yaml`** mirror the English catalogs of the TUI and Desktop apps. The key
  sets are exported to `locales/_keys.tui.json` and `locales/_keys.desktop.json` in the Hermes repo,
  which is what the validator checks against.
- YAML values must be **text**. A number, list, `true`/`false` or empty value is rejected.
- Never use YAML reserved words (`on`, `off`, `yes`, `no`) as keys.

### Placeholders

- Core (Python) strings use **named** placeholders exactly as English does: `"⏳ Draining {count} active agent(s)..."`.
  Keep every `{name}` from the English value; a missing or misspelled placeholder makes Hermes fall
  back to the untranslated string for that key.
- TUI and Desktop entries whose English value is a *function* (it takes arguments) are written in YAML
  as strings with **positional** placeholders: `"{0} of {1} sessions"`.

### Validate

```bash
hermes plugins validate ./hermes-lang-pl
```

The validator checks that every declared id has `locales/<id>.yaml`, that each file parses and is
text-only (a non-text value is an **error**), and **warns** — listing them — about keys that do not
exist in the English catalog for that surface (they are harmless but do nothing). The `.tui.yaml` /
`.desktop.yaml` checks run when the corresponding `_keys.*.json` export is present.

### Register from Python (optional)

A plugin that already has Python code can register catalogs itself:

```python
from pathlib import Path

def register(ctx):
    here = Path(__file__).parent
    ctx.register_locale("pl", here / "locales" / "pl.yaml", endonym="Polski")
    ctx.register_locale("pl", {"approval": {"denied": "      ✗ Odrzucono"}})          # dict, nested or flat
    ctx.register_locale("pl", here / "locales" / "pl.tui.yaml", surface="tui")
    ctx.register_locale_dir(here / "locales")                                         # everything at once
```

`register_locale(lang, source, *, endonym=None, rtl=False, surface="core")` takes a YAML path or a
mapping; `surface` is `core`, `tui` or `desktop`. Registration is undone when the plugin unloads and
never changes `display.language`.

## Personal overrides without a plugin

Drop a partial file into your Hermes home:

```yaml
# ~/.hermes/locales/en.yaml — only the keys you want to change
approval:
  denied: "      ✗ Nope."
```

The overlay applies to that profile only (`~/.hermes/profiles/<name>/locales/` for a named profile).
Overlay files for a language Hermes does not bundle make that language selectable too. Edits are read
on the next start or the next `display.language` change.

## Troubleshooting

- **`Unknown language ... for display.language`** — the pack is not installed or not enabled
  (`hermes plugins list`), or the id in `provides_locales` does not match the file name
  (`pl` needs `locales/pl.yaml`).
- **A translated string still shows in English** — the key is not in the English catalog (run
  `hermes plugins validate`; unknown keys are listed), or the placeholder set differs from English.
- **Language listed but the TUI/Desktop is still English** — the pack has no `.tui.yaml` /
  `.desktop.yaml`; those surfaces render English plus whatever the pack provides. For the 16 bundled
  languages the TUI ships its own `locales/<lang>.tui.yaml` in the Hermes tree (the Desktop bundles
  its translations in-app), so a pack for one of those only needs the keys it wants to override.
