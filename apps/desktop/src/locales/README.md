# Locale files

UI translations for the Hermes Desktop renderer. The Python-side CLI/Gateway
has its own `locales/*.yaml` tree at the repo root — this directory only
covers the React shell.

## Layout

```
locales/
├── en/translation.json       # source of truth (English)
├── zh-CN/translation.json    # 简体中文
└── locales.test.ts           # parity guard (run in CI)
```

One JSON file per language, single namespace (`translation`). Keys are
flat, `namespace:subsection.path` style:

```json
{
  "common.close": "Close",
  "settings.appearance.mode.dark": "Dark"
}
```

The leading segment (`common`, `settings`, `chat`, …) groups related
strings but is purely organizational — i18next treats every key as a
single flat string.

## Adding a new language

1. Copy `en/translation.json` to `locales/<lang>/translation.json` (use a
   BCP-47 code: `ja`, `ko`, `zh-TW`, `pt-BR`, …).
2. Translate the values. **Keep every key**, and **do not add or remove
   keys** — the parity test will fail.
3. If your translation uses `{{placeholder}}` interpolation, keep the
   placeholder names identical to the English file.
4. Add the language to `SUPPORTED_LANGUAGES` in `src/lib/i18n.ts` and to
   `LANGUAGE_LABELS` next to it (use the **native** name of the language).
5. Add the language to the picker grid in
   `src/app/settings/appearance-settings.tsx`.
6. Add the file path to the `others` list in
   `src/locales/locales.test.ts` so the parity test runs against it.
7. Run `npm run test:ui -- src/locales/locales.test.ts` and `npm run
   type-check` before opening a PR.

## Adding a new key

1. Add the key to **every** locale file in the same commit. The parity
   test will fail otherwise.
2. Use it in the React component with `const { t } = useTranslation()`
   followed by `t('namespace:key.path')`.

## Conventions

- **Voice.** Match the tone of the English file: short, neutral, present
  tense ("Save", not "Saving"). Settings copy tends to be a little more
  conversational than chat — keep that.
- **Plurals.** i18next supports `_one` / `_other` / `_many` suffixes:
  ```json
  { "session.count_one": "{{count}} session", "session.count_other": "{{count}} sessions" }
  ```
- **Punctuation.** End sentences with a period only when the original
  English does.
- **Code in strings.** Wrap product/code names in code formatting in the
  React source (`<code>`). Don't try to encode it in the translation
  file.

## Verification

```bash
npm run test:ui -- src/locales/locales.test.ts   # parity
npm run type-check                                # types
npm run lint                                      # style
npm run dev                                       # manual smoke (toggle language in Settings → Appearance)
```

The smoke test: open Settings → Appearance, switch language, all visible
labels should swap to the new language **without** showing any raw key
names (a raw key means a missing translation). Switch back to English —
the layout should be byte-identical to a pre-i18n screenshot.
