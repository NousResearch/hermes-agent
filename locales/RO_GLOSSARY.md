# Romanian (ro) localisation — consistency decisions

## Source

Maintained on the fork at https://github.com/idshdx/hermes-agent, branch
`feature/ro-locale` (upstream: `NousResearch/hermes-agent`). Keep local work on
that branch: `hermes update` is configured with
`updates.parked_branch_strategy: update_in_place` so upstream `main` is merged
into it rather than switching the checkout away.

Applies to all four surfaces: `web/src/i18n/ro.ts`, `locales/ro.yaml` (core agent),
`locales/ro.tui.yaml` (TUI), `locales/ro.desktop.yaml` (desktop).

## Register

**Informal (`tu`)** is the target register. The web catalog was hand-refined in this
register (`Salvează`, `Șterge`, `Anulează`, `Rulează`) and the other three catalogs
were normalised to match: 2nd-person-plural imperative verbs → informal singular
(`Trimiteți` → `Trimite`, `Deschideți` → `Deschide`, `selectați` → `selectează`,
`Conectați-vă` → `Conectează-te`).

Verb-level unification is complete across all four catalogs (imperatives and `să`
subjunctives). Known residual: possessive/object pronouns still use the polite form
in a number of strings (`dvs.`, `vă`, e.g. `mesajul dvs.`, `vă păstrează editările`).
Converting these requires noun-gender/number agreement per string and is left for
human review rather than a risky bulk edit.

## Agent chat responses

Selecting Romanian (`display.language: ro`) makes the agent reply in Romanian in
chats. `agent/system_prompt.py::_reply_language_guidance()` adds

> Always reply to the user in Română (ro). Preserve code, command names, product
> names, URLs, and quoted user content when appropriate.

to the stable system prompt, so it is captured when a session starts. This is on by
default; `display.reply_in_language: false` opts out and keeps responses in English.
Existing sessions keep the prompt they started with — start a new session (or run
`hermes sessions refresh-prompts`) after switching languages to adopt the change.

## Glossary

| English | Romanian | Notes |
|---------|----------|-------|
| Agent | Agent | Keep |
| API / CLI / TUI / GUI / MCP | same | Acronyms, keep |
| Cancel | Anulează | |
| Chat | Chat | Keep |
| Close | Închide | |
| Config / Configuration | Config / Configurație | "Config" in nav labels |
| Confirm | Confirmă | |
| Cron | Cron | Keep (tech term) |
| Dashboard | Panou | |
| Delete | Șterge | |
| Disabled / Enabled | Dezactivat / Activat | |
| Error | Eroare | |
| Gateway | Gateway | Keep (tech term) |
| Keys | Chei | |
| Loading… | Se încarcă… | |
| Logs | Jurnale | Never "Bușteni" |
| Match (search) | Rezultat / potrivire | Never "meci" (sports) |
| Model | Model | Same word |
| Plugin | Plugin | Keep |
| Profile | Profil | |
| Running (process) | Rulează / în curs | Never "Alergarea" |
| Save | Salvează | |
| Session | Sesiune | |
| Settings | Setări | |
| Skill | Abilitate | Keep "Skill" only as a product term |
| Slug (URL) | Slug | Keep |
| Steer | Redirecționează | Never "viraj" (road curve) |
| Stopped | Oprit | |
| Token (ML/API) | Token / Tokeni (pl.) | Never "Jetoane" (casino chips) |
| Tool | Instrument | |
| Turn (conversation) | Rundă | Never "viraj" (road curve) |
| Update | Actualizare / Actualizează | |

## Key names are never translated

`Enter`, `Esc`, `Tab`, `Shift`, `Ctrl`, `Ctrl+C`, `Ctrl+U`, `Cmd`, arrow keys and
single-letter hotkeys (`d`, `e`, `x`, `q`) stay in English inside hint strings.
"Enter" as a key was previously mistranslated to "Introduceți"; the verb "enter"
(e.g. "Enter a custom amount") is correctly `Introdu` / `Introduceți`.

Product/brand names (`Hermes`, `Nous Research`, `Claude Code`, `OpenAI`, …) stay
as-is.

## Placeholders and function leaves

- Named placeholders (`{count}`, `{name}`, `{0}`, `{1}`) must match the English
  source exactly, including count and order.
- `web/src/i18n/ro.ts` function leaves must keep the English call signature;
  YAML packs express function leaves as strings with positional `{0}`, `{1}`.
- Arrays (e.g. `cron.scheduleModes.weekdaysShort`) are JSON-encoded in YAML packs
  and decoded by `adaptStringOverrides()` in `apps/shared/src/i18n.ts`.
- `{s}` is a plural suffix marker consumed by the UI (e.g. `skillCount`). Romanian
  uses a fixed plural form instead; singular counts read "1 abilități", an accepted
  limitation until plural-aware formatting is added.

## Regeneration

After regenerating catalogs with `node scripts/_generate_ro_catalogs.mjs`, re-apply
the manual corrections recorded here and in the Phase 1–6 plan. Set `HERMES_PYTHON`
and `HERMES_HOME` to point the generator at a non-default interpreter or profile.
