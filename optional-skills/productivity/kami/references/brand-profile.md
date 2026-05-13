# Brand Profile Reference

Full specification for loading and applying user brand profiles. Referenced from SKILL.md Step 0.

## Profile locations

1. `~/.config/kami/brand.md` (global user config, XDG-compliant, preferred)
2. `~/.kami/brand.md` (legacy fallback)

If found, parse the YAML frontmatter for structured fields and treat the Markdown body below the frontmatter as freeform habit notes. If no profile exists, continue without interruption.

## Layer A: Placeholder substitution

Apply at template edit time: replace matching `{{...}}` placeholders in the HTML body with profile values before building.

| Profile field | Template placeholder(s) |
|---|---|
| `name` | `{{NAME}}`, `{{作者}}`, `{{AUTHOR}}` |
| `role_title` | `{{ROLE_TITLE}}` |
| `email` | `{{EMAIL}}` |
| `website` | `{{WEBSITE}}`, `{{PERSONAL_SITE}}` |
| `github` | `{{GITHUB_URL}}` (expand to full URL: `https://github.com/<value>`) |
| `x` | `{{X_URL}}` (expand to `https://x.com/<value>`) |
| `city` | `{{CITY}}` |
| `country` | `{{COUNTRY}}` |
| `company` | `{{COMPANY}}` |
| `tagline` | `{{TAGLINE}}` |

Rule: explicit instructions in the current conversation always override profile values (e.g. "sign it as Alex" beats `name: Tang`). Profile fills slots only; if the template has no matching placeholder, do not insert the field.

## Layer B: Session defaults

Apply when the current request is ambiguous. Use profile fields to fill in missing signal silently; do not skip clarification questions for genuinely unclear intent (e.g. "make a document" still warrants asking which type).

| Ambiguous signal | Profile field | Behavior |
|---|---|---|
| No language stated | `language` | Use `cn` / `en` / `ja` path |
| No doc type stated | `default_doc_type` | Use as fallback, still ask if truly ambiguous |
| No output format stated | `output_format` | `pdf` / `pptx` / `both` |
| No page size stated | `page_size` | `a4` or `letter` |
| Currency context unclear | `currency_locale` | `USD` -> $, M/B; `CNY` -> 元, 万/亿 |
| Tone unclear | `tone` | `formal` -> deferential register; `casual` -> relaxed; `balanced` -> default |
| Footer unclear | `footer_note` | Append standing disclaimer if present |
| TOC decision unclear | `always_include_toc` | `true` -> auto-add TOC in long-doc / portfolio |

## Layer C: Visual customization

Apply after template content is filled, before calling `build.py`:

- `brand_color`: edit the `--brand` CSS variable in the template `<style>` block. Warn if the hue departs significantly from ink-blue; the warm palette constraint (parchment + neutrals) remains in force regardless.
- `logo`: insert file path into any `<img src="...">` logo slot in one-pager / portfolio / slides cover.

## Layer D: Habit notes

Treat the Markdown body below the YAML frontmatter as additional voice and style context, merged with `references/writing.md` guidance. When a note names a specific document type ("equity reports: dense, cite inline"), apply it when that doc type is active. Habit notes are advisory: if the current document's content does not fit a stated preference, follow the content.

## Guardrails (preserve editorial judgment)

These six rules apply to every layer above. Weight them equally with the field table.

1. **Profile is a fallback, not a first resort.** When the current request already carries tone, urgency, or subject-specific signal, use that signal. Profile fills gaps only.
2. **Editorial judgment can override any layer silently.** If a profile field would harm this specific document, skip it. No explanation needed. Example: `tone: formal` in profile, but the user hands over a playful changelog; follow the changelog's voice.
3. **Vary surface details across documents.** Even when author, brand_color, and company all come from profile, each document's opening hook, chart selection, section order, and density must be chosen fresh for the current content. Two equity reports should not look like clones.
4. **Profile fills slots, never introduces new content.** If the template has no `{{EMAIL}}` placeholder, do not insert a contact line because `email` exists in profile.
5. **Apply silently, do not announce.** Never write "applied your profile defaults" or any similar note in the output. Profile disappears into the background.
6. **Habit notes are advisory, not mandatory.** They describe tendencies ("I prefer dense reports"), not hard rules. When content does not fit, follow the content.

## Precedence

```
explicit prompt  >  editorial judgment for this document  >  habit notes  >  frontmatter defaults  >  built-in defaults
```
