---
name: obsidian
description: Read, search, create, and edit notes in the Obsidian vault.
---

# Obsidian Skill

Use this skill for filesystem-first work on Obsidian-owned notes: reading,
listing, searching, creating, appending, editing, and adding wikilinks. Do not
use Obsidian as the default discovery engine for code, configuration, runtime
state, or another source whose authoritative location is already known.

## When to Use

- The user asks to read, find, create, or edit an Obsidian note.
- The requested artifact belongs in the vault.
- The authoritative source is unknown and vault evidence is genuinely relevant.

## Prerequisites

Resolve the authoritative source before accessing the vault. Resolve the vault
path only when the source-resolution procedure selects Obsidian.

The documented vault-path convention is the `OBSIDIAN_VAULT_PATH` environment
variable, for example from `${HERMES_HOME:-~/.hermes}/.env`. If it is unset,
use `~/Documents/Obsidian Vault`.

File tools do not expand shell variables. Pass a concrete absolute path to
`read_file`, `write_file`, `patch`, or `search_files`. Vault paths may contain
spaces, which is another reason to prefer file tools over shell commands.

If the selected vault path is unknown, `terminal` is acceptable for resolving
`OBSIDIAN_VAULT_PATH` or checking whether the fallback exists. Once resolved,
switch back to file tools.

## How to Run

### Source resolution

Apply this lookup order before any vault discovery:

1. **Explicit source** — If the user provides a path, repository, URL,
   filename, pull request, or exact authoritative source, access it directly.
   Use it for the requested evidence without inferring claims it does not
   support.
2. **Known canonical home** — Use the source's owning canonical location:
   - Project governance (including MAIOS) → the canonical GitHub repository or
     a verified local checkout.
   - Code and configuration → the owning repository or verified machine config.
   - Runtime state → the designated runtime location.
   - Obsidian-owned notes and exploratory knowledge → Obsidian.
3. **Narrow source index** — When the exact location is not known, use the
   owning repository map, manifest, or source-specific index.
4. **Obsidian discovery** — Use vault indexes and search only when the artifact
   belongs in Obsidian, the user requests vault research, or the source is
   unknown and vault evidence is genuinely relevant.
5. **Broad discovery** — Search broadly only as the final fallback.

Enforce these rules:

- Never scan Obsidian merely because a task concerns system configuration.
- Never reconstruct repository or configuration evidence from vault notes when
  the live authoritative source is available.
- Do not repeat a vault search after resolving the direct source.
- For a routing-hook investigation, search known Codex hook, config, and plugin
  locations directly unless evidence shows the target hook is vault-owned.
- If a direct source is inaccessible, disclose that fact and use one deliberate
  fallback; do not silently change authority.
- Preserve vault privacy and organization rules.
- Preserve one canonical home for each artifact. Patch the generator or source
  that owns generated copies, regenerate them, and verify parity instead of
  editing an installed or cached copy into permanent divergence.

### Quick reference

| Operation | Preferred tool |
|---|---|
| Read a note | `read_file` |
| List notes | `search_files` with `target: "files"` |
| Search note contents | `search_files` with `target: "content"` |
| Create or rewrite a note | `write_file` |
| Focused edit or anchored append | `patch` |

## Procedure

1. Apply the source-resolution order. Stop if a direct non-vault source answers
   the request.
2. If Obsidian is selected, resolve the concrete absolute vault path.
3. Use the narrowest appropriate vault operation:
   - Read a known note with `read_file`.
   - List Markdown notes with `search_files`, `target: "files"`, and
     `pattern: "*.md"` under the selected folder.
   - Search note contents with `search_files`, `target: "content"`, the content
     regex as `pattern`, and `file_glob: "*.md"` when appropriate.
   - Create or rewrite a note with `write_file` and complete Markdown content.
   - Use `patch` for a focused edit or an anchored append with stable context.
4. For a simple append without stable context, use `terminal` only when it is
   the clearest safe option.
5. Use `[[Note Name]]` syntax for Obsidian wikilinks.

## Pitfalls

- A system topic is not evidence that its source lives in Obsidian.
- Vault notes may summarize stale repository or machine state; prefer the live
  authoritative source whenever it is available.
- Do not pass unresolved variables such as `$OBSIDIAN_VAULT_PATH` to file tools.
- Do not broaden a vault search after a direct source has been resolved.
- Do not overwrite a generated copy without updating its canonical source and
  running the owning regeneration path.

## Verification

Validate source selection with these cases:

- Exact hook file or config location known → access it directly with zero vault
  lookup.
- Exact GitHub repository and pull request known → use GitHub directly with zero
  vault lookup.
- User asks for an Obsidian note → use the vault first.
- Source genuinely unknown but likely recorded in notes → use a narrow vault
  index, then search.
- Direct source unavailable → disclose the failure and use one deliberate
  fallback.
- Generated skill copy → regenerate it through the owning sync path and verify
  that it matches the canonical source.
