# Project context files

Hermes can add one project-context type to the system prompt. `SOUL.md` is separate identity input and does not participate in this priority list.

## Discovery and precedence

The first context type with non-empty content wins:

| Priority | Type | Search scope |
|---|---|---|
| 1 | `.hermes.md`, then `HERMES.md` | Nearest match from the working directory up to the Git root |
| 2 | `AGENTS.md` family | Every directory from the Git root to the working directory |
| 3 | `CLAUDE.md`, then `claude.md` | Working directory only |
| 4 | `.cursorrules` and `.cursor/rules/*.mdc` | Working directory only. Non-empty Cursor files are concatenated |

Without a Git root, Hermes checks only the working directory for the `AGENTS.md` family. This prevents an unrelated file in a home or temporary parent directory from gaining prompt authority.

For each directory in an `AGENTS.md` chain, Hermes tries these names in order:

1. `AGENTS.override.md`
2. `AGENTS.md`
3. `agents.md`

The first non-empty readable file wins for that directory. An empty or unreadable higher-priority file falls through to the next name. Hermes merges selected files root first and working directory last, so deeper guidance has later precedence. Byte-identical content is included once. `AGENTS.override.md` is therefore a same-directory replacement, not an extra layer beside `AGENTS.md`.

Only the winning non-empty context type loads. A non-empty `.hermes.md` shadows the entire `AGENTS.md` chain. `CLAUDE.md` and Cursor rules load only when neither earlier type has non-empty content.

## Progressive subdirectory hints

The startup prompt contains the `AGENTS.md` chain only through the initial working directory. When a tool later enters or accesses a deeper directory, `agent/subdirectory_hints.py` may attach that directory's local hint file to the tool result. Hints stay inside the working tree, skip excluded dependency/cache/archive directories, honor `AGENTS.override.md`, and deduplicate identical content. They do not rebuild the system prompt.

## Size and truncation

`context_file_max_chars` in `config.yaml`, when set to a positive number, is the cap. Otherwise Hermes derives the cap from the model context window:

- 6% of the window after converting tokens to the prompt builder's character estimate.
- A 20,000-character floor.
- A 500,000-character ceiling.
- 20,000 characters when the context length is unknown.

Hermes applies the cap to each selected file section. It also applies the same cap to the merged `AGENTS.md` chain, so depth cannot multiply the budget without limit. Truncation keeps the head and tail, inserts a marker, names the source path for `read_file`, logs the event, and queues a session warning. Progressive subdirectory hints use their own fixed preview limit and warning behavior.

Keep universal rules at the repository root and local non-inferable rules in nested files. Put architecture, tutorials, inventories, and process detail in maintained docs behind precise pointers.

## Security and control

Project context passes through the prompt-injection scanner before inclusion. A blocked project file becomes a `[BLOCKED: ...]` marker rather than executable prompt text. Symlinked subdirectory hints must resolve inside the working tree and may not target denied paths.

`hermes --ignore-rules` disables automatic project-context loading for that session along with the other user customization layers documented by `hermes --help`. Use it to isolate configuration problems, not as a permanent fix.

## Implementation references

- Discovery, precedence, and truncation: `agent/prompt_builder.py`
- Manifest used by context inspection: `agent/context_file_sources.py`
- Later directory hints: `agent/subdirectory_hints.py`
- Prompt ordering: `website/docs/developer-guide/prompt-assembly.md`