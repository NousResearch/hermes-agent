# Orient

Find the existing shape before you edit. Do not rewrite from a generated wiki.

## First 60 seconds

1. Read root `AGENTS.md` and `CONTRIBUTING.md`. Search open issues and PRs.
2. If GitNexus MCP tools are already present: `query` the concept, `context` on the symbol, `impact` before the first edit. Do not run `gitnexus analyze` or `setup` unless a human asked.
3. Else if Serena (or another LSP MCP) is present: `find_symbol`, then `find_referencing_symbols`.
4. Else: `rg` the name, open the files, optionally `ast-grep` for a structural pattern.
5. For a third-party library API, use Context7 or upstream docs. Do not guess signatures from memory.
6. Name the blast radius in one sentence (who calls this, which tests, which sibling surfaces). Then edit.

DeepWiki, cluster dumps, and packed trees (repomix/gitingest) are orientation only. They are not evidence.

GitNexus is PolyForm Noncommercial — use it if the maintainer already installed it. Do not vendor it. Serena is the MIT LSP fallback.

## Do not

- Index the repo or rewrite `AGENTS.md` / `CLAUDE.md` as a side effect of a claim.
- Duplicate instruction files (`CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md`) when `AGENTS.md` already exists.
- Treat a wiki answer as `tests.green`.
- Invent a second loop named after a tool.
