# Citation formats per output target

The ledger is format-agnostic: `sources.py render --style ...` emits the block,
this file says where it goes and what the inline marker looks like.

## Markdown / chat answers

Inline `[n]` immediately after the sentence. Block at the end:

```
## Sources

[1] https://example.com/a — Page title
[2] https://example.com/b
```

`--style plain` gives a bare `Sources:` header for chat replies where a
markdown heading would be noise.

## PDF via LaTeX (`latex-pdf-report` skill)

Use `--style footnotes` and map each id to `\footnote{}` at first use, or keep
numeric markers and emit an endnotes section. For a bibliography-shaped report,
`--style bibtex` writes `@misc` entries keyed `source<N>`; cite them with
`\cite{source3}` and let BibTeX render the list.

Do not mix: either numeric `[n]` + a Sources section, or `\cite{}` + BibTeX.
Two numbering systems in one document is worse than none.

## Word (.docx, `docx` skill)

Real footnotes are preferred over inline brackets in prose documents intended
for human editing — reviewers expect Word footnotes. Keep the ledger ids as the
footnote numbers so `verify` still works on a markdown source-of-truth, and
generate the .docx from that markdown.

## Slides (.pptx, `powerpoint` skill)

Inline `[n]` in the bullet, one "Sources" slide at the end rendered with
`--style plain`. Never put a URL in a body bullet — it wrecks the layout and
can't be clicked in a projected deck.

## Spreadsheets (.xlsx)

Add a `source` column holding the id, plus a `Sources` sheet built from
`render --style plain`. Do not paste URLs into data cells.

## Wiki / multi-page output (`llm-wiki`, Obsidian)

Per-page Sources block, ids shared across pages from one ledger. Because ids
are ledger identities, `[7]` means the same page everywhere in the wiki — that
consistency is the reason not to reset the ledger between pages of one build.

## Research papers

Hand off to the `research-paper-writing` skill. Export with
`--style bibtex` into `references.bib`, then follow that skill's citation
verification (it greps `\cite{...}` against the .bib). The ledger's job ends at
producing verified URL entries; venue formatting is that skill's domain.

## Code and config artifacts

No citations inside generated code. If provenance matters, put it in the
commit message, the PR body, or a doc header — not in comments scattered
through source.
