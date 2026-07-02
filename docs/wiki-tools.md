# Wiki Tools

The Hermes pilot treats `yallaplay-wiki/` as Claudio's durable knowledge base. These tools are intentionally small and safe.

## Search

```bash
python3 tools/wiki_search.py "season pass"
python3 tools/wiki_search.py warehouse --domain analytics --limit 10
python3 tools/wiki_search.py slack --domain collaboration --files-only
python3 tools/wiki_search.py "formula dsl" --context 1
```

Output format is clickable and compact:

```text
path/to/page.md:42: matching line
```

Domains are convenience filters over the wiki's current domain layout, not strict permissions.

## New Page Template

Print a template without writing:

```bash
python3 tools/wiki_new.py finding "Rummy economy regression" --print
```

Create a draft page:

```bash
python3 tools/wiki_new.py methodology "Retention cohort freshness check"
python3 tools/wiki_new.py tool "Bedrock" --dir tools/product
```

The helper refuses to overwrite existing files unless `--force` is passed.

## Capture Rules

Save only reusable knowledge:

- Definitions: business or metric concepts.
- Methodology: repeatable workflows, caveats, or bias traps.
- Findings: concrete results with future value.
- Queries: reusable SQL patterns.
- Schema: durable table/column behavior.

Do not save secrets, personal data, unsupported speculation, or one-off scratch output.
