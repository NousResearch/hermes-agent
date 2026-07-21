# Workflow Lab Analyzer

`scripts/workflow_lab_analyze.py` builds a private DuckDB database from local
developer workflow traces and emits an aggregate Markdown report. It is intended
for periodic workflow-improvement labs where raw history must remain local.

## Data sources

When present, the analyzer ingests:

- shell history: `~/.zsh_history`, `~/.bash_history`, fish history
- Atuin SQLite history metadata
- Codex, Claude Code, and Hermes JSONL session summaries under the standard home
  directories plus project-local `.codex`, `.claude`, and `.hermes`
- git repository metadata under a configurable root

The script also records local tool availability, including `ladybugs`,
`ladybug`, and related Python imports, so runs can distinguish missing local
packages from workflow-relevant tools.

## Privacy model

Raw history/session payloads are never printed to stdout. The generated
`analysis.md` contains aggregate counts and redacted short summaries only.
Obvious token/API-key/password patterns are replaced before rows are written to
`events_redacted.jsonl` and `workflow.duckdb`.

Still treat the output directory as private: command histories can contain
sensitive paths, hostnames, and project names even after redaction.

## Usage

DuckDB can be supplied either by the Python package or by a throwaway virtualenv:

```bash
uv venv /tmp/workflow-lab-venv
uv pip install --python /tmp/workflow-lab-venv/bin/python duckdb
/tmp/workflow-lab-venv/bin/python scripts/workflow_lab_analyze.py \
  --output ~/.hermes/workflow-lab/$(date +%F)
```

Optional flags:

- `--repo-root PATH` — scan a narrower tree for git repositories.
- `--no-git-scan` — skip repository metadata discovery for a faster run.

Outputs:

- `workflow.duckdb` — queryable local event database.
- `events_redacted.jsonl` — redacted normalized rows used to build the database.
- `analysis.md` — aggregate findings and candidate improvements.

## Example follow-up queries

```sql
select source, count(*) from events group by 1 order by 2 desc;
select left(text, 120), count(*)
from events
where status <> '' and try_cast(status as bigint) <> 0
group by 1
having count(*) > 1
order by 2 desc;
```
