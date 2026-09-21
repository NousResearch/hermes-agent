# Nous Discord Archive — source scripts

Queryable local index of the Nous Research Discord (auto-archived every 6h by
teknium1/nous-discord-archive). This directory is the VERSION-CONTROLLED
source of truth for the pipeline.

## Files

| File | Purpose |
|------|---------|
| `index.py` | Builds the SQLite FTS5 index from the archive clone (incremental by mtime+size; `--force` for full rebuild) |
| `search.py` | Search CLI with author/channel/thread/date filters |
| `pull_and_diff.py` | Pull + reindex + new-message diff engine — the cron's data collector |
| `digest-wrapper.sh` | Cron wrapper: execs pull_and_diff.py, stdout feeds the LLM digest prompt |

## Deployed locations (runtime)

These source files are deployed (copied) to:

- `~/.hermes/nous-archive/` — index.py, search.py, pull_and_diff.py
- `~/.hermes/scripts/nous-archive-digest.sh` — the cron script (the repo copy
  here is named `digest-wrapper.sh` to avoid confusion; the deployed name is
  what the cron `script:` field references)

The skill `skills/research/nous-discord-archive/` (root of this repo) is
deployed to `~/.hermes/skills/research/nous-discord-archive/`.

**After changing any source file here, copy it to the deployed location.**

## NOT committed (runtime state)

- `nous_archive.db` — the 376MB SQLite index (build it with index.py)
- `last_sha` — last processed upstream commit
- `last_diff.txt` — full raw diff of messages since last run

These are gitignored/absent by design. The cron recreates them.

## Cron

Job `nous-archive-digest` (id 53921a891657) runs every 6h (04/10/16/22 UK),
delivers to Discord #hermes-discord. Full output spec lives in the skill at
`references/digest-output-spec.md`.
