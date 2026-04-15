## [2.0.0] - 2026-04-13

### Changed
- Complete rewrite of the dream cycle architecture
- Replaced single-pass background loop with a deterministic four-phase pipeline: Orient, Gather, Classify, File
- Replaced `$AGENT_DATA_ROOT` paths with `{agent_root}/commons/` convention for platform portability
- Signal emission now uses journal signal payload field (standard OCAS pattern) instead of direct write to Elephas intake directory
- `references/classification.md` moved to `references/` directory

### Added
- Incremental cursor: ingestion log tracks last processed run_id per skill; mid-run termination loses no filed work
- Re-emergence detection: topics skipped 3+ times in subsequent journals are auto-promoted
- Two-pass stale handling: MemPalace entries require two consecutive contradictions before invalidation
- Change magnitude gates: >30% file rate warns; >50% halts and stages for operator review
- Hibernation protection: skip cycle entirely if no new journals for 7 consecutive days
- Relevance scoring model with additive signals and penalties (see `references/classification.md`)
- Wing/room assignment taxonomy for all OCAS skill domains
- `lucid:update` cron job (midnight daily self-update)
- `lucid.update` command and self-update procedure
- MemPalace declared as required MCP in frontmatter
- Standard `hermes:` and `openclaw:` frontmatter structure

### Removed
- `dream-loop.py`, `init-script.py`, `cron-command.sh` scripts (agent now reasons directly)
- `ocas-layer`, `ocas-visibility`, `ocas-skill-type`, `ocas-cron` custom frontmatter fields
- `compatibility:` non-standard frontmatter field (moved to `requires.mcp`)

## [1.0.0] - 2026-04-09

### Added
- Initial implementation: nightly dream loop reviewing session interactions for MemPalace/Chronicle ingestion
- Dual-phase process: Fresh Scan and Weak Signal Recirculation
- Background execution mode to avoid timeout on long-running analysis
