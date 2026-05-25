# Implementation Decisions

Updated: 2026-05-26

## Trend Discovery Center

Decision: implement as a bundled standalone plugin, not by rewriting Hermes core.

Reason:

- keeps this business-trend system isolated and maintainable
- uses existing Hermes plugin CLI surface
- avoids hardcoding a one-off project into `run_agent.py`, `cli.py`, or gateway core

Decision: use SQLite under `~/.hermes/trend-discovery/`.

Reason:

- local-first and profile-scoped
- no VPS/Postgres dependency required for the initial reliable system
- easy for future AI agents to inspect and back up

Decision: use macOS `launchd` on this machine.

Reason:

- user asked for the system to actually run
- this host is macOS
- LaunchAgents continue running on schedule without keeping the chat open

Decision: configure macOS notifications with local fallback.

Reason:

- provides real user-visible notification on this Mac
- local receipt logs still capture delivery attempts if notification fails

Decision: keep Open Crawl and n8n optional.

Reason:

- the original failure mode was over-reliance on Open Crawl and n8n
- source adapters should fail independently without taking down the pipeline
- empty optional adapter URLs must be marked skipped, not success

Decision: expose explicit operator modes and source administration.

Reason:

- future AI agents and the user need to see how the system is controlled
- launchd schedule alone is not enough; operators need mode/status/source/log commands
- `trend-discovery ops` is the top-level operational summary
