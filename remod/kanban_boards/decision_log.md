# Jade Rebranding - Decision Log

| Date | Decision | Rationale | Alternatives Considered |
|------|----------|-----------|------------------------|
| 2026-05-13 | Use string substitutions only | Keeps upstream merges clean; Option A vs full rename | Full rename (Option B) - rejected due to merge complexity |
| 2026-05-13 | Keep cli.py, main.py unchanged | Explicitly forbidden by AGENTS.md | Would break plugin compatibility |
| 2026-05-13 | Change "⚕" caduceus to "◆" diamond | Clean visual rebrand without complex styling | "⚡", "⚔", "✦" - less unique |
| 2026-05-13 | Keep hermes_cli package name | Would break imports, plugins, config references | jade_cli - too disruptive |
| 2026-05-13 | "Jade — Executive Intelligence for Oracule Zero" tagline | Matches executive/orchestrator positioning | Other options tested, this kept |
| 2026-05-13 | Do NOT modify run_agent.py, gateway/run.py | Core logic files - any change would be logic, not display | None |

## Key Constraints Confirmed
- No Python module/package renames (breakage risk)
- No import path changes
- No config key renames (user compatibility)
- No command name changes (invocation mechanics)