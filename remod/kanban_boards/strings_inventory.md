# Jade Rebranding - Strings Inventory

## Intentionally Left (Safe to Keep)

These are NOT to be changed because they are:
- Python import/module names (would break installation)
- Environment variables (API compatibility)
- Package names (PyPI publishing)
- Internal function/variable names (not user-facing)
- Config keys (user configs would break)

| Pattern | Reason | Examples |
|---------|--------|----------|
| hermes_cli package dir | Python package name | from hermes_cli import ... |
| get_hermes_home() | Internal function | config paths |
| HERMES_HOME env var | Config compatibility | user configs |
| hermes-agent PyPI name | Package identity | installation |
| run_agent.py, cli.py, gateway/run.py, hermes_cli/main.py | FORBIDDEN FILES | AGENTS.md rules |

## Needs Manual Review

To be filled after running grep:

`
Files to check after grep:
- [ ] Any remaining "Hermes" strings
- [ ] Any remaining "NousResearch" strings
- [ ] Any remaining caduceus icons
- [ ] Any remaining caduceus references
`

## Already Changed

| File | Line | Before | After |
|------|------|--------|-------|
| hermes_cli/banner.py | 472 | Nous Research | Oracule Zero |
| hermes_cli/banner.py | doc | Hermes update | Jade update |
| hermes_cli/default_soul.py | 4 | Hermes Agent, Nous Research | Jade, Oracule Zero |
| hermes_cli/skin_engine.py | all skins | Hermes Agent | Jade |
| hermes_cli/status.py | 97 | Hermes Agent Status | Jade Status |
| hermes_cli/setup.py | 180 | Hermes Setup | Jade Setup |
| hermes_cli/setup.py | 3097 | Hermes Agent Setup | Jade Setup |
| hermes_cli/tools_config.py | 2393 | Hermes Tool Configuration | Jade Tool Configuration |
| hermes_cli/uninstall.py | 455 | Hermes Agent Uninstaller | Jade Uninstaller |
| hermes_cli/gateway.py | 3171 | Hermes Gateway | Jade Gateway |
| ui-tui/src/theme.ts | 239-247 | Hermes Agent | Jade |
| ui-tui/src/branding.tsx | 52 | NOUS HERMES | JADE |
| ui-tui/src/branding.tsx | 56 | Nous Research | Oracule Zero |
| ui-tui/src/branding.tsx | 227 | Nous Research | Oracule Zero |