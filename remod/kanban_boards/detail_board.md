# Jade Rebranding - Detail Board

## Phase 1: Python CLI Branding (High Priority)

### TODO
- [ ] README.md - Title, description, intro section (keep install instructions)

### IN PROGRESS
- [ ] hermes_cli/config.py line 4707 - "Hermes Configuration" header
- [ ] hermes_cli/config.py line 266 - "Hermes Agent" comment (if still exists)

### DONE
- [x] hermes_cli/banner.py (lines 472, 130, 287)
- [x] hermes_cli/default_soul.py (line 4)
- [x] hermes_cli/skin_engine.py (all 6 skin dicts)
- [x] hermes_cli/status.py (lines 97)
- [x] hermes_cli/setup.py (lines 180, 2087, 3097, 3133)
- [x] hermes_cli/tools_config.py (line 2393)
- [x] hermes_cli/uninstall.py (lines 455, 679)
- [x] hermes_cli/gateway.py (lines 1253, 3171)

## Phase 2: TUI Branding (Medium Priority)

### DONE
- [x] ui-tui/src/theme.ts (lines 239-247)
- [x] ui-tui/src/components/branding.tsx (lines 52, 56, 227)
- [x] ui-tui/src/components/appChrome.tsx (line 30)
- [x] ui-tui/src/components/appLayout.tsx (line 318)

### TODO (Requires ASCII Art Work)
- [ ] ui-tui/src/banner.ts - LOGO_ART "HERMES AGENT" -> "JADE AGENT"
- [ ] ui-tui/src/banner.ts - CADUCEUS_ART -> diamond/art update

## Phase 3: Verification (Final)

### TODO
- [ ] Grep entire repo for "Hermes" (case-sensitive)
- [ ] Grep entire repo for "NousResearch" (case-sensitive)
- [ ] Categorize results into:
  - (a) Intentionally kept (imports, package names, env vars)
  - (b) Needs manual review
- [ ] Delete remod/ docs files (they are session notes, not needed)
- [ ] Final verification: git diff summary