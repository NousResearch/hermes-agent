# Jade Rebranding - Risk Register

| Risk | Probability | Impact | Mitigation | Status |
|------|-------------|--------|------------|--------|
| Forbidden file accidentally modified | Low | Critical | Verify git diff before each commit | Avoided |
| Import breakage from module renames | Medium | High | DO NOT rename modules, keep hermes_cli | Mitigated |
| Upstream merge conflicts in next cycle | Low | Medium | Stick to Option A (display strings only) | Managed |
| TUI ASCII art mismatch | Medium | Medium | Redesign LOGO_ART after core changes | Open |
| User config breakage | Low | High | Keep HERMES_HOME, HERMES_* env var names | Mitigated |
| Plugin compatibility lost | Low | High | Do not change plugin API surface | Mitigated |
| Branding inconsistency (some strings missed) | High | Low | Full repo grep sweep | Active |
| README/docs drift from code | Medium | Low | Update docs after string changes | Open |
| Skill catalog references to Hermes | Low | Medium | Need to audit skills/ directory | Open |

## Top 3 Action Items
1. Complete full-repo "Hermes" grep sweep
2. Complete README.md updates
3. Complete skill file audit