```
 ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓██████▓▒░░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓███████▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░    ░▒▓██████▓▒░░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░  
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░   ░▒▓█▓▒░   ░▒▓██████▓▒░░▒▓████████▓▒░▒▓█▓▒░    ░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░  

                A U T O L Y C U S
              v e r s i o n  0 . 0 . 5-beta
```

---

## Release Notes — Autolycus v0.0.5-beta

**Release Date:** May 6, 2026  
**Platform:** Linux (aarch64/ARM64, x86_64), FreeBSD, macOS  
**License:** GPL v2  
**Status:** Beta — upstream sync integration verified

### About This Release

Autolycus v0.0.5-beta is the first release built on the new upstream sync pipeline. It integrates 100+ commits from upstream Hermes Agent v0.11.0 while preserving our Autolycus TUI branding, custom provider support, and cross-platform installer. The fork now stays current with upstream innovations automatically via an hourly sync cron job.

This release has been verified working on aarch64 (Radxa Rock 5B running Armbian), proving ARM64 compatibility.

---

### What's New

- **Upstream v0.11.0 integration:** Full rebase onto Hermes Agent v0.11.0, bringing in:
  - Internationalization (i18n) support with French and Turkish locale files
  - SSE token batching and error handling for Open WebUI performance
  - Kanban multi-agent board dispatcher with dashboard plugin
  - Doctor diagnostic tool improvements
  - New terminal backends (Lightpanda browser engine)
  - 100+ bug fixes, security patches, and feature enhancements

- **Automated upstream sync pipeline:** Hourly cron job fetches upstream changes, rebases our fork, resolves known conflict patterns (README branding, custom functions), and pushes — keeping the fork always current

- **aarch64/ARM64 verified:** Successfully tested on Radxa Rock 5B (Armbian Linux) with full TUI functionality

- **TUI version update:** Banner now shows v0.0.5

---

### Technical Details

- **Rebase strategy:** Our 31 Autolycus-specific commits rebased cleanly on top of upstream's 100+ new commits
- **Conflict resolution:** README.md (Autolycus branding preserved), cli.py (our custom functions kept), hermes_cli/main.py (upstream's latest taken)
- **Preserved customizations:** `build_plan_path` function, `custom_provider_slug` function, Autolycus README branding
- **Git identity:** Commits authored as waym0reom3ga, committed as Waymore

---

### Known Issues

- Beta status — upstream sync still being refined for edge cases
- Some bundled skills may need manual reset if you have local overrides

---

### Upgrade Instructions

```bash
# Pull latest
cd ~/compiled/autolycus-agent && git pull

# Reinstall
./scripts/install-autolycus.sh

# Start
hermes
```

---

*Autolycus by Technetia Inc — The self-improving AI agent*
