# RTK - Rust Token Killer (Codex CLI)

Usage: token-optimized CLI proxy for shell commands.

Rule
- Prefer `rtk` wrappers for shell commands in this repo.
- Prefer the narrowest wrapper that fits the job.
- Use raw shell only when `rtk` would hide needed detail.

High-value commands
```bash
rtk git status
rtk git diff
rtk git log --oneline --decorate -n 20
rtk pytest -q
rtk test --help
rtk err <command>         # show only errors/warnings
rtk summary <command>     # compress verbose command output
rtk log ~/.hermes/logs/gateway.log
rtk tree -L 1
rtk ls
rtk read path/to/file
rtk grep 'pattern' path/to/dir
rtk json file.json
```

Repo guidance
- For Hermes gateway troubleshooting, prefer `rtk log ~/.hermes/logs/gateway.log` over raw `journalctl` when the file already has the signal you need.
- For very verbose commands, prefer `rtk summary ...` first, then rerun raw only if needed.
- If a command is repeatedly noisy, improve `.rtk/filters.toml` instead of re-reading the same boilerplate.

Meta commands
```bash
rtk gain            # token savings analytics
rtk gain --history   # recent command savings
rtk proxy <cmd>     # run raw command without filtering but track usage
```

Verification
```bash
rtk --version
rtk gain
which rtk
```
