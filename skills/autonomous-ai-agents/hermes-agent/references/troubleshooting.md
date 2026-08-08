# Troubleshooting

### Voice not working
1. Check `stt.enabled: true` in config.yaml
2. Verify provider: `pip install faster-whisper` or set API key
3. In gateway: `/restart`. In CLI: exit and relaunch.

### Tool not available
1. `hermes tools` — check if toolset is enabled for your platform
2. Some tools need env vars (check `.env`)
3. `/reset` after enabling tools

### Model/provider issues
1. `hermes doctor` — check config and dependencies
2. `hermes auth` — re-authenticate OAuth providers (or `hermes auth add <provider>`)
3. Check `.env` has the right API key
4. **Copilot 403**: `gh auth login` tokens do NOT work for Copilot API. You must use the Copilot-specific OAuth device code flow via `hermes model` → GitHub Copilot.

### Changes not taking effect
- **Tools/skills:** `/reset` starts a new session with updated toolset
- **Config changes:** In gateway: `/restart`. In CLI: exit and relaunch.
- **Code changes:** Restart the CLI or gateway process

### Skills not showing
1. `hermes skills list` — verify installed
2. `hermes skills config` — check platform enablement
3. Load explicitly: `hermes -s name` (or the skill's own `/<name>` slash command)

### Gateway issues
Check logs first:
```bash
grep -i "failed to send\|error" ~/.hermes/logs/gateway.log | tail -20
```

Common gateway problems:
- **Gateway dies on SSH logout**: Enable linger: `sudo loginctl enable-linger $USER`
- **Gateway dies on WSL2 close**: WSL2 requires `systemd=true` in `/etc/wsl.conf` for systemd services to work. Without it, gateway falls back to `nohup` (dies when session closes).
- **Gateway crash loop**: Reset the failed state: `systemctl --user reset-failed hermes-gateway`

### Platform-specific issues
- **Discord bot silent**: Must enable **Message Content Intent** in Bot → Privileged Gateway Intents.
- **Slack bot only works in DMs**: Must subscribe to `message.channels` event. Without it, the bot ignores public channels.
- **Windows-specific issues** (`Alt+Enter` newline, WinError 10106, UTF-8 BOM config, line endings): see `references/windows-quirks.md`.

### Computer use not working (cua-driver)

`hermes computer-use doctor` is the first stop — it probes the cua-driver
binary, TCC grants, and daemon autostart registration.

1. **Capture sees elements but bounds are all `(0,0,0,0)`** — the installed
   cua-driver is too old (pre-0.19, `get_window_state` has no structured
   frames). Upgrade with `hermes computer-use install --upgrade`, or from
   upstream: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh)"`.
2. **`AXPress` clicks work but coordinate clicks fail** — same root cause as
   above: old driver. Real frames arrive with 0.19+.
3. **Daemon dies on reboot / after login** — register autostart so `serve`
   comes up at every interactive logon:
   ```bash
   cua-driver autostart enable      # macOS: LaunchAgent; Windows: Scheduled Task
   cua-driver autostart status      # verify "registered (running)"
   ```
   On macOS the LaunchAgent label matches the driver's TCC identity
   (`com.trycua.driver`), so Accessibility / Screen Recording grants persist
   across logons. `hermes computer-use doctor` reports `daemon_autostart` to
   catch this.
4. **macOS permissions granted but still can't capture** — the daemon must
   run under the CLI identity: `~/.local/bin/cua-driver serve` (not
   `open -a CuaDriver --args serve`). Grant Accessibility + Screen Recording
   to both `cua-driver` and `CuaDriver` in System Settings → Privacy &
   Security. After a driver upgrade, re-grant: a new binary has a new
   code-signing hash (TCC grants reset by design).
5. **`cua-driver permissions status` says `daemon_running: false` while
   capture works** — known probe quirk when the daemon is launchd-managed
   (the CLI checks for a daemon under its own identity; launchd's XPC
   identity differs). Trust `launchctl print gui/$(id -u)/com.trycua.driver`
   and real capture bounds instead.
6. **macOS upgrade blocked by "Operation not permitted" on `cp/mv` of
   CuaDriver.app** — TCC app-integrity protection: authorized app contents
   cannot be overwritten (but `rm` is allowed). Delete the app and rebuild
   from the release binary, then re-grant TCC.

### Auxiliary models not working
If `auxiliary` tasks (vision, compression, session_search) fail silently, the `auto` provider can't find a backend. Either set `OPENROUTER_API_KEY` or `GOOGLE_API_KEY`, or explicitly configure each auxiliary task's provider:
```bash
hermes config set auxiliary.vision.provider <your_provider>
hermes config set auxiliary.vision.model <model_name>
```

### "Reset permissions" / auto-approving everything
See `references/security-privacy.md` — wipe the "Always allow" stores, don't touch yolo mode.

