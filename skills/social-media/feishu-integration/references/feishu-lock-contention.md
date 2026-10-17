# Feishu Websocket Lock Contention

## Problem

After making feishu-related code changes and restarting the Gateway, the error log shows:

```
ERROR gateway.platforms.feishu: [Feishu] Another local Hermes gateway is already using this Feishu app_id. Stop the other gateway before starting a second Feishu websocket client.
WARNING gateway.run: ✗ feishu failed to connect
```

The Gateway starts with only 1 platform (api_server) instead of including feishu.

## Root Cause

Feishu uses a **scoped lock file** (`~/.hermes/scoped_locks/feishu-app-id`) to prevent multiple Gateway instances from connecting to the same Feishu app simultaneously. If an old process (even from a different profile like `--profile siyue`) is still running, it holds the websocket connection and the lock persists.

## Fix Procedure

1. **Find ALL hermes processes**:
   ```bash
   ps aux | grep 'hermes.*gateway\|hermes_cli.main' | grep -v grep
   ```

2. **Kill them all** (use -9 if normal kill doesn't work):
   ```bash
   kill -9 <pid1> <pid2> ...
   ```

3. **Unload launchd service** (if running via LaunchAgent):
   ```bash
   launchctl unload ~/Library/LaunchAgents/ai.hermes.gateway.plist
   # For named profiles:
   launchctl unload ~/Library/LaunchAgents/ai.hermes.gateway-<profile>.plist
   ```

4. **Wait for process cleanup**:
   ```bash
   sleep 2
   ps aux | grep 'hermes.*gateway' | grep -v grep  # should be empty
   ```

5. **Reload and start Gateway**:
   ```bash
   launchctl load ~/Library/LaunchAgents/ai.hermes.gateway.plist
   ```

6. **Verify feishu is connected**:
   ```bash
   tail -30 ~/.hermes/logs/gateway.log | grep 'feishu\|platform'
   # Should show feishu in the platform list (not just api_server)
   ```

## Prevention

- Always kill ALL hermes processes before restarting when making feishu changes
- Check for `--profile <name>` processes that may be running independently
- The scoped lock file (`~/.hermes/scoped_locks/feishu-app-id`) is automatically cleaned up when the owning process exits — no manual cleanup needed

## Related

- Gateway restart procedure (SKILL.md Pitfalls section)
- Config.yaml platforms list configuration
