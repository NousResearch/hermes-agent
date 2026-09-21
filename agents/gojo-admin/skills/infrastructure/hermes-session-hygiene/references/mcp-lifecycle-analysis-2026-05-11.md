# MCP Lifecycle Analysis — May 11 2026

## Live PID Audit (2026-05-11 18:35)

### Gateway stack (KEEP)
```
3487705       1 158912 Ssl       57:45 ?        python -m hermes_cli.main gateway run --replace
3487709 3487705 52796 Ssl        58:02 ?        uv tool uvx workspace-mcp
3487711 3487705 121436 Ssl       58:02 ?        node ms-365-mcp-server --preset mail,calendar
3487713 3487705 72972 Ssl        58:02 ?        node nanobanana-mcp/dist/index.js
3487757 3487709 125260 Sl        58:01 ?        python workspace-mcp (child of uvx)
```
Total: ~396MB across 4 processes. Correctly daemonised (PPID=1). Keep.

### Idle CLI stack 1 — pts/0 (REMOVABLE)
```
3340351 3340344 458476 Sl+    08:18:44 pts/0    python3 hermes (bare CLI, no subcommand)
3455443 3340351 122956 Ssl    02:42:33 ?        node ms-365-mcp-server --preset mail,calendar
3438591 3340351 74528 Ssl     03:33:35 ?        node nanobanana-mcp/dist/index.js
3455518 3340351 46288 Ssl     02:42:41 ?        uv tool uvx workspace-mcp
3455539 3455518 124224 Sl     02:42:41 ?        python workspace-mcp (child of uvx)
```
Total: ~689MB across 5 processes. Idle 8h+. SSH session alive (pts/0).

### Idle CLI stack 2 — pts/4 (REMOVABLE — this session)
```
3501646 3498091 242152 Rl+        13:17 pts/4    python3 hermes chat
3502524 3501646 55680 Ssl        10:33 ?        uv tool uvx workspace-mcp
3502526 3501646 116928 Ssl       10:33 ?        node ms-365-mcp-server --preset mail,calendar
3502528 3501646 71896 Ssl        10:33 ?        node nanobanana-mcp/dist/index.js
3502598 3502524 131652 Sl        10:32 ?        python workspace-mcp (child of uvx)
```
Total: ~482MB across 5 processes. Active at time of audit but would be reclaimable after this session ends.

### Orphan npm exec (REMVABLE)
```
3443461 3443443 90564 Sl+     03:18:48 pts/2    npm exec @ycse/nanobanana-mcp
3443525 3443461  1916 S+      03:18:47 pts/2    sh -c nanobanana-mcp
3443526 3443525 71548 Sl+     03:18:47 pts/2    node nanobanana-mcp (via npx, NOT the standard path)
```
Total: ~90MB. Manual leftover from abandoned `npm exec` command. Not part of a Hermes stack.

## System Memory at Time of Audit
```
Mem:   7.6Gi total,   3.4Gi used,  420Mi free
Swap:  4.0Gi total,   1.6Gi used,  2.4Gi free
```
Total reclaimable: ~1.26GB (689 + 482 + 90). This would bring used memory to ~2.1Gi and free to ~1.7Gi.

## Key Observations

1. **No PPID-1 orphans found.** All MCP children were correctly parented. The "orphan" problem in prior reports was either stale data, or a gateway restart self-healed them before this audit.

2. **The bigger problem is idle duplicates, not orphans.** SSH sessions stay alive for hours/days. The Hermes CLI and its MCP children are technically alive and parented — but functionally idle.

3. **The previous report referenced PID 3420935 which was already dead by audit time.** Always reconcile live before acting on stale reports.

4. **PIDs shift after every gateway restart.** Never hardcode PIDs in cleanup scripts.

## Gateway Health
- PID 3487705 owns port 8000 (workspace-mcp OAuth callback server)
- Dashboard (PID 1692448) owns port 9119
- Gateway lock file is correctly held
- No reparented MCP children to PID 1

## The nanobanana-mcp duplication on pts/2

The `npm exec @ycse/nanobanana-mcp` at PID 3443461 is interesting — it's running nanobanana-mcp through npx from ~/.npm/_npx/, NOT from the standard path (~/.hermes/node-mcps/node_modules/@ycse/nanobanana-mcp/). This suggests someone ran a manual test or npm install from a raw shell. It's completely detached from any Hermes process tree. Safe to kill.
