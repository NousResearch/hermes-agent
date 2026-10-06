---
name: fleet-ssh
description: Manage multiple machines via SSH — run commands, check status, deploy files, and monitor health across a fleet
version: 1.0.0
author: het4rk
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [SSH, Fleet, DevOps, Remote, Management, Monitoring]
    requires_toolsets: [terminal]
    config:
      - key: fleet.hosts_file
        description: "Path to fleet hosts inventory file"
        default: "~/.hermes/fleet/hosts.yaml"
        prompt: "Path to your fleet hosts file"
---

# Fleet SSH Management

Manage multiple machines via SSH — run commands across your entire fleet, check health, deploy files, and monitor uptime, load, and disk usage in parallel.

## When to Use

- Run a shell command across multiple machines simultaneously
- Check the health of your fleet (uptime, load, disk, memory)
- Deploy a file or script to all hosts
- Tail logs from a remote machine
- Check if a process is running across fleet nodes
- Coordinate tasks on homelab servers, mining rigs, or distributed compute nodes

## Prerequisites

- SSH key-based authentication configured for all hosts (no password prompts)
- Python 3.7+ with `pyyaml` installed (`pip install pyyaml`)
- `ssh` and `scp` available in `$PATH`

Verify SSH key auth works:
```bash
ssh -o BatchMode=yes user@host "echo ok"
```

## Fleet Inventory

Create your hosts inventory at `~/.hermes/fleet/hosts.yaml` (or the path set in config):

```yaml
hosts:
  - name: node0
    host: 10.0.0.1
    user: admin
  - name: node1
    host: 10.0.0.2
    user: admin
  - name: node2
    host: 10.0.0.3
    user: deploy
    port: 2222        # optional, defaults to 22
    tags: [gpu, mining]
```

A template is available at `templates/hosts.yaml` in this skill directory.

## Quick Reference

| Task | Command |
|------|---------|
| Run command on one host | `ssh user@host "command"` |
| Run command on all hosts | See procedure below |
| Check fleet health | `python3 scripts/fleet_status.py` |
| Deploy file to all hosts | `for` loop with `scp` — see procedure |
| Tail remote logs | `ssh user@host "tail -f /var/log/syslog"` |
| Check process across fleet | `ssh user@host "pgrep -x proc_name"` |

## Procedures

### 1. Run Command on a Single Host

```bash
ssh admin@10.0.0.1 "uptime"
```

With a non-default port:
```bash
ssh -p 2222 admin@10.0.0.3 "df -h /"
```

### 2. Run Command Across All Hosts (Parallel)

Parse `hosts.yaml` and fire background SSH jobs, then wait for all to finish:

```bash
#!/usr/bin/env bash
# run_all.sh <command>
HOSTS_FILE="${FLEET_HOSTS_FILE:-$HOME/.hermes/fleet/hosts.yaml}"
CMD="$*"

# Parse YAML with Python (avoids dependency on yq)
eval "$(python3 - <<'EOF'
import yaml, sys
with open("$HOSTS_FILE") as f:
    data = yaml.safe_load(f)
for h in data["hosts"]:
    port = h.get("port", 22)
    print(f'ssh -p {port} -o ConnectTimeout=10 -o BatchMode=yes {h["user"]}@{h["host"]} "$CMD" &')
print("wait")
EOF
)"
```

Or run it inline as a one-liner (requires `yq`):
```bash
HOSTS_FILE=~/.hermes/fleet/hosts.yaml
CMD="uptime"
yq -r '.hosts[] | "\(.user)@\(.host)"' "$HOSTS_FILE" | \
  xargs -P8 -I{} ssh -o ConnectTimeout=10 -o BatchMode=yes {} "$CMD"
```

### 3. Check Fleet Health

Use the bundled script to get uptime, load, disk, and memory for every host:

```bash
python3 skills/devops/fleet-ssh/scripts/fleet_status.py
# or after copying to PATH:
fleet_status.py
```

Sample output:
```
Fleet Health Report — 2024-01-15 14:32:01
═══════════════════════════════════════════════════════════════
Host      Address      Status  Uptime          Load   Disk%  Mem%
─────────────────────────────────────────────────────────────────
node0     10.0.0.1     UP      3 days, 4:12    0.42   34%    61%
node1     10.0.0.2     UP      1 day, 22:05    1.20   67%    45%
node2     10.0.0.3     DOWN    —               —      —      —
═══════════════════════════════════════════════════════════════
2/3 hosts reachable
```

To use a custom hosts file:
```bash
FLEET_HOSTS_FILE=/path/to/hosts.yaml python3 scripts/fleet_status.py
```

### 4. Deploy File to All Hosts

```bash
#!/usr/bin/env bash
# deploy_file.sh <local_path> <remote_path>
LOCAL="$1"
REMOTE="$2"
HOSTS_FILE="${FLEET_HOSTS_FILE:-$HOME/.hermes/fleet/hosts.yaml}"

python3 - <<EOF
import yaml, subprocess, sys

with open("$HOSTS_FILE") as f:
    data = yaml.safe_load(f)

procs = []
for h in data["hosts"]:
    port = str(h.get("port", 22))
    dest = f'{h["user"]}@{h["host"]}:{REMOTE}'
    cmd = ["scp", "-P", port, "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "$LOCAL", dest]
    print(f"Deploying to {h['name']} ({dest})...")
    procs.append((h["name"], subprocess.Popen(cmd)))

for name, p in procs:
    rc = p.wait()
    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
    print(f"  {name}: {status}")
EOF
```

### 5. Tail Remote Logs

Tail a log on a specific host (interactive, Ctrl-C to stop):
```bash
ssh admin@10.0.0.1 "tail -f /var/log/syslog"
```

Tail the same log across multiple hosts (one terminal pane each, requires `tmux`):
```bash
HOSTS_FILE=~/.hermes/fleet/hosts.yaml
python3 -c "
import yaml
with open('$HOSTS_FILE') as f:
    hosts = yaml.safe_load(f)['hosts']
for i, h in enumerate(hosts):
    cmd = f'tmux new-window -n {h[\"name\"]} \"ssh {h[\"user\"]}@{h[\"host\"]} tail -f /var/log/syslog\"'
    import os; os.system(cmd)
"
```

### 6. Check if a Process is Running Across Fleet

```bash
PROCESS="nginx"
HOSTS_FILE="${FLEET_HOSTS_FILE:-$HOME/.hermes/fleet/hosts.yaml}"

python3 - <<EOF
import yaml, subprocess

with open("$HOSTS_FILE") as f:
    hosts = yaml.safe_load(f)["hosts"]

for h in hosts:
    port = str(h.get("port", 22))
    result = subprocess.run(
        ["ssh", "-p", port, "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
         f'{h["user"]}@{h["host"]}', f"pgrep -x $PROCESS > /dev/null && echo RUNNING || echo NOT RUNNING"],
        capture_output=True, text=True, timeout=10
    )
    status = result.stdout.strip() or "UNREACHABLE"
    print(f"{h['name']:12s} {h['host']:15s}  $PROCESS: {status}")
EOF
```

## Pitfalls

- **SSH key auth required**: Password prompts will hang in parallel mode. Use `ssh-copy-id user@host` to install your key first.
- **ConnectTimeout**: Always set `-o ConnectTimeout=10` (or similar) to avoid blocking on offline hosts.
- **Output interleaving**: Parallel SSH output can interleave. Prefix each line with the hostname or collect output per-host before printing.
- **Known hosts**: First-time connections prompt for host key confirmation. Pre-populate `~/.ssh/known_hosts` with `ssh-keyscan`:
  ```bash
  ssh-keyscan -H 10.0.0.1 10.0.0.2 10.0.0.3 >> ~/.ssh/known_hosts
  ```
- **StrictHostKeyChecking**: For trusted internal networks, add `-o StrictHostKeyChecking=accept-new` to auto-accept new host keys.
- **Sudo commands**: Use `ssh user@host "sudo command"` only if passwordless sudo is configured; otherwise it will hang.

## Verification

After setting up your `hosts.yaml`, verify the skill is working:

```bash
# 1. Confirm SSH key auth works for each host
python3 -c "
import yaml, subprocess
with open('$HOME/.hermes/fleet/hosts.yaml') as f:
    for h in yaml.safe_load(f)['hosts']:
        r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                            f'{h[\"user\"]}@{h[\"host\"]}', 'echo ok'],
                           capture_output=True, text=True)
        print(h['name'], '✓' if r.stdout.strip() == 'ok' else '✗ ' + r.stderr.strip())
"

# 2. Run fleet health check
python3 skills/devops/fleet-ssh/scripts/fleet_status.py

# 3. Run a test command across all hosts
FLEET_HOSTS_FILE=~/.hermes/fleet/hosts.yaml python3 -c "
import yaml, subprocess
with open('$HOME/.hermes/fleet/hosts.yaml') as f:
    for h in yaml.safe_load(f)['hosts']:
        r = subprocess.run(['ssh', f'{h[\"user\"]}@{h[\"host\"]}', 'hostname'],
                           capture_output=True, text=True)
        print(h['name'], '->', r.stdout.strip())
"
```
