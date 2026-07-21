---
name: linux-security-guardian
description: |-
  Linux server security health check — 5-dimension assessment covering SSH hardening,
  firewall & network, system hardening, user privileges, and operational security.
  Outputs a scored report with prioritized fix commands. Like a health checkup for your server.
platforms: [linux]
category: security
triggers:
  - "security check"
  - "security health check"
  - "server security audit"
  - "hardening check"
  - "Linux security scan"
  - "检查服务器安全"
  - "安全体检"
  - "服务器安全加固"
toolsets:
  - terminal
  - file
---

# 🛡️ Linux Security Guardian

> **One-liner**: A lightweight security health check for Linux servers — like running a system
> diagnostic, but for security. Covers SSH, firewall, system hardening, user privileges, and
> operational security. Outputs a scored report with prioritized fix commands.

**Target users**: Developers, small-team ops, self-hosters who manage Linux servers
**Time**: First run ~5 min (to understand results), subsequent runs ~30 sec
**Output**: Scored report (0-100) + prioritized fix list

---

## When to Use

| Scenario | When | Frequency |
|----------|------|-----------|
| 🆕 New server | Just provisioned a VPS or bare metal | Once |
| 📅 Routine | Weekly/monthly maintenance | Weekly |
| 🔔 Incident | After suspicious log entries or alerts | On-demand |
| 🏗️ Pre-deploy | Before launching a new service | Per deploy |
| 🔄 Post-change | After kernel upgrade or major software install | Per change |

## Quick Start

Just say:

> "Run a security health check on this server"

The agent will automatically:
1. Check SSH configuration (port, auth methods, root login, keys)
2. Audit network/firewall state (ufw status, open ports)
3. Scan system hardening (updates, file permissions, npm)
4. Review user privileges (empty passwords, sudo users, failed logins)
5. Verify operational security (fail2ban, auto-updates, logging)
6. Calculate a composite score
7. Output a prioritized fix list

## Security Scoring

| Score | Grade | Label | Meaning |
|-------|-------|-------|---------|
| 90-100 | A | 🛡️ Excellent | Baseline security in place |
| 70-89 | B | ⚠️ Good | Minor optimizations available |
| 50-69 | C | 🔶 Fair | Notable risks — recommend fixes |
| 0-49 | D | 🔴 Poor | Critical gaps — fix immediately |

**Weights**: SSH 25% / Network 25% / System 20% / User 20% / Ops 10%

## Check Details

### ① SSH Security (25%)

| Check | Pass | Fail | Fix |
|-------|------|------|-----|
| Non-default port | Port ≠ 22 | Using port 22 | Edit `/etc/ssh/sshd_config` |
| Password auth disabled | `PasswordAuthentication no` | Password login allowed | `sed -i 's/yes$/no/'` on config |
| Root login disabled | `PermitRootLogin no` | Root can SSH directly | Same file, change to `no` |
| Key auth configured | `~/.ssh/authorized_keys` exists | No keys | `ssh-keygen` then copy pubkey |
| Protocol version | Protocol 2 only | v1 enabled | Default modern SSH |

### ② Network & Firewall (25%)

| Check | Pass | Fail | Fix |
|-------|------|------|-----|
| Firewall active | `ufw active` or iptables rules | No firewall | `ufw enable` |
| Minimal open ports | Only necessary ports open | Unexpected listeners | `ufw deny <port>` |
| Services bound locally | DB/cache on 127.0.0.1 | Services on 0.0.0.0 | Config bind address |

### ③ System Hardening (20%)

| Check | Pass | Fail | Fix |
|-------|------|------|-----|
| System updates | `apt list --upgradable` empty | Pending updates | `apt update && apt upgrade` |
| Shadow permissions | `/etc/shadow` 600/640 | Too permissive | `chmod 640 /etc/shadow` |
| /tmp sticky bit | Sticky bit set | Missing | `chmod +t /tmp` |
| npm global packages | No known malicious packages | Blocklisted package found | `npm uninstall -g <pkg>` |

### ④ User Privileges (20%)

| Check | Pass | Fail | Fix |
|-------|------|------|-----|
| No empty passwords | All users have passwords | Empty password found | `passwd -l <user>` |
| Sudo group minimal | Expected users only | Unexpected sudoer | `gpasswd -d <user> sudo` |
| Failed logins | No recent spikes | Many recent failures | Check `/var/log/auth.log` |

### ⑤ Operational Security (10%)

| Check | Pass | Fail | Fix |
|-------|------|------|-----|
| fail2ban running | `systemctl is-active fail2ban` = active | Not running | `systemctl enable --now fail2ban` |
| Auto-updates | `unattended-upgrades` configured | Not configured | `apt install unattended-upgrades` |
| Logging active | rsyslog/journald running | Logging stopped | `systemctl start rsyslog` |

## Example Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🛡️ Linux Security Guardian
  Server: 203.0.113.42
  Time: 2026-05-18 22:30 UTC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Score: 72/100 (B — Good)

① SSH Security:      75/100  ⚠️
  ✅ Port changed (non-22)
  ✅ Root login disabled
  ❌ Password login still allowed
  ✅ SSH keys configured

② Network & Firewall: 80/100  ⚠️
  ✅ ufw active
  ⚠️ Port 3000 exposed (close if unused)

③ System Hardening:  60/100  🔶
  ❌ 12 pending security updates
  ✅ /etc/shadow permissions OK
  ✅ npm packages clean

④ User Privileges:   80/100  ⚠️
  ✅ No empty-password users
  ✅ Sudo group looks correct

⑤ Ops Security:      60/100  🔶
  ✅ fail2ban running
  ❌ Auto-updates not configured

📋 Fix Priority:
  [HIGH]   Disable SSH password login
  [HIGH]   Install security updates
  [MEDIUM] Configure unattended-upgrades
```

## Verification

After applying fixes, re-run the check to confirm the score improved.
For SSH changes, test login in a separate session before closing the active one.

## Pitfalls

- **SSH lockout risk**: When disabling password auth, ensure key auth works FIRST.
  Keep the active SSH session open while testing a new one.
- **ufw reset warning**: If a cloud provider's metadata service uses a specific port,
  `ufw enable` may block it. Whitelist cloud metadata IPs (e.g. 169.254.169.254).
- **fail2ban false bans**: If you SSH from variable IPs, set `ignoreip` in jail.local.
- **npm global packages**: Some packages legitimately use postinstall scripts.
  Always review the script content before flagging as malicious.
