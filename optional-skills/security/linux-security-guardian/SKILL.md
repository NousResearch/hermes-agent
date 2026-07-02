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
1. Detect distro/firewall tool (Debian/Ubuntu, RHEL/CentOS, Arch, etc.)
2. Check SSH configuration (port, auth methods, root login, keys)
3. Audit network/firewall state (ufw/iptables/firewalld, open ports)
4. Scan system hardening (updates, file permissions)
5. Review user privileges (empty passwords, sudo users, failed logins)
6. Verify operational security (fail2ban, auto-updates, logging)
7. Calculate a composite score
8. Output a prioritized fix list

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
| Password auth disabled | `PasswordAuthentication no` | Password login allowed | `sudo sed -i 's/^PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && sudo systemctl reload sshd` |
| Root login disabled | `PermitRootLogin no` | Root can SSH directly | `sudo sed -i 's/^PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config && sudo systemctl reload sshd` |
| Key auth configured | `~/.ssh/authorized_keys` exists | No keys | `ssh-keygen -t ed25519` then copy pubkey |
| Protocol version | Protocol 2 only | v1 enabled | Default modern SSH |

**Scoring rubric**:
- Port ≠ 22: +5
- Password auth disabled: +5
- Root login disabled: +5
- Key auth configured: +5
- Protocol 2 only: +5

### ② Network & Firewall (25%)

| Check | Pass | Fail | Fix |
|-------|------|------|-----|
| Firewall active | ufw/iptables/firewalld active | No firewall | `sudo ufw enable` or equivalent |
| Minimal open ports | Only necessary ports open | Unexpected listeners | `sudo ufw deny <port>` or equivalent |
| Services bound locally | DB/cache on 127.0.0.1 | Services on 0.0.0.0 | Config bind address |

**Distro-aware commands**:
- Debian/Ubuntu: `ufw status`, `ufw enable`, `ufw deny`
- RHEL/CentOS: `firewall-cmd --state`, `firewall-cmd --add-port`, `systemctl enable firewalld`
- Arch: `iptables -L`, `iptables -A`

**Scoring rubric**:
- Firewall active: +8
- Minimal open ports: +9
- Services bound locally: +8

### ③ System Hardening (20%)

| Check | Pass | Fail | Fix |
|-------|------|------|-----|
| System updates | `apt list --upgradable` empty or equivalent | Pending updates | `sudo apt update && sudo apt upgrade -y` or equivalent |
| Shadow permissions | `/etc/shadow` 600/640 | Too permissive | `sudo chmod 640 /etc/shadow` |
| /tmp sticky bit | Sticky bit set | Missing | `sudo chmod +t /tmp` |
| npm global packages | No known malicious packages | Blocklisted package found | `npm uninstall -g <pkg>` |

**Distro-aware package manager**:
- Debian/Ubuntu: `apt`
- RHEL/CentOS/Fedora: `dnf` or `yum`
- Arch: `pacman`

**Scoring rubric**:
- System updates: +5
- Shadow permissions: +5
- /tmp sticky bit: +5
- npm packages clean: +5

### ④ User Privileges (20%)

| Check | Pass | Fail | Fix |
|-------|------|------|-----|
| No empty passwords | All users have passwords | Empty password found | `sudo passwd -l <user>` |
| Sudo group minimal | Expected users only | Unexpected sudoer | `sudo gpasswd -d <user> sudo` |
| Failed logins | No recent spikes | Many recent failures | Check `/var/log/auth.log` or equivalent |

**Scoring rubric**:
- No empty passwords: +7
- Sudo group minimal: +7
- Failed logins normal: +6

### ⑤ Operational Security (10%)

| Check | Pass | Fail | Fix |
|-------|------|------|-----|
| fail2ban running | `systemctl is-active fail2ban` = active | Not running | `sudo systemctl enable --now fail2ban` |
| Auto-updates | `unattended-upgrades` configured | Not configured | `sudo apt install unattended-upgrades` or equivalent |
| Logging active | rsyslog/journald running | Logging stopped | `sudo systemctl start rsyslog` |

**Scoring rubric**:
- fail2ban running: +3
- Auto-updates: +4
- Logging active: +3

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
  ❌ Password login still allowed (-5)
  ✅ SSH keys configured

② Network & Firewall: 80/100  ⚠️
  ✅ ufw active
  ⚠️ Port 3000 exposed (-8)

③ System Hardening:  60/100  🔶
  ❌ 12 pending security updates (-5)
  ✅ /etc/shadow permissions OK
  ✅ npm packages clean

④ User Privileges:   80/100  ⚠️
  ✅ No empty-password users
  ✅ Sudo group looks correct

⑤ Ops Security:      60/100  🔶
  ✅ fail2ban running
  ❌ Auto-updates not configured (-4)

📋 Fix Priority:
  [HIGH]   Disable SSH password login
  [HIGH]   Install security updates
  [MEDIUM] Configure unattended-upgrades
```

## Deterministic Audit Procedure

To ensure consistent reports across runs, follow this exact sequence:

1. **Distro detection** (once per run):
   ```bash
   if [ -f /etc/debian_version ]; then
     PKG="apt"
     FIREWALL="ufw"
   elif [ -f /etc/redhat-release ]; then
     PKG="dnf"
     FIREWALL="firewall-cmd"
   elif [ -f /etc/arch-release ]; then
     PKG="pacman"
     FIREWALL="iptables"
   else
     PKG="unknown"
     FIREWALL="unknown"
   fi
   ```

2. **SSH audit** (read-only):
   ```bash
   sshd -T | grep -E "^(passwordauthentication|permitrootlogin|port |protocol)"
   ```

3. **Network audit**:
   ```bash
   $FIREWALL status 2>/dev/null || iptables -L 2>/dev/null || echo "no firewall"
   ss -tulpn
   ```

4. **System audit**:
   ```bash
   $PKG list --upgradable 2>/dev/null || echo "package manager unavailable"
   stat -c "%a %n" /etc/shadow
   ls -ld /tmp
   ```

5. **User audit**:
   ```bash
   cat /etc/shadow | awk -F: '($2 == "") {print $1}'
   getent group sudo | awk -F: '{print $4}' | tr ',' '\n'
   lastb | head -20
   ```

6. **Ops audit**:
   ```bash
   systemctl is-active fail2ban
   systemctl is-active unattended-upgrades 2>/dev/null || echo "not installed"
   systemctl is-active rsyslog || systemctl is-active systemd-journald
   ```

7. **Score calculation**:
   - Sum per-check rubric points
   - Apply weights: SSH×0.25, Network×0.25, System×0.20, User×0.20, Ops×0.10
   - Round to nearest integer

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
- **Distro differences**: This skill assumes systemd-based distros. For OpenRC, Alpine, etc.,
  replace `systemctl` with equivalent service managers.
