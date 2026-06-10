---
name: linux-security-hardening
description: "Linux Mint system administration: security hardening (fail2ban, auditd, Lynis, firewall, network audit), desktop configuration (keybindings, clipboard, emoji picker for Cinnamon + XFCE), and validation."
triggers:
  - security hardening
  - linux security
  - improve security
  - fail2ban
  - firewall security
  - system hardening
  - fortalecer segurança
  - segurança linux
  - lynis
  - linux mint
  - keybinding
  - keyboard shortcut
  - clipboard history
  - emoji picker
  - desktop configuration
  - cinnamon
  - xfce
---

# Linux Security Hardening

Comprehensive Linux security hardening workflow: baseline audit, immediate fixes, security tool installation, and validation with Lynis.

**Workflow preference**: Execute in batches ("manda bala"). User wants direct action with minimal ceremony. Document failures and successes.

## Phase 1: Baseline Audit

Run BEFORE making changes to establish current state.

**Quick way**: Use the bundled audit script:
```bash
bash ~/.hermes/skills/devops/linux-security-hardening/scripts/baseline-audit.sh
```

**Or run manually**:

```bash
# OS and firewall
echo "=== OS ===" && cat /etc/os-release | head -5 && echo -e "\n=== FIREWALL ===" && sudo ufw status 2>/dev/null || echo "ufw not active"

# SSH and open ports
echo -e "\n=== SSH ===" && systemctl is-active sshd 2>/dev/null || echo "ssh not running"
echo -e "\n=== OPEN PORTS ===" && sudo ss -tlnp 2>/dev/null | head -20

# Security tools
echo -e "\n=== FAIL2BAN ===" && systemctl is-active fail2ban 2>/dev/null || echo "fail2ban not installed/active"
echo -e "\n=== AUTO UPDATES ===" && systemctl is-active unattended-upgrades 2>/dev/null || echo "unattended-upgrades not active"

# Users and SUID
echo -e "\n=== USERS WITH SHELL ===" && grep -v '/nologin\|/false' /etc/passwd
echo -e "\n=== SUID FILES ===" && find /usr -perm -4000 -type f 2>/dev/null | head -20

# Recent logins
echo -e "\n=== LAST LOGINS ===" && last -5 2>/dev/null

# Kernel params
echo -e "\n=== KERNEL PARAMS ===" && sysctl net.ipv4.ip_forward 2>/dev/null && sysctl net.ipv4.conf.all.send_redirects 2>/dev/null && sysctl kernel.kptr_restrict 2>/dev/null

# Password quality
echo -e "\n=== PAM PW QUALITY ===" && cat /etc/security/pwquality.conf 2>/dev/null | grep -v '^#' | grep -v '^$' | head -10

# Installed security packages
echo -e "\n=== INSTALLED SECURITY PKGS ===" && dpkg -l | grep -iE 'fail2ban|libpam-pwquality|auditd|apparmor|clamav|rkhunter|lynis' 2>/dev/null

# Docker exposure check
echo -e "\n=== DOCKER EXPOSED ===" && sudo ss -tlnp | grep docker | grep -v '127.0.0'

# SSH key perms
echo -e "\n=== SSH KEY AUTH ===" && ls -la ~/.ssh/ 2>/dev/null
```

**Key indicators**:
- UFW status (default deny incoming?)
- Open ports in 0.0.0.0 (should be 127.0.0.1 if local-only)
- SSH enabled but not used?
- Fail2ban installed/active?
- Auto-updates enabled?
- Kernel params (ip_forward, send_redirects)
- SUID files (unusual binaries with SUID?)

## Phase 1b: Network-Level Audit

Run AFTER host baseline to map the network surface. Critical for home/office LANs.

**Reference**: See `references/home-network-audit.md` for the full checklist including WiFi security, router hardening, DNS, and device inventory methodology.

### Router/Gateway Audit

```bash
# Identify gateway
GATEWAY=$(ip route | grep default | awk '{print $3}')
echo "Gateway: $GATEWAY"

# Scan router ports (bash fallback if nmap not installed)
echo "=== ROUTER PORTS ==="
for port in 22 23 53 80 443 8080 8443; do
  timeout 2 bash -c "echo >/dev/tcp/$GATEWAY/$port" 2>/dev/null && echo "$GATEWAY:$port OPEN" || true
done

# Check if admin panel uses HTTPS or just HTTP
curl -sk "https://$GATEWAY" -o /dev/null -w "HTTPS: %{http_code}\n" 2>/dev/null
curl -sk "http://$GATEWAY" -o /dev/null -w "HTTP: %{http_code}\n" 2>/dev/null
```

**Critical findings to flag:**
- Port 23 (Telnet) OPEN → must disable in router admin panel
- Port 80 (HTTP admin) without 443 (HTTPS) → credentials in cleartext
- Port 53 (DNS) open to WAN → DNS amplification attack vector
- Port 445 (SMB) open → Windows file sharing exposed, common attack vector
- Port 548 (AFP) open → Apple file sharing, rarely needed on routers
- More than 5 open ports → router is likely running unnecessary services

**Router hardening checklist** (manual, via admin panel):
1. Change default admin password
2. Disable Telnet (port 23) — ALWAYS
3. Disable UPnP unless explicitly needed
4. Enable WPA3 or WPA2-AES (never WPA-TKIP or WEP)
5. Disable WPS (Wi-Fi Protected Setup) — vulnerable to brute force
6. Set guest network for visitors (isolates from main LAN)
7. Update firmware to latest
8. Change DNS to filtered provider (1.1.1.2/1.0.0.2 for malware blocking, or 9.9.9.9)
9. Disable remote management (admin from WAN)
10. Enable firewall/spi on router if available

**Device-specific guides**: See `references/` for ISP-router-specific hardening with exact menu paths:
- `intelbras-wifiber-ax1800v.md` — Intelbras WiFiber AX1800V (Vero/Ultrawave/AmericaNet). Covers WPS, UPnP, DNS, ACL, TR-069, firewall, and the Telnet port 23 ISP-firmware problem.

### WiFi Security Audit

```bash
# List nearby networks + security type
nmcli -f SSID,SIGNAL,FREQ,SECURITY dev wifi list 2>/dev/null

# Check current connection details
nmcli -f 802-11-wireless-security.key-mgmt,802-11-wireless-security.group,802-11-wireless-security.pairwise,802-11-wireless-security.proto \
  connection show "NETWORK_NAME" 2>/dev/null

# Check WiFi interface power management (saves battery but can cause issues)
iwconfig 2>/dev/null | grep -E 'ESSID|Power|Frequency'
```

**WiFi security matrix:**

| Config | Bom | Ruim | Critico |
|--------|-----|------|---------|
| Security | WPA3, WPA2-AES (CCMP) | WPA-TKIP | WEP, Open |
| Key mgmt | wpa-psk, sae | wpa-eap (ok for enterprise) | None |
| Channel | 5GHz preferred (less interference) | 2.4GHz only | — |
| WPS | Disabled | Enabled | — |
| Guest network | Configured + isolated | Not configured | — |

**Pitfall**: Routers often ship with WPS enabled by default. WPS has a design flaw that allows offline brute-force of the PIN in hours. ALWAYS disable.

**Pitfall**: `nmcli connection show "NETWORK"` fails silently when the machine is connected via wired Ethernet (not WiFi). The WiFi interface shows `ESSID:off/any` and `Not-Associated`. In this case, cipher/key-mgmt details are unavailable from the host — must check the router admin panel directly instead.

**Pitfall**: WPA2-TKIP is broken since 2008. If router shows `pairwise=TKIP` or `group=TKIP`, it's vulnerable. Must use CCMP/AES only.

### DNS Security Check

```bash
# Current DNS resolver
cat /etc/resolv.conf 2>/dev/null

# Test DNS leak
nslookup google.com
# If server is 127.0.0.53 → systemd-resolved (check upstream)
resolvectl status 2>/dev/null | grep "DNS Servers"
```

**Recommended DNS providers** (privacy + malware filtering):
- Cloudflare Family: 1.1.1.2 / 1.0.0.2 (blocks malware + adult content)
- Quad9: 9.9.9.9 / 149.112.112.112 (threat intelligence filtered)
- Cloudflare: 1.1.1.1 / 1.0.0.1 (fast, no filtering)

**Set DNS on router** (affects ALL devices on network):
1. Router admin → DHCP/DNS settings
2. Set primary: 1.1.1.2, secondary: 1.0.0.2
3. Save and reboot router

### LAN Device Discovery

```bash
# ARP table (shows recently-seen devices)
ip neigh show | grep -v '^172\.|^10\.|^fe80'

# Full subnet scan (if nmap available)
nmap -sn 192.168.1.0/24 2>/dev/null || echo "Install nmap for full scan"

# Alternative: ping sweep with bash
for i in $(seq 1 254); do
  ping -c1 -W1 "192.168.1.$i" &>/dev/null && echo "192.168.1.$i ALIVE" &
done; wait
```

**Device inventory**: Document every device found with:
- IP address
- MAC address (first 3 octets = vendor — use `macchanger -l` or online lookup)
- Expected role (phone, laptop, printer, IoT, etc.)
- Whether it should be on the network

**Pitfall**: Unknown devices on the LAN are either neighbors leeching WiFi or compromised IoT devices. Investigate immediately.

### Docker Exposure Analysis

```bash
# Containers binding to 0.0.0.0 (LAN-accessible)
echo "=== DOCKER EXPOSED TO LAN ==="
docker ps --format '{{.Names}}: {{.Ports}}' 2>/dev/null | grep '0.0.0.0'

# Containers properly bound to localhost
echo "=== DOCKER LOCALHOST ONLY ==="
docker ps --format '{{.Names}}: {{.Ports}}' 2>/dev/null | grep '127.0.0.1'
```

**Rule**: DBs (Postgres, MySQL, Redis) and internal tools MUST bind to 127.0.0.1. Only web servers meant for LAN testing may use 0.0.0.0 — and only behind UFW.

### Fail2ban Jail Check

```bash
# Service running ≠ doing something. Check actual jails:
sudo fail2ban-client status 2>/dev/null
# If output shows "No jails" or only empty list → fail2ban is decorative
```

**Pitfall**: Default fail2ban install has NO jails. You must create `/etc/fail2ban/jail.local` with at least `[sshd]` enabled. Without jails, the service runs but blocks nothing.

**Pitfall**: `docker ps --format '{{.Ports}}'` output can be ambiguous — it shows `0.0.0.0` even when the compose file says `127.0.0.1` if the container was started from a stale build. Use `docker port <container_name>` for the definitive answer. Also cross-check with the compose file itself.

### Service Exposure Matrix

Cross-reference `ss -tlnp` with expected bindings:

```bash
# Everything on 0.0.0.0 that shouldn't be
ss -tlnp | grep '0.0.0.0' | grep -v -E ':(53|631)\b'
```

Flag any port on 0.0.0.0 that isn't intentionally exposed (like a dev web server for mobile testing).

### Avahi/mDNS Check

```bash
# Avahi broadcasts hostname and services to entire LAN
systemctl is-active avahi-daemon 2>/dev/null
avahi-browse -alr 2>/dev/null | head -20
```

**Pitfall**: Avahi (mDNS/Bonjour) broadcasts your hostname and available services (SSH, CUPS, etc.) to everyone on the LAN. On a shared or public network, this leaks info. Disable with `sudo systemctl disable --now avahi-daemon` if not needed (printer discovery, AirPlay, etc.).

---

## Phase 2: Immediate Fixes (No Downtime)

Apply before installing tools. Low-risk corrections.

### Fix SSH File Permissions

```bash
chmod 600 ~/.ssh/id_ed25519 ~/.ssh/agent.env ~/.ssh/known_hosts
```

**Common issues**: agent.env, known_hosts often have 664/775. Should be 600.

### Clean Up UFW Rules

Remove phantom rules (e.g., SSH if sshd not running):

```bash
sudo ufw delete allow 22/tcp
```

**Check**: Compare `systemctl is-active sshd` with UFW rules. If service not running, rule is unnecessary.

### Disable IP Forwarding

Only needed if routing traffic (rare for dev machines):

```bash
sudo sysctl -w net.ipv4.ip_forward=0
sudo sed -i 's/net.ipv4.ip_forward = 1/net.ipv4.ip_forward = 0/' /etc/sysctl.conf
```

**Persist**: Changes must go into `/etc/sysctl.conf` to survive reboot.

### Disable ICMP Redirects

Prevent routing attacks:

```bash
sudo sysctl -w net.ipv4.conf.all.send_redirects=0
sudo sysctl -w net.ipv4.conf.default.send_redirects=0
echo "net.ipv4.conf.all.send_redirects=0" | sudo tee -a /etc/sysctl.conf
echo "net.ipv4.conf.default.send_redirects=0" | sudo tee -a /etc/sysctl.conf
```

### Kernel Hardening

**Recommended**: Use the bundled template for complete hardening:
```bash
sudo cp ~/.hermes/skills/devops/linux-security-hardening/templates/network-hardening.conf /etc/sysctl.d/60-network-hardening.conf
sudo sysctl --system
```

**Or add individual params manually**:
```bash
sudo bash -c 'cat >> /etc/sysctl.conf << "EOF"
net.ipv4.conf.all.accept_source_route=0
net.ipv6.conf.all.accept_source_route=0
net.ipv4.conf.all.accept_redirects=0
net.ipv6.conf.all.accept_redirects=0
net.ipv4.icmp_echo_ignore_broadcasts=1
EOF
'
sudo sysctl -p
```

### Unused Kernel Modules Blacklist

Reduce attack surface by blacklisting unused protocols:

```bash
sudo cp ~/.hermes/skills/devops/linux-security-hardening/templates/modprobe-blacklist.conf /etc/modprobe.d/security-blacklist.conf
```

This disables DCCP, SCTP, RDS, TIPC, and unused filesystems. USB storage and Bluetooth are commented out — uncomment if not needed.
```

## Phase 3: Security Tool Installation

Install and configure security tools. Use `sudo bash -c` with heredoc for multi-line configs (avoids sudo password prompts and quoting issues).

### Fail2ban (Brute Force Protection)

```bash
# Install
sudo apt install -y fail2ban

# Configure with heredoc (CRITICAL: use heredoc, not echo/tee chain)
sudo bash -c 'cat > /etc/fail2ban/jail.local << "EOF"
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
destemail = user@example.com
action = %(action_mwl)s
EOF
'

# Enable and start
sudo systemctl enable fail2ban
sudo systemctl restart fail2ban
sudo fail2ban-client status
```

**Pitfall**: Using echo/tee chain for Fail2ban config fails due to shell interpretation of `%(action_mwl)s`. MUST use heredoc with quoted delimiter (`<< "EOF"`).

**Common failure**: `'%' must be followed by '%' or '('` - indicates escaping issue. Fix: use heredoc.

### Unattended-Upgrades (Auto Security Patches)

Install ONLY security updates automatically, never reboot without permission:

```bash
# Install (usually pre-installed)
sudo apt install -y unattended-upgrades

# Configure security-only
sudo bash -c 'cat > /etc/apt/apt.conf.d/50unattended-upgrades-custom << "EOF"
Unattended-Upgrade::Allowed-Origins {"Ubuntu:${distro_codename}-security";};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::Remove-Unused-Kernel-Packages "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF
'

# Enable
sudo systemctl enable --now unattended-upgrades
```

**Check status**: `systemctl is-active unattended-upgrades`

### PAM Password Quality (libpam-pwquality)

Enforce strong passwords:

```bash
# Install
sudo apt install -y libpam-pwquality

# Configure
sudo bash -c 'cat > /etc/security/pwquality.conf << "EOF"
password requisite pam_pwquality.so retry=3 minlen=12 ucredit=-1 lcredit=-1 dcredit=-1 ocredit=-1 difok=3
EOF
'
```

**Requirements**:
- Minlen: 12 characters
- ucredit: -1 (at least 1 uppercase)
- lcredit: -1 (at least 1 lowercase)
- dcredit: -1 (at least 1 digit)
- ocredit: -1 (at least 1 special char)
- difok: 3 (at least 3 chars different from previous password)

### Auditd (System Auditing)

Log all system events for forensics:

```bash
# Install
sudo apt install -y auditd

# Enable
sudo systemctl enable auditd
sudo systemctl start auditd
sudo systemctl is-active auditd
```

**Note**: Default installation has empty ruleset. Define rules in `/etc/audit/rules.d/audit.rules`:

```bash
sudo bash -c 'cat > /etc/audit/rules.d/audit.rules << "EOF"
# Login e autenticacao
-w /var/log/auth.log -p wa -k auth_log
-w /etc/ssh/sshd_config -p wa -k ssh_config
-w /etc/passwd -p wa -k passwd_changes
-w /etc/shadow -p wa -k shadow_changes
-w /etc/group -p wa -k group_changes
-w /etc/sudoers -p wa -k sudoers_changes
-w /etc/sudoers.d/ -p wa -k sudoers_d_changes

# Execucao de comandos privilegiados
-a always,exit -F arch=b64 -S execve -F euid=0 -F auid>=1000 -F auid!=4294967295 -k root_cmds

# Modificacao de sistema
-w /bin/ -p wa -k bin_changes
-w /sbin/ -p wa -k sbin_changes
-w /usr/bin/ -p wa -k usr_bin_changes
-w /usr/sbin/ -p wa -k usr_sbin_changes

# Rede
-a exit,always -F arch=b64 -S bind -F auid>=1000 -k network_bind

# Docker
-w /etc/docker/ -p wa -k docker_config
-w /var/lib/docker/ -p wa -k docker_data

# Cron
-w /etc/crontab -p wa -k crontab
-w /etc/cron.d/ -p wa -k cron_d

# Modulos kernel
-w /etc/modules -p wa -k kernel_modules
-a always,exit -F arch=b64 -S init_module -S delete_module -k kernel_modules_syscall
EOF
'
sudo augenrules --load
sudo systemctl restart auditd
```

**Verify**: `sudo auditctl -s` should show `enabled 1`. `sudo auditctl -l | wc -l` counts rules.

## Phase 4: Validation with Lynis

Scan system to measure hardening progress.

```bash
# Install
sudo apt install -y lynis

# Run scan (NOTE: use --no-colors, not --no-color)
# Timeout: 180s minimum for full scan
sudo lynis audit system --no-colors 2>&1 | tail -100
```

**Pitfall**: `--quick --no-color` is invalid. Use `--no-colors` only.

**Pitfall**: Default 60s timeout is insufficient. Need 180s for full scan.

**Interpret results**:
- Hardening Index: 65+ = good baseline
- 80+ = excellent
- Look for "Ways to improve" section
- Check Components status (Firewall [V], Malware scanner [X])

## Common Security Issues to Address

### Docker Exposed Ports

Docker containers binding to `0.0.0.0` are accessible externally:

```bash
sudo ss -tlnp | grep docker | grep '0.0.0.0'
```

**Fix**: Edit `docker-compose.yml` to bind to `127.0.0.1`:

```yaml
services:
  app:
    ports:
      - "127.0.0.1:3000:3000"  # NOT "3000:3000"
```

**Restart**: `docker-compose down && docker-compose up -d`

**Tradeoff: LAN access for dev testing**: Some containers (e.g., web dev server) may need LAN access for mobile testing. In that case, keep that ONE port on `0.0.0.0` and ensure UFW allows only the LAN subnet (e.g., `192.168.1.0/24`). All other containers (DBs, SonarQube, etc.) stay on `127.0.0.1`.

### SSH Key Permissions

Check and fix:

```bash
ls -la ~/.ssh/
# Should show:
# -rw------- 1 user user 464 id_ed25519
# -rw-r--r-- 1 user user 101 id_ed25519.pub
# -rw------- 1 user user 831 known_hosts

chmod 600 ~/.ssh/id_ed25519 ~/.ssh/agent.env ~/.ssh/known_hosts
```

### Known Hosts Backup

Remove old backup with wrong perms:

```bash
rm -f ~/.ssh/known_hosts.old
```

### Password Expiration

Check root/user password expiration:

```bash
passwd -S root
passwd -S $USER
```

**Format**: `username P 2026-04-28 0 99999 7 -1`
- Third field: 0 = no min days
- Fourth field: 99999 = no max days (should be lower for production)

## Optional Hardening (Nice-to-Have)

### Legal Banners

Warn unauthorized users before login (include legal reference for Brazil):

```bash
sudo bash -c 'cat > /etc/issue << "EOF"

  ╔══════════════════════════════════════════════╗
  ║  ACESSO RESTRITO - SISTEMA PRIVADO           ║
  ║                                              ║
  ║  Este sistema e de uso exclusivo do          ║
  ║  proprietario. Acesso nao autorizado         ║
  ║  constitui crime (Art. 154-A, CP).           ║
  ║  Todas as atividades sao monitoradas.        ║
  ╚══════════════════════════════════════════════╝

EOF
'
sudo cp /etc/issue /etc/issue.net
```

### Malware Scanner

Install rkhunter (rootkit detection):

```bash
sudo apt install -y rkhunter
# Create baseline of system files (MUST run before first check)
sudo rkhunter --propupd
sudo rkhunter --check --skip-keypress
```

**Verify**: Check `/var/log/rkhunter.log` for results. Warnings about `lwp-request`, `shared memory segments`, and `hidden files` are normal on Docker/desktop systems.

### System Accounting

Install sysstat for resource usage metrics:

```bash
sudo apt install -y sysstat
# CRITICAL: enable data collection in config (default is disabled)
sudo sed -i 's/ENABLED="false"/ENABLED="true"/' /etc/default/sysstat
sudo systemctl enable --now sysstat
```

**Verify**: `sar -u 1 1` should show CPU stats. If "Cannot open /var/log/sysstat", data collection is still disabled.

### File Integrity Monitoring

Install AIDE (Advanced Intrusion Detection Environment):

```bash
# Install
sudo apt install -y aide

# Init baseline (TAKES 15+ MIN on 120GB disk — run with high timeout or before a break)
sudo aide --init --config /etc/aide/aide.conf
sudo cp /var/lib/aide/aide.db.new /var/lib/aide/aide.db

# Verify
sudo aide --check --config /etc/aide/aide.conf
```

**IMPORTANT**: AIDE init scans the entire filesystem and can take 15+ minutes on a 120GB NVMe. Plan accordingly — do NOT attempt with low timeouts or in background (sudo needs tty, background processes fail with "a terminal is required to read the password").

### Compiler Hardening

Restrict compiler access to root:

```bash
# gcc/g++ are symlinks (gcc -> gcc-13). chmod the REAL binary, not the symlink
sudo chmod 700 /usr/bin/x86_64-linux-gnu-gcc-13 2>/dev/null
sudo chmod 700 /usr/bin/x86_64-linux-gnu-g++-13 2>/dev/null
sudo chmod 700 /usr/bin/make 2>/dev/null
sudo chmod 700 /usr/bin/ld 2>/dev/null
sudo chmod 700 /usr/bin/as 2>/dev/null
```

**Verify**: `stat -c '%a %n' /usr/bin/x86_64-linux-gnu-gcc-13` should show `700`. The symlinks in `/usr/bin/gcc` stay 777 (that's fine, they point to the restricted binary).

## Final Validation

After all fixes:

```bash
# Kernel params
sysctl net.ipv4.ip_forward net.ipv4.conf.all.send_redirects 2>/dev/null

# UFW status
sudo ufw status | head -10

# Services
systemctl is-active fail2ban
systemctl is-active unattended-upgrades
systemctl is-active auditd

# SSH perms
ls -la ~/.ssh/ | grep -E 'id_ed25519|agent.env|known_hosts'

# PAM pwquality
cat /etc/security/pwquality.conf
```

## Troubleshooting

### Fail2ban Won't Start

Error: `'%' must be followed by '%' or '('`

**Cause**: Echo/tee chain misinterpreted `%(action_mwl)s`

**Fix**: Use heredoc with quoted delimiter:

```bash
sudo bash -c 'cat > /etc/fail2ban/jail.local << "EOF"
action = %(action_mwl)s
EOF
'
```

### Lynis Timeout Error

Scan dies after 60s.

**Fix**: Increase timeout to 180s+ or use no timeout for full scan:

```bash
sudo lynis audit system --no-colors 2>&1 | tail -100
```

### UFW Rules Not Applying

Service restart required:

```bash
sudo ufw reload
```

### Kernel Params Not Persisting

Changes lost after reboot.

**Fix**: Ensure params in `/etc/sysctl.conf`:

```bash
cat /etc/sysctl.conf | grep ip_forward
```

If missing, add and apply:

```bash
echo "net.ipv4.ip_forward = 0" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## Security Score Target

- **Good baseline**: Hardening Index 65+
- **Excellent**: Hardening Index 80+
- **Focus on**: Firewall, auto-updates, malware scanner, strong passwords

## Weekly Security Scan (Cron)

Automated weekly scan combining rkhunter + security updates + login audit + fail2ban status.

Script lives at `/opt/scripts/security-scan.sh` (user home, not /root).

```bash
# Add to root crontab (domingo 08:00)
sudo bash -c '(crontab -l 2>/dev/null; echo "0 8 * * 0 /opt/scripts/security-scan.sh") | crontab -'

# Run manually
sudo /opt/scripts/security-scan.sh
```

**Log**: `/var/log/security-scan.log` (auto-rotates at 1MB)

## Support Files

### Desktop Configuration Scripts
Ready-to-use scripts for desktop automation:
- `scripts/type-parenleft.sh` — xdotool workaround for broken `(` key
- `scripts/emoji-picker.sh` — rofi-based emoji picker with custom emoji list
- `scripts/clipboard-menu.sh` — CopyQ clipboard history trigger

### Desktop Configuration Reference
Complete desktop setup guide for Cinnamon + XFCE including keybindings, clipboard, emoji picker:
```bash
cat ~/.hermes/skills/devops/linux-security-hardening/references/desktop-configuration.md
```

### Baseline Audit Script
Automated script for Phase 1 baseline audit. Run before hardening:
```bash
bash ~/.hermes/skills/devops/linux-security-hardening/scripts/baseline-audit.sh
```

### Auditd Rules Template
Ready-to-use auditd rules for desktop/dev machines. Copy to system:
```bash
sudo cp ~/.hermes/skills/devops/linux-security-hardening/templates/audit-rules.conf /etc/audit/rules.d/audit.rules
sudo augenrules --load
sudo systemctl restart auditd
```

Covers: auth, root commands, system binaries, network binds, Docker, cron, kernel modules.

### Common Security Issues Reference
Detailed troubleshooting guide for specific security issues and fixes:
```bash
cat ~/.hermes/skills/devops/linux-security-hardening/references/security-issues.md
```

Covers: Docker exposure, SSH permissions, Fail2ban config issues, kernel params, and more.

### Home Network Audit
Complete checklist for auditing home/residential networks: router ports, WiFi security, DNS, device inventory, Docker exposure, Avahi/mDNS:
```bash
cat ~/.hermes/skills/devops/linux-security-hardening/references/home-network-audit.md
```

### Intelbras WiFiber AX1800V Hardening
Device-specific hardening guide for ISP-issued Intelbras WiFiber AX1800V (Vero/Ultrawave/AmericaNet):
```bash
cat ~/.hermes/skills/devops/linux-security-hardening/references/intelbras-wifiber-ax1800v.md
```
Covers: exact menu paths for WPS, UPnP, DNS, ACL, TR-069, firewall; SupAdmin credentials; Telnet port 23 workaround; inMesh notes.

### Kali Tools Catalog
Reference of Kali Linux tools for deeper security auditing, organized by category:
```bash
cat ~/.hermes/skills/devops/linux-security-hardening/references/kali-tools-catalog.md
```

Covers: reconnaissance, vulnerability scanning, Wi-Fi audit, password cracking, MITM, exploit frameworks, forensics. Includes which tools install directly on Mint vs need Kali.

### Pentest Tools — Local Setup (No sudo)
How to install Nuclei, SQLMap, testssl.sh, dirsearch, subfinder, ffuf in ~/tools/ without root:
```bash
cat ~/.hermes/skills/devops/linux-security-hardening/references/pentest-tools-setup.md
```

Covers: git clone installs, binary downloads, Docker-based tools (ZAP, Nikto, OpenVAS), symlinks, PATH setup, and pitfalls.

### Network Hardening Sysctl Template
Production-ready sysctl config for kernel + network hardening:
```bash
sudo cp ~/.hermes/skills/devops/linux-security-hardening/templates/network-hardening.conf /etc/sysctl.d/60-network-hardening.conf
sudo sysctl --system
```

Covers: ICMP hardening, redirect prevention, source routing, TCP hardening, kernel pointer restriction, Yama ptrace scope.

### Modprobe Blacklist Template
Blacklist unused kernel modules to reduce attack surface:
```bash
sudo cp ~/.hermes/skills/devops/linux-security-hardening/templates/modprobe-blacklist.conf /etc/modprobe.d/security-blacklist.conf
```

Covers: DCCP, SCTP, RDS, TIPC, unused filesystems, USB storage (optional), Bluetooth (optional).

## Desktop Configuration (Cinnamon + XFCE)

Configure desktop features: custom keybindings, clipboard history, emoji picker, and broken key workarounds. **Critical:** Cinnamon uses `dconf`, XFCE uses `xfconf-query` — check with `echo $XDG_CURRENT_DESKTOP` before registering shortcuts.

**Keybindings:** Register via dconf (Cinnamon) or xfconf-query (XFCE). Both support Super, Alt, Ctrl, Shift modifiers. Cinnamon requires updating `custom-list` after adding shortcuts.

**Clipboard:** CopyQ (`sudo apt install copyq xclip`) — tray icon, search, persistent history. Map to Super+V.

**Emoji Picker:** rofi + custom emoji list + xdotool. Map to Super+.. See `scripts/emoji-picker.sh`.

**Broken Keys:** xdotool workaround (`xdotool key --clearmodifiers parenleft`) or GTK hex input (Ctrl+Shift+U → hex code → Enter).

**Scripts:** `scripts/type-parenleft.sh`, `scripts/emoji-picker.sh`, `scripts/clipboard-menu.sh`

See `references/desktop-configuration.md` for full details including dconf paths, xfconf-query syntax, binding syntax, and diagnostic commands.

## Remote Access — RustDesk Server (Self-Hosted)

Deploy a self-hosted remote desktop solution as an alternative to TeamViewer/AnyDesk. RustDesk Server OSS consists of two services: `hbbs` (rendezvous/registration) and `hbbr` (relay).

### Quick Setup (Docker Compose)

```yaml
# docker-compose.yml
services:
  hbbs:
    container_name: rustdesk-hbbs
    image: rustdesk/rustdesk-server:latest
    command: hbbs
    volumes:
      - ./rustdesk-data:/root
    network_mode: host
    restart: unless-stopped

  hbbr:
    container_name: rustdesk-hbbr
    image: rustdesk/rustdesk-server:latest
    command: hbbr
    volumes:
      - ./rustdesk-data:/root
    network_mode: host
    restart: unless-stopped
```

**Key:** `network_mode: host` is REQUIRED — NAT traversal fails with port mapping. Both services bind to `0.0.0.0` by design (they need LAN/WAN access for relay).

### Retrieve Public Key

```bash
cat ./rustdesk-data/id_ed25519.pub
```

The public key is needed by all client devices to connect.

### Firewall (UFW)

```bash
# hbbs (rendezvous)
sudo ufw allow 21115/tcp  # NAT type test
sudo ufw allow 21116/tcp  # TCP hole punch + web client
sudo ufw allow 21116/udp  # ID registration + heartbeat

# hbbr (relay)
sudo ufw allow 21117/tcp  # relay traffic
sudo ufw allow 21118/tcp  # web client (optional)
sudo ufw allow 21119/tcp  # web client (optional)
sudo ufw reload
```

### Client Configuration

In RustDesk client: Settings → Network → ID/Relay Server → enter server IP. Paste the public key.

### Security Considerations

- **Bind to LAN only** if no remote access needed from outside: not possible with standard image (hardcoded `0.0.0.0`). Use UFW to restrict access instead.
- **Key-based auth**: Always set the public key on clients — prevents unauthorized relay usage.
- **Keep keys safe**: `id_ed25519` (private) and `id_ed25519.pub` (public) in `./rustdesk-data/`. Backup the private key.
- **Docker restart policy**: `unless-stopped` ensures auto-start on boot.

### Troubleshooting

- **"Not ready" in client**: Check that both containers are running (`docker ps`), UFW rules are applied, and public key matches.
- **Can connect on LAN but not remotely**: Check router port forwarding for 21115-21119, verify external IP with `curl ifconfig.me`.
- **Container exits immediately**: Check `docker logs rustdesk-hbbs` — usually a port conflict or missing permissions on data volume.

### See Also

- `references/intelbras-wifiber-ax1800v.md` for router port forwarding on common ISP routers
- RustDesk docs: https://rustdesk.com/docs/en/self-host/

## Pitfalls

- **Fail2ban config escaping**: MUST use heredoc (`<< "EOF"`) for config with `%(action_mwl)s`. Echo/tee chain fails.
- **Lynis scan options**: `--quick --no-color` is invalid. Use `--no-colors` only.
- **Lynis timeout**: Default 60s insufficient. Need 180s for full scan.
- **Kernel persistence**: Changes via `sysctl -w` don't persist. Must edit `/etc/sysctl.conf`.
- **Docker re-enables ip_forward**: Docker sets `net.ipv4.ip_forward=1` automatically when containers start. This is NORMAL — Docker needs it for container networking. UFW blocks external routing, so it's safe.
- **sysstat disabled by default**: Must edit `/etc/default/sysstat` to set `ENABLED="true"`. Without this, sysstat runs but collects nothing.
- **Compilers are symlinks**: `gcc` -> `gcc-13`. chmod the target binary, not the symlink.
- **apt lock contention**: If `apt install` times out, a stale apt process may hold the lock. Fix: `sudo kill -9 $(sudo lsof -t /var/lib/dpkg/lock-frontend)` then `sudo dpkg --configure -a`.
- **Rkhunter baseline**: Run `sudo rkhunter --propupd` after install to create file property baseline. Without it, first scan reports false positives.
- **AIDE init timeout**: `aide --init` scans the entire filesystem and takes 15+ minutes on a 120GB disk. Do NOT run with low timeouts. Also cannot run in background because `sudo` requires a tty — background processes fail with "a terminal is required to read the password". Run foreground with timeout 600s+.
- **Docker binding**: `0.0.0.0` exposes externally. Use `127.0.0.1` for local-only services.
- **SSH file perms**: agent.env, known_hosts often have wrong perms (664/775). Fix to 600.
- **UFW phantom rules**: Remove rules for disabled services (e.g., SSH if sshd not running).
- **Unattended-upgrades reboot**: Set `Automatic-Reboot "false"` to avoid unexpected reboots.
- **Fail2ban empty jails**: Default install has NO jails. Service runs but blocks nothing. Always verify with `fail2ban-client status` after install.
- **Postfix banner disclosure**: Default smtpd_banner leaks OS/software name. Run `postconf -e 'smtpd_banner = $myhostname ESMTP'` and `postconf -e 'disable_vrfy_command = yes'`.
- **Router Telnet open**: Many ISP routers ship with port 23 (Telnet) enabled. Telnet sends credentials in cleartext. Disable via router admin panel (usually http://192.168.1.1).
- **accept_redirects default**: Ubuntu/Mint ships with `net.ipv4.conf.all.accept_redirects=1`, enabling ICMP redirect attacks. Must set to 0. **Pitfall**: even after copying the hardening template to `/etc/sysctl.d/`, `sysctl --system` may not apply it immediately. Verify with `sysctl net.ipv4.conf.all.accept_redirects` and force with `sudo sysctl -w net.ipv4.conf.all.accept_redirects=0 net.ipv4.conf.default.accept_redirects=0` if needed.
- **Unused kernel protocols**: DCCP, SCTP, RDS, TIPC load by default but are rarely needed. Blacklist via modprobe to reduce attack surface.
- **Bash port-scan fallback**: When nmap isn't installed, use `timeout 2 bash -c "echo >/dev/tcp/HOST/PORT"` for quick port checks. Not as thorough but requires zero dependencies.
- **Docker 0.0.0.0 by default**: `docker-compose.yml` ports like `"3000:3000"` bind to 0.0.0.0. Always use `"127.0.0.1:3000:3000"` for local services.
- **Router port scan false positives**: When scanning router IP with Python socket, ports like 8888/9999/5432 may be HOST Docker containers binding to 0.0.0.0, not router services. Always cross-reference with `ss -tlnp` and `docker ps` on the host.
- **Docker port verification**: `docker ps --format '{{.Ports}}'` can show misleading `0.0.0.0` even when compose file says `127.0.0.1` (stale container). Use `docker port <container>` for definitive bind address. Cross-check with the compose file.
- **WiFi audit on wired machines**: `nmcli connection show "NETWORK"` returns nothing when machine connects via wired Ethernet. WiFi interface shows `ESSID:off/any`. Must check cipher/WPA details from router admin panel instead.
- **Router with 10+ open ports**: Usually means router is running unnecessary services (FTP, Telnet, SMB, AFP, rpcbind). Disable everything except DNS (53), HTTP/HTTPS admin (80/443), and optionally SSH (22) restricted to LAN.
- **WiFi WPA2-TKIP**: Broken since 2008. If `nmcli connection show` reveals `pairwise=TKIP`, the network is vulnerable. Must use CCMP/AES only.
- **WPS enabled by default**: Most routers ship with WPS on. The PIN-based WPS protocol has a design flaw allowing offline brute-force in hours. ALWAYS disable in router admin panel.
- **Avahi/mDNS info leak**: avahi-daemon broadcasts hostname and running services to entire LAN. Disable on shared/public networks: `sudo systemctl disable --now avahi-daemon`.
- **ISP router SupAdmin credentials**: Many ISP-issued routers (Intelbras WiFiber, Zyxel, etc.) have a hidden super-admin account (e.g., SupAdmin/AdminSup3031). These credentials are found in forums, not manuals. If user-level admin lacks options (e.g., can't disable Telnet), try SupAdmin login.
- **ISP router Telnet not in UI**: Some ISP routers run Telnet (port 23) via custom firmware with no UI toggle. Workaround: use router ACL to limit access, firewall Filtro IP/Porta to block port 23, or accept LAN-only risk with ACL configured.
- **Router TR-069 tradeoff**: TR-069 lets ISP remotely manage the router (firmware, config). Disabling it improves privacy/security but breaks remote support. User decides based on whether they self-manage.
- **Intelbras WPS UI is confusing**: In WiFiber AX1800V, "Desativar WPS" checkbox must be ENABLED (checked) to disable WPS. The label reads "se habilitado, sera desativada a funcao WPS" — counter-intuitive.
- **Lynis without sudo**: Lynis runs without root but skips many tests (RLS checks, boot loader, password hashing, firewall rules, auditd rules, encryption). Useful for quick overview but incomplete — always run with sudo for full audit.
- **accept_redirects not applying via sysctl --system**: Even with correct values in `/etc/sysctl.d/60-network-hardening.conf`, `sysctl --system` may not apply `net.ipv4.conf.all.accept_redirects=0`. Always verify with `sysctl net.ipv4.conf.all.accept_redirects` after applying. If still 1, force with `sudo sysctl -w net.ipv4.conf.all.accept_redirects=0 net.ipv4.conf.default.accept_redirects=0 net.ipv6.conf.all.accept_redirects=0 net.ipv6.conf.default.accept_redirects=0`.
- **ZRAM lost after swapoff -a**: `swapoff -a` disables zram too. `swapon -a` only reactivates fstab entries (swapfile), NOT zram managed by systemd. Always run `sudo systemctl restart zram.service` after swap recyle. Verify with `zramctl && swapon --show`.
- **Postfix VRFY command**: Default allows email user enumeration. Run `postconf -e 'disable_vrfy_command = yes'` to disable.
