---
name: cifs-mount-watchdog
description: Diagnose and fix CIFS/SMB mounts that fail on boot due to Tailscale/network race conditions, and install a systemd watchdog timer to auto-recover. Use when a service (e.g. Ollama) crashes with permission denied on a network-mounted path.
tags: [cifs, smb, mount, systemd, tailscale, ollama, watchdog, race-condition]
---

# CIFS Mount Watchdog

## When to use
- A service crashes with `permission denied` or `no such file` on a path under `/mnt/`
- `mount | grep cifs` shows the share is not mounted despite being in `/etc/fstab`
- fstab uses `_netdev` + `x-systemd.requires=tailscaled.service` (Tailscale-dependent mounts)
- You want automatic recovery without manual intervention after reboot or Tailscale reconnect

---

## Step 1 — Diagnose

```bash
journalctl -u <service>.service -n 50 --no-pager
```

Look for: `mkdir /mnt/xxx: permission denied` or `no such file or directory`

Then check mount status:
```bash
mount | grep cifs           # is the share actually mounted?
findmnt -t cifs             # cleaner view
ls -la /mnt/                # does the mountpoint directory exist?
ping <truenas-ip>           # is the NAS reachable?
```

If mount is missing but NAS is reachable → race condition on boot.

---

## Step 2 — Immediate fix (manual remount)

```bash
sudo mount /mnt/ssd_llm     # uses fstab entry
sudo systemctl restart ollama
systemctl is-active ollama
```

---

## Step 3 — Install watchdog script

Create `/usr/local/bin/mount-watchdog.sh`:

```bash
sudo tee /usr/local/bin/mount-watchdog.sh > /dev/null << 'EOF'
#!/bin/bash
LOGFILE="/var/log/mount-watchdog.log"
MOUNTS=("/mnt/ssd_llm" "/mnt/ssd_vm")
REMOUNTED=0

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOGFILE"
}

for MOUNTPOINT in "${MOUNTS[@]}"; do
    if ! findmnt -t cifs "$MOUNTPOINT" > /dev/null 2>&1; then
        log "WARN: $MOUNTPOINT niet gemount, probeer te mounten..."
        if mount "$MOUNTPOINT" >> "$LOGFILE" 2>&1; then
            log "OK: $MOUNTPOINT succesvol gemount"
            REMOUNTED=1
        else
            log "ERROR: mounten van $MOUNTPOINT mislukt"
        fi
    fi
done

if [ "$REMOUNTED" -eq 1 ]; then
    if findmnt -t cifs /mnt/ssd_llm > /dev/null 2>&1; then
        log "INFO: ssd_llm hersteld, Ollama herstarten..."
        systemctl restart ollama
        log "INFO: Ollama herstart"
    fi
fi
EOF
sudo chmod +x /usr/local/bin/mount-watchdog.sh
```

---

## Step 4 — systemd service unit

```bash
sudo tee /etc/systemd/system/mount-watchdog.service > /dev/null << 'EOF'
[Unit]
Description=CIFS Mount Watchdog (ssd_llm / ssd_vm)
After=tailscaled.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/mount-watchdog.sh
StandardOutput=journal
StandardError=journal
EOF
```

---

## Step 5 — systemd timer unit (every 1 minute)

```bash
sudo tee /etc/systemd/system/mount-watchdog.timer > /dev/null << 'EOF'
[Unit]
Description=Run CIFS Mount Watchdog every minute
Requires=tailscaled.service

[Timer]
OnBootSec=2min
OnUnitActiveSec=1min
Unit=mount-watchdog.service

[Install]
WantedBy=timers.target
EOF
```

---

## Step 6 — Enable and verify

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mount-watchdog.timer
systemctl list-timers mount-watchdog.timer --no-pager
# Wait ~1 minute, then:
journalctl -u mount-watchdog -n 10 --no-pager
```

Expected: `Finished mount-watchdog.service` every minute. If a mount was restored, log shows `OK: ... succesvol gemount` and `Ollama herstart`.

---

## Alternatief scenario: OLLAMA_MODELS pad corrupt (mount aanwezig, modellen kapot)

Soms is de mount wél aanwezig maar bevat de modellen-map corrupte of ontbrekende manifests. Symptoom: `curl http://127.0.0.1:11434/api/tags` geeft `{"models":[]}` of modellen worden niet gevonden (404). De echte modellen staan dan lokaal op `/usr/share/ollama/.ollama`.

**Diagnose:**
```bash
curl -s http://127.0.0.1:11434/api/tags           # {"models":[]} = probleem
ls /usr/share/ollama/.ollama/models/manifests/     # echte modellen hier?
cat /etc/systemd/system/ollama.service.d/override.conf  # welk pad?
ls /mnt/ssd_llm/ollama/                            # kapote inhoud?
```

**Fix: override naar lokaal pad redirecten:**
```bash
# sudo -n tee werkt als sudoers tee/systemctl toestaat zonder wachtwoord
sudo -n tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null << 'EOF'
[Service]
Environment="OLLAMA_MODELS=/usr/share/ollama/.ollama/models"
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF

sudo -n systemctl daemon-reload
sudo -n systemctl restart ollama
sleep 4
curl -s http://127.0.0.1:11434/api/tags   # verwacht: modellen zichtbaar
```

**Kapotte SSD-map veilig hernoemen (als file owner, geen sudo nodig):**
```bash
mv /mnt/ssd_llm/ollama /mnt/ssd_llm/ollama.broken-backup
```

**Let op:** `sudo -n` (non-interactive) werkt alleen voor specifieke commands die via sudoers NOPASSWD zijn toegestaan (bijv. `tee`, `systemctl`). `sudo -n mv` of `sudo -n ls` op willekeurige paden werkt mogelijk niet — check dan eerst of de user eigenaar is van de map.

---

## Pitfalls

- **sudo via SSH heredoc**: gebruik `sudo -n tee /path > /dev/null << 'EOF'` — niet `sudo bash -c "cat > /path" << 'EOF'` (dat werkt niet betrouwbaar over SSH)
- **sudo -n vs interactief**: `sudo -n` faalt met exit code 1 als het wachtwoord vereist is. Test eerst met `sudo -n systemctl status ollama` — als dat werkt, werken systemctl/tee ook.
- **Ollama models path**: in `/etc/systemd/system/ollama.service.d/override.conf` als `OLLAMA_MODELS=`. Kan naar SSD wijzen terwijl echte modellen lokaal staan.
- **nofail in fstab**: zorgt dat boot niet blokkeert als mount faalt — goed voor uptime, maar betekent dat de service soms start zonder mount. De watchdog compenseert dit.
- **ssd_vm zonder CIFS**: ssd_vm kan lokale content hebben (bijv. `pluto/` map) zonder actieve CIFS mount — dit is normaal als die share niet nodig is voor actieve services.
- **Rechten watchdog script**: de systemd service draait als root (geen `User=` opgegeven), dus `mount` en `systemctl restart` werken zonder sudo.
