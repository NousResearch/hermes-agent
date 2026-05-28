# linux-nat Baseline Checklist

Updated: 2026-05-28
Scope: Read-only baseline checks before deploying Hermes Team AI Platform to `linux-nat`.

## Current Access Status

Local check found:

- `linux-nat` resolves to `linux-nat.tail40e9e7.ts.net`
- Tailscale IP is `100.70.103.59`
- SSH alias exists
- SSH failed in current session because no SSH identity is loaded

Observed local output:

```text
The agent has no identities.
rattanasak@linux-nat: Permission denied (publickey).
```

Before live baseline, load or provide the correct SSH key:

```bash
ssh-add -l
ssh-add ~/.ssh/<key-for-linux-nat>
ssh linux-nat@linux-nat.tail40e9e7.ts.net
```

If the host uses a different username, update the command accordingly.

## Safety Rules

These checks are read-only unless explicitly marked otherwise.

Do not run:

- `docker system prune`
- `docker compose down`
- `systemctl restart`
- firewall changes
- package upgrades
- file deletion
- permission changes

without separate approval.

## 1. Identity And Host

```bash
hostname
whoami
id
date
uptime
uname -a
cat /etc/os-release
```

Record:

- hostname
- username
- OS version
- kernel
- uptime
- timezone

## 2. CPU, RAM, Disk

```bash
lscpu | sed -n '1,30p'
free -h
swapon --show
df -hT
lsblk -f
```

Go/no-go thresholds:

| Metric | Go | Caution | No-Go |
|---|---:|---:|---:|
| root disk usage | < 75% | 75-85% | > 85% |
| available RAM | > 12 GiB | 6-12 GiB | < 6 GiB |
| swap active pressure | low/none | intermittent | sustained high |
| load average | stable | near CPU count | above CPU count sustained |

## 3. Network And NAT

```bash
ip -4 addr show scope global
ip route
ss -tulpen | sed -n '1,160p'
tailscale status
tailscale ip -4
curl -4 -s --max-time 5 ifconfig.me || true
```

Record:

- public egress IP
- Tailscale IP
- listening ports
- whether ports bind to `0.0.0.0`, `127.0.0.1`, or Tailscale/internal IP
- Tailscale backend state

Expected design:

- management over Tailscale
- browser dashboards through Cloudflare Access
- no new public admin ports

## 4. Cloudflare Tunnel

```bash
systemctl list-units '*cloudflared*' --no-pager
pgrep -a cloudflared || true
cloudflared tunnel list 2>/dev/null || true
cloudflared tunnel info 2>/dev/null || true
```

Record:

- whether cloudflared is installed
- service name
- tunnel id/name
- mapped hostnames
- active status

## 5. Docker Runtime

```bash
docker version
docker info | sed -n '1,80p'
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | sed -n '1,160p'
docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}' | sed -n '1,160p'
docker system df
```

Record:

- container count
- unhealthy containers
- top CPU containers
- top RAM containers
- Docker disk usage
- storage driver
- cgroup version

No-go conditions:

- many unhealthy critical containers before pilot
- disk pressure from images/volumes
- Docker daemon instability

## 6. Existing Production Services

```bash
docker ps --filter health=unhealthy --format 'table {{.Names}}\t{{.Status}}'
docker ps --filter status=exited --format 'table {{.Names}}\t{{.Status}}'
```

For known services, record:

- EmailHunter
- ScanlyIQ
- OpenClaw
- Jigsaw
- GodsEye
- Infisical
- Logto
- Portainer
- Duplicati
- monitoring services

## 7. Systemd Health

```bash
systemctl --failed --no-pager
systemctl list-timers --all --no-pager | sed -n '1,160p'
journalctl -p warning..alert --since '24 hours ago' --no-pager | tail -200
```

Record:

- failed units
- important timers
- repeated warnings/errors

## 8. Cron Jobs

```bash
crontab -l 2>/dev/null || true
sudo crontab -l 2>/dev/null || true
ls -la /etc/cron.* /etc/cron.d 2>/dev/null
```

Record:

- backup jobs
- watchdogs
- monitoring jobs
- project-specific jobs

Do not edit cron during baseline.

## 9. Monitoring Stack

Known from docs:

- Prometheus
- Grafana
- Loki
- Alertmanager
- Blackbox exporter
- cAdvisor
- Lark Alert Bridge

Checks:

```bash
docker ps --format '{{.Names}}' | grep -Ei 'prometheus|grafana|loki|alert|blackbox|cadvisor|exporter' || true
curl -fsS http://127.0.0.1:9092/-/healthy 2>/dev/null || true
curl -fsS http://127.0.0.1:9093/-/healthy 2>/dev/null || true
```

Record:

- active monitoring containers
- Prometheus health
- Alertmanager health
- alert delivery path

## 10. Backup Status

Known tools from docs:

- Duplicati
- DB backup scripts
- R2/offsite backup references

Checks:

```bash
find /home/linux-nat -maxdepth 3 -iname '*backup*' -type f 2>/dev/null | sed -n '1,120p'
find /srv -maxdepth 4 -iname '*backup*' -type f 2>/dev/null | sed -n '1,120p'
docker ps --format '{{.Names}}' | grep -Ei 'duplicati|backup' || true
```

Record:

- latest DB backup
- latest project config backup
- offsite backup path
- restore test status

No-go:

- no backup for existing production data before pilot changes

## 11. Security Baseline

```bash
sudo ufw status verbose 2>/dev/null || true
sudo nft list ruleset 2>/dev/null | sed -n '1,200p' || true
systemctl is-active fail2ban 2>/dev/null || true
systemctl is-active crowdsec 2>/dev/null || true
last -n 20
```

Record:

- firewall status
- Fail2ban status
- CrowdSec status
- suspicious login activity
- SSH password auth policy, if accessible

Do not change firewall during baseline.

## 12. Hermes Readiness Checks

After baseline only:

```bash
python3 --version
node --version 2>/dev/null || true
npm --version 2>/dev/null || true
git --version
docker compose version
```

Need for Hermes:

- Python runtime
- Git
- Node only if building dashboard/TUI/web assets
- Docker Compose for project runtime

## Baseline Report Template

```text
Date:
Host:
User:
Tailscale IP:
Public egress IP:
OS:
CPU:
RAM total:
RAM available:
Swap:
Disk root:
Docker containers:
Unhealthy containers:
Failed systemd units:
Cloudflare tunnel:
Backup status:
Monitoring status:
Risks:
Go/No-Go:
```

## Baseline Go/No-Go Decision

Go when:

- SSH over Tailscale works
- disk is below no-go threshold
- enough RAM is available
- Docker is healthy
- existing critical services are understood
- backups are identified
- monitoring is active

No-go when:

- cannot access server reliably
- root disk above 85%
- critical backup status unknown
- existing production services are unstable
- pilot would require opening public admin ports

