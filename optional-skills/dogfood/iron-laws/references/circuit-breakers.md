# Circuit Breakers — full reference

The complete list of operation categories that **must trigger Law 2**
(Stop before you break) before execution. Each row says: what counts,
what to show, what to ask.

For each category, the agent must produce in its message:

1. The **exact command/path** that will change
2. The **reason** this change is needed
3. A **rollback path** (or "irreversible" if there's none)
4. An **explicit yes** from the user, *for this exact change*

---

## 1. Auth & credentials

| Operation | Show | Ask | Why this matters |
|-----------|------|-----|-------------------|
| Change root user password | target user + new value (once) | "OK to set root to T10_Z0n1#?" | Lock-out is unrecoverable from remote |
| Edit `/etc/shadow` | diff | explicit | One wrong char = no login |
| Rotate API token / revoke | token name + scope | explicit | May break downstream deps |
| Disable 2FA / sudo NOPASSWD | file + line | explicit | One-shot security regression |

**Rule of thumb:** anything ending in your ability to log in must
have a console escape. Proxmox console, IPMI, cloud-init serial.

---

## 2. System paths

| Path | Why dangerous | Default action |
|------|---------------|----------------|
| `/etc/**` | System config | Backup before edit (`cp <file> <file>.bak`) |
| `/var/lib/**` | Service data | Snapshot / dump first |
| `/boot/**` | Bootloader | Don't touch without test plan |
| `/usr/lib/**` | Libs removed = broken deps | Confirm package, not file |
| `/usr/local/bin/**` | Often manually installed scripts | List prior commands first |

---

## 3. Storage

| Operation | Show | Ask |
|-----------|------|-----|
| `rm -rf <path>` | full path, with `ls -la` confirming contents | "delete these N files?" |
| `mv` over existing | full source + destination | explicit |
| `dd`, `wipefs`, `mkfs.*` | device target (`/dev/sdX`) | explicit + double-check |
| Partition resize | current + target size | explicit |
| `rsync --delete` | source + destination + `--dry-run` first | explicit |
| `btrfs` subvolume ops | target subvol + flags | explicit |

---

## 4. Network

| Operation | Show | Ask |
|-----------|------|-----|
| `iptables -F` (flush) | current ruleset | explicit (will lock you out mid-edit) |
| UFW / `nft` deletes | current ruleset | explicit |
| Routing table edits | `ip route` before/after | explicit |
| DNS server change | service + new upstream | explicit |
| Interface down | interface name + remote path | explicit |

**Special case:** any change that may remove the user's own SSH
session. Always test from a *second* session, or schedule with a
deadman switch.

---

## 5. Service lifecycle

| Operation | Show | Ask |
|-----------|------|-----|
| `systemctl stop` | service name + dependents | explicit |
| `systemctl restart` | service name | confirm unless trivial app |
| `systemctl mask` / `disable` | service name | explicit (persists across reboots) |
| `kill -9` | PID + ps line | confirm process is target, not system |
| `reboot` / `shutdown` | host + reason | explicit + warn if SSH'd in |
| `pct stop` / `qm stop` | ID + uptime | explicit unless emergency |

---

## 6. Container / VM

| Operation | Show | Ask |
|-----------|------|-----|
| `docker rm` / `podman rm` | name + image + status | explicit |
| `docker system prune -a` | preview output first | explicit |
| `pct destroy` / `qm destroy` | ID + description | **double-explicit** (no undo) |
| `docker network rm` | name + attached containers | explicit |
| `docker volume rm` | name + mountpoint + size | explicit |
| `kubectl delete` | resource + namespace | explicit |

---

## 7. Git history

| Operation | Show | Ask |
|-----------|------|-----|
| `git push --force` (with or without `-f`) | branch + remote + recent log | explicit |
| `git reset --hard` | current HEAD + target HEAD | explicit |
| `git clean -fd` | preview list | explicit |
| `git filter-branch` / `git filter-repo` | scope | explicit |
| `rm -rf .git` | path | explicit (drops everything) |

---

## 8. User-facing production

| Operation | Show | Ask |
|-----------|------|-----|
| Push to public repo | branch + diff summary | explicit |
| Deploy to prod | service + version + rollback plan | explicit |
| DNS record change | zone + record | explicit + warn propagation delay |
| TLS cert rotation | domain + new issuer | explicit |
| Public chat post / API write | target + content | explicit (irreversible) |

---

## The one-line form

For any of the above, your message to the user should contain:

```
About to: <exact command>
Will affect: <files / services / users>
To undo: <rollback path>
OK to proceed?
```

If you can't fill any of those four lines honestly, stop.
