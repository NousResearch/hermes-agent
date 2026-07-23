# Root Audit and Backup Retention

This reference absorbs the former `util-vps-cleanup` workflow into Genie. Use it when disk usage spikes, `/root` looks messy, or a cleanup needs attribution rather than routine cache/log removal.

## Policy

Only one historical backup may remain on the VPS at a time, in addition to current live data.

- Live data is not a historical backup.
- Local full `state.db` copies are not valid retained backups by default; they duplicate live data and are the primary disk-balloon failure mode. Keep one only when explicitly requested.
- Keep the newest valid historical backup by default.
- Delete or archive older historical backups after confirming there is a newer valid backup.
- Report any violation explicitly: path, size, timestamp, and recommended deletion target.
- Do not create a second historical backup as part of cleanup unless the operation is specifically a backup replacement and the older backup is removed after verification.

Common historical-backup paths:

- `/root/backup/*`
- `/root/backups/*`
- `/root/.hermes/state-snapshots/*`
- `/root/.hermes/migrations/*/backups/*`
- profile DB `.bak-*` files under `/root/.hermes/profiles/*/commons/db/`

## When routine Genie cleanup is insufficient

Run this audit flow before or after `genie.py --assess` when:

- filesystem usage jumped unexpectedly
- a backup directory is large
- duplicate repos or worktrees may exist
- `/root` has multiple similarly named repos
- the user asks why disk usage changed
- deleting a candidate requires checking whether running systems reference it

## Phase 1 — Inventory

```bash
df -hT /
du -xhd1 /root 2>/dev/null | sort -hr | head -60
du -xhd1 / 2>/dev/null | sort -hr | head -60
find / -xdev -type f -size +100M -mtime -3 -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort -k1,2 -r | head -100
```

Also inspect likely large subtrees:

```bash
du -xhd2 /root/.hermes /root/projects /root/backup /root/backups /var 2>/dev/null | sort -hr | head -100
```

## Phase 2 — Backup retention audit

Find all historical backup candidates and enforce the one-backup rule.

```bash
find /root/backup /root/backups /root/.hermes/state-snapshots /root/.hermes/migrations -xdev -mindepth 1 -maxdepth 4 \
  \( -type f -o -type d \) -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort -r | head -200

find /root -xdev \( -name 'state.db*' -o -name 'chronicle.db*' -o -name 'chroma.sqlite3*' -o -name '*.bak-*' \) \
  -printf '%TY-%Tm-%Td %TH:%TM %s %p -> %l\n' 2>/dev/null | sort -r | head -200
```

For SQLite backups, prefer `sqlite3 backup.db 'PRAGMA integrity_check;'` before treating one as valid. For plain file/directory backups, at minimum verify nonzero size and expected key files.

If there are multiple historical backups:

1. identify newest valid backup to keep
2. list older backups as reclaimable
3. remove older backups when cleanup has been authorized, or move them to trash when uncertain
4. verify `df -h /` after removal

## Phase 3 — Duplicate repo detection

Git repos and `node_modules` often explain multi-GB drift.

```bash
du -sh /root/*/ /root/projects/*/ /root/projects/github-staging/*/ 2>/dev/null | sort -hr | head -80
```

For suspicious pairs, compare remotes and HEADs:

```bash
compare_repos() {
  local a="$1" b="$2"
  echo "=== $a ==="; git -C "$a" remote -v 2>/dev/null | head -1; git -C "$a" log --oneline -1 2>/dev/null
  echo "=== $b ==="; git -C "$b" remote -v 2>/dev/null | head -1; git -C "$b" log --oneline -1 2>/dev/null
  stat -c '%y %n' "$a" "$b"
}
```

Same remote + same HEAD = duplicate candidate. Prefer keeping the newer active path, referenced path, or top-level canonical path.

## Phase 4 — Reference checks before moving/deleting

Do not blanket-grep all of `.hermes`; target specific reference sources.

```bash
grep -rl '/root/' /etc/systemd/system/ 2>/dev/null | grep -v hermes || true
ps aux | grep '/root/' | grep -v grep | grep -v '.hermes' || true
grep -n '/root/' /root/.bashrc /root/.profile /root/.zshrc 2>/dev/null | grep -v '#' || true
crontab -l 2>/dev/null || true
hermes cron list 2>/dev/null || true
grep '/root/' /root/.hermes/config.yaml /root/.hermes/profiles/*/config.yaml 2>/dev/null || true
```

For loose Python files, check actual import statements, not substring matches.

## Phase 5 — Classification

- **Live runtime:** referenced by processes, services, active configs, or Hermes runtime. Keep.
- **Historical backup:** backup/snapshot/`.bak-*`. Enforce one historical backup total on the VPS.
- **Git staging:** repo-only state pushed to GitHub. Usually safe to move/delete if duplicate and clean.
- **Project:** experiments or third-party clones. Safe only when not active and re-clonable.
- **Trash:** caches, stale extracts, old logs, empty dirs, one-shot scripts, duplicate repos with same remote + same HEAD.

When uncertain, move to a dated trash directory first:

```bash
TRASH="/root/.trash/$(date +%Y-%m-%d)"
mkdir -p "$TRASH"
mv /path/to/candidate "$TRASH/"
```

Then verify:

```bash
df -h /
systemctl is-active hermes-gateway 2>/dev/null || true
hermes status 2>/dev/null || true
```

## Known pitfall — symlinked state.db backups

`/root/.hermes/state.db` may be a symlink to `/root/.hermes/profiles/indigo/state.db`. A backup script that uses `stat -c%s` on the symlink sees the symlink length, not the 12GB+ target. Use `readlink -f` and `stat -L` before copying. If available space is less than target size plus margin, skip the local state backup instead of filling the disk.

## Known pitfall — partial backup newer than complete backup

Retention must not blindly keep the newest backup directory. A failed retry can create a tiny newer directory that is not a valid restore point, while the previous larger directory is the newest complete backup. When enforcing the one-historical-backup rule, rank candidates by completeness first, recency second. At minimum, score whether expected key files are present (`state.db`, `chroma.sqlite3`, `chronicle.lbug`, `weave.lbug`, `styx.db`, `transactions.db`, `mempalace.tar.gz`) before deleting older candidates.
