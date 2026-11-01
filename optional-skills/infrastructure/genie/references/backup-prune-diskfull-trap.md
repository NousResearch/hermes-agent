# backup prune / disk-full trap

## STATUS: FIX APPLIED 2026-07-14 (verify before re-patching)

The live backup writer is `backup_all_hermes_data.sh` (run via the ocas-custodian
wrapper), **not** `backup_system.sh`. `backup_system.sh` targets `/root/backups`
(plural) and is unused. Earlier scans blamed it for "zero retention logic,"
which was wrong. Follow the real exec chain before assigning root cause.

## Root cause of /root/backup bloat

`backup_all_hermes_data.sh` copies the 14 GB profile `state.db`
(`/root/.hermes/.../state.db`, symlink) at line 51, then has its prune line
(`find /root/backup -mtime +3 -exec rm -rf`) at line 114. The script runs under
`set -euo pipefail`. On a near-full disk, the `cp` hits ENOSPC, the script aborts,
and the prune **never runs**. Partial `active-dbs-*` directories accumulate,
all of them missing `state.db`, which is the ENOSPC-abort signature.

## The fix

1. Prune moved to **before** the 14 GB copy.
2. Free-space guard added: `readlink -f` + `stat -L`; skip the copy if
   `avail < size * 1.1`.
3. `trap cleanup_partial ERR` removes the partial directory on failure.

## Verification recipe (run first)

```bash
# 1. Prune is now above the copy in the script:
grep -n "find /root/backup -mtime" /root/indigo-repo/scripts/backup_all_hermes_data.sh
#    → line number must be LOWER than the state.db cp line.

# 2. Free-space guard present:
grep -n "avail < size" /root/indigo-repo/scripts/backup_all_hermes_data.sh
#    → must return a match.

# 3. Partial-cleanup trap present:
grep -n "trap cleanup_partial ERR" /root/indigo-repo/scripts/backup_all_hermes_data.sh
#    → must return a match.
```

If all three checks pass, the fix is intact. Do **not** re-apply it, because
doing so would duplicate the guards. The separate TASK-015 (state.db FTS-trigram bloat,
VACUUM) is a distinct open task and is not covered by this fix.
