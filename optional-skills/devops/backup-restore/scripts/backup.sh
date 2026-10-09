#!/usr/bin/env bash
set -euo pipefail

# === Configuration: edit these before first use ===
REPO_DIR="${HERMES_HOME:-$HOME/.hermes}"          # Git repo to push backups to
BACKUP_DIR="${REPO_DIR}/backups"                   # Where tarballs are stored
RETENTION_WEEKS=4                                  # Auto-prune older than this

TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
TARBALL="${BACKUP_DIR}/hermes-backup-${TIMESTAMP}.tar.gz"
MANIFEST="${BACKUP_DIR}/manifest-${TIMESTAMP}.txt"

# === Sources to backup (add/remove as needed) ===
SOURCES=(
  "${HERMES_HOME:-$HOME/.hermes}/config.yaml"
  "${HERMES_HOME:-$HOME/.hermes}/SOUL.md"
  "${HERMES_HOME:-$HOME/.hermes}/profiles"
  "${HERMES_HOME:-$HOME/.hermes}/skills"
)

# === Exclude patterns ===
EXCLUDES=(
  "--exclude=*_key.txt"
  "--exclude=*_credentials"
  "--exclude=.env"
  "--exclude=.env.*"
  "--exclude=__pycache__"
  "--exclude=*.pyc"
  "--exclude=.git"
  "--exclude=node_modules"
  "--exclude=*.tar.gz"
)

echo "=== Backup: ${TIMESTAMP} ==="

mkdir -p "${BACKUP_DIR}"

for src in "${SOURCES[@]}"; do
  [ ! -e "$src" ] && echo "WARNING: Source not found: $src"
done

echo "Creating tarball..."
# Use -C for relative paths so tarball is portable
PARENT_DIR="$(cd "$(dirname "${REPO_DIR}")" && pwd)"
BASE_NAME="$(basename "${REPO_DIR}")"
tar czf "${TARBALL}" "${EXCLUDES[@]}" -C "${PARENT_DIR}" \
  "${SOURCES[@]#${PARENT_DIR}/}" 2>/dev/null

echo "Writing manifest..."
{
  echo "Backup: ${TIMESTAMP}"
  echo "Host: $(hostname)"
  echo "Sources:"
  for src in "${SOURCES[@]}"; do
    if [ -e "$src" ]; then
      size=$(du -sh "$src" 2>/dev/null | cut -f1)
      echo "  ${src}  (${size})"
    else
      echo "  ${src}  (MISSING)"
    fi
  done
  echo ""
  echo "Tarball size: $(du -h "${TARBALL}" | cut -f1)"
  echo "Tarball: ${TARBALL}"
} > "${MANIFEST}"

echo "Committing to git..."
cd "${REPO_DIR}"
git pull --rebase origin main 2>/dev/null || true
git add backups/
git commit -m "backup ${TIMESTAMP}" --quiet || echo "Nothing new to commit"

echo "Pushing to git..."
git push origin main 2>/dev/null || echo "WARNING: git push failed (offline?). Backup saved locally."

echo "Pruning backups older than ${RETENTION_WEEKS} weeks..."
find "${BACKUP_DIR}" -name "hermes-backup-*.tar.gz" -mtime +$((RETENTION_WEEKS * 7)) -delete 2>/dev/null || true
find "${BACKUP_DIR}" -name "manifest-*.txt" -mtime +$((RETENTION_WEEKS * 7)) -delete 2>/dev/null || true

echo ""
echo "=== Backup complete ==="
echo "Tarball: ${TARBALL}"
echo "Size:    $(du -h "${TARBALL}" | cut -f1)"
ls -lh "${BACKUP_DIR}" | grep hermes-backup
