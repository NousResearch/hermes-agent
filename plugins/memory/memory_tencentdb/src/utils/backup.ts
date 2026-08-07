/**
 * BackupManager: generic file/directory backup utility.
 *
 * Provides two backup modes:
 *   - `backupFile(src, category, tag, maxKeep)` — copy a single file
 *   - `backupDirectory(src, category, tag, maxKeep)` — copy an entire directory
 *
 * All backups land under `<backupRoot>/<category>/` with timestamped names.
 * After each backup, entries beyond `maxKeep` are automatically pruned
 * (oldest first, by lexicographic order on the timestamp-embedded name).
 */

import fs from "node:fs/promises";
import path from "node:path";

export class BackupManager {
  private backupRoot: string;

  /**
   * @param backupRoot - Absolute path to the root backup directory
   *                     (e.g. `<dataDir>/.backup`).
   */
  constructor(backupRoot: string) {
    this.backupRoot = backupRoot;
  }

  /**
   * Backup a single file.
   *
   * Destination: `<backupRoot>/<category>/<category>_<timestamp>_<tag>.<ext>`
   *
   * @param srcFile   - Absolute path to the source file
   * @param category  - Logical grouping (e.g. "persona")
   * @param tag       - Additional identifier (e.g. "offset42")
   * @param maxKeep   - Max backup files to retain in this category (0 = unlimited)
   */
  async backupFile(
    srcFile: string,
    category: string,
    tag: string,
    maxKeep: number,
  ): Promise<void> {
    try {
      await fs.access(srcFile);
    } catch {
      return; // Source file doesn't exist, nothing to backup
    }

    const destDir = path.join(this.backupRoot, category);
    await fs.mkdir(destDir, { recursive: true });

    const ext = path.extname(srcFile); // e.g. ".md"
    const timestamp = formatTimestamp(new Date());
    const destName = `${category}_${timestamp}_${tag}${ext}`;
    await fs.copyFile(srcFile, path.join(destDir, destName));

    if (maxKeep > 0) {
      await pruneOldEntries(destDir, maxKeep, "file");
    }
  }

  /**
   * Backup an entire directory (shallow copy of all files).
   *
   * Destination: `<backupRoot>/<category>/<category>_<timestamp>_<tag>/`
   *
   * @param srcDir    - Absolute path to the source directory
   * @param category  - Logical grouping (e.g. "scene_blocks")
   * @param tag       - Additional identifier (e.g. "offset42")
   * @param maxKeep   - Max backup directories to retain in this category (0 = unlimited)
   */
  async backupDirectory(
    srcDir: string,
    category: string,
    tag: string,
    maxKeep: number,
  ): Promise<void> {
    let entries: import("node:fs").Dirent[];
    try {
      entries = await fs.readdir(srcDir, { withFileTypes: true });
    } catch {
      return; // Source directory doesn't exist
    }

    // Only backup regular files (skip subdirectories to avoid EISDIR errors)
    const files = entries.filter((e) => e.isFile()).map((e) => e.name);
    if (files.length === 0) return;

    const parentDir = path.join(this.backupRoot, category);
    const timestamp = formatTimestamp(new Date());
    const destDir = path.join(parentDir, `${category}_${timestamp}_${tag}`);
    await fs.mkdir(destDir, { recursive: true });

    for (const file of files) {
      await fs.copyFile(path.join(srcDir, file), path.join(destDir, file));
    }

    if (maxKeep > 0) {
      await pruneOldEntries(parentDir, maxKeep, "directory");
    }
  }

  /**
   * Find the latest backup directory for a category.
   *
   * Backup directory names are `<category>_<timestamp>_<tag>` where the
   * timestamp is `YYYYMMDD_HHmmss` (lexicographic order = chronological order),
   * so the lexicographically largest entry is the most recent one.
   *
   * @param category - Logical grouping (e.g. "scene_blocks")
   * @returns Absolute path to the latest backup directory, or undefined if none.
   */
  async findLatestBackup(category: string): Promise<string | undefined> {
    const parentDir = path.join(this.backupRoot, category);
    let entries: import("node:fs").Dirent[];
    try {
      entries = await fs.readdir(parentDir, { withFileTypes: true });
    } catch {
      return undefined; // No backup directory yet
    }
    const dirs = entries.filter((e) => e.isDirectory()).map((e) => e.name);
    if (dirs.length === 0) return undefined;
    dirs.sort(); // ascending — oldest first; last = newest
    return path.join(parentDir, dirs[dirs.length - 1]);
  }

  /**
   * Restore the latest backup of `category` into `destDir`.
   *
   * Strategy:
   *   1. Find the latest backup directory; if none exists, do nothing
   *      (fail-soft: never clobber the destination when there is no
   *      ground truth to restore from).
   *   2. Wipe `destDir` and recreate it.
   *   3. Copy every regular file from the backup directory into `destDir`.
   *
   * @param category - Logical grouping (e.g. "scene_blocks")
   * @param destDir  - Absolute path to the directory to restore into
   * @returns `{ restored: true, from }` when a backup was applied,
   *          `{ restored: false }` when no backup was found.
   * @throws  Lets fs errors during wipe/copy propagate so callers can decide
   *          whether to fail-soft (log) or fail-hard.
   */
  async restoreLatestDirectory(
    category: string,
    destDir: string,
  ): Promise<{ restored: boolean; from?: string }> {
    const from = await this.findLatestBackup(category);
    if (!from) return { restored: false };

    // Wipe the destination first so any partial LLM writes are removed,
    // then recreate the directory and copy regular files back.
    await fs.rm(destDir, { recursive: true, force: true });
    await fs.mkdir(destDir, { recursive: true });

    const entries = await fs.readdir(from, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isFile()) continue;
      await fs.copyFile(path.join(from, entry.name), path.join(destDir, entry.name));
    }

    return { restored: true, from };
  }
}

// ============================
// Helpers
// ============================

function formatTimestamp(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return [
    d.getFullYear(),
    pad(d.getMonth() + 1),
    pad(d.getDate()),
    "_",
    pad(d.getHours()),
    pad(d.getMinutes()),
    pad(d.getSeconds()),
  ].join("");
}

/**
 * Keep only the newest `maxKeep` entries in a directory.
 * Entries are sorted by name ascending (oldest first) since backup names
 * embed timestamps, so lexicographic order = chronological order.
 *
 * @param dir     - Directory containing the backup entries
 * @param maxKeep - Number of entries to retain
 * @param kind    - "file" to unlink, "directory" to rm -rf
 */
async function pruneOldEntries(
  dir: string,
  maxKeep: number,
  kind: "file" | "directory",
): Promise<void> {
  let entries: string[];
  try {
    entries = await fs.readdir(dir);
  } catch {
    return;
  }

  entries.sort(); // ascending — oldest first
  const toRemove = entries.slice(0, Math.max(0, entries.length - maxKeep));

  for (const name of toRemove) {
    try {
      if (kind === "file") {
        await fs.unlink(path.join(dir, name));
      } else {
        await fs.rm(path.join(dir, name), { recursive: true, force: true });
      }
    } catch {
      // best-effort
    }
  }
}
