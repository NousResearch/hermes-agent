/**
 * LocalStorageBackend — file-system based implementation of IStorageBackend.
 *
 * Used for local development and "free" mode where no COS is available.
 * Maps object keys to local file paths under a configurable root directory.
 */

import { readFile, writeFile, mkdir, readdir, unlink, stat, rm, appendFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join, dirname, sep, resolve } from "node:path";
import type {
  IStorageBackend,
  StorageObject,
  PutObjectOptions,
  ListObjectsOptions,
  ListResult,
  ListEntry,
  StorageLogger,
} from "./types.js";

const TAG = "[storage][local]";

export interface LocalStorageBackendOptions {
  /** Root directory for all stored files. */
  rootDir: string;
  /** Logger instance. */
  logger?: StorageLogger;
}

export class LocalStorageBackend implements IStorageBackend {
  readonly type = "local" as const;
  private readonly rootDir: string;
  private readonly logger?: StorageLogger;

  constructor(opts: LocalStorageBackendOptions | string) {
    if (typeof opts === "string") {
      // Backwards-compatible: accept plain string rootDir
      this.rootDir = opts;
    } else {
      this.rootDir = opts.rootDir;
      this.logger = opts.logger;
    }
  }

  /**
   * Resolve a storage key to an absolute filesystem path under rootDir.
   *
   * CR-6 fix (2026-05-19): rejects path-traversal attempts. Without this guard
   * a key containing "../" or absolute paths could read/write arbitrary files
   * on disk (e.g. getObject("../../../etc/passwd") would resolve outside
   * rootDir). Affects standalone mode where user-controllable fields like
   * instanceId / sceneName / sessionKey end up in the key.
   *
   * Rejected:
   * - Empty key
   * - Keys containing NUL (\0) — POSIX/Linux path terminator, can confuse
   *   downstream tooling (sqlite, file managers).
   * - Keys with leading "/" or "\" (absolute paths).
   * - Keys whose resolved path falls outside rootDir (../ traversal).
   */
  private resolvePath(key: string): string {
    if (!key || typeof key !== "string") {
      throw new Error(`Invalid storage key: ${JSON.stringify(key)}`);
    }
    if (key.includes("\0")) {
      throw new Error("Storage key must not contain NUL character");
    }
    if (key.startsWith("/") || key.startsWith("\\")) {
      throw new Error(`Storage key must be relative, got absolute: ${key}`);
    }

    // Normalize key separators to OS path separators
    const normalized = key.split("/").join(sep);

    // Compute the absolute resolved path; resolve() collapses ".." segments.
    const absRoot = resolve(this.rootDir);
    const absResolved = resolve(absRoot, normalized);

    // Ensure the resolved path stays inside rootDir. Append sep so that
    // a key like "../rootDir2/foo" (which resolves to a sibling directory
    // whose name happens to start with rootDir's name) is also rejected.
    const rootWithSep = absRoot.endsWith(sep) ? absRoot : absRoot + sep;
    if (absResolved !== absRoot && !absResolved.startsWith(rootWithSep)) {
      throw new Error(`Path traversal rejected: key "${key}" escapes rootDir`);
    }

    return absResolved;
  }

  async putObject(key: string, content: string | Buffer, opts?: PutObjectOptions): Promise<void> {
    const filePath = this.resolvePath(key);
    await mkdir(dirname(filePath), { recursive: true });

    const buf = typeof content === "string" ? Buffer.from(content, "utf-8") : content;
    await writeFile(filePath, buf);

    // Store metadata as a sidecar .meta.json file if metadata is provided
    if (opts?.contentType || (opts?.metadata && Object.keys(opts.metadata).length > 0)) {
      const metaPath = filePath + ".meta.json";
      await writeFile(metaPath, JSON.stringify({
        contentType: opts.contentType,
        metadata: opts.metadata,
      }));
    }

    this.logger?.debug?.(`${TAG} putObject: ${key} (${buf.length} bytes)`);
  }

  /**
   * Append content to the end of a file. CR-1 fix (2026-05-19): uses POSIX
   * fs.appendFile (with O_APPEND flag) which is atomic per call on local fs,
   * so concurrent appendObject calls to the same key do not race.
   */
  async appendObject(key: string, content: string | Buffer): Promise<void> {
    const filePath = this.resolvePath(key);
    await mkdir(dirname(filePath), { recursive: true });

    const buf = typeof content === "string" ? Buffer.from(content, "utf-8") : content;
    // fs.appendFile uses O_APPEND which provides POSIX atomic-append guarantees
    // for writes up to PIPE_BUF (4096 bytes on Linux). Larger writes may be
    // interleaved but never overwritten — each chunk lands at the end.
    await appendFile(filePath, buf);

    this.logger?.debug?.(`${TAG} appendObject: ${key} (+${buf.length} bytes)`);
  }

  async getObject(key: string): Promise<StorageObject | null> {
    const filePath = this.resolvePath(key);

    if (!existsSync(filePath)) {
      return null;
    }

    try {
      const [content, stats] = await Promise.all([
        readFile(filePath),
        stat(filePath),
      ]);

      // Try to read sidecar metadata
      let contentType: string | undefined;
      let metadata: Record<string, string> | undefined;
      const metaPath = filePath + ".meta.json";
      if (existsSync(metaPath)) {
        try {
          const metaRaw = await readFile(metaPath, "utf-8");
          const meta = JSON.parse(metaRaw);
          contentType = meta.contentType;
          metadata = meta.metadata;
        } catch {
          // ignore corrupt meta file
        }
      }

      return {
        key,
        content,
        contentType,
        metadata,
        lastModified: stats.mtime,
        size: stats.size,
      };
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") {
        return null;
      }
      throw err;
    }
  }

  async exists(key: string): Promise<boolean> {
    const filePath = this.resolvePath(key);
    return existsSync(filePath);
  }

  async listObjects(prefix: string, opts?: ListObjectsOptions): Promise<ListResult> {
    const dirPath = this.resolvePath(prefix);
    const maxKeys = opts?.maxKeys ?? 100;
    const recursive = opts?.recursive ?? false;

    if (!existsSync(dirPath)) {
      return { entries: [], total: 0 };
    }

    // Collect all matching entries (up to a reasonable ceiling)
    const allEntries: ListEntry[] = [];
    await this.walkDir(dirPath, prefix, allEntries, recursive, 10_000);

    // Simple marker-based pagination: skip entries until marker is found
    let startIdx = 0;
    if (opts?.marker) {
      const markerIdx = allEntries.findIndex(e => e.key === opts.marker);
      if (markerIdx >= 0) {
        startIdx = markerIdx + 1;
      }
    }

    const page = allEntries.slice(startIdx, startIdx + maxKeys);
    const hasMore = startIdx + maxKeys < allEntries.length;

    return {
      entries: page,
      nextMarker: hasMore ? page[page.length - 1]?.key : undefined,
      total: allEntries.length,
    };
  }

  async deleteObject(key: string): Promise<void> {
    const filePath = this.resolvePath(key);

    try {
      await unlink(filePath);
      // Also remove sidecar metadata if exists
      const metaPath = filePath + ".meta.json";
      if (existsSync(metaPath)) {
        await unlink(metaPath);
      }
      this.logger?.debug?.(`${TAG} deleteObject: ${key}`);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== "ENOENT") {
        throw err;
      }
      // Idempotent: ignore if file doesn't exist
    }
  }

  async deleteByPrefix(prefix: string): Promise<number> {
    const dirPath = this.resolvePath(prefix);

    if (!existsSync(dirPath)) {
      return 0;
    }

    // Count files before deleting
    const entries: ListEntry[] = [];
    await this.walkDir(dirPath, prefix, entries, true, Number.MAX_SAFE_INTEGER);
    const fileCount = entries.filter(e => !e.isDirectory).length;

    await rm(dirPath, { recursive: true, force: true });
    this.logger?.debug?.(`${TAG} deleteByPrefix: ${prefix} (${fileCount} files)`);

    return fileCount;
  }

  // ── Private helpers ──────────────────────────────────────

  private async walkDir(
    dirPath: string,
    keyPrefix: string,
    entries: ListEntry[],
    recursive: boolean,
    limit: number,
  ): Promise<void> {
    if (entries.length >= limit) return;

    let items: string[];
    try {
      items = await readdir(dirPath);
    } catch {
      return;
    }

    for (const item of items) {
      if (entries.length >= limit) break;
      // Skip sidecar metadata files
      if (item.endsWith(".meta.json")) continue;

      const fullPath = join(dirPath, item);
      const itemKey = keyPrefix.endsWith("/")
        ? `${keyPrefix}${item}`
        : `${keyPrefix}/${item}`;

      try {
        const stats = await stat(fullPath);

        if (stats.isDirectory()) {
          entries.push({
            key: itemKey + "/",
            size: 0,
            lastModified: stats.mtime,
            isDirectory: true,
          });
          if (recursive) {
            await this.walkDir(fullPath, itemKey + "/", entries, true, limit);
          }
        } else {
          entries.push({
            key: itemKey,
            size: stats.size,
            lastModified: stats.mtime,
            isDirectory: false,
          });
        }
      } catch {
        // skip inaccessible items
      }
    }
  }
}
