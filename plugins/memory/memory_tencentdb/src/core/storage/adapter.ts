/**
 * StorageAdapter — adapts IStorageBackend to a fs-like interface
 * for compatibility with upper-layer code.
 *
 * Progressive migration strategy:
 *   Existing L2/L3/Recall code uses `fs.readFile(path.join(dataDir, ...))`.
 *   StorageAdapter provides equivalent method signatures, internally delegating
 *   to IStorageBackend, so existing code only needs to swap the import.
 *
 * Eventually, callers may inline IStorageBackend calls directly and
 * this adapter can be removed.
 */

import type { IStorageBackend, StorageObject, ListEntry, ListObjectsOptions, ListResult, PutObjectOptions } from "./types.js";

class ScopedStorageBackend implements IStorageBackend {
  readonly type: "local" | "cos";
  private readonly prefix: string;

  constructor(private readonly base: IStorageBackend, prefix: string) {
    this.type = base.type;
    const normalized = prefix.replace(/^\/+/, "").replace(/\/+/g, "/");
    this.prefix = normalized && !normalized.endsWith("/") ? `${normalized}/` : normalized;
  }

  private key(key: string): string {
    if (typeof key !== "string" || key.includes("\0") || key.startsWith("/") || key.startsWith("\\")) {
      throw new Error(`Invalid scoped storage key: ${JSON.stringify(key)}`);
    }
    const normalized = key.replace(/^\/+/, "").replace(/\\+/g, "/").replace(/\/+/g, "/");
    if (normalized.split("/").some((part) => part === "..")) {
      throw new Error(`Path traversal rejected in scoped storage key: ${key}`);
    }
    return `${this.prefix}${normalized}`;
  }

  private unkey(key: string): string {
    return key.startsWith(this.prefix) ? key.slice(this.prefix.length) : key;
  }

  async putObject(key: string, content: string | Buffer, opts?: PutObjectOptions): Promise<void> {
    return this.base.putObject(this.key(key), content, opts);
  }

  async appendObject(key: string, content: string | Buffer): Promise<void> {
    return this.base.appendObject(this.key(key), content);
  }

  async getObject(key: string): Promise<StorageObject | null> {
    const obj = await this.base.getObject(this.key(key));
    return obj ? { ...obj, key: this.unkey(obj.key) } : null;
  }

  async exists(key: string): Promise<boolean> {
    return this.base.exists(this.key(key));
  }

  async listObjects(prefix: string, opts?: ListObjectsOptions): Promise<ListResult> {
    const result = await this.base.listObjects(this.key(prefix), opts);
    return {
      ...result,
      entries: result.entries.map((entry) => ({ ...entry, key: this.unkey(entry.key) })),
    };
  }

  async deleteObject(key: string): Promise<void> {
    return this.base.deleteObject(this.key(key));
  }

  async deleteByPrefix(prefix: string): Promise<number> {
    return this.base.deleteByPrefix(this.key(prefix));
  }
}

export function createScopedStorageAdapter(base: StorageAdapter, prefix: string): StorageAdapter {
  if (!prefix) return base;
  return new StorageAdapter(new ScopedStorageBackend(base.getBackend(), prefix));
}

export class StorageAdapter {
  constructor(private backend: IStorageBackend) {}

  get type() { return this.backend.type; }

  // ── fs.readFile replacement ──

  async readFile(key: string): Promise<string | null> {
    const obj = await this.backend.getObject(key);
    if (!obj) return null;
    return obj.content.toString("utf-8");
  }

  async readFileOrThrow(key: string): Promise<string> {
    const content = await this.readFile(key);
    if (content === null) throw new Error(`File not found: ${key}`);
    return content;
  }

  async readFileBuffer(key: string): Promise<Buffer | null> {
    const obj = await this.backend.getObject(key);
    if (!obj) return null;
    return obj.content;
  }

  // ── fs.writeFile replacement ──

  async writeFile(key: string, content: string | Buffer): Promise<void> {
    return this.backend.putObject(key, content);
  }

  // ── fs.appendFile replacement — atomic via backend.appendObject (CR-1 fix) ──

  /**
   * Append to a storage object atomically.
   *
   * CR-1 fix (2026-05-19): previously implemented as read-modify-write
   * (readFile + concat + putObject), which lost data under concurrency
   * (audit/exp1 reproduced 99% loss at 100 parallel writes). Now delegates
   * to backend.appendObject which uses:
   *   - LocalStorageBackend: POSIX fs.appendFile (O_APPEND atomic)
   *   - CosStorageBackend: COS Append Object API (server-side atomic + 409 retry)
   */
  async appendFile(key: string, content: string): Promise<void> {
    return this.backend.appendObject(key, content);
  }

  // ── fs.readdir replacement ──

  async readdir(prefix: string, suffix?: string): Promise<ListEntry[]> {
    const result = await this.backend.listObjects(prefix, { maxKeys: 10000 });
    if (!suffix) return result.entries;
    return result.entries.filter(e => e.key.endsWith(suffix));
  }

  async readdirNames(prefix: string, suffix?: string): Promise<string[]> {
    const entries = await this.readdir(prefix, suffix);
    return entries
      .filter(e => !e.isDirectory)
      .map((e) => {
        // Return filename without prefix
        const name = e.key.startsWith(prefix) ? e.key.slice(prefix.length) : e.key;
        return name;
      });
  }

  // ── fs.unlink replacement ──

  async unlink(key: string): Promise<void> {
    return this.backend.deleteObject(key);
  }

  // ── fs.rm (recursive) replacement ──

  async rmdir(prefix: string): Promise<void> {
    await this.backend.deleteByPrefix(prefix);
  }

  // ── fs.mkdir (recursive) replacement ──
  // No-op for object storage (directories are implicit).
  // For local backend, putObject auto-creates parent dirs.

  async mkdir(_prefix: string): Promise<void> {
    // No-op: directories are created implicitly on putObject
  }

  // ── fs.access replacement ──

  async exists(key: string): Promise<boolean> {
    return this.backend.exists(key);
  }

  // ── fs.stat replacement ──

  async stat(key: string): Promise<{ key: string; size: number; lastModified: number; createdAt: number } | null> {
    const obj = await this.backend.getObject(key);
    if (!obj) return null;
    const lastModified = obj.lastModified?.getTime() ?? Date.now();
    return {
      key,
      size: obj.size ?? obj.content.length,
      lastModified,
      createdAt: lastModified,
    };
  }

  // ── fs.rename replacement ──

  async rename(sourceKey: string, destKey: string): Promise<void> {
    // CR-8 partial fix (2026-05-19): preserve contentType + metadata across rename.
    // The 3-step (get → put → delete) is still NOT atomic; if the process is killed
    // between put and delete, both source and dest will exist (data duplication).
    // A complete fix requires a native renameObject in IStorageBackend (using
    // POSIX fs.rename for local + COS x-cos-copy-source for remote). Tracked as
    // long-term work — see audit report H-6 (persona.md backup rotation).
    const obj = await this.backend.getObject(sourceKey);
    if (!obj) throw new Error(`Source not found: ${sourceKey}`);
    await this.backend.putObject(destKey, obj.content, {
      contentType: obj.contentType,
      metadata: obj.metadata,
    });
    await this.backend.deleteObject(sourceKey);
  }

  // ── fs.copyFile replacement ──

  async copyFile(sourceKey: string, destKey: string): Promise<void> {
    // CR-8 partial fix (2026-05-19): preserve contentType + metadata across copy.
    const obj = await this.backend.getObject(sourceKey);
    if (!obj) throw new Error(`Source not found: ${sourceKey}`);
    await this.backend.putObject(destKey, obj.content, {
      contentType: obj.contentType,
      metadata: obj.metadata,
    });
  }

  // ── fs.cp -r replacement (new in v2 redesign for skill versioning) ──

  /**
   * Recursively copy all objects under `srcPrefix` to `dstPrefix`.
   *
   * 路径映射：每个 src 下的 object key `${srcPrefix}/path/to/x` 都会被复制到
   * `${dstPrefix}/path/to/x`。`prefix` 之间的相对路径保持不变。
   *
   * 行为：
   *   - srcPrefix 下没有任何对象 → throw `STORAGE_NOT_FOUND: <srcPrefix>`
   *   - dstPrefix 已存在 ≥1 个对象 → 默认 throw `DESTINATION_EXISTS: <dstPrefix>`
   *   - 传入 `{ overwrite: true }` 时允许覆盖（dst 端可能存在残留也照写）
   *
   * 在 Phase 5 之后被 skill-versioning 使用：每次 owner 改动触发新版本时，
   * 把上一版本目录拷贝到新版本目录，再应用本次资源变更（write/remove）。
   */
  async copyTree(
    srcPrefix: string,
    dstPrefix: string,
    opts: { overwrite?: boolean } = {},
  ): Promise<void> {
    const srcEntries = await this.backend.listObjects(srcPrefix, {
      maxKeys: 100_000,
      recursive: true,
    });

    // src 下没有任何对象 → 视为不存在
    const srcFiles = srcEntries.entries.filter((e) => !e.isDirectory);
    const srcExists = await this.backend.exists(srcPrefix);
    if (srcFiles.length === 0 && !srcExists) {
      throw new Error(`STORAGE_NOT_FOUND: ${srcPrefix}`);
    }

    if (!opts.overwrite) {
      const dstEntries = await this.backend.listObjects(dstPrefix, {
        maxKeys: 1000,
        recursive: true,
      });
      if (dstEntries.entries.some((e) => !e.isDirectory)) {
        throw new Error(`DESTINATION_EXISTS: ${dstPrefix}`);
      }
    }

    const srcNorm = srcPrefix.endsWith("/") ? srcPrefix : srcPrefix + "/";
    const dstNorm = dstPrefix.endsWith("/") ? dstPrefix : dstPrefix + "/";

    for (const entry of srcFiles) {
      // 计算相对路径
      let rel = entry.key;
      if (rel.startsWith(srcNorm)) rel = rel.slice(srcNorm.length);
      else if (rel === srcPrefix) rel = "";
      const dstKey = `${dstNorm}${rel}`;

      const obj = await this.backend.getObject(entry.key);
      if (!obj) continue;
      await this.backend.putObject(dstKey, obj.content, {
        contentType: obj.contentType,
        metadata: obj.metadata,
      });
    }
  }

  // ── Direct backend access ──

  getBackend(): IStorageBackend {
    return this.backend;
  }
}
