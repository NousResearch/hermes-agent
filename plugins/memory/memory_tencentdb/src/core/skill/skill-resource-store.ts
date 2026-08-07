/**
 * SkillResourceStore — 资源字节读写（仅 files/ 子树）
 *
 * 与旧 SkillContentManager 的差异：
 *   - 路径键改为 `<skill_id>/v<version>/files/<relative_path>`（不再用 name）
 *   - 不再管理 SKILL.md（DB 是权威源）
 *   - manifest 由 `listResources` 直接从磁盘列出（不需要 DB 同步）
 *
 * 设计文档对应：§2.4 物理 storage 布局；§3.5.9~3.5.11 接口。
 */

import type { StorageAdapter } from "../storage/adapter.js";
import type { SkillManifestEntry } from "./types.js";

const STORAGE_PREFIX = "skills/";
const FILES_SUBDIR = "files";
const DEFAULT_MAX_RESOURCE_SIZE_BYTES = 5_000_000;
const DEFAULT_MAX_SKILL_TOTAL_BYTES = 50_000_000;

export type ResourceErrorCode = "INVALID_PATH" | "RESOURCE_TOO_LARGE";

export class SkillResourceError extends Error {
  constructor(public readonly code: ResourceErrorCode, message?: string) {
    super(message ? `${code}: ${message}` : code);
    this.name = "SkillResourceError";
  }
}

export interface SkillResourcePayload {
  path: string;
  content: string;
  encoding: "utf-8" | "base64";
  mime_type?: string;
  is_executable?: boolean;
}

export interface SkillResourceReadResult {
  path: string;
  content: string;
  encoding: "utf-8" | "base64";
  size_bytes: number;
  mime_type: string;
  is_executable: boolean;
  version: number;
}

export interface SkillResourceStoreOptions {
  storage: StorageAdapter;
  maxResourceSizeBytes?: number;
  /** 整 skill 资源总大小上限。默认 50 MB（设计 §3.5.1）。 */
  maxSkillTotalBytes?: number;
}

export class SkillResourceStore {
  private readonly storage: StorageAdapter;
  private readonly maxBytes: number;
  private readonly maxTotalBytes: number;

  constructor(opts: SkillResourceStoreOptions) {
    this.storage = opts.storage;
    this.maxBytes = opts.maxResourceSizeBytes ?? DEFAULT_MAX_RESOURCE_SIZE_BYTES;
    this.maxTotalBytes = opts.maxSkillTotalBytes ?? DEFAULT_MAX_SKILL_TOTAL_BYTES;
  }

  /** 整 skill 总字节数上限（外部用于聚合校验前查阅）。 */
  getMaxSkillTotalBytes(): number {
    return this.maxTotalBytes;
  }

  /**
   * 估算 payload 的解码后字节大小，不写盘。
   * 用于 SkillVersioning 在落盘前做整 skill 总大小聚合校验。
   */
  estimatePayloadSize(payload: SkillResourcePayload): number {
    return decodeContent(payload.content, payload.encoding).length;
  }

  /**
   * 聚合校验：当前 manifest + 新写入 - 要删除/被覆盖 ≤ maxTotalBytes。
   * 超限抛 RESOURCE_TOO_LARGE，由 versioning 在落盘前调用。
   */
  assertTotalSize(
    currentManifest: SkillManifestEntry[],
    toWrite: SkillResourcePayload[] = [],
    toRemove: string[] = [],
  ): void {
    const removed = new Set(toRemove);
    const writePaths = new Set(toWrite.map((p) => p.path));
    let total = 0;
    for (const m of currentManifest) {
      if (removed.has(m.path)) continue;
      if (writePaths.has(m.path)) continue; // will be replaced
      total += m.size_bytes;
    }
    for (const p of toWrite) {
      total += this.estimatePayloadSize(p);
    }
    if (total > this.maxTotalBytes) {
      throw new SkillResourceError(
        "RESOURCE_TOO_LARGE",
        `total skill size ${total} bytes exceeds max ${this.maxTotalBytes} bytes`,
      );
    }
  }

  /** 写入资源字节，校验 path 和 size。如果 path 已存在则覆盖。 */
  async writeResource(skillId: string, version: number, payload: SkillResourcePayload): Promise<SkillManifestEntry> {
    this.assertPath(payload.path);
    const buf = decodeContent(payload.content, payload.encoding);
    if (buf.length > this.maxBytes) {
      throw new SkillResourceError(
        "RESOURCE_TOO_LARGE",
        `${payload.path} (${buf.length} bytes) exceeds max ${this.maxBytes} bytes`,
      );
    }
    const key = this.fileKey(skillId, version, payload.path);
    const mime = payload.mime_type ?? guessMime(payload.path);
    const isExec = payload.is_executable ?? false;
    // 用 backend.putObject 直接写以保留 metadata（is_executable）
    await this.storage.getBackend().putObject(key, buf, {
      contentType: mime,
      metadata: { is_executable: isExec ? "1" : "0" },
    });
    return {
      path: payload.path,
      size_bytes: buf.length,
      mime_type: mime,
      is_executable: isExec,
    };
  }

  /** 读取资源字节；不存在 → null。 */
  async readResource(
    skillId: string,
    version: number,
    path: string,
    encoding: "utf-8" | "base64",
  ): Promise<SkillResourceReadResult | null> {
    this.assertPath(path);
    const key = this.fileKey(skillId, version, path);
    const buf = await this.storage.readFileBuffer(key);
    if (!buf) return null;
    return {
      path,
      content: encoding === "base64" ? buf.toString("base64") : buf.toString("utf-8"),
      encoding,
      size_bytes: buf.length,
      mime_type: guessMime(path),
      is_executable: false, // 不从 storage 元数据反推（local-fs 与 COS 行为不一致）
      version,
    };
  }

  /** 删除资源（幂等：不存在不抛错）。 */
  async removeResource(skillId: string, version: number, path: string): Promise<void> {
    this.assertPath(path);
    const key = this.fileKey(skillId, version, path);
    try {
      await this.storage.unlink(key);
    } catch {
      /* idempotent */
    }
  }

  /** 列出某版本目录下的全部资源元信息（不含字节）。 */
  async listResources(skillId: string, version: number): Promise<SkillManifestEntry[]> {
    const prefix = this.filesPrefix(skillId, version);
    const result = await this.storage.getBackend().listObjects(prefix, {
      recursive: true,
      maxKeys: 100_000,
    });
    const out: SkillManifestEntry[] = [];
    for (const e of result.entries) {
      if (e.isDirectory) continue;
      const path = e.key.startsWith(prefix) ? e.key.slice(prefix.length) : e.key;
      // 读元数据拿到 is_executable / contentType（开销比读字节小很多）
      const obj = await this.storage.getBackend().getObject(e.key);
      const isExec = obj?.metadata?.is_executable === "1";
      const mime = obj?.contentType ?? guessMime(path);
      out.push({
        path,
        size_bytes: e.size,
        mime_type: mime,
        is_executable: isExec,
      });
    }
    return out;
  }

  /** 拿到该版本目录的相对前缀，便于 skill-versioning 调 storage.copyTree。 */
  versionDir(skillId: string, version: number): string {
    return `${STORAGE_PREFIX}${skillId}/v${version}`;
  }

  // ── helpers ──

  private fileKey(skillId: string, version: number, path: string): string {
    return `${this.filesPrefix(skillId, version)}${path}`;
  }

  private filesPrefix(skillId: string, version: number): string {
    return `${this.versionDir(skillId, version)}/${FILES_SUBDIR}/`;
  }

  private assertPath(path: string): void {
    if (!path) throw new SkillResourceError("INVALID_PATH", "empty path");
    if (path.startsWith("/") || path.startsWith("\\"))
      throw new SkillResourceError("INVALID_PATH", `absolute path not allowed: ${path}`);
    if (path.includes("\0"))
      throw new SkillResourceError("INVALID_PATH", `NUL not allowed: ${path}`);
    // ".." 段防越界
    const segs = path.split(/[\\/]/);
    if (segs.some((s) => s === "..")) {
      throw new SkillResourceError("INVALID_PATH", `traversal not allowed: ${path}`);
    }
  }
}

// ═════════════════════════════════════════════════════════════════════
//  独立工具
// ═════════════════════════════════════════════════════════════════════

function decodeContent(content: string, encoding: "utf-8" | "base64"): Buffer {
  return encoding === "base64" ? Buffer.from(content, "base64") : Buffer.from(content, "utf-8");
}

const MIME_BY_EXT: Record<string, string> = {
  ".sh": "text/x-shellscript",
  ".bash": "text/x-shellscript",
  ".js": "application/javascript",
  ".mjs": "application/javascript",
  ".ts": "application/typescript",
  ".json": "application/json",
  ".md": "text/markdown",
  ".txt": "text/plain",
  ".yml": "application/yaml",
  ".yaml": "application/yaml",
  ".py": "text/x-python",
  ".html": "text/html",
  ".css": "text/css",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
};

function guessMime(path: string): string {
  const idx = path.lastIndexOf(".");
  if (idx < 0) return "application/octet-stream";
  const ext = path.slice(idx).toLowerCase();
  return MIME_BY_EXT[ext] ?? "application/octet-stream";
}
