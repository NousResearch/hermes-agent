/**
 * File Storage Abstraction Layer — Core Types & Interfaces.
 *
 * This module defines the file storage contracts for L2 scenario files,
 * L3 persona files, checkpoints, and (future) media uploads.
 *
 * Design principles:
 * 1. **Backend-agnostic**: Upper layers depend only on IStorageBackend —
 *    never on COS SDK or fs internals directly.
 * 2. **Async-first**: All methods return Promises (COS is inherently async,
 *    local-fs adapter wraps with async for uniformity).
 * 3. **Extensible**: Interface is minimal for v1; future phases add
 *    presigned URLs, STS credential generation, multipart upload, etc.
 *
 * Relationship to IMemoryStore (src/core/store/types.ts):
 *   - IMemoryStore  = database abstraction (L0/L1 structured data → VDB/SQLite)
 *   - IStorageBackend = file storage abstraction (L2/L3 Markdown files → COS/local-fs)
 *   Both are parallel, not replacements of each other.
 */

// ============================
// Storage Object Types
// ============================

/** Options for writing an object. */
export interface PutObjectOptions {
  /** MIME content type, e.g. "text/markdown", "application/json". */
  contentType?: string;
  /** Custom metadata key-value pairs. */
  metadata?: Record<string, string>;
}

/** A retrieved storage object. */
export interface StorageObject {
  /** Full object key (path). */
  key: string;
  /** Object content as a Buffer. */
  content: Buffer;
  /** MIME content type. */
  contentType?: string;
  /** Custom metadata. */
  metadata?: Record<string, string>;
  /** Last modification time. */
  lastModified?: Date;
  /** Object size in bytes. */
  size?: number;
}

/** Options for listing objects. */
export interface ListObjectsOptions {
  /** Maximum number of entries to return. Default: 100. */
  maxKeys?: number;
  /** Pagination marker from a previous ListResult. */
  marker?: string;
  /** Whether to list recursively into subdirectories. Default: false. */
  recursive?: boolean;
}

/** A single entry in a list result. */
export interface ListEntry {
  /** Full object key (path) or directory prefix. */
  key: string;
  /** Size in bytes (0 for directories). */
  size: number;
  /** Last modification time. */
  lastModified: Date;
  /** Whether this entry represents a directory (common prefix). */
  isDirectory: boolean;
}

/** Result of a list operation. */
export interface ListResult {
  /** Listed entries (files and/or directories). */
  entries: ListEntry[];
  /** Marker for fetching the next page; undefined if no more pages. */
  nextMarker?: string;
  /** Total count of entries matching the prefix (if available). */
  total?: number;
}

// ============================
// IStorageBackend — The Core Abstraction
// ============================

/**
 * Unified file storage interface for L2/L3 Markdown files, checkpoints, etc.
 *
 * Implementations:
 * - `LocalStorageBackend` (local-backend.ts) — local filesystem, for dev/free mode
 * - `CosStorageBackend`  (cos-backend.ts)   — Tencent Cloud COS, for production
 *
 * All methods are async. Errors are thrown as exceptions (unlike IMemoryStore
 * which swallows errors); callers should handle appropriately.
 */
export interface IStorageBackend {
  /** Storage backend identifier for logging/diagnostics. */
  readonly type: "local" | "cos";

  /**
   * Write an object (create or overwrite).
   * @param key   Object key (path), e.g. "scenes/work/2026Q1.md"
   * @param content String or Buffer content
   * @param opts  Optional content type and metadata
   */
  putObject(key: string, content: string | Buffer, opts?: PutObjectOptions): Promise<void>;

  /**
   * Append content to the end of an object.
   *
   * Semantics (CR-1 fix, 2026-05-19):
   * - LocalStorageBackend: uses POSIX fs.appendFile — atomic per-call.
   * - CosStorageBackend: uses COS Append Object API (`?append&position=N`) —
   *   atomic per-call, server-side position-checked. Concurrent appends to
   *   the same key are detected and retried (the second one gets 409
   *   AppendPositionErrorException and retries with fresh position).
   *
   * Once an object is created via appendObject in COS, it becomes type
   * `appendable` and CANNOT be overwritten by putObject. Callers must
   * stick to one access pattern per key (use APPENDABLE_KEY_PREFIXES guard
   * in CosStorageBackend to enforce this at runtime).
   *
   * @param key     Object key (path), e.g. "records/2026-05-20.jsonl"
   * @param content String or Buffer to append (atomic)
   */
  appendObject(key: string, content: string | Buffer): Promise<void>;

  /**
   * Read an object by key.
   * @returns The object, or null if not found.
   */
  getObject(key: string): Promise<StorageObject | null>;

  /**
   * Check whether an object exists.
   */
  exists(key: string): Promise<boolean>;

  /**
   * List objects under a prefix.
   * @param prefix Key prefix, e.g. "scenes/" or "conversations/sess_abc/"
   * @param opts   Pagination and recursion options
   */
  listObjects(prefix: string, opts?: ListObjectsOptions): Promise<ListResult>;

  /**
   * Delete a single object.
   * Does not throw if the object does not exist (idempotent).
   */
  deleteObject(key: string): Promise<void>;

  /**
   * Delete all objects under a prefix (for instance destruction).
   * @returns Number of objects deleted.
   */
  deleteByPrefix(prefix: string): Promise<number>;
}

// ============================
// Credential Types (for COS backend)
// ============================

/** COS access credentials obtained from a configured credential source. */
export interface CosCredential {
  /** SecretId (AK) — identifies the caller. */
  secretId: string;
  /** SecretKey (SK) — used by COS SDK to sign requests. */
  secretKey: string;
  /** Session token for STS temporary credentials (future). */
  token?: string;
  /** COS bucket name, e.g. "tdai-memory-data-1234567890". */
  bucket: string;
  /** COS region, e.g. "ap-guangzhou". */
  region: string;
  /** Key path prefix for this instance, e.g. "tenant_alice/space_001/". */
  prefix: string;
  /** Credential expiration timestamp in ms (undefined = never expires). */
  expiresAt?: number;
}

/**
 * Credential provider abstraction — decouples COS credential sourcing
 * from the storage backend. Supports caching and auto-refresh.
 *
 * Implementations:
 * - MockCredentialProvider  — local dev / testing
 * - ConfigCredentialProvider — reads from config file
 * - custom credential provider — calls a deployment-specific credential source
 */
export interface ICredentialProvider {
  /**
   * Get current valid COS credentials (may return cached).
   * Automatically refreshes if cache is expired.
   */
  getCosCredential(): Promise<CosCredential>;

  /**
   * Force invalidate cached credentials.
   * Call this when COS returns 403 (credential rejected).
   */
  invalidate(): void;
}

// ============================
// Storage Configuration
// ============================

/** Configuration for creating a storage backend. */
export interface StorageBackendConfig {
  /** Backend type: "local" for dev, "cos" for production. */
  type: "local" | "cos";

  /** Local backend: root directory for file storage. */
  localRootDir?: string;

  /** COS backend: credential provider instance. */
  credentialProvider?: ICredentialProvider;
}

/** Minimal logger interface for storage operations. */
export interface StorageLogger {
  debug?: (message: string) => void;
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
}

// ============================
// Storage Path Constants (retained from original design)
// ============================

/**
 * Storage path conventions.
 *
 * COS full path: {pathPrefix}/{key}
 *
 * key directory structure:
 *   conversations/{YYYY-MM-DD}.jsonl   — L0 conversation records
 *   records/{YYYY-MM-DD}.jsonl         — L1 memory records
 *   scene_blocks/{name}.md             — L2 scene blocks
 *   persona.md                         — L3 user persona
 *   .metadata/scene_index.json         — scene index
 *   .metadata/checkpoint.json          — pipeline checkpoint
 *   .metadata/manifest.json            — metadata manifest
 *   .metadata/instance_id              — instance ID
 *   .backup/persona/                   — persona backups
 *   .backup/scene_blocks/              — scene block backups
 */
export const StoragePaths = {
  /** L3 core memory. Chat mode stores persona; code mode stores Team Operating Doctrine in the same file. */
  persona: "persona.md",
  /** L2 scene blocks directory */
  sceneBlocksDir: "scene_blocks/",
  /** L0 conversations directory */
  conversationsDir: "conversations/",
  /** L1 memory records directory */
  recordsDir: "records/",
  /** Metadata directory */
  metadataDir: ".metadata/",
  /** Scene index */
  sceneIndex: ".metadata/scene_index.json",
  /** Pipeline checkpoint */
  checkpoint: ".metadata/checkpoint.json",
  /** Metadata manifest */
  manifest: ".metadata/manifest.json",
  /** Instance ID */
  instanceId: ".metadata/instance_id",
  /** Backup directory */
  backupDir: ".backup/",

  /** Build scene block path */
  sceneBlock: (name: string) => `scene_blocks/${name}.md`,
  /** Build conversation JSONL path */
  conversation: (date: string) => `conversations/${date}.jsonl`,
  /** Build memory record JSONL path */
  record: (date: string) => `records/${date}.jsonl`,
  /** Build persona backup path */
  personaBackup: (index: number) => `.backup/persona/persona.${index}.md`,
  /** Build scene block backup path */
  sceneBlockBackup: (index: number, name: string) => `.backup/scene_blocks/${index}/${name}.md`,
} as const;
