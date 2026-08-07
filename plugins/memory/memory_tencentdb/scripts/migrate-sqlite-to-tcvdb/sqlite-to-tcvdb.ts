import fs from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";
import type { MemoryRecord } from "../../src/core/record/l1-writer.js";
import { listLocalProfiles } from "../../src/core/profile/profile-sync.js";
import { createBM25Encoder } from "../../src/core/store/bm25-local.js";
import { VectorStore, type L0RecordRow } from "../../src/core/store/sqlite.js";
import { TcvdbMemoryStore } from "../../src/core/store/tcvdb.js";
import type { L0Record, L1RecordRow, ProfileRecord, ProfileSyncRecord, StoreInitResult } from "../../src/core/store/types.js";
import { readManifest } from "../../src/utils/manifest.js";
import {
  rewriteMigrationManifest as rewriteMigrationManifestDefault,
  type RewriteMigrationManifestResult,
} from "./manifest-write.js";
import {
  writeMigrationPluginConfig as writeMigrationPluginConfigDefault,
} from "./config-write.js";

export const DEFAULT_MIGRATION_PLUGIN_ID = "memory-tencentdb";
export const ALL_MIGRATION_LAYERS = ["l0", "l1", "l2", "l3"] as const;

const TAG = "[memory-tdai][migrate]";

function log(message: string): void {
  process.stderr.write(`${TAG} ${message}\n`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export type MigrationLayer = (typeof ALL_MIGRATION_LAYERS)[number];
export type Bm25Language = "zh" | "en";

export interface ResolvedMigrationCliOptions {
  pluginDataDir: string;
  sqlitePath: string;
  openclawConfigPath: string;
  pluginId: string;
  layers: MigrationLayer[];
  applyConfig: boolean;
  configBackup: boolean;
  rewriteManifest: boolean;
  failIfTargetNonempty: boolean;
  verifyCounts: boolean;
  dryRun: boolean;
  yes: boolean;
  bm25Enabled: boolean;
  bm25Language: Bm25Language;
  summaryJsonPath?: string;
  jobId?: string;
  tcvdb: {
    url: string;
    username: string;
    apiKey: string;
    database: string;
    alias: string;
    embeddingModel: string;
    timeout: number;
    caPemPath?: string;
  };
}

export interface MigrationPreflightSummary {
  pluginId: string;
  dryRun: boolean;
  layers: MigrationLayer[];
  paths: {
    pluginDataDir: string;
    sqlitePath: string;
    openclawConfigPath: string;
  };
  source: {
    l0Count: number;
    l1Count: number;
    profileCount: number;
    manifestExists: boolean;
    manifestStoreType: "sqlite" | "tcvdb" | null;
  };
  target: {
    url: string;
    username: string;
    database: string;
    alias: string;
    embeddingModel: string;
    timeout: number;
    bm25Enabled: boolean;
    bm25Language: Bm25Language;
  };
  options: {
    applyConfig: boolean;
    configBackup: boolean;
    rewriteManifest: boolean;
    failIfTargetNonempty: boolean;
    verifyCounts: boolean;
    yes: boolean;
  };
  migration?: {
    l0Migrated: number;
    l1Migrated: number;
    profileMigrated: number;
    targetL0Count: number;
    targetL1Count: number;
    targetProfileCount: number;
    configWritten: boolean;
    manifestWritten: boolean;
    manifestBackupPath?: string;
  };
}

export const DEFAULT_MIGRATION_PAGE_SIZE = 50;

export interface MigrationTargetStore {
  init(providerInfo?: unknown): Promise<StoreInitResult> | StoreInitResult;
  isDegraded(): boolean;
  close(): void;
  upsertL1(record: MemoryRecord, embedding?: Float32Array): Promise<boolean> | boolean;
  upsertL0(record: L0Record, embedding?: Float32Array): Promise<boolean> | boolean;
  upsertL1Batch?(records: MemoryRecord[]): Promise<number>;
  upsertL0Batch?(records: L0Record[]): Promise<number>;
  countL1(): Promise<number> | number;
  countL0(): Promise<number> | number;
  pullProfiles?(): Promise<ProfileRecord[]>;
  syncProfiles?(records: ProfileSyncRecord[]): Promise<void>;
}

export interface RunMigrationCliDeps {
  createTargetStore?: (options: ResolvedMigrationCliOptions) => MigrationTargetStore;
  writeMigrationPluginConfig?: typeof writeMigrationPluginConfigDefault;
  rewriteMigrationManifest?: (params: {
    dataDir: string;
    tcvdbUrl: string;
    tcvdbDatabase: string;
    tcvdbAlias?: string;
  }) => Promise<RewriteMigrationManifestResult>;
  verifyDelayMs?: number;
}

function getRequiredString(values: Record<string, string | boolean | undefined>, key: string): string {
  const value = values[key];
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`Missing required option --${key}`);
  }
  return value.trim();
}

function getOptionalString(values: Record<string, string | boolean | undefined>, key: string): string | undefined {
  const value = values[key];
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed === "" ? undefined : trimmed;
}

function resolveBooleanOption(
  values: Record<string, string | boolean | undefined>,
  key: string,
  defaultValue: boolean,
): boolean {
  const value = values[key];
  return typeof value === "boolean" ? value : defaultValue;
}

function resolveTcvdbApiKey(values: Record<string, string | boolean | undefined>): string {
  const directApiKey = getOptionalString(values, "tcvdb-api-key");
  const apiKeyEnvName = getOptionalString(values, "tcvdb-api-key-env");

  if (directApiKey && apiKeyEnvName) {
    throw new Error("Provide either --tcvdb-api-key or --tcvdb-api-key-env, not both");
  }
  if (directApiKey) {
    return directApiKey;
  }
  if (!apiKeyEnvName) {
    throw new Error("Missing required TCVDB API key input: use --tcvdb-api-key or --tcvdb-api-key-env");
  }

  const envValue = process.env[apiKeyEnvName]?.trim();
  if (!envValue) {
    throw new Error(`Environment variable ${apiKeyEnvName} is empty or not set`);
  }
  return envValue;
}

function parseLayers(rawLayers: string | undefined): MigrationLayer[] {
  if (!rawLayers) return [...ALL_MIGRATION_LAYERS];

  const values = rawLayers
    .split(",")
    .map((layer) => layer.trim())
    .filter(Boolean);

  if (values.length === 0) {
    throw new Error("--layers must include at least one layer");
  }

  const uniqueLayers = [...new Set(values)];
  const invalid = uniqueLayers.filter(
    (layer): layer is string => !ALL_MIGRATION_LAYERS.includes(layer as MigrationLayer),
  );
  if (invalid.length > 0) {
    throw new Error(`Unsupported layer(s): ${invalid.join(", ")}`);
  }

  return uniqueLayers as MigrationLayer[];
}

function parseBm25Language(rawLanguage: string | undefined): Bm25Language {
  if (!rawLanguage) return "zh";
  if (rawLanguage === "zh" || rawLanguage === "en") {
    return rawLanguage;
  }
  throw new Error(`Unsupported --bm25-language value: ${rawLanguage}`);
}

function parseTimeout(rawValue: string | undefined): number {
  if (!rawValue) return 10000;
  const timeout = Number(rawValue);
  if (!Number.isFinite(timeout) || timeout <= 0) {
    throw new Error(`Invalid --tcvdb-timeout-ms value: ${rawValue}`);
  }
  return timeout;
}

/**
 * Build a "nothing to migrate" summary — used when source data dir or sqlite
 * file doesn't exist (e.g. fresh deployment that hasn't captured any data yet).
 */
function buildEmptySummary(options: ResolvedMigrationCliOptions): MigrationPreflightSummary {
  return {
    pluginId: options.pluginId,
    dryRun: options.dryRun,
    layers: options.layers,
    paths: {
      pluginDataDir: options.pluginDataDir,
      sqlitePath: options.sqlitePath,
      openclawConfigPath: options.openclawConfigPath,
    },
    source: {
      l0Count: 0,
      l1Count: 0,
      profileCount: 0,
      manifestExists: false,
      manifestStoreType: null,
    },
    target: {
      url: options.tcvdb.url,
      username: options.tcvdb.username,
      database: options.tcvdb.database,
      alias: options.tcvdb.alias,
      embeddingModel: options.tcvdb.embeddingModel,
      timeout: options.tcvdb.timeout,
      bm25Enabled: options.bm25Enabled,
      bm25Language: options.bm25Language,
    },
    options: {
      applyConfig: options.applyConfig,
      configBackup: options.configBackup,
      rewriteManifest: options.rewriteManifest,
      failIfTargetNonempty: options.failIfTargetNonempty,
      verifyCounts: options.verifyCounts,
      yes: options.yes,
    },
  };
}

async function ensureReadablePath(filePath: string, label: string): Promise<void> {
  try {
    await fs.access(filePath);
  } catch {
    throw new Error(`${label} does not exist or is not accessible: ${filePath}`);
  }
}

async function ensureReadableDirectory(dirPath: string, label: string): Promise<void> {
  const stat = await fs.stat(dirPath).catch(() => null);
  if (!stat?.isDirectory()) {
    throw new Error(`${label} is not a directory: ${dirPath}`);
  }
}

function safeParseMetadata(raw: string): Record<string, unknown> {
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

function compactTimestamps(row: L1RecordRow): string[] {
  return [...new Set([row.timestamp_str, row.timestamp_start, row.timestamp_end].filter(Boolean))];
}

function mapL1RowToMemoryRecord(row: L1RecordRow): MemoryRecord {
  const timestamps = compactTimestamps(row);
  const fallbackIso = row.updated_time || row.created_time || row.timestamp_end || row.timestamp_str || new Date(0).toISOString();
  return {
    id: row.record_id,
    content: row.content,
    type: row.type as MemoryRecord["type"],
    priority: row.priority,
    scene_name: row.scene_name,
    source_message_ids: [],
    metadata: safeParseMetadata(row.metadata_json),
    timestamps,
    createdAt: row.created_time || fallbackIso,
    updatedAt: row.updated_time || row.created_time || fallbackIso,
    sessionKey: row.session_key || "",
    sessionId: row.session_id || "",
  };
}

function mapL0RowToRecord(row: L0RecordRow): L0Record {
  return {
    id: row.record_id,
    sessionKey: row.session_key,
    sessionId: row.session_id || "",
    role: row.role,
    messageText: row.message_text,
    recordedAt: row.recorded_at || "",
    timestamp: row.timestamp ?? 0,
  };
}

function createTargetStoreDefault(options: ResolvedMigrationCliOptions): MigrationTargetStore {
  const bm25Encoder = createBM25Encoder(
    {
      enabled: options.bm25Enabled,
      language: options.bm25Language,
    },
  );

  return new TcvdbMemoryStore({
    url: options.tcvdb.url,
    username: options.tcvdb.username,
    apiKey: options.tcvdb.apiKey,
    database: options.tcvdb.database,
    embeddingModel: options.tcvdb.embeddingModel,
    timeout: options.tcvdb.timeout,
    caPemPath: options.tcvdb.caPemPath,
    bm25Encoder,
  });
}

async function ensureTargetIsEmpty(
  options: ResolvedMigrationCliOptions,
  targetStore: MigrationTargetStore,
): Promise<void> {
  if (!options.failIfTargetNonempty) return;

  const [existingL1, existingL0, existingProfiles] = await Promise.all([
    Promise.resolve(targetStore.countL1()),
    Promise.resolve(targetStore.countL0()),
    targetStore.pullProfiles ? targetStore.pullProfiles().then((records) => records.length) : Promise.resolve(0),
  ]);

  if (existingL1 > 0 || existingL0 > 0 || existingProfiles > 0) {
    throw new Error(
      `Target store is not empty (L1=${existingL1}, L0=${existingL0}, profiles=${existingProfiles})`,
    );
  }
}

async function migrateL1Records(sourceStore: VectorStore, targetStore: MigrationTargetStore, pageSize: number): Promise<number> {
  let migrated = 0;
  let cursor = "";
  const useBatch = typeof targetStore.upsertL1Batch === "function";

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const rows = sourceStore.queryL1RecordsCursor(cursor, pageSize);
    if (rows.length === 0) break;

    const records = rows.map(mapL1RowToMemoryRecord);

    if (useBatch) {
      const count = await targetStore.upsertL1Batch!(records);
      if (count === 0) {
        throw new Error(`Failed to batch migrate L1 records (cursor=${cursor}, page=${rows.length})`);
      }
    } else {
      for (const record of records) {
        const ok = await Promise.resolve(targetStore.upsertL1(record));
        if (!ok) throw new Error(`Failed to migrate L1 record ${record.id}`);
      }
    }

    migrated += rows.length;
    cursor = rows[rows.length - 1].record_id;
    log(`L1: 已迁移 ${migrated} 条...`);

    if (rows.length < pageSize) break;
  }

  log(`L1: 迁移完成，共 ${migrated} 条`);
  return migrated;
}

async function migrateL0Records(sourceStore: VectorStore, targetStore: MigrationTargetStore, pageSize: number): Promise<number> {
  let migrated = 0;
  let cursor = "";
  const useBatch = typeof targetStore.upsertL0Batch === "function";

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const rows = sourceStore.queryL0RecordsCursor(cursor, pageSize);
    if (rows.length === 0) break;

    const records = rows.map(mapL0RowToRecord);

    if (useBatch) {
      const count = await targetStore.upsertL0Batch!(records);
      if (count === 0) {
        throw new Error(`Failed to batch migrate L0 records (cursor=${cursor}, page=${rows.length})`);
      }
    } else {
      for (const record of records) {
        const ok = await Promise.resolve(targetStore.upsertL0(record));
        if (!ok) throw new Error(`Failed to migrate L0 record ${record.id}`);
      }
    }

    migrated += rows.length;
    cursor = rows[rows.length - 1].record_id;
    log(`L0: 已迁移 ${migrated} 条...`);

    if (rows.length < pageSize) break;
  }

  log(`L0: 迁移完成，共 ${migrated} 条`);
  return migrated;
}

async function migrateProfiles(
  pluginDataDir: string,
  targetStore: MigrationTargetStore,
): Promise<number> {
  const profiles = await listLocalProfiles(pluginDataDir);
  if (profiles.length === 0) {
    log("Profiles: 无本地 profile，跳过");
    return 0;
  }
  if (!targetStore.syncProfiles) {
    throw new Error("Target store does not support profile sync");
  }
  log(`Profiles: 发现 ${profiles.length} 个本地 profile，开始同步...`);
  await targetStore.syncProfiles(profiles);
  log(`Profiles: 同步完成，共 ${profiles.length} 个`);
  return profiles.length;
}

async function verifyMigratedCounts(
  summary: MigrationPreflightSummary,
  targetStore: MigrationTargetStore,
  delayMs: number,
): Promise<{ l1Count: number; l0Count: number; profileCount: number }> {
  if (delayMs > 0) {
    log(`等待 ${Math.round(delayMs / 1000)} 秒让远端数据落盘...`);
    await sleep(delayMs);
  }
  log("开始校验迁移数量...");

  const [l1Count, l0Count, profileCount] = await Promise.all([
    Promise.resolve(targetStore.countL1()),
    Promise.resolve(targetStore.countL0()),
    targetStore.pullProfiles ? targetStore.pullProfiles().then((records) => records.length) : Promise.resolve(0),
  ]);

  log(`校验结果: L1=${l1Count}/${summary.source.l1Count}, L0=${l0Count}/${summary.source.l0Count}, Profiles=${profileCount}/${summary.source.profileCount}`);

  if (l1Count !== summary.source.l1Count) {
    throw new Error(`L1 count verification failed: source=${summary.source.l1Count}, target=${l1Count}`);
  }
  if (l0Count !== summary.source.l0Count) {
    throw new Error(`L0 count verification failed: source=${summary.source.l0Count}, target=${l0Count}`);
  }
  if (profileCount !== summary.source.profileCount) {
    throw new Error(
      `Profile count verification failed: source=${summary.source.profileCount}, target=${profileCount}`,
    );
  }

  log("校验通过");
  return { l1Count, l0Count, profileCount };
}

async function writeSummaryJson(
  summaryJsonPath: string | undefined,
  summary: MigrationPreflightSummary,
): Promise<void> {
  if (!summaryJsonPath) return;
  await fs.mkdir(path.dirname(summaryJsonPath), { recursive: true });
  await fs.writeFile(summaryJsonPath, `${JSON.stringify(summary, null, 2)}\n`, "utf-8");
}

function printUsageAndExit(): never {
  const text = `
SQLite → 腾讯云向量数据库迁移工具

Usage:
  migrate-sqlite-to-tcvdb [options]

Required:
  --plugin-data-dir <path>       插件数据目录路径
  --openclaw-config-path <path>  openclaw.json 配置文件路径
  --tcvdb-url <url>              TCVDB 服务地址
  --tcvdb-username <name>        TCVDB 用户名
  --tcvdb-database <name>        TCVDB 数据库名
  --tcvdb-embedding-model <name> Embedding 模型名称
  --tcvdb-api-key <key>          TCVDB API 密钥（明文，与 --tcvdb-api-key-env 二选一）
  --tcvdb-api-key-env <var>      包含 API 密钥的环境变量名

Optional:
  --sqlite-path <path>           SQLite 数据库路径（默认: <plugin-data-dir>/vectors.db）
  --plugin-id <id>               写入配置时使用的插件 ID（默认: memory-tencentdb）
  --layers <l0,l1,l2,l3>         要迁移的层，逗号分隔（默认: l0,l1,l2,l3）
  --tcvdb-alias <name>           用户自定义别名
  --tcvdb-timeout-ms <ms>        请求超时（默认: 10000）
  --tcvdb-ca-pem <path>          CA 证书 PEM 文件路径（HTTPS 连接时使用）
  --bm25-language <zh|en>        BM25 分词语言（默认: zh）
  --summary-json-path <path>     将迁移摘要写入此文件
  --job-id <id>                  迁移任务 ID（用于追踪）

Flags:
  --dry-run                      仅预览，不执行写入
  --yes                          跳过交互确认
  --no-apply-config              不自动更新 openclaw.json
  --no-config-backup             写入配置前不备份
  --no-rewrite-manifest          不更新 manifest.json
  --no-fail-if-target-nonempty   目标库非空时不中止
  --no-verify-counts             迁移后不校验记录数
  --no-bm25-enabled              禁用 BM25 稀疏向量

  -h, --help                     显示此帮助信息

Examples:
  # 预检模式
  migrate-sqlite-to-tcvdb \\
    --plugin-data-dir ~/.openclaw/memory-tdai \\
    --openclaw-config-path ~/.openclaw/openclaw.json \\
    --tcvdb-url http://127.0.0.1:80 --tcvdb-username root \\
    --tcvdb-api-key-env TCVDB_API_KEY \\
    --tcvdb-database agent_memory_prod \\
    --tcvdb-embedding-model bge-large-zh \\
    --dry-run

  # 正式迁移
  migrate-sqlite-to-tcvdb \\
    --plugin-data-dir ~/.openclaw/memory-tdai \\
    --openclaw-config-path ~/.openclaw/openclaw.json \\
    --tcvdb-url http://127.0.0.1:80 --tcvdb-username root \\
    --tcvdb-api-key-env TCVDB_API_KEY \\
    --tcvdb-database agent_memory_prod \\
    --tcvdb-embedding-model bge-large-zh \\
    --yes
`.trimStart();
  process.stdout.write(text);
  process.exit(0);
}

export function resolveMigrationCliOptions(argv: string[]): ResolvedMigrationCliOptions {
  const { values } = parseArgs({
    args: argv,
    strict: true,
    allowPositionals: false,
    allowNegative: true,
    options: {
      help: { type: "boolean", short: "h" },
      "plugin-data-dir": { type: "string" },
      "sqlite-path": { type: "string" },
      "openclaw-config-path": { type: "string" },
      "plugin-id": { type: "string" },
      layers: { type: "string" },
      "apply-config": { type: "boolean" },
      "config-backup": { type: "boolean" },
      "rewrite-manifest": { type: "boolean" },
      "fail-if-target-nonempty": { type: "boolean" },
      "verify-counts": { type: "boolean" },
      "dry-run": { type: "boolean" },
      yes: { type: "boolean" },
      "bm25-enabled": { type: "boolean" },
      "bm25-language": { type: "string" },
      "summary-json-path": { type: "string" },
      "job-id": { type: "string" },
      "tcvdb-url": { type: "string" },
      "tcvdb-username": { type: "string" },
      "tcvdb-api-key": { type: "string" },
      "tcvdb-api-key-env": { type: "string" },
      "tcvdb-database": { type: "string" },
      "tcvdb-alias": { type: "string" },
      "tcvdb-embedding-model": { type: "string" },
      "tcvdb-timeout-ms": { type: "string" },
      "tcvdb-ca-pem": { type: "string" },
    },
  });

  if (values.help) {
    printUsageAndExit();
  }

  const pluginDataDir = path.resolve(getRequiredString(values, "plugin-data-dir"));
  const sqlitePath = path.resolve(
    getOptionalString(values, "sqlite-path") ?? path.join(pluginDataDir, "vectors.db"),
  );
  const summaryJsonPath = getOptionalString(values, "summary-json-path");

  return {
    pluginDataDir,
    sqlitePath,
    openclawConfigPath: path.resolve(getRequiredString(values, "openclaw-config-path")),
    pluginId: getOptionalString(values, "plugin-id") ?? DEFAULT_MIGRATION_PLUGIN_ID,
    layers: parseLayers(getOptionalString(values, "layers")),
    applyConfig: resolveBooleanOption(values, "apply-config", true),
    configBackup: resolveBooleanOption(values, "config-backup", true),
    rewriteManifest: resolveBooleanOption(values, "rewrite-manifest", true),
    failIfTargetNonempty: resolveBooleanOption(values, "fail-if-target-nonempty", true),
    verifyCounts: resolveBooleanOption(values, "verify-counts", true),
    dryRun: resolveBooleanOption(values, "dry-run", false),
    yes: resolveBooleanOption(values, "yes", false),
    bm25Enabled: resolveBooleanOption(values, "bm25-enabled", true),
    bm25Language: parseBm25Language(getOptionalString(values, "bm25-language")),
    summaryJsonPath: summaryJsonPath ? path.resolve(summaryJsonPath) : undefined,
    jobId: getOptionalString(values, "job-id"),
    tcvdb: {
      url: getRequiredString(values, "tcvdb-url"),
      username: getRequiredString(values, "tcvdb-username"),
      apiKey: resolveTcvdbApiKey(values),
      database: getRequiredString(values, "tcvdb-database"),
      alias: getOptionalString(values, "tcvdb-alias") ?? "",
      embeddingModel: getRequiredString(values, "tcvdb-embedding-model"),
      timeout: parseTimeout(getOptionalString(values, "tcvdb-timeout-ms")),
      caPemPath: getOptionalString(values, "tcvdb-ca-pem"),
    },
  };
}

export async function collectMigrationPreflight(
  options: ResolvedMigrationCliOptions,
): Promise<MigrationPreflightSummary> {
  // ── 优雅处理"数据目录 / sqlite 不存在"的场景 ──
  // 如果源数据路径不存在（全新部署、尚未有任何 capture），不报错——
  // 返回"全零"summary，让调用方判断是否跳过迁移。
  const dirExists = await fs.stat(options.pluginDataDir).then(s => s.isDirectory()).catch(() => false);
  if (!dirExists) {
    log(`plugin data directory 不存在，无需迁移: ${options.pluginDataDir}`);
    return buildEmptySummary(options);
  }
  const sqliteExists = await fs.access(options.sqlitePath).then(() => true).catch(() => false);
  if (!sqliteExists) {
    log(`sqlite database 不存在，无需迁移: ${options.sqlitePath}`);
    return buildEmptySummary(options);
  }
  await ensureReadablePath(options.openclawConfigPath, "OpenClaw config file");

  const store = new VectorStore(options.sqlitePath, 0);
  const initResult = store.init();
  if (store.isDegraded()) {
    store.close();
    throw new Error(`Failed to open sqlite store for migration preflight: ${initResult.reason ?? "unknown error"}`);
  }

  try {
    const [profiles, manifest] = await Promise.all([
      listLocalProfiles(options.pluginDataDir),
      Promise.resolve(readManifest(options.pluginDataDir)),
    ]);

    return {
      pluginId: options.pluginId,
      dryRun: options.dryRun,
      layers: options.layers,
      paths: {
        pluginDataDir: options.pluginDataDir,
        sqlitePath: options.sqlitePath,
        openclawConfigPath: options.openclawConfigPath,
      },
      source: {
        l0Count: store.countL0(),
        l1Count: store.countL1(),
        profileCount: profiles.length,
        manifestExists: manifest !== null,
        manifestStoreType: manifest?.store.type ?? null,
      },
      target: {
        url: options.tcvdb.url,
        username: options.tcvdb.username,
        database: options.tcvdb.database,
        alias: options.tcvdb.alias,
        embeddingModel: options.tcvdb.embeddingModel,
        timeout: options.tcvdb.timeout,
        bm25Enabled: options.bm25Enabled,
        bm25Language: options.bm25Language,
      },
      options: {
        applyConfig: options.applyConfig,
        configBackup: options.configBackup,
        rewriteManifest: options.rewriteManifest,
        failIfTargetNonempty: options.failIfTargetNonempty,
        verifyCounts: options.verifyCounts,
        yes: options.yes,
      },
    };
  } finally {
    store.close();
  }
}

export async function runMigrationCli(
  argv: string[],
  deps: RunMigrationCliDeps = {},
): Promise<MigrationPreflightSummary> {
  const options = resolveMigrationCliOptions(argv);
  log("开始预检...");
  const summary = await collectMigrationPreflight(options);
  log(`预检完成: 源数据 L1=${summary.source.l1Count}, L0=${summary.source.l0Count}, Profiles=${summary.source.profileCount}`);
  log(`目标: ${summary.target.url} / ${summary.target.database}`);

  const hasSourceData = summary.source.l0Count > 0 || summary.source.l1Count > 0 || summary.source.profileCount > 0;

  if (!hasSourceData) {
    log("源数据为空，跳过数据迁移。");
  }

  if (options.dryRun) {
    log("预检模式 (dry-run)，不执行写入");
    await writeSummaryJson(options.summaryJsonPath, summary);
    return summary;
  }

  const createTargetStore = deps.createTargetStore ?? createTargetStoreDefault;
  const writeMigrationPluginConfig = deps.writeMigrationPluginConfig ?? writeMigrationPluginConfigDefault;
  const rewriteMigrationManifest = deps.rewriteMigrationManifest ?? (async (params) =>
    await rewriteMigrationManifestDefault(params));

  const migration = {
    l1Migrated: 0,
    l0Migrated: 0,
    profileMigrated: 0,
    targetL1Count: 0,
    targetL0Count: 0,
    targetProfileCount: 0,
    configWritten: false,
    manifestWritten: false,
    manifestBackupPath: undefined as string | undefined,
  };

  if (hasSourceData) {
    log("打开 SQLite 源库...");
    const sourceStore = new VectorStore(options.sqlitePath, 0);
    const sourceInitResult = sourceStore.init();
    if (sourceStore.isDegraded()) {
      sourceStore.close();
      throw new Error(`Failed to reopen sqlite source store: ${sourceInitResult.reason ?? "unknown error"}`);
    }

    log("初始化目标库...");
    const targetStore = createTargetStore(options);

    try {
      await Promise.resolve(targetStore.init());
      if (targetStore.isDegraded()) {
        throw new Error("Target store entered degraded mode during initialization");
      }

      await ensureTargetIsEmpty(options, targetStore);

      const pageSize = DEFAULT_MIGRATION_PAGE_SIZE;
      log(`分页大小: ${pageSize} 条/批`);

      migration.l1Migrated = options.layers.includes("l1") ? await migrateL1Records(sourceStore, targetStore, pageSize) : 0;
      migration.l0Migrated = options.layers.includes("l0") ? await migrateL0Records(sourceStore, targetStore, pageSize) : 0;
      migration.profileMigrated = options.layers.includes("l2") || options.layers.includes("l3")
        ? await migrateProfiles(options.pluginDataDir, targetStore)
        : 0;

      const verifyDelayMs = deps.verifyDelayMs ?? 10_000;

      const verifiedCounts = options.verifyCounts
        ? await verifyMigratedCounts(summary, targetStore, verifyDelayMs)
        : {
            l1Count: await Promise.resolve(targetStore.countL1()),
            l0Count: await Promise.resolve(targetStore.countL0()),
            profileCount: targetStore.pullProfiles ? (await targetStore.pullProfiles()).length : 0,
          };

      migration.targetL1Count = verifiedCounts.l1Count;
      migration.targetL0Count = verifiedCounts.l0Count;
      migration.targetProfileCount = verifiedCounts.profileCount;
    } finally {
      sourceStore.close();
      targetStore.close();
    }
  }

  if (options.applyConfig) {
    log(`写入配置到 ${options.openclawConfigPath} ...`);
    await writeMigrationPluginConfig({
      configPath: options.openclawConfigPath,
      pluginId: options.pluginId,
      tcvdb: {
        url: options.tcvdb.url,
        username: options.tcvdb.username,
        apiKey: options.tcvdb.apiKey,
        database: options.tcvdb.database,
        alias: options.tcvdb.alias,
        embeddingModel: options.tcvdb.embeddingModel,
        timeout: options.tcvdb.timeout,
      },
      bm25: {
        enabled: options.bm25Enabled,
        language: options.bm25Language,
      },
    });
    migration.configWritten = true;
    log("配置写入完成");
  }

  if (options.rewriteManifest) {
    log("更新 manifest...");
    const manifestResult = await rewriteMigrationManifest({
      dataDir: options.pluginDataDir,
      tcvdbUrl: options.tcvdb.url,
      tcvdbDatabase: options.tcvdb.database,
      tcvdbAlias: options.tcvdb.alias || undefined,
    });
    migration.manifestWritten = manifestResult.created || manifestResult.updated;
    migration.manifestBackupPath = manifestResult.backupPath;
    log(`Manifest ${manifestResult.created ? "创建" : "更新"}完成${manifestResult.backupPath ? `，备份: ${manifestResult.backupPath}` : ""}`);
  }

  summary.migration = {
    l1Migrated: migration.l1Migrated,
    l0Migrated: migration.l0Migrated,
    profileMigrated: migration.profileMigrated,
    targetL1Count: migration.targetL1Count,
    targetL0Count: migration.targetL0Count,
    targetProfileCount: migration.targetProfileCount,
    configWritten: migration.configWritten,
    manifestWritten: migration.manifestWritten,
    manifestBackupPath: migration.manifestBackupPath,
  };

  await writeSummaryJson(options.summaryJsonPath, summary);
  log("迁移全部完成!");
  return summary;
}
