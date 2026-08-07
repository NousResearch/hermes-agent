/**
 * SQLite 实现的 IMetadataStore。
 *
 * 对应设计文档 §6.1 / §6.2。基于 Node 内置 `node:sqlite`（DatabaseSync，Node 22+）。
 * 单连接隐式串行 + 显式 BEGIN/COMMIT 保证复合写入原子性。
 *
 * 表前缀 `meta_`，与内核已有 entity_* / l0 / l1 表隔离。
 */

import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
import path from "node:path";
import type { DatabaseSync, SQLInputValue } from "node:sqlite";
import { mapTeamMemberWithProfile } from "./team-member-view.js";
import { generateId, generateRelationId, ID_PREFIX } from "../utils/id-generator.js";
import {
  isSqliteRelationIdCollision,
  runWithGeneratedRelationId,
  RELATION_ID_RETRY_LIMIT,
} from "./relation-id-insert.js";
import { generateUserKey } from "../utils/crypto.js";
import { isUserKeyExpired } from "../utils/user-key.js";
import type {
  UserEntity,
  UserKeyEntity,
  TeamEntity,
  TeamMemberEntity,
  TeamMemberView,
  AgentEntity,
  TaskEntity,
  TaskAgentEntity,
  ParticipationLogEntity,
  AppendParticipationLogInput,
  ParticipationLogFilter,
  AssetEntity,
  FixedAssetBindingEntity,
  AclEntity,
  CreateUserInput,
  CreateUserKeyInput,
  CreateTeamInput,
  AddTeamMemberInput,
  CreateAgentInput,
  CreateTaskInput,
  CreateAssetInput,
  FixedAssetBindingInput,
  GrantAclInput,
  AgentFilter,
  TaskFilter,
  AssetFilter,
  BatchDeleteResult,
  ListPage,
  PaginationParams,
  InstanceUserListFilter,
  AgentFixedAssetCountRow,
  AssetType,
  ConfigParamEntity,
  UpsertConfigParamInput,
  ListConfigParamsFilter,
} from "../types.js";
import { DEFAULT_PAGINATION } from "../pagination.js";
import { buildChatMemoryAssetId } from "../utils/chat-memory-asset.js";

const require = createRequire(import.meta.url);
function requireNodeSqlite(): typeof import("node:sqlite") {
  return require("node:sqlite") as typeof import("node:sqlite");
}

function nowIso(): string {
  return new Date().toISOString();
}

const PK_RETRY_LIMIT = 3;

/** Returns true if the error is a SQLite UNIQUE constraint failure on a generated primary key column. */
function isPkCollision(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return /UNIQUE constraint failed: meta_\w+\.(user_id|team_id|agent_id|task_id|asset_id|acl_id|key_id)\b/.test(msg);
}

function isStorePkCollision(err: unknown): boolean {
  return isPkCollision(err) || isSqliteRelationIdCollision(err);
}

type Row = Record<string, SQLInputValue>;

export class SqliteMetadataStore implements IMetadataStore {
  private db!: DatabaseSync;
  private readonly dbPath: string;
  private initialized = false;

  /** @param dbPath 文件路径或 ":memory:"。 */
  constructor(dbPath: string) {
    this.dbPath = dbPath;
  }

  init(): void {
    if (this.initialized) return;
    const { DatabaseSync: DbSync } = requireNodeSqlite();
    if (this.dbPath !== ":memory:") {
      mkdirSync(path.dirname(this.dbPath), { recursive: true });
    }
    this.db = new DbSync(this.dbPath);
    this.db.exec("PRAGMA busy_timeout = 5000");
    this.db.exec("PRAGMA foreign_keys = ON");
    this.createSchema();
    this.initialized = true;
  }

  close(): void {
    if (this.initialized) {
      this.db.close();
      this.initialized = false;
    }
  }

  // ============================================================
  // Schema
  // ============================================================
  private createSchema(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS meta_users (
        user_id TEXT PRIMARY KEY,
        password TEXT,
        auth_provider TEXT NOT NULL,
        external_id TEXT NOT NULL,
        username TEXT NOT NULL,
        display_name TEXT,
        email TEXT,
        raw_profile_json TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL DEFAULT 'active',
        user_type TEXT NOT NULL DEFAULT 'normal',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
      );
      CREATE UNIQUE INDEX IF NOT EXISTS ux_meta_users_system_admin ON meta_users(user_type) WHERE user_type = 'system_admin';
      CREATE INDEX IF NOT EXISTS idx_meta_users_auth_username ON meta_users(auth_provider, username);
      CREATE INDEX IF NOT EXISTS idx_meta_users_auth_external ON meta_users(auth_provider, external_id);
      CREATE INDEX IF NOT EXISTS idx_meta_users_email ON meta_users(email) WHERE email IS NOT NULL;
      CREATE TABLE IF NOT EXISTS meta_user_keys (
        key_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        key_value TEXT NOT NULL UNIQUE,
        name TEXT,
        status TEXT NOT NULL DEFAULT 'active',
        is_default INTEGER NOT NULL DEFAULT 0,
        last_used_at TEXT,
        expires_at TEXT,
        created_at TEXT NOT NULL,
        revoked_at TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}'
      );
      CREATE INDEX IF NOT EXISTS idx_meta_user_keys_user ON meta_user_keys(user_id, status);
      CREATE TABLE IF NOT EXISTS meta_teams (
        team_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        owner_user_id TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
      );
      CREATE TABLE IF NOT EXISTS meta_team_members (
        id TEXT PRIMARY KEY,
        team_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'member',
        joined_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active',
        UNIQUE(team_id, user_id)
      );
      CREATE TABLE IF NOT EXISTS meta_agents (
        agent_id TEXT PRIMARY KEY,
        team_id TEXT NOT NULL,
        owner_user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT,
        prompt TEXT,
        visibility TEXT NOT NULL DEFAULT 'team',
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
      );
      CREATE INDEX IF NOT EXISTS idx_meta_agents_team_status ON meta_agents(team_id, status, created_at DESC);
      CREATE TABLE IF NOT EXISTS meta_tasks (
        task_id TEXT PRIMARY KEY,
        team_id TEXT NOT NULL,
        creator_user_id TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT,
        source_type TEXT NOT NULL DEFAULT 'manual',
        source_url TEXT,
        status TEXT NOT NULL DEFAULT 'running',
        auto_assign_floating_assets INTEGER NOT NULL DEFAULT 0,
        risk_level TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
      );
      CREATE INDEX IF NOT EXISTS idx_meta_tasks_team_status ON meta_tasks(team_id, status, created_at DESC);
      CREATE TABLE IF NOT EXISTS meta_task_agents (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        role_in_task TEXT,
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        UNIQUE(task_id, agent_id)
      );
      CREATE TABLE IF NOT EXISTS meta_participation_logs (
        id TEXT PRIMARY KEY,
        team_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'unknown',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS meta_assets (
        asset_id TEXT PRIMARY KEY,
        team_id TEXT NOT NULL,
        asset_type TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT,
        owner_user_id TEXT NOT NULL,
        source_type TEXT NOT NULL,
        source_ref TEXT,
        version INTEGER NOT NULL DEFAULT 1,
        visibility TEXT NOT NULL DEFAULT 'team',
        status TEXT NOT NULL DEFAULT 'draft',
        confidence REAL,
        expires_at TEXT,
        last_used_at TEXT,
        usage_count INTEGER NOT NULL DEFAULT 0,
        content_ref TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
      );
      CREATE INDEX IF NOT EXISTS idx_meta_assets_team_status ON meta_assets(team_id, status, created_at DESC);
      CREATE TABLE IF NOT EXISTS meta_agent_fixed_assets (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        asset_id TEXT NOT NULL,
        asset_type TEXT NOT NULL,
        injection_mode TEXT NOT NULL DEFAULT 'summary',
        priority INTEGER NOT NULL DEFAULT 50,
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(agent_id, asset_id)
      );
      CREATE TABLE IF NOT EXISTS meta_asset_acl (
        id TEXT PRIMARY KEY,
        asset_id TEXT NOT NULL,
        subject_type TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        permission TEXT NOT NULL,
        effect TEXT NOT NULL DEFAULT 'allow',
        granted_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(asset_id, subject_type, subject_id, permission)
      );
      CREATE INDEX IF NOT EXISTS idx_meta_users_created ON meta_users(created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_meta_user_keys_user_created ON meta_user_keys(user_id, created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_meta_teams_created ON meta_teams(created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_meta_members_user_status ON meta_team_members(user_id, status);
      CREATE INDEX IF NOT EXISTS idx_meta_members_team_status_joined ON meta_team_members(team_id, status, joined_at DESC);
      CREATE INDEX IF NOT EXISTS idx_meta_agents_owner_status_created ON meta_agents(owner_user_id, status, created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_meta_tasks_creator_status_created ON meta_tasks(creator_user_id, status, created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_meta_task_agents_task_status_created ON meta_task_agents(task_id, status, created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_meta_pl_team_created ON meta_participation_logs(team_id, created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_meta_pl_team_task_agent_created ON meta_participation_logs(team_id, task_id, agent_id, created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_meta_pl_team_user_created ON meta_participation_logs(team_id, user_id, created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_meta_pl_team_dims_created ON meta_participation_logs(team_id, task_id, agent_id, user_id, created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_meta_fixed_agent_prio_created ON meta_agent_fixed_assets(agent_id, priority DESC, created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_meta_acl_asset_created ON meta_asset_acl(asset_id, created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_meta_acl_subject_created ON meta_asset_acl(subject_type, subject_id, created_at DESC);

      CREATE TABLE IF NOT EXISTS meta_config_params (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scope TEXT NOT NULL CHECK (scope IN ('global', 'user')),
        user_id TEXT,
        module TEXT NOT NULL,
        param_name TEXT NOT NULL,
        param_value TEXT NOT NULL,
        description TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK (
          (scope = 'global' AND user_id IS NULL) OR
          (scope = 'user' AND user_id IS NOT NULL)
        )
      );
      CREATE UNIQUE INDEX IF NOT EXISTS ux_meta_config_params_global
        ON meta_config_params(module, param_name) WHERE scope = 'global';
      CREATE UNIQUE INDEX IF NOT EXISTS ux_meta_config_params_user
        ON meta_config_params(user_id, module, param_name) WHERE scope = 'user';
      CREATE INDEX IF NOT EXISTS idx_meta_config_params_module
        ON meta_config_params(module);
    `);
    this.migrateUserTypeColumn();
    this.migrateLegacyUserKeys();
  }

  private migrateUserTypeColumn(): void {
    const hasCol = this.all<{ name: string }>(
      "SELECT name FROM pragma_table_info('meta_users') WHERE name = 'user_type'",
    );
    if (hasCol.length === 0) {
      try {
        this.db.exec(`ALTER TABLE meta_users ADD COLUMN user_type TEXT NOT NULL DEFAULT 'normal'`);
      } catch {
        /* column may exist from concurrent init */
      }
    }
    this.db.exec(`DROP INDEX IF EXISTS ux_meta_users_single_system_admin`);
  }

  /** 存量库：若 meta_users 仍有 user_key 列，回填 meta_user_keys 后不再写入该列。 */
  private migrateLegacyUserKeys(): void {
    const hasUserKeyCol = this.all<{ name: string }>(
      "SELECT name FROM pragma_table_info('meta_users') WHERE name = 'user_key'",
    );
    if (hasUserKeyCol.length === 0) return;

    const rows = this.all(
      `SELECT u.user_id, u.user_key, u.created_at FROM meta_users u
       WHERE u.user_key IS NOT NULL AND u.user_key != ''
         AND NOT EXISTS (SELECT 1 FROM meta_user_keys k WHERE k.user_id = u.user_id)`,
    );
    for (const row of rows) {
      this.insertUserKeyRow({
        user_id: String(row.user_id),
        key_value: String(row.user_key),
        is_default: true,
        created_at: String(row.created_at),
      });
    }
  }

  private insertUserKeyRow(input: {
    user_id: string;
    key_value: string;
    name?: string | null;
    is_default?: boolean;
    expires_at?: string | null;
    created_at?: string;
    metadata_json?: string;
  }): UserKeyEntity {
    const now = input.created_at ?? nowIso();
    for (let attempt = 0; attempt < PK_RETRY_LIMIT; attempt++) {
      const keyId = generateId(ID_PREFIX.userKey);
      try {
        this.run(
          `INSERT INTO meta_user_keys
            (key_id, user_id, key_value, name, status, is_default, last_used_at, expires_at, created_at, revoked_at, metadata_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)`,
          keyId,
          input.user_id,
          input.key_value,
          input.name ?? null,
          "active",
          input.is_default ? 1 : 0,
          null,
          input.expires_at ?? null,
          now,
          null,
          input.metadata_json ?? "{}",
        );
        return this.getUserKeyById(keyId)!;
      } catch (err) {
        if (isPkCollision(err)) continue;
        throw err;
      }
    }
    throw new Error("user key PK collision after max retries");
  }

  private mapUserKey(row: Row | null | undefined): UserKeyEntity | null {
    if (!row) return null;
    return {
      key_id: String(row.key_id),
      user_id: String(row.user_id),
      key_value: String(row.key_value),
      name: row.name != null ? String(row.name) : null,
      status: String(row.status) as UserKeyEntity["status"],
      is_default: Number(row.is_default) === 1,
      last_used_at: row.last_used_at != null ? String(row.last_used_at) : null,
      expires_at: row.expires_at != null ? String(row.expires_at) : null,
      created_at: String(row.created_at),
      revoked_at: row.revoked_at != null ? String(row.revoked_at) : null,
      metadata_json: String(row.metadata_json ?? "{}"),
    };
  }

  private tx<T>(fn: () => T): T {
    this.db.exec("BEGIN");
    try {
      const result = fn();
      this.db.exec("COMMIT");
      return result;
    } catch (e) {
      try {
        this.db.exec("ROLLBACK");
      } catch {
        /* ignore */
      }
      throw e;
    }
  }

  private get<T = Row>(sql: string, ...params: SQLInputValue[]): T | null {
    const row = this.db.prepare(sql).get(...params) as T | undefined;
    return row ?? null;
  }

  private all<T = Row>(sql: string, ...params: SQLInputValue[]): T[] {
    return this.db.prepare(sql).all(...params) as T[];
  }

  private run(sql: string, ...params: SQLInputValue[]): void {
    this.db.prepare(sql).run(...params);
  }

  private selectList<T>(
    countSql: string,
    countParams: SQLInputValue[],
    dataSql: string,
    dataParams: SQLInputValue[],
    pagination: PaginationParams | null | undefined,
    mapper: (row: Row) => T | null,
  ): ListPage<T> {
    const totalRow = this.get<{ c: number }>(countSql, ...countParams);
    const total = Number(totalRow?.c ?? 0);
    const p = pagination ?? DEFAULT_PAGINATION;
    const rows = this.all(`${dataSql} LIMIT ? OFFSET ?`, ...dataParams, p.limit, p.offset);
    const items: T[] = [];
    for (const r of rows) {
      const mapped = mapper(r);
      if (mapped) items.push(mapped);
    }
    return { items, total };
  }

  // ============================================================
  // User
  // ============================================================
  createUser(input: CreateUserInput): UserEntity {
    const now = nowIso();
    const defaultKeyValue = input.default_key_value ?? generateUserKey();
    for (let attempt = 0; attempt < PK_RETRY_LIMIT; attempt++) {
      const userId = input.user_id ?? generateId(ID_PREFIX.user);
      try {
        this.tx(() => {
          this.run(
            `INSERT INTO meta_users
              (user_id, password, auth_provider, external_id, username,
               display_name, email, raw_profile_json, status, user_type, created_at, updated_at, metadata_json)
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)`,
            userId,
            input.password ?? null,
            input.auth_provider as string,
            input.external_id as string,
            input.username as string,
            input.display_name ?? null,
            input.email ?? null,
            input.raw_profile_json ?? "{}",
            input.status ?? "active",
            input.user_type ?? "normal",
            now,
            now,
            input.metadata_json ?? "{}",
          );
          this.insertUserKeyRow({
            user_id: userId,
            key_value: defaultKeyValue,
            is_default: true,
            created_at: now,
          });
        });
        return this.getUserById(userId)!;
      } catch (err) {
        if (isPkCollision(err) && !input.user_id) continue;
        throw err;
      }
    }
    throw new Error("PK collision after max retries");
  }

  getUserById(userId: string): UserEntity | null {
    return this.mapUser(this.get("SELECT * FROM meta_users WHERE user_id = ?", userId));
  }

  getUserByKey(userKey: string): UserEntity | null {
    const keyRow = this.mapUserKey(
      this.get("SELECT * FROM meta_user_keys WHERE key_value = ? AND status = 'active'", userKey),
    );
    if (!keyRow || isUserKeyExpired(keyRow.expires_at)) return null;
    this.touchUserKeyUsage(keyRow.key_id);
    return this.getUserById(keyRow.user_id);
  }

  getDefaultUserKey(userId: string): UserKeyEntity | null {
    return this.mapUserKey(
      this.get(
        "SELECT * FROM meta_user_keys WHERE user_id = ? AND is_default = 1 AND status = 'active' LIMIT 1",
        userId,
      ),
    );
  }

  getUserByUsername(authProvider: string, username: string): UserEntity | null {
    return this.mapUser(
      this.get("SELECT * FROM meta_users WHERE auth_provider = ? AND username = ?", authProvider, username),
    );
  }

  getUserByEmail(email: string): UserEntity | null {
    return this.mapUser(this.get("SELECT * FROM meta_users WHERE email = ?", email));
  }

  getUserByExternalId(authProvider: string, externalId: string): UserEntity | null {
    return this.mapUser(
      this.get(
        "SELECT * FROM meta_users WHERE auth_provider = ? AND external_id = ?",
        authProvider,
        externalId,
      ),
    );
  }

  updateUser(userId: string, patch: Partial<UserEntity>): UserEntity | null {
    const allowed = ["password", "display_name", "email", "raw_profile_json", "status", "metadata_json", "username"] as const;
    this.applyUpdate("meta_users", "user_id", userId, allowed, patch);
    return this.getUserById(userId);
  }

  deleteUsers(userIds: string[]): BatchDeleteResult {
    const result = this.batchDelete("meta_users", "user_id", userIds);
    if (result.deleted_ids.length > 0) {
      const ph = result.deleted_ids.map(() => "?").join(",");
      this.run(`DELETE FROM meta_user_keys WHERE user_id IN (${ph})`, ...result.deleted_ids);
      this.run(`DELETE FROM meta_team_members WHERE user_id IN (${ph})`, ...result.deleted_ids);
      this.run(`DELETE FROM meta_asset_acl WHERE subject_type = 'user' AND subject_id IN (${ph})`, ...result.deleted_ids);
    }
    return result;
  }

  listUsersByTeam(
    teamId: string,
    pagination?: PaginationParams | null,
    filter?: InstanceUserListFilter,
  ): ListPage<UserEntity> {
    let base =
      "FROM meta_users u JOIN meta_team_members m ON m.user_id = u.user_id WHERE m.team_id = ? AND m.status = 'active'";
    const params: SQLInputValue[] = [teamId];
    if (filter?.user_ids?.length) {
      base += ` AND u.user_id IN (${filter.user_ids.map(() => "?").join(",")})`;
      params.push(...filter.user_ids);
    }
    if (filter?.username) {
      base += " AND u.username = ?";
      params.push(filter.username);
    }
    return this.selectList(
      `SELECT COUNT(*) AS c ${base}`,
      params,
      `SELECT u.* ${base} ORDER BY u.created_at DESC`,
      params,
      pagination,
      (r) => this.mapUser(r),
    );
  }

  listUsers(
    pagination?: PaginationParams | null,
    filter?: InstanceUserListFilter,
  ): ListPage<UserEntity> {
    let where = "WHERE 1=1";
    const params: SQLInputValue[] = [];
    if (filter?.status) {
      where += " AND status = ?";
      params.push(filter.status);
    }
    if (filter?.user_type) {
      where += " AND user_type = ?";
      params.push(filter.user_type);
    }
    if (filter?.user_ids?.length) {
      where += ` AND user_id IN (${filter.user_ids.map(() => "?").join(",")})`;
      params.push(...filter.user_ids);
    }
    if (filter?.username) {
      where += " AND username = ?";
      params.push(filter.username);
    }
    return this.selectList(
      `SELECT COUNT(*) AS c FROM meta_users ${where}`,
      params,
      `SELECT * FROM meta_users ${where} ORDER BY created_at DESC`,
      params,
      pagination,
      (r) => this.mapUser(r),
    );
  }

  countUsers(): number {
    const row = this.get<{ c: number }>("SELECT COUNT(*) AS c FROM meta_users");
    return row?.c ?? 0;
  }

  countSystemAdmins(): number {
    const row = this.get<{ c: number }>("SELECT COUNT(*) AS c FROM meta_users WHERE user_type = 'system_admin'");
    return row?.c ?? 0;
  }

  countTeams(): number {
    const row = this.get<{ c: number }>("SELECT COUNT(*) AS c FROM meta_teams");
    return row?.c ?? 0;
  }

  // ============================================================
  // UserKey
  // ============================================================
  createUserKey(input: CreateUserKeyInput): UserKeyEntity {
    if (input.is_default) {
      this.run(
        "UPDATE meta_user_keys SET is_default = 0 WHERE user_id = ? AND status = 'active'",
        input.user_id,
      );
    }
    const keyValue = input.key_value ?? generateUserKey();
    return this.insertUserKeyRow({
      user_id: input.user_id,
      key_value: keyValue,
      name: input.name,
      is_default: input.is_default ?? false,
      expires_at: input.expires_at,
      metadata_json: input.metadata_json,
    });
  }

  getUserKeyById(keyId: string): UserKeyEntity | null {
    return this.mapUserKey(this.get("SELECT * FROM meta_user_keys WHERE key_id = ?", keyId));
  }

  listUserKeys(userId: string, pagination?: PaginationParams | null): ListPage<UserKeyEntity> {
    const base = "FROM meta_user_keys WHERE user_id = ?";
    return this.selectList(
      `SELECT COUNT(*) AS c ${base}`,
      [userId],
      `SELECT * ${base} ORDER BY created_at DESC`,
      [userId],
      pagination,
      (r) => this.mapUserKey(r),
    );
  }

  countActiveUserKeys(userId: string): number {
    const row = this.get(
      "SELECT COUNT(*) AS c FROM meta_user_keys WHERE user_id = ? AND status = 'active'",
      userId,
    );
    return Number(row?.c ?? 0);
  }

  revokeUserKey(keyId: string, options?: { promoteNextDefault?: boolean }): UserKeyEntity | null {
    const promoteNextDefault = options?.promoteNextDefault ?? true;
    const existing = this.getUserKeyById(keyId);
    if (!existing) return null;

    if (existing.is_default && promoteNextDefault) {
      this.run("UPDATE meta_user_keys SET is_default = 0 WHERE key_id = ?", keyId);
      const next = this.get(
        `SELECT * FROM meta_user_keys WHERE user_id = ? AND status = 'active' AND key_id != ? ORDER BY created_at ASC LIMIT 1`,
        existing.user_id,
        keyId,
      );
      if (next) {
        this.run("UPDATE meta_user_keys SET is_default = 1 WHERE key_id = ?", String(next.key_id));
      }
    }

    this.run("DELETE FROM meta_user_keys WHERE key_id = ?", keyId);
    return existing;
  }

  updateUserKey(
    keyId: string,
    patch: Partial<Pick<UserKeyEntity, "name" | "expires_at" | "is_default" | "metadata_json">>,
  ): UserKeyEntity | null {
    const existing = this.getUserKeyById(keyId);
    if (!existing) return null;
    if (patch.is_default === true) {
      this.run(
        "UPDATE meta_user_keys SET is_default = 0 WHERE user_id = ? AND status = 'active'",
        existing.user_id,
      );
    }
    const allowed = ["name", "expires_at", "is_default", "metadata_json"] as const;
    this.applyUpdate("meta_user_keys", "key_id", keyId, allowed, {
      ...patch,
      is_default: patch.is_default === undefined ? undefined : patch.is_default ? 1 : 0,
    } as Partial<UserKeyEntity>);
    return this.getUserKeyById(keyId);
  }

  touchUserKeyUsage(keyId: string): void {
    this.run("UPDATE meta_user_keys SET last_used_at = ? WHERE key_id = ?", nowIso(), keyId);
  }

  revokeAllUserKeysForUser(userId: string): void {
    this.run(
      "DELETE FROM meta_user_keys WHERE user_id = ? AND status = 'active'",
      userId,
    );
  }

  // ============================================================
  // Team（自动 admin 成员）
  // ============================================================
  createTeam(input: CreateTeamInput): TeamEntity {
    const now = nowIso();
    for (let attempt = 0; attempt < PK_RETRY_LIMIT; attempt++) {
      const teamId = input.team_id ?? generateId(ID_PREFIX.team);
      try {
        return this.tx(() => {
          this.run(
            `INSERT INTO meta_teams
              (team_id, name, description, owner_user_id, status, created_at, updated_at, metadata_json)
             VALUES (?,?,?,?,?,?,?,?)`,
            teamId,
            input.name,
            input.description ?? null,
            input.owner_user_id,
            input.status ?? "active",
            now,
            now,
            input.metadata_json ?? "{}",
          );
          this.run(
            `INSERT INTO meta_team_members (id, team_id, user_id, role, joined_at, status)
             VALUES (?,?,?,?,?,?)`,
            generateRelationId(),
            teamId,
            input.owner_user_id,
            "admin",
            now,
            "active",
          );
          return this.getTeamById(teamId)!;
        });
      } catch (err) {
        if (isStorePkCollision(err) && !input.team_id) continue;
        if (isSqliteRelationIdCollision(err)) continue;
        throw err;
      }
    }
    throw new Error("PK collision after max retries");
  }

  getTeamById(teamId: string): TeamEntity | null {
    return this.mapTeam(this.get("SELECT * FROM meta_teams WHERE team_id = ?", teamId));
  }

  updateTeam(teamId: string, patch: Partial<TeamEntity>): TeamEntity | null {
    const allowed = ["name", "description", "status", "metadata_json"] as const;
    this.applyUpdate("meta_teams", "team_id", teamId, allowed, patch);
    return this.getTeamById(teamId);
  }

  deleteTeams(teamIds: string[]): BatchDeleteResult {
    const result = this.batchDelete("meta_teams", "team_id", teamIds);
    if (result.deleted_ids.length > 0) {
      const ph = result.deleted_ids.map(() => "?").join(",");
      this.run(`DELETE FROM meta_team_members WHERE team_id IN (${ph})`, ...result.deleted_ids);
      this.run(`DELETE FROM meta_agents WHERE team_id IN (${ph})`, ...result.deleted_ids);
      this.run(`DELETE FROM meta_tasks WHERE team_id IN (${ph})`, ...result.deleted_ids);
      this.run(`DELETE FROM meta_assets WHERE team_id IN (${ph})`, ...result.deleted_ids);
    }
    return result;
  }

  listTeamsByUser(userId: string, pagination?: PaginationParams | null, filter?: { name?: string }): ListPage<TeamEntity> {
    let base =
      "FROM meta_teams t JOIN meta_team_members m ON m.team_id = t.team_id WHERE m.user_id = ? AND m.status = 'active'";
    const params: SQLInputValue[] = [userId];
    if (filter?.name) {
      base += " AND t.name = ?";
      params.push(filter.name);
    }
    return this.selectList(
      `SELECT COUNT(*) AS c ${base}`,
      params,
      `SELECT t.* ${base} ORDER BY t.created_at DESC`,
      params,
      pagination,
      (r) => this.mapTeam(r),
    );
  }

  // ============================================================
  // TeamMember
  // ============================================================
  addTeamMember(input: AddTeamMemberInput): TeamMemberEntity {
    const now = nowIso();
    runWithGeneratedRelationId(input.id, isSqliteRelationIdCollision, (id) => {
      this.run(
        `INSERT INTO meta_team_members (id, team_id, user_id, role, joined_at, status)
       VALUES (?,?,?,?,?,?)
       ON CONFLICT(team_id, user_id) DO UPDATE SET role = excluded.role, status = excluded.status`,
        id,
        input.team_id,
        input.user_id,
        input.role ?? "member",
        now,
        input.status ?? "active",
      );
    });
    return this.getTeamMember(input.team_id, input.user_id)!;
  }

  removeTeamMember(teamId: string, userId: string): void {
    this.run("DELETE FROM meta_team_members WHERE team_id = ? AND user_id = ?", teamId, userId);
  }

  listTeamMembers(teamId: string, pagination?: PaginationParams | null): ListPage<TeamMemberEntity> {
    const base = "FROM meta_team_members WHERE team_id = ? AND status = 'active'";
    return this.selectList(
      `SELECT COUNT(*) AS c ${base}`,
      [teamId],
      `SELECT * ${base} ORDER BY joined_at DESC`,
      [teamId],
      pagination,
      (r) => r as unknown as TeamMemberEntity,
    );
  }

  getTeamMember(teamId: string, userId: string): TeamMemberEntity | null {
    return this.get<TeamMemberEntity>(
      "SELECT * FROM meta_team_members WHERE team_id = ? AND user_id = ?",
      teamId,
      userId,
    );
  }

  listTeamMembersWithProfile(teamId: string, pagination?: PaginationParams | null): ListPage<TeamMemberView> {
    const base =
      "FROM meta_team_members m LEFT JOIN meta_users u ON u.user_id = m.user_id WHERE m.team_id = ? AND m.status = 'active'";
    return this.selectList(
      `SELECT COUNT(*) AS c ${base}`,
      [teamId],
      `SELECT m.id, m.team_id, m.user_id, m.role, m.joined_at, m.status, COALESCE(u.username, '') AS username ${base} ORDER BY m.joined_at DESC`,
      [teamId],
      pagination,
      (r) => mapTeamMemberWithProfile(r as unknown as TeamMemberEntity & { username?: string }),
    );
  }

  getTeamMemberWithProfile(teamId: string, userId: string): TeamMemberView | null {
    const row = this.get<TeamMemberEntity & { username?: string }>(
      `SELECT m.id, m.team_id, m.user_id, m.role, m.joined_at, m.status, COALESCE(u.username, '') AS username
       FROM meta_team_members m
       LEFT JOIN meta_users u ON u.user_id = m.user_id
       WHERE m.team_id = ? AND m.user_id = ?`,
      teamId,
      userId,
    );
    return row ? mapTeamMemberWithProfile(row) : null;
  }

  // ============================================================
  // Agent
  // ============================================================
  createAgent(input: CreateAgentInput): AgentEntity {
    const now = nowIso();
    for (let attempt = 0; attempt < PK_RETRY_LIMIT; attempt++) {
      const agentId = input.agent_id ?? generateId(ID_PREFIX.agent);
      try {
        this.run(
          `INSERT INTO meta_agents
            (agent_id, team_id, owner_user_id, name, description, prompt, visibility, status, created_at, updated_at, metadata_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)`,
          agentId,
          input.team_id,
          input.owner_user_id,
          input.name,
          input.description ?? null,
          input.prompt ?? null,
          input.visibility ?? "team",
          input.status ?? "active",
          now,
          now,
          input.metadata_json ?? "{}",
        );
        return this.getAgentById(agentId)!;
      } catch (err) {
        if (isPkCollision(err) && !input.agent_id) continue;
        throw err;
      }
    }
    throw new Error("PK collision after max retries");
  }

  getAgentById(agentId: string): AgentEntity | null {
    return this.mapAgent(this.get("SELECT * FROM meta_agents WHERE agent_id = ?", agentId));
  }

  updateAgent(agentId: string, patch: Partial<AgentEntity>): AgentEntity | null {
    const allowed = ["name", "description", "prompt", "visibility", "status", "metadata_json"] as const;
    this.applyUpdate("meta_agents", "agent_id", agentId, allowed, patch);
    return this.getAgentById(agentId);
  }

  deleteAgents(agentIds: string[]): BatchDeleteResult {
    const existingAgents = agentIds
      .map((agentId) => this.getAgentById(agentId))
      .filter((agent): agent is AgentEntity => !!agent);
    const selfMemoryByAgent = new Map(
      existingAgents.map((agent) => [agent.agent_id, buildChatMemoryAssetId(agent.team_id, agent.agent_id)]),
    );

    const result = this.batchDelete("meta_agents", "agent_id", agentIds);
    if (result.deleted_ids.length > 0) {
      const ph = result.deleted_ids.map(() => "?").join(",");
      this.run(`DELETE FROM meta_task_agents WHERE agent_id IN (${ph})`, ...result.deleted_ids);
      this.run(`DELETE FROM meta_agent_fixed_assets WHERE agent_id IN (${ph})`, ...result.deleted_ids);
      const selfMemoryAssetIds = result.deleted_ids
        .map((agentId) => selfMemoryByAgent.get(agentId))
        .filter((assetId): assetId is string => !!assetId);
      if (selfMemoryAssetIds.length > 0) {
        this.deleteAssets(selfMemoryAssetIds);
      }
    }
    return result;
  }

  listAgentsByTeam(teamId: string, pagination?: PaginationParams | null, filter?: AgentFilter): ListPage<AgentEntity> {
    let where = "WHERE team_id = ?";
    const params: SQLInputValue[] = [teamId];
    if (filter?.status) {
      where += " AND status = ?";
      params.push(filter.status);
    }
    if (filter?.owner_user_id) {
      where += " AND owner_user_id = ?";
      params.push(filter.owner_user_id);
    }
    if (filter?.name) {
      where += " AND name = ?";
      params.push(filter.name);
    }
    return this.selectList(
      `SELECT COUNT(*) AS c FROM meta_agents ${where}`,
      params,
      `SELECT * FROM meta_agents ${where} ORDER BY created_at DESC`,
      params,
      pagination,
      (r) => this.mapAgent(r),
    );
  }

  listAgentsByOwner(userId: string, pagination?: PaginationParams | null, filter?: AgentFilter): ListPage<AgentEntity> {
    let where = "WHERE owner_user_id = ?";
    const params: SQLInputValue[] = [userId];
    if (filter?.status) {
      where += " AND status = ?";
      params.push(filter.status);
    }
    if (filter?.name) {
      where += " AND name = ?";
      params.push(filter.name);
    }
    return this.selectList(
      `SELECT COUNT(*) AS c FROM meta_agents ${where}`,
      params,
      `SELECT * FROM meta_agents ${where} ORDER BY created_at DESC`,
      params,
      pagination,
      (r) => this.mapAgent(r),
    );
  }

  // ============================================================
  // Task（含 linkAgents 原子）
  // ============================================================
  createTask(input: CreateTaskInput): TaskEntity {
    const now = nowIso();
    for (let attempt = 0; attempt < PK_RETRY_LIMIT; attempt++) {
      const taskId = input.task_id ?? generateId(ID_PREFIX.task);
      try {
        return this.tx(() => {
          this.run(
            `INSERT INTO meta_tasks
              (task_id, team_id, creator_user_id, title, description, source_type, source_url,
               status, auto_assign_floating_assets, risk_level, created_at, updated_at, metadata_json)
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)`,
            taskId,
            input.team_id,
            input.creator_user_id,
            input.title,
            input.description ?? null,
            input.source_type ?? "manual",
            input.source_url ?? null,
            input.status ?? "running",
            input.auto_assign_floating_assets ? 1 : 0,
            input.risk_level ?? null,
            now,
            now,
            input.metadata_json ?? "{}",
          );
          for (const link of input.linked_agents ?? []) {
            this.run(
              `INSERT INTO meta_task_agents (id, task_id, agent_id, role_in_task, status, created_at)
               VALUES (?,?,?,?,?,?)`,
              generateRelationId(),
              taskId,
              link.agent_id,
              link.role_in_task ?? null,
              "active",
              now,
            );
          }
          return this.getTaskById(taskId)!;
        });
      } catch (err) {
        if (isStorePkCollision(err) && !input.task_id) continue;
        if (isSqliteRelationIdCollision(err)) continue;
        throw err;
      }
    }
    throw new Error("PK collision after max retries");
  }

  getTaskById(taskId: string): TaskEntity | null {
    return this.mapTask(this.get("SELECT * FROM meta_tasks WHERE task_id = ?", taskId));
  }

  updateTask(taskId: string, patch: Partial<TaskEntity>): TaskEntity | null {
    const allowed = ["title", "description", "source_type", "source_url", "status", "auto_assign_floating_assets", "risk_level", "metadata_json"] as const;
    const normalized: Record<string, SQLInputValue> = {};
    for (const k of allowed) {
      if (k in patch && (patch as Record<string, unknown>)[k] !== undefined) {
        const v = (patch as Record<string, unknown>)[k];
        normalized[k] = k === "auto_assign_floating_assets" ? (v ? 1 : 0) : (v as SQLInputValue);
      }
    }
    this.applyUpdateRaw("meta_tasks", "task_id", taskId, normalized);
    return this.getTaskById(taskId);
  }

  deleteTasks(taskIds: string[]): BatchDeleteResult {
    const result = this.batchDelete("meta_tasks", "task_id", taskIds);
    if (result.deleted_ids.length > 0) {
      const ph = result.deleted_ids.map(() => "?").join(",");
      this.run(`DELETE FROM meta_task_agents WHERE task_id IN (${ph})`, ...result.deleted_ids);
    }
    return result;
  }

  listTasksByTeam(teamId: string, pagination?: PaginationParams | null, filter?: TaskFilter): ListPage<TaskEntity> {
    let where = "WHERE team_id = ?";
    const params: SQLInputValue[] = [teamId];
    if (filter?.status) {
      where += " AND status = ?";
      params.push(filter.status);
    }
    if (filter?.creator_user_id) {
      where += " AND creator_user_id = ?";
      params.push(filter.creator_user_id);
    }
    if (filter?.title) {
      where += " AND title = ?";
      params.push(filter.title);
    }
    return this.selectList(
      `SELECT COUNT(*) AS c FROM meta_tasks ${where}`,
      params,
      `SELECT * FROM meta_tasks ${where} ORDER BY created_at DESC`,
      params,
      pagination,
      (r) => this.mapTask(r),
    );
  }

  listTasks(filter: TaskFilter, pagination?: PaginationParams | null): ListPage<TaskEntity> {
    let where = "WHERE 1=1";
    const params: SQLInputValue[] = [];
    if (filter.status) {
      where += " AND status = ?";
      params.push(filter.status);
    }
    if (filter.creator_user_id) {
      where += " AND creator_user_id = ?";
      params.push(filter.creator_user_id);
    }
    if (filter.title) {
      where += " AND title = ?";
      params.push(filter.title);
    }
    return this.selectList(
      `SELECT COUNT(*) AS c FROM meta_tasks ${where}`,
      params,
      `SELECT * FROM meta_tasks ${where} ORDER BY created_at DESC`,
      params,
      pagination,
      (r) => this.mapTask(r),
    );
  }

  // ============================================================
  // TaskAgent
  // ============================================================
  linkTaskAgent(taskId: string, agentId: string, roleInTask?: string): TaskAgentEntity {
    const now = nowIso();
    runWithGeneratedRelationId(undefined, isSqliteRelationIdCollision, (id) => {
      this.run(
        `INSERT INTO meta_task_agents (id, task_id, agent_id, role_in_task, status, created_at)
       VALUES (?,?,?,?,?,?)
       ON CONFLICT(task_id, agent_id) DO UPDATE SET role_in_task = excluded.role_in_task, status = 'active'`,
        id,
        taskId,
        agentId,
        roleInTask ?? null,
        "active",
        now,
      );
    });
    return this.get<TaskAgentEntity>(
      "SELECT * FROM meta_task_agents WHERE task_id = ? AND agent_id = ?",
      taskId,
      agentId,
    )!;
  }

  unlinkTaskAgent(taskId: string, agentId: string): void {
    this.run("DELETE FROM meta_task_agents WHERE task_id = ? AND agent_id = ?", taskId, agentId);
  }

  listTaskAgents(taskId: string, pagination?: PaginationParams | null): ListPage<TaskAgentEntity> {
    const base = "FROM meta_task_agents WHERE task_id = ? AND status = 'active'";
    return this.selectList(
      `SELECT COUNT(*) AS c ${base}`,
      [taskId],
      `SELECT * ${base} ORDER BY created_at DESC`,
      [taskId],
      pagination,
      (r) => r as unknown as TaskAgentEntity,
    );
  }

  // ============================================================
  // ParticipationLog
  // ============================================================
  appendParticipationLog(input: AppendParticipationLogInput): ParticipationLogEntity {
    const now = nowIso();
    const createdAt = input.created_at ?? now;
    const entity: ParticipationLogEntity = {
      id: generateRelationId(),
      team_id: input.team_id,
      task_id: input.task_id,
      agent_id: input.agent_id,
      user_id: input.user_id,
      source: input.source ?? "unknown",
      metadata_json: input.metadata_json ?? "{}",
      created_at: createdAt,
      updated_at: createdAt,
    };
    this.run(
      `INSERT INTO meta_participation_logs
        (id, team_id, task_id, agent_id, user_id, source, metadata_json, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      entity.id,
      entity.team_id,
      entity.task_id,
      entity.agent_id,
      entity.user_id,
      entity.source,
      entity.metadata_json,
      entity.created_at,
      entity.updated_at,
    );
    return entity;
  }

  listParticipationLogs(
    filter: ParticipationLogFilter,
    pagination?: PaginationParams | null,
  ): ListPage<ParticipationLogEntity> {
    const { sql, params } = this.buildParticipationLogWhere(filter);
    if (filter.dedupe) {
      const dedupeIds = `
        SELECT id FROM (
          SELECT id, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC, id DESC) AS rn
          FROM meta_participation_logs
          WHERE ${sql}
        ) WHERE rn = 1
      `;
      return this.selectList(
        `SELECT COUNT(*) AS c FROM (${dedupeIds})`,
        params,
        `SELECT * FROM meta_participation_logs WHERE id IN (${dedupeIds}) ORDER BY created_at DESC, id DESC`,
        params,
        pagination,
        (r) => this.mapParticipationLog(r),
      );
    }
    const base = `FROM meta_participation_logs WHERE ${sql}`;
    return this.selectList(
      `SELECT COUNT(*) AS c ${base}`,
      params,
      `SELECT * ${base} ORDER BY created_at DESC, id DESC`,
      params,
      pagination,
      (r) => this.mapParticipationLog(r),
    );
  }

  private buildParticipationLogWhere(filter: ParticipationLogFilter): { sql: string; params: SQLInputValue[] } {
    const conditions = ["team_id = ?"];
    const params: SQLInputValue[] = [filter.team_id];
    if (filter.task_id) {
      conditions.push("task_id = ?");
      params.push(filter.task_id);
    }
    if (filter.agent_id) {
      conditions.push("agent_id = ?");
      params.push(filter.agent_id);
    }
    if (filter.user_id) {
      conditions.push("user_id = ?");
      params.push(filter.user_id);
    }
    if (filter.created_after) {
      conditions.push("created_at >= ?");
      params.push(filter.created_after);
    }
    if (filter.created_before) {
      conditions.push("created_at <= ?");
      params.push(filter.created_before);
    }
    return { sql: conditions.join(" AND "), params };
  }

  private mapParticipationLog(row: Row | null): ParticipationLogEntity | null {
    if (!row) return null;
    const r = row as Record<string, unknown>;
    return {
      id: String(r.id),
      team_id: String(r.team_id),
      task_id: String(r.task_id),
      agent_id: String(r.agent_id),
      user_id: String(r.user_id),
      source: String(r.source),
      metadata_json: String(r.metadata_json),
      created_at: String(r.created_at),
      updated_at: String(r.updated_at),
    };
  }

  // ============================================================
  // Asset
  // ============================================================
  createAsset(input: CreateAssetInput): AssetEntity {
    const now = nowIso();
    const assetId = input.asset_id;
    this.run(
      `INSERT INTO meta_assets
        (asset_id, team_id, asset_type, name, description, owner_user_id, source_type, source_ref,
         version, visibility, status, confidence, expires_at, last_used_at, usage_count, content_ref,
         created_at, updated_at, metadata_json)
       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
      assetId,
      input.team_id,
      input.asset_type,
      input.name,
      input.description ?? null,
      input.owner_user_id,
      input.source_type,
      input.source_ref ?? null,
      1,
      input.visibility ?? "team",
      input.status ?? "draft",
      input.confidence ?? null,
      input.expires_at ?? null,
      null,
      0,
      input.content_ref ?? null,
      now,
      now,
      input.metadata_json ?? "{}",
    );
    return this.getAssetById(assetId)!;
  }

  getAssetById(assetId: string): AssetEntity | null {
    return this.mapAsset(this.get("SELECT * FROM meta_assets WHERE asset_id = ?", assetId));
  }

  updateAsset(assetId: string, patch: Partial<AssetEntity>): AssetEntity | null {
    const allowed = ["name", "description", "visibility", "status", "confidence", "expires_at", "content_ref", "version", "source_ref", "metadata_json"] as const;
    this.applyUpdate("meta_assets", "asset_id", assetId, allowed, patch);
    return this.getAssetById(assetId);
  }

  deleteAssets(assetIds: string[]): BatchDeleteResult {
    // 物理删除 meta_assets，并级联清理绑定与 ACL。
    // 已不存在视为幂等成功（Skill 钩子/handler 双通道会二次调用）。
    const result: BatchDeleteResult = { deleted_ids: [], failed: [] };
    for (const id of assetIds) {
      const existing = this.getAssetById(id);
      if (!existing) {
        result.deleted_ids.push(id);
        continue;
      }
      this.tx(() => {
        this.run("DELETE FROM meta_agent_fixed_assets WHERE asset_id = ?", id);
        this.run("DELETE FROM meta_asset_acl WHERE asset_id = ?", id);
        this.run("DELETE FROM meta_assets WHERE asset_id = ?", id);
      });
      result.deleted_ids.push(id);
    }
    return result;
  }

  listAssetsByTeam(teamId: string, pagination?: PaginationParams | null, filter?: AssetFilter): ListPage<AssetEntity> {
    let where = "WHERE team_id = ?";
    const params: SQLInputValue[] = [teamId];
    if (filter?.asset_type) {
      where += " AND asset_type = ?";
      params.push(filter.asset_type);
    }
    if (filter?.status) {
      where += " AND status = ?";
      params.push(filter.status);
    }
    if (filter?.owner_user_id) {
      where += " AND owner_user_id = ?";
      params.push(filter.owner_user_id);
    }
    if (filter?.visibility) {
      where += " AND visibility = ?";
      params.push(filter.visibility);
    }
    return this.selectList(
      `SELECT COUNT(*) AS c FROM meta_assets ${where}`,
      params,
      `SELECT * FROM meta_assets ${where} ORDER BY created_at DESC`,
      params,
      pagination,
      (r) => this.mapAsset(r),
    );
  }

  touchAssetUsage(assetId: string): void {
    this.run(
      "UPDATE meta_assets SET usage_count = usage_count + 1, last_used_at = ? WHERE asset_id = ?",
      nowIso(),
      assetId,
    );
  }

  // ============================================================
  // AgentFixedAsset（全量替换）
  // ============================================================
  setAgentFixedAssets(agentId: string, bindings: FixedAssetBindingInput[]): void {
    const now = nowIso();
    for (let attempt = 0; attempt < RELATION_ID_RETRY_LIMIT; attempt++) {
      try {
        this.tx(() => {
          this.run("DELETE FROM meta_agent_fixed_assets WHERE agent_id = ?", agentId);
          for (const b of bindings) {
            this.run(
              `INSERT INTO meta_agent_fixed_assets
            (id, agent_id, asset_id, asset_type, injection_mode, priority, created_by, created_at)
           VALUES (?,?,?,?,?,?,?,?)`,
              generateRelationId(),
              agentId,
              b.asset_id,
              b.asset_type,
              b.injection_mode ?? "summary",
              b.priority ?? 50,
              b.created_by,
              now,
            );
          }
        });
        return;
      } catch (err) {
        if (isSqliteRelationIdCollision(err)) continue;
        throw err;
      }
    }
    throw new Error("relation id collision after max retries");
  }

  addAgentFixedAsset(agentId: string, b: FixedAssetBindingInput): void {
    // UNIQUE(agent_id, asset_id) 已在 schema 里定义（sqlite-adapter.ts:227 段），
    // INSERT OR IGNORE 命中冲突时 no-op，天然幂等。
    this.run(
      `INSERT OR IGNORE INTO meta_agent_fixed_assets
        (id, agent_id, asset_id, asset_type, injection_mode, priority, created_by, created_at)
       VALUES (?,?,?,?,?,?,?,?)`,
      generateRelationId(),
      agentId,
      b.asset_id,
      b.asset_type,
      b.injection_mode ?? "summary",
      b.priority ?? 50,
      b.created_by,
      nowIso(),
    );
  }

  listAgentFixedAssets(agentId: string, pagination?: PaginationParams | null): ListPage<FixedAssetBindingEntity> {
    const base = "FROM meta_agent_fixed_assets WHERE agent_id = ?";
    return this.selectList(
      `SELECT COUNT(*) AS c ${base}`,
      [agentId],
      `SELECT * ${base} ORDER BY priority DESC, created_at DESC`,
      [agentId],
      pagination,
      (r) => r as unknown as FixedAssetBindingEntity,
    );
  }

  getAgentFixedAsset(agentId: string, assetId: string): FixedAssetBindingEntity | null {
    return this.get<FixedAssetBindingEntity>(
      "SELECT * FROM meta_agent_fixed_assets WHERE agent_id = ? AND asset_id = ?",
      agentId,
      assetId,
    );
  }

  summarizeAgentFixedAssetsByAgents(
    agentIds: string[],
    options?: { assetId?: string },
  ): AgentFixedAssetCountRow[] {
    if (agentIds.length === 0) return [];
    const ph = agentIds.map(() => "?").join(",");
    const params: SQLInputValue[] = [...agentIds];
    let sql =
      `SELECT agent_id, asset_type, COUNT(DISTINCT asset_id) AS cnt
       FROM meta_agent_fixed_assets
       WHERE agent_id IN (${ph})`;
    if (options?.assetId) {
      sql += ` AND asset_id = ?`;
      params.push(options.assetId);
    }
    sql += ` GROUP BY agent_id, asset_type`;
    const rows = this.all<{ agent_id: string; asset_type: string; cnt: number | bigint }>(sql, ...params);
    return rows.map((r) => ({
      agent_id: r.agent_id,
      asset_type: r.asset_type as AssetType,
      cnt: Number(r.cnt),
    }));
  }

  // ============================================================
  // ACL
  // ============================================================
  grantAcl(input: GrantAclInput): AclEntity {
    const now = nowIso();
    runWithGeneratedRelationId(input.id, isSqliteRelationIdCollision, (id) => {
      this.run(
        `INSERT INTO meta_asset_acl
        (id, asset_id, subject_type, subject_id, permission, effect, granted_by, created_at, updated_at)
       VALUES (?,?,?,?,?,?,?,?,?)
       ON CONFLICT(asset_id, subject_type, subject_id, permission)
       DO UPDATE SET effect = excluded.effect, granted_by = excluded.granted_by, updated_at = excluded.updated_at`,
        id,
        input.asset_id,
        input.subject_type,
        input.subject_id,
        input.permission,
        input.effect ?? "allow",
        input.granted_by,
        now,
        now,
      );
    });
    return this.get<AclEntity>(
      "SELECT * FROM meta_asset_acl WHERE asset_id = ? AND subject_type = ? AND subject_id = ? AND permission = ?",
      input.asset_id,
      input.subject_type,
      input.subject_id,
      input.permission,
    )!;
  }

  getAclById(id: string): AclEntity | null {
    return this.get<AclEntity>("SELECT * FROM meta_asset_acl WHERE id = ?", id);
  }

  revokeAcl(id: string): void {
    this.run("DELETE FROM meta_asset_acl WHERE id = ?", id);
  }

  listAclByAsset(assetId: string, pagination?: PaginationParams | null): ListPage<AclEntity> {
    const base = "FROM meta_asset_acl WHERE asset_id = ?";
    return this.selectList(
      `SELECT COUNT(*) AS c ${base}`,
      [assetId],
      `SELECT * ${base} ORDER BY created_at DESC`,
      [assetId],
      pagination,
      (r) => r as unknown as AclEntity,
    );
  }

  listAclBySubject(subjectType: string, subjectId: string, pagination?: PaginationParams | null): ListPage<AclEntity> {
    const base = "FROM meta_asset_acl WHERE subject_type = ? AND subject_id = ?";
    return this.selectList(
      `SELECT COUNT(*) AS c ${base}`,
      [subjectType, subjectId],
      `SELECT * ${base} ORDER BY created_at DESC`,
      [subjectType, subjectId],
      pagination,
      (r) => r as unknown as AclEntity,
    );
  }

  // ============================================================
  // Helpers
  // ============================================================
  private applyUpdate<T>(
    table: string,
    pkCol: string,
    pkVal: string,
    allowed: readonly string[],
    patch: Partial<T>,
  ): void {
    const fields: Record<string, SQLInputValue> = {};
    for (const k of allowed) {
      const v = (patch as Record<string, unknown>)[k];
      if (k in (patch as object) && v !== undefined) {
        fields[k] = v as SQLInputValue;
      }
    }
    this.applyUpdateRaw(table, pkCol, pkVal, fields);
  }

  private applyUpdateRaw(
    table: string,
    pkCol: string,
    pkVal: string,
    fields: Record<string, SQLInputValue>,
  ): void {
    const keys = Object.keys(fields);
    const hasUpdatedAt = ["meta_users", "meta_teams", "meta_agents", "meta_tasks", "meta_assets"].includes(table);
    if (keys.length === 0 && !hasUpdatedAt) return;
    const sets = keys.map((k) => `${k} = ?`);
    const params: SQLInputValue[] = keys.map((k) => fields[k]);
    if (hasUpdatedAt) {
      sets.push("updated_at = ?");
      params.push(nowIso());
    }
    if (sets.length === 0) return;
    params.push(pkVal);
    this.run(`UPDATE ${table} SET ${sets.join(", ")} WHERE ${pkCol} = ?`, ...params);
  }

  private batchDelete(table: string, pkCol: string, ids: string[]): BatchDeleteResult {
    const result: BatchDeleteResult = { deleted_ids: [], failed: [] };
    for (const id of ids) {
      const exists = this.get(`SELECT ${pkCol} FROM ${table} WHERE ${pkCol} = ?`, id);
      if (!exists) {
        result.failed.push({ id, reason: "not_found" });
        continue;
      }
      this.run(`DELETE FROM ${table} WHERE ${pkCol} = ?`, id);
      result.deleted_ids.push(id);
    }
    return result;
  }

  // ── Row mappers ──
  private mapUser(row: Row | null): UserEntity | null {
    if (!row) return null;
    const r = row as Record<string, unknown>;
    return {
      user_id: String(r.user_id),
      password: r.password != null ? String(r.password) : null,
      auth_provider: String(r.auth_provider),
      external_id: String(r.external_id),
      username: String(r.username),
      display_name: r.display_name != null ? String(r.display_name) : null,
      email: r.email != null ? String(r.email) : null,
      raw_profile_json: String(r.raw_profile_json ?? "{}"),
      status: String(r.status) as UserEntity["status"],
      user_type: (String(r.user_type ?? "normal") as UserEntity["user_type"]),
      created_at: String(r.created_at),
      updated_at: String(r.updated_at),
      metadata_json: String(r.metadata_json ?? "{}"),
    };
  }

  private mapTeam(row: Row | null): TeamEntity | null {
    if (!row) return null;
    const r = row as Record<string, unknown>;
    return {
      team_id: String(r.team_id),
      name: String(r.name),
      description: r.description != null ? String(r.description) : null,
      owner_user_id: String(r.owner_user_id),
      status: String(r.status) as TeamEntity["status"],
      created_at: String(r.created_at),
      updated_at: String(r.updated_at),
      metadata_json: String(r.metadata_json ?? "{}"),
    };
  }

  private mapAgent(row: Row | null): AgentEntity | null {
    if (!row) return null;
    const r = row as Record<string, unknown>;
    return {
      agent_id: String(r.agent_id),
      team_id: String(r.team_id),
      owner_user_id: String(r.owner_user_id),
      name: String(r.name),
      description: r.description != null ? String(r.description) : null,
      prompt: r.prompt != null ? String(r.prompt) : null,
      visibility: String(r.visibility ?? "team") as AgentEntity["visibility"],
      status: String(r.status) as AgentEntity["status"],
      created_at: String(r.created_at),
      updated_at: String(r.updated_at),
      metadata_json: String(r.metadata_json ?? "{}"),
    };
  }

  private mapTask(row: Row | null): TaskEntity | null {
    if (!row) return null;
    const r = row as Record<string, unknown>;
    return {
      task_id: String(r.task_id),
      team_id: String(r.team_id),
      creator_user_id: String(r.creator_user_id),
      title: String(r.title),
      description: r.description != null ? String(r.description) : null,
      source_type: String(r.source_type) as TaskEntity["source_type"],
      source_url: r.source_url != null ? String(r.source_url) : null,
      status: String(r.status) as TaskEntity["status"],
      auto_assign_floating_assets: Boolean(r.auto_assign_floating_assets),
      risk_level: r.risk_level != null ? String(r.risk_level) : null,
      created_at: String(r.created_at),
      updated_at: String(r.updated_at),
      metadata_json: String(r.metadata_json ?? "{}"),
    };
  }

  private mapAsset(row: Row | null): AssetEntity | null {
    if (!row) return null;
    const r = row as Record<string, unknown>;
    return {
      asset_id: String(r.asset_id),
      team_id: String(r.team_id),
      asset_type: String(r.asset_type) as AssetEntity["asset_type"],
      name: String(r.name),
      description: r.description != null ? String(r.description) : null,
      owner_user_id: String(r.owner_user_id),
      source_type: String(r.source_type),
      source_ref: r.source_ref != null ? String(r.source_ref) : null,
      version: Number(r.version ?? 1),
      visibility: String(r.visibility) as AssetEntity["visibility"],
      status: String(r.status) as AssetEntity["status"],
      confidence: r.confidence != null ? Number(r.confidence) : null,
      expires_at: r.expires_at != null ? String(r.expires_at) : null,
      last_used_at: r.last_used_at != null ? String(r.last_used_at) : null,
      usage_count: Number(r.usage_count ?? 0),
      content_ref: r.content_ref != null ? String(r.content_ref) : null,
      created_at: String(r.created_at),
      updated_at: String(r.updated_at),
      metadata_json: String(r.metadata_json ?? "{}"),
    };
  }

  // ============================================================
  // ConfigParam
  // ============================================================

  getConfigParam(
    scope: "global" | "user",
    userId: string | null,
    module: string,
    paramName: string,
  ): ConfigParamEntity | null {
    let row: Row | null;
    if (scope === "global") {
      row = this.get<Row>(
        `SELECT * FROM meta_config_params WHERE scope = 'global' AND module = ? AND param_name = ?`,
        module, paramName,
      );
    } else {
      row = this.get<Row>(
        `SELECT * FROM meta_config_params WHERE scope = 'user' AND user_id = ? AND module = ? AND param_name = ?`,
        userId!, module, paramName,
      );
    }
    return this.mapConfigParam(row);
  }

  upsertConfigParam(input: UpsertConfigParamInput): ConfigParamEntity {
    const now = nowIso();
    if (input.scope === "global") {
      this.db.exec("BEGIN");
      try {
        const existing = this.get<Row>(
          `SELECT id FROM meta_config_params WHERE scope = 'global' AND module = ? AND param_name = ?`,
          input.module, input.param_name,
        );
        if (existing) {
          this.run(
            `UPDATE meta_config_params SET param_value = ?, description = ?, updated_at = ? WHERE id = ?`,
            input.param_value, input.description, now, (existing as any).id,
          );
        } else {
          this.run(
            `INSERT INTO meta_config_params (scope, user_id, module, param_name, param_value, description, created_at, updated_at)
             VALUES ('global', NULL, ?, ?, ?, ?, ?, ?)`,
            input.module, input.param_name, input.param_value, input.description, now, now,
          );
        }
        this.db.exec("COMMIT");
      } catch (err) {
        this.db.exec("ROLLBACK");
        throw err;
      }
    } else {
      this.db.exec("BEGIN");
      try {
        const existing = this.get<Row>(
          `SELECT id FROM meta_config_params WHERE scope = 'user' AND user_id = ? AND module = ? AND param_name = ?`,
          input.user_id!, input.module, input.param_name,
        );
        if (existing) {
          this.run(
            `UPDATE meta_config_params SET param_value = ?, description = ?, updated_at = ? WHERE id = ?`,
            input.param_value, input.description, now, (existing as any).id,
          );
        } else {
          this.run(
            `INSERT INTO meta_config_params (scope, user_id, module, param_name, param_value, description, created_at, updated_at)
             VALUES ('user', ?, ?, ?, ?, ?, ?, ?)`,
            input.user_id!, input.module, input.param_name, input.param_value, input.description, now, now,
          );
        }
        this.db.exec("COMMIT");
      } catch (err) {
        this.db.exec("ROLLBACK");
        throw err;
      }
    }

    const result = this.getConfigParam(
      input.scope,
      input.scope === "user" ? input.user_id! : null,
      input.module,
      input.param_name,
    );
    return result!;
  }

  listConfigParams(filter: ListConfigParamsFilter): ConfigParamEntity[] {
    const conditions: string[] = [`module = ?`];
    const params: SQLInputValue[] = [filter.module];

    if (filter.scope) {
      conditions.push(`scope = ?`);
      params.push(filter.scope);
    }
    if (filter.userId) {
      conditions.push(`(scope = 'global' OR (scope = 'user' AND user_id = ?))`);
      params.push(filter.userId);
    }
    if (filter.paramNames && filter.paramNames.length > 0) {
      const placeholders = filter.paramNames.map(() => "?").join(", ");
      conditions.push(`param_name IN (${placeholders})`);
      params.push(...filter.paramNames);
    }

    const sql = `SELECT * FROM meta_config_params WHERE ${conditions.join(" AND ")} ORDER BY scope ASC, param_name ASC`;
    const rows = this.all<Row>(sql, ...params);
    return rows.map((r) => this.mapConfigParam(r)!);
  }

  private mapConfigParam(row: Row | null): ConfigParamEntity | null {
    if (!row) return null;
    const r = row as Record<string, unknown>;
    return {
      id: Number(r.id),
      scope: String(r.scope) as ConfigParamEntity["scope"],
      user_id: r.user_id != null ? String(r.user_id) : null,
      module: String(r.module),
      param_name: String(r.param_name),
      param_value: String(r.param_value),
      description: String(r.description),
      created_at: String(r.created_at),
      updated_at: String(r.updated_at),
    };
  }
}
