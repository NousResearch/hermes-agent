/**
 * MetadataService — 元数据业务编排层。
 *
 * 对应设计文档 §2 / §7 / §10。在 IMetadataStore 之上叠加业务约束：
 *   - createAgent/createTask/createAsset 校验 team 存在
 *   - createTask linkAgents 校验 agent 同 team
 *   - setAgentFixedAssets 用 canBindAsset 校验 visibility
 *   - listAgentFixedAssetsWithDetail 聚合 agent + 详情 +（可选）visibility 过滤 + touchUsage
 *   - checkAssetPermission 懒加载 ACL（角色默认覆盖时不查表）
 *   - user_key 生成/刷新
 *
 * 不感知具体后端（SQLite / MongoDB），保证存储可切换。
 */

import type { IMetadataStore } from "../store/interface.js";
import {
  checkPermission,
  canBindAsset,
  roleDefaultCovers,
  type PermCheckResult,
  type PermCheckLogger,
} from "./permission-checker.js";
import {
  maskUserKey, isUserKeyExpired, DEFAULT_MAX_ACTIVE_USER_KEYS,
} from "../utils/user-key.js";
import {
  lookupMemorySystemUser,
  isMemorySystemUserKey,
  toMemorySystemVerifyUser,
  type MemorySystemUserConfig,
} from "../system-user.js";
import { resolveUserId } from "./resolve-user-id.js";
import type { V3AuthContext } from "../router/auth.js";
import { DEFAULT_INSTANCE_ID, DEFAULT_AUTH_PROVIDER } from "../constants.js";
import {
  canManageUsers,
  canViewUser,
  filterVisibleUsers,
  isSystemAdminUser,
  toPublicUser,
} from "./user-visibility.js";
import type {
  UserEntity,
  UserPublic,
  UserKeyEntity,
  UserKeyPublic,
  UserKeyCreated,
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
  AgentFixedAssetSummary,
  AgentFixedAssetSummaryResult,
  FixedAssetTypeCounts,
  SummarizeAgentFixedAssetsParams,
  AclEntity,
  CreateUserInput,
  InitAdminInput,
  InitAdminResult,
  CreateUserApiResult,
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
  AssetType,
  AssetVisibility,
  AssetStatus,
  InjectionMode,
  Permission,
  UserType,
  PaginatedResult,
  PaginationParams,
  InstanceUserListFilter,
  UserListFilter,
} from "../types.js";
import { formatListResult, paginateArray, resolvePagination, wrapPaginated, DEFAULT_PAGINATION } from "../pagination.js";
import { generateId, ID_PREFIX } from "../utils/id-generator.js";
import { buildChatMemoryAssetId } from "../utils/chat-memory-asset.js";

/** 业务校验错误，带可映射到 HTTP 状态的 code。 */
export class MetadataError extends Error {
  constructor(
    public readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "MetadataError";
  }
}

/** Detect unique constraint violation (SQLite UNIQUE or MongoDB E11000) on a specific column. */
function isUniqueViolation(err: unknown, column?: string): boolean {
  if (!(err instanceof Error)) return false;
  const msg = err.message;
  if (/UNIQUE constraint failed/.test(msg)) {
    return column ? msg.includes(column) : true;
  }
  if ((err as any).code === 11000) {
    if (!column) return true;
    const kp = (err as any).keyPattern;
    return kp ? column in kp : msg.includes(column);
  }
  return false;
}

export interface AgentBasicData {
  agent_id: string;
  team_id: string;
  owner_user_id: string;
  prompt: string | null;
  visibility: AssetVisibility;
  status: string;
}

export interface AgentAssetView {
  asset_id: string;
  asset_type: AssetType;
  name: string;
  description: string | null;
  status: AssetStatus;
  visibility: AssetVisibility;
  injection_mode: InjectionMode;
  priority: number;
  created_at: string;
}

export interface ListWithDetailParams {
  agent_id: string;
  apply_visibility_filter?: boolean;
  touch_usage?: boolean;
  limit?: number;
  offset?: number;
}

export interface AgentFixedAssetDetailResult {
  agent: AgentBasicData;
  items: AgentAssetView[];
  total: number;
  limit: number;
  offset: number;
}

export interface CheckPermissionParams {
  user_id?: string;
  user_key?: string;
  asset_id: string;
  action: Permission;
  agent_id?: string;
}

export interface ListAccessibleAssetsParams {
  user_id?: string;
  user_key?: string;
  team_id?: string;
  action?: Permission;
  asset_type?: AssetType;
  agent_id?: string;
  /**
   * 可选的 visibility 白名单，用于在权限判定后再做二次过滤。
   * 例：`["team"]` = 只返回团队可见的（管控页"团队资产"tab 用）；
   * 不传 = 返回所有可访问的（含自己的 private）。
   * 目的：让前端从 HTTP 响应上就拿不到不需要的数据（安全 + 减小载荷）。
   */
  visibility?: AssetEntity["visibility"] | AssetEntity["visibility"][];
  limit?: number;
  offset?: number;
}

const FILTERED_STATUSES: AssetStatus[] = ["archived", "deprecated", "failed"];

export interface MetadataQuotaLimits {
  maxUsersPerInstance: number;
  maxTeamsPerInstance: number;
}

export const DEFAULT_METADATA_QUOTA_LIMITS: MetadataQuotaLimits = {
  maxUsersPerInstance: 500,
  maxTeamsPerInstance: 100,
};

export class MetadataService {
  private readonly quota: MetadataQuotaLimits;
  private readonly memorySystemUser?: MemorySystemUserConfig;
  private _configParams?: import("./config-param-service.js").IConfigParamService;

  /**
   * 进程内 LRU 缓存：已确认（在 store 里存在的） chat_memory 资产 id 集合。
   *
   * 命中即可跳过 getAssetById + createAsset + addAgentFixedAsset 三次 DB 往返。
   * 未命中就走完整 ensure 流程，成功后写入缓存。跨进程/多 pod 时各自维护自己
   * 的 LRU —— 一致性由 store 主键约束保证。
   *
   * 用 Map 的插入序 + 达到上限时淘汰最老项。达不到严格 LRU（不做 touch），
   * 但对于这个"只写一次、后续都命中"的场景足够：一旦确认存在，条目要么持续
   * 命中要么被更新的 team+agent 挤走再重新走一遍 DB —— 冷淘汰的代价只是一次
   * 数据库查询。maxSize 由 CHAT_MEMORY_ENSURE_CACHE_SIZE 控制。
   */
  private readonly ensuredChatMemoryAssets = new Map<string, true>();
  private static readonly CHAT_MEMORY_ENSURE_CACHE_SIZE = 4096;

  /**
   * skill 资产登记的进程内 LRU：key = skill_id（即 asset_id）。
   * 语义与 ensuredChatMemoryAssets 一致，短路 ensureSkillAsset 的重复 create+bind。
   */
  private readonly ensuredSkillAssets = new Map<string, true>();
  private static readonly SKILL_ENSURE_CACHE_SIZE = 4096;

  constructor(
    private readonly store: IMetadataStore,
    private readonly instanceId: string = DEFAULT_INSTANCE_ID,
    private readonly logger: PermCheckLogger = { debug: () => {} },
    quotaLimits?: Partial<MetadataQuotaLimits>,
    memorySystemUser?: MemorySystemUserConfig,
  ) {
    this.quota = { ...DEFAULT_METADATA_QUOTA_LIMITS, ...quotaLimits };
    this.memorySystemUser = memorySystemUser;
  }

  get scopedInstanceId(): string {
    return this.instanceId;
  }

  get configParams(): import("./config-param-service.js").IConfigParamService {
    if (!this._configParams) {
      throw new Error("ConfigParamService not initialized. Call setConfigParamService() after store.init().");
    }
    return this._configParams;
  }

  setConfigParamService(svc: import("./config-param-service.js").IConfigParamService): void {
    this._configParams = svc;
  }

  /** memory 静态 key 仅用于 auth/verify body，不可作 Header 鉴权。 */
  isConfiguredMemorySystemUserKey(userKey: string): boolean {
    return isMemorySystemUserKey(userKey, this.memorySystemUser);
  }

  private pag(input: { limit?: number; offset?: number }): PaginationParams {
    return resolvePagination(input);
  }

  private async allAclRecords(assetId: string): Promise<AclEntity[]> {
    const out: AclEntity[] = [];
    let offset = 0;
    const limit = 100;
    while (true) {
      const page = await this.store.listAclByAsset(assetId, { limit, offset });
      out.push(...page.items);
      if (offset + page.items.length >= page.total) break;
      offset += limit;
    }
    return out;
  }

  /** internal：按实例分页列出用户（含 system_admin，不脱敏）。 */
  async listUsersByInstance(
    instanceId: string,
    pagination: PaginationParams,
    filter?: InstanceUserListFilter,
  ): Promise<PaginatedResult<UserEntity>> {
    void instanceId;
    const page = await this.store.listUsers(pagination, filter);
    return formatListResult(page, pagination);
  }

  /** 校验用户存在，否则抛 not_found。 */
  private async requireUser(userId: string): Promise<UserEntity> {
    const user = await this.getUserById(userId);
    if (!user) throw new MetadataError("user_not_found", `user not found: ${userId}`);
    return user;
  }

  get rawStore(): IMetadataStore {
    return this.store;
  }

  private async assertUserQuota(): Promise<void> {
    const count = await this.store.countUsers();
    const limit = this._configParams
      ? await this._configParams.getEffectiveInt("quota", "max_users_per_instance")
      : this.quota.maxUsersPerInstance;
    if (count >= limit) {
      throw new MetadataError(
        "user_limit_exceeded",
        `user limit ${limit} reached for instance ${this.instanceId} (current: ${count})`,
      );
    }
  }

  private async assertTeamQuota(): Promise<void> {
    const count = await this.store.countTeams();
    const limit = this._configParams
      ? await this._configParams.getEffectiveInt("quota", "max_teams_per_instance")
      : this.quota.maxTeamsPerInstance;
    if (count >= limit) {
      throw new MetadataError(
        "team_limit_exceeded",
        `team limit ${limit} reached for instance ${this.instanceId} (current: ${count})`,
      );
    }
  }

  // ============================================================
  // User（含 user_key 生成/刷新）
  // ============================================================
  async initAdminUser(input: InitAdminInput): Promise<InitAdminResult> {
    if ((await this.store.countUsers()) > 0) {
      throw new MetadataError("already_initialized", "system already has users; init-admin requires empty database");
    }
    if ((await this.store.countSystemAdmins()) > 0) {
      throw new MetadataError("already_initialized", "system_admin already exists");
    }
    const created = await this.createUserWithType(
      { username: input.username, default_key_value: input.user_key },
      "system_admin",
    );
    return { user_id: created.user_id, user_key: created.default_user_key };
  }

  async createNormalUser(input: CreateUserInput): Promise<CreateUserApiResult> {
    return this.createUserWithType(input, "normal");
  }

  /** 未传 auth_provider / external_id 时补默认值（local / user_id）。 */
  private resolveCreateUserInput(
    input: CreateUserInput,
  ): CreateUserInput & { auth_provider: string; external_id: string } {
    const authProvider = input.auth_provider?.trim() || DEFAULT_AUTH_PROVIDER;
    const externalId = input.external_id?.trim();
    if (externalId) {
      return { ...input, auth_provider: authProvider, external_id: externalId };
    }
    const userId = input.user_id ?? generateId(ID_PREFIX.user);
    return { ...input, auth_provider: authProvider, user_id: userId, external_id: userId };
  }

  private async createUserWithType(input: CreateUserInput, userType: UserType): Promise<CreateUserApiResult> {
    const resolved = this.resolveCreateUserInput(input);
    await this.assertUserQuota();
    const user = await this.store.createUser({
      ...resolved,
      password: null,
      user_type: userType,
    });
    const defaultKey = await this.store.getDefaultUserKey(user.user_id);
    if (!defaultKey) {
      throw new MetadataError("internal_error", "default user key not created");
    }
    return {
      user_id: user.user_id,
      user_type: user.user_type,
      created_at: user.created_at,
      default_user_key: defaultKey.key_value,
    };
  }

  async getUserForCaller(userId: string, ctx: V3AuthContext): Promise<UserPublic> {
    const user = await this.getUserById(userId);
    if (!user || !canViewUser(user, ctx)) {
      throw new MetadataError("user_not_found", `user not found: ${userId}`);
    }
    return toPublicUser(user, ctx);
  }

  async getUserById(userId: string): Promise<UserEntity | null> {
    return this.store.getUserById(userId);
  }

  async getUserByKey(userKey: string): Promise<UserEntity | null> {
    return this.store.getUserByKey(userKey);
  }

  async getUserByExternalId(authProvider: string, externalId: string): Promise<UserEntity | null> {
    return this.store.getUserByExternalId(authProvider, externalId);
  }

  async deleteUsersForCaller(userIds: string[], ctx: V3AuthContext): Promise<BatchDeleteResult> {
    if (!canManageUsers(ctx)) {
      throw new MetadataError("permission_denied", "user management requires system admin");
    }
    let deletingSystemAdmins = 0;
    for (const id of userIds) {
      const u = await this.getUserById(id);
      if (u && isSystemAdminUser(u)) deletingSystemAdmins++;
    }
    const totalAdmins = await this.store.countSystemAdmins();
    if (totalAdmins > 0 && totalAdmins - deletingSystemAdmins < 1) {
      throw new MetadataError("last_system_admin", "cannot delete the last system_admin user");
    }
    return this.deleteUsers(userIds);
  }

  async deleteUsers(userIds: string[]): Promise<BatchDeleteResult> {
    return this.store.deleteUsers(userIds);
  }

  async listUsersForCaller(
    input: { team_id?: string } & UserListFilter,
    ctx: V3AuthContext,
    pagination: PaginationParams,
  ): Promise<PaginatedResult<UserPublic>> {
    const filtersPresent = !!(input.user_ids?.length || input.username);
    const storeFilter = this.buildUserListStoreFilter(input);

    if (!input.team_id) {
      if (!ctx.isSystemAdmin) {
        throw new MetadataError("missing_team_id", "team_id is required for non-system-admin callers");
      }
      const page = await this.store.listUsers(pagination, storeFilter);
      const items = filterVisibleUsers(page.items, ctx);
      return formatListResult({ items, total: page.total }, pagination);
    }

    const teamId = input.team_id;

    if (ctx.isSystemAdmin) {
      const page = await this.store.listUsersByTeam(teamId, pagination, storeFilter);
      const items = filterVisibleUsers(page.items, ctx);
      return formatListResult({ items, total: page.total }, pagination);
    }

    if (!ctx.userId) {
      throw new MetadataError("permission_denied", "authentication required");
    }

    const member = await this.store.getTeamMember(teamId, ctx.userId);
    if (!member || member.status !== "active") {
      throw new MetadataError("permission_denied", "not a team member");
    }

    const isTeamAdmin = member.role === "admin";
    if (isTeamAdmin) {
      const page = await this.store.listUsersByTeam(teamId, pagination, storeFilter);
      const items = filterVisibleUsers(page.items, ctx, { allowTeamPeers: true });
      return formatListResult({ items, total: page.total }, pagination);
    }

    if (filtersPresent) {
      throw new MetadataError("filter_not_allowed", "filters are not allowed for normal team members");
    }

    const self = await this.store.getUserById(ctx.userId);
    if (!self) {
      throw new MetadataError("user_not_found", `user not found: ${ctx.userId}`);
    }
    const visible = filterVisibleUsers([self], ctx);
    if (pagination.offset > 0) {
      return wrapPaginated([], 1, pagination);
    }
    return wrapPaginated(visible, 1, pagination);
  }

  private buildUserListStoreFilter(input: UserListFilter): InstanceUserListFilter | undefined {
    const filter: InstanceUserListFilter = {};
    if (input.user_ids?.length) filter.user_ids = input.user_ids;
    if (input.username) filter.username = input.username;
    return Object.keys(filter).length ? filter : undefined;
  }

  /** @deprecated 使用 listUsersForCaller */
  async listUsersByTeamForCaller(
    teamId: string,
    ctx: V3AuthContext,
    pagination: PaginationParams,
  ): Promise<PaginatedResult<UserPublic>> {
    return this.listUsersForCaller({ team_id: teamId }, ctx, pagination);
  }

  async listUsersByTeam(teamId: string, pagination: PaginationParams = DEFAULT_PAGINATION): Promise<PaginatedResult<UserEntity>> {
    const page = await this.store.listUsersByTeam(teamId, pagination);
    return formatListResult(page, pagination);
  }

  assertCanManageUsers(ctx: V3AuthContext): void {
    if (!canManageUsers(ctx)) {
      throw new MetadataError("permission_denied", "user management requires system admin");
    }
  }

  canManageUserScope(userId: string, ctx: V3AuthContext): boolean {
    return ctx.isAdmin || ctx.isSystemAdmin || ctx.userId === userId;
  }

  assertUserScope(userId: string, callerUserId?: string, isAdmin = false, isSystemAdmin = false): void {
    if (isAdmin || isSystemAdmin || userId === callerUserId) return;
    throw new MetadataError("permission_denied", "cannot access another user's keys");
  }

  assertCallerIsOwner(targetUserId: string, callerId: string): void {
    if (targetUserId !== callerId) {
      throw new MetadataError("permission_denied", "user_id does not match caller");
    }
  }

  async verifyAuthForCaller(userKey: string, ctx: V3AuthContext): Promise<{ valid: boolean; user: UserPublic | null }> {
    const user = await this.verifyAuth(userKey);
    if (!user) return { valid: false, user: null };
    const visibilityCtx: V3AuthContext = ctx.userId
      ? ctx
      : {
          token: userKey,
          userId: user.user_id,
          isAdmin: false,
          isSystemAdmin: user.user_type === "system_admin",
        };
    if (!canViewUser(user, visibilityCtx)) {
      return { valid: true, user: null };
    }
    if (this.memorySystemUser && user.user_id === this.memorySystemUser.userId) {
      return {
        valid: true,
        user: {
          user_id: user.user_id,
          user_type: user.user_type,
          username: user.username,
          created_at: user.created_at,
        },
      };
    }
    return { valid: true, user: toPublicUser(user, visibilityCtx) };
  }

  /** 校验 user_key 并返回对应用户（无效返回 null）。 */
  async verifyAuth(userKey: string): Promise<UserEntity | null> {
    if (!userKey) return null;
    const configured = lookupMemorySystemUser(userKey, this.instanceId, this.memorySystemUser);
    if (configured) return configured;
    return this.store.getUserByKey(userKey);
  }

  toPublicUserKey(entity: UserKeyEntity): UserKeyPublic {
    return {
      key_id: entity.key_id,
      user_id: entity.user_id,
      key_prefix: maskUserKey(entity.key_value),
      name: entity.name ?? null,
      status: entity.status,
      is_default: entity.is_default,
      last_used_at: entity.last_used_at ?? null,
      expires_at: entity.expires_at ?? null,
      created_at: entity.created_at,
      revoked_at: entity.revoked_at ?? null,
    };
  }

  async createUserKey(
    userId: string,
    input: { name?: string | null; expires_at?: string | null },
  ): Promise<UserKeyCreated> {
    await this.requireUser(userId);

    const active = await this.store.countActiveUserKeys(userId);
    if (active >= this.maxActiveUserKeys) {
      throw new MetadataError("key_limit_exceeded", `active user key limit ${this.maxActiveUserKeys} reached`);
    }

    const entity = await this.store.createUserKey({
      user_id: userId,
      name: input.name,
      expires_at: input.expires_at,
      is_default: false,
    });
    return { ...this.toPublicUserKey(entity), key_value: entity.key_value };
  }

  async listUserKeys(userId: string, pagination: PaginationParams = DEFAULT_PAGINATION): Promise<PaginatedResult<UserKeyPublic>> {
    await this.requireUser(userId);
    const page = await this.store.listUserKeys(userId, pagination);
    const items = page.items.map((k) => this.toPublicUserKey(k));
    return formatListResult({ items, total: page.total }, pagination);
  }

  async getUserKey(keyId: string): Promise<UserKeyPublic> {
    const entity = await this.store.getUserKeyById(keyId);
    if (!entity) throw new MetadataError("user_key_not_found", `user key not found: ${keyId}`);
    return this.toPublicUserKey(entity);
  }

  /** 校验调用方有权访问该 key（本人、system_admin 或 bootstrap），返回脱敏详情。 */
  async getUserKeyForCaller(
    keyId: string,
    callerUserId?: string,
    isAdmin = false,
    isSystemAdmin = false,
  ): Promise<UserKeyPublic> {
    const entity = await this.store.getUserKeyById(keyId);
    if (!entity) throw new MetadataError("user_key_not_found", `user key not found: ${keyId}`);
    const owner = await this.getUserById(entity.user_id);
    if (!owner) {
      throw new MetadataError("user_key_not_found", `user key not found: ${keyId}`);
    }
    if (!isAdmin && !isSystemAdmin && entity.user_id !== callerUserId) {
      throw new MetadataError("permission_denied", "cannot access another user's key");
    }
    if (isSystemAdminUser(owner) && !isAdmin && callerUserId !== owner.user_id) {
      throw new MetadataError("user_key_not_found", `user key not found: ${keyId}`);
    }
    return this.toPublicUserKey(entity);
  }

  async revokeUserKey(keyId: string): Promise<void> {
    const entity = await this.store.getUserKeyById(keyId);
    if (!entity) throw new MetadataError("user_key_not_found", `user key not found: ${keyId}`);
    if (!(await this.getUserById(entity.user_id))) {
      throw new MetadataError("user_key_not_found", `user key not found: ${keyId}`);
    }

    const active = await this.store.countActiveUserKeys(entity.user_id);
    if (active <= 1) {
      throw new MetadataError("last_key_cannot_revoke", "cannot revoke the last active user key");
    }

    console.info(
      `[META] revokeUserKey: user_id=${entity.user_id} key_id=${entity.key_id} key_prefix=${maskUserKey(entity.key_value)}`,
    );
    await this.store.revokeUserKey(keyId, { promoteNextDefault: true });
  }

  async updateUserKey(
    keyId: string,
    patch: { name?: string | null; expires_at?: string | null },
  ): Promise<UserKeyPublic> {
    const existing = await this.store.getUserKeyById(keyId);
    if (!existing) throw new MetadataError("user_key_not_found", `user key not found: ${keyId}`);
    if (!(await this.getUserById(existing.user_id))) {
      throw new MetadataError("user_key_not_found", `user key not found: ${keyId}`);
    }

    const updated = await this.store.updateUserKey(keyId, patch);
    if (!updated) throw new MetadataError("user_key_not_found", `user key not found: ${keyId}`);
    return this.toPublicUserKey(updated);
  }

  private get maxActiveUserKeys(): number {
    const fromEnv = Number(process.env.TDAI_USER_KEY_MAX_ACTIVE);
    return Number.isFinite(fromEnv) && fromEnv > 0 ? fromEnv : DEFAULT_MAX_ACTIVE_USER_KEYS;
  }

  // ============================================================
  // Team
  // ============================================================
  async createTeam(input: CreateTeamInput): Promise<TeamEntity> {
    await this.assertTeamQuota();
    return this.store.createTeam(input);
  }

  async getTeamById(teamId: string): Promise<TeamEntity | null> {
    return this.store.getTeamById(teamId);
  }

  async updateTeam(teamId: string, patch: Partial<TeamEntity>): Promise<TeamEntity> {
    if (!(await this.getTeamById(teamId))) throw new MetadataError("team_not_found", `team not found: ${teamId}`);
    const updated = await this.store.updateTeam(teamId, patch);
    if (!updated) throw new MetadataError("team_not_found", `team not found: ${teamId}`);
    return updated;
  }

  async deleteTeams(teamIds: string[]): Promise<BatchDeleteResult> {
    return this.store.deleteTeams(teamIds);
  }

  async listTeamsByUser(userId: string, pagination: PaginationParams = DEFAULT_PAGINATION, filter?: { name?: string }): Promise<PaginatedResult<TeamEntity>> {
    const page = await this.store.listTeamsByUser(userId, pagination, filter);
    const items = page.items;
    return formatListResult({ items, total: page.total }, pagination);
  }

  // ============================================================
  // TeamMember
  // ============================================================
  async addTeamMember(input: AddTeamMemberInput): Promise<TeamMemberEntity> {
    const team = await this.getTeamById(input.team_id);
    if (!team) throw new MetadataError("team_not_found", `team not found: ${input.team_id}`);
    const reqRole = input.role ?? "member";
    // owner 由 createTeam 固定为 admin；禁止经 add upsert 降级，否则会出现
    // 「仍是 owner 但 role≠admin」——面板当 admin、team-member/add 却 403。
    if (input.user_id === team.owner_user_id && reqRole !== "admin") {
      throw new MetadataError("permission_denied", "cannot demote team owner");
    }
    const existing = await this.store.getTeamMember(input.team_id, input.user_id);
    if (existing?.status === "active" && existing.role === reqRole) {
      throw new MetadataError(
        "member_already_exists",
        `member already exists: ${input.team_id}/${input.user_id}`,
      );
    }
    return this.store.addTeamMember({ ...input, role: reqRole });
  }

  async removeTeamMember(teamId: string, userId: string): Promise<void> {
    await this.store.removeTeamMember(teamId, userId);
  }

  async listTeamMembers(teamId: string, pagination: PaginationParams = DEFAULT_PAGINATION): Promise<PaginatedResult<TeamMemberEntity>> {
    const page = await this.store.listTeamMembers(teamId, pagination);
    return formatListResult(page, pagination);
  }

  async getTeamMember(teamId: string, userId: string): Promise<TeamMemberEntity | null> {
    return this.store.getTeamMember(teamId, userId);
  }

  // ============================================================
  // Agent（校验 team 存在）
  // ============================================================
  async createAgent(input: CreateAgentInput): Promise<AgentEntity> {
    await this.assertTeamExists(input.team_id);
    const agent = await this.store.createAgent(input);
    // 建 agent 的同一事务边界外，立即 mint 该 agent 的 chat_memory 资产 +
    // 绑定到 fixed_assets。avoids Bug 2：首次对话触发时 asset 还不存在 →
    // profile-memory-injector 首个 session prewarm 走 fallback 到 tools-only、
    // 且 session_init 缓存策略下当 session 内永远读不到 L3。
    //
    // 幂等：ensureChatMemoryAsset 内部对已存在 asset / 已存在 binding 走 no-op。
    // 失败非致命：agent 已建成功，chat_memory 只是"更早准备好"，
    // 即便这里失败，/conversation/add 那条链路依然会重试 ensure，故此处仅 log warn。
    try {
      await this.ensureChatMemoryAsset({
        team_id: agent.team_id,
        agent_id: agent.agent_id,
      });
    } catch (err) {
      // 这里没有专用 warn logger（service 层只有 PermCheckLogger.debug），
      // 用 console.warn 与 v2-router.handleConversationAdd 里同类 catch 保持一致。
      console.warn(
        `[META] createAgent: ensureChatMemoryAsset failed (agent=${agent.agent_id} team=${agent.team_id}): ` +
        (err instanceof Error ? err.message : String(err)),
      );
    }
    return agent;
  }

  async getAgentById(agentId: string): Promise<AgentEntity | null> {
    return this.store.getAgentById(agentId);
  }

  async updateAgent(agentId: string, patch: Partial<AgentEntity>): Promise<AgentEntity> {
    if (!(await this.getAgentById(agentId))) throw new MetadataError("agent_not_found", `agent not found: ${agentId}`);
    const updated = await this.store.updateAgent(agentId, patch);
    if (!updated) throw new MetadataError("agent_not_found", `agent not found: ${agentId}`);
    return updated;
  }

  async deleteAgents(agentIds: string[]): Promise<BatchDeleteResult> {
    return this.store.deleteAgents(agentIds);
  }

  async listAgentsByTeam(
    teamId: string,
    pagination: PaginationParams = DEFAULT_PAGINATION,
    filter?: AgentFilter,
  ): Promise<PaginatedResult<AgentEntity>> {
    const page = await this.store.listAgentsByTeam(teamId, pagination, filter);
    const items = page.items;
    return formatListResult({ items, total: page.total }, pagination);
  }

  async listAgentsByOwner(
    userId: string,
    pagination: PaginationParams = DEFAULT_PAGINATION,
    filter?: AgentFilter,
  ): Promise<PaginatedResult<AgentEntity>> {
    const page = await this.store.listAgentsByOwner(userId, pagination, filter);
    const items = page.items;
    return formatListResult({ items, total: page.total }, pagination);
  }

  /** 归档（软关闭）agent：status → inactive，并归档其自身 chat_memory 资产。 */
  async archiveAgent(agentId: string): Promise<AgentEntity> {
    const existing = await this.getAgentById(agentId);
    if (!existing) throw new MetadataError("agent_not_found", `agent not found: ${agentId}`);
    const archived = await this.updateAgent(agentId, { status: "inactive" });
    const selfMemoryAssetId = buildChatMemoryAssetId(existing.team_id, existing.agent_id);
    await this.store.deleteAssets([selfMemoryAssetId]);
    return archived;
  }

  // ============================================================
  // Task（校验 team 存在 + linked agents 同 team）
  // ============================================================
  async createTask(input: CreateTaskInput): Promise<TaskEntity> {
    await this.assertTeamExists(input.team_id);
    for (const link of input.linked_agents ?? []) {
      const agent = await this.getAgentById(link.agent_id);
      if (!agent) {
        throw new MetadataError("agent_not_found", `agent not found: ${link.agent_id}`);
      }
      if (agent.team_id !== input.team_id) {
        throw new MetadataError(
          "agent_team_mismatch",
          `agent ${link.agent_id} not in team ${input.team_id}`,
        );
      }
    }
    return this.store.createTask(input);
  }

  async getTaskById(taskId: string): Promise<TaskEntity | null> {
    return this.store.getTaskById(taskId);
  }

  async updateTask(taskId: string, patch: Partial<TaskEntity>): Promise<TaskEntity> {
    if (!(await this.getTaskById(taskId))) throw new MetadataError("task_not_found", `task not found: ${taskId}`);
    const updated = await this.store.updateTask(taskId, patch);
    if (!updated) throw new MetadataError("task_not_found", `task not found: ${taskId}`);
    return updated;
  }

  async deleteTasks(taskIds: string[]): Promise<BatchDeleteResult> {
    return this.store.deleteTasks(taskIds);
  }

  async listTasksByTeam(
    teamId: string,
    pagination: PaginationParams = DEFAULT_PAGINATION,
    filter?: TaskFilter,
  ): Promise<PaginatedResult<TaskEntity>> {
    const page = await this.store.listTasksByTeam(teamId, pagination, filter);
    const items = page.items;
    return formatListResult({ items, total: page.total }, pagination);
  }

  async listTasks(filter: TaskFilter, pagination: PaginationParams = DEFAULT_PAGINATION): Promise<PaginatedResult<TaskEntity>> {
    const page = await this.store.listTasks(filter, pagination);
    const items = page.items;
    return formatListResult({ items, total: page.total }, pagination);
  }

  /** 归档（软关闭）task：status → completed。 */
  async archiveTask(taskId: string): Promise<TaskEntity> {
    return this.updateTask(taskId, { status: "completed" });
  }

  // ============================================================
  // TaskAgent
  // ============================================================
  async linkTaskAgent(taskId: string, agentId: string, roleInTask?: string): Promise<TaskAgentEntity> {
    const task = await this.getTaskById(taskId);
    if (!task) throw new MetadataError("task_not_found", `task not found: ${taskId}`);
    const agent = await this.getAgentById(agentId);
    if (!agent) throw new MetadataError("agent_not_found", `agent not found: ${agentId}`);
    if (agent.team_id !== task.team_id) {
      throw new MetadataError("agent_team_mismatch", `agent ${agentId} not in team ${task.team_id}`);
    }
    return this.store.linkTaskAgent(taskId, agentId, roleInTask);
  }

  async unlinkTaskAgent(taskId: string, agentId: string): Promise<void> {
    if (!(await this.getTaskById(taskId))) throw new MetadataError("task_not_found", `task not found: ${taskId}`);
    await this.store.unlinkTaskAgent(taskId, agentId);
  }

  async listTaskAgents(taskId: string, pagination: PaginationParams = DEFAULT_PAGINATION): Promise<PaginatedResult<TaskAgentEntity>> {
    const page = await this.store.listTaskAgents(taskId, pagination);
    return formatListResult(page, pagination);
  }

  // ============================================================
  // ParticipationLog
  // ============================================================
  async appendParticipationLog(input: AppendParticipationLogInput): Promise<ParticipationLogEntity> {
    await this.assertParticipationContext(input.team_id, input.task_id, input.agent_id, input.user_id);
    return this.store.appendParticipationLog(input);
  }

  async listParticipationLogs(
    filter: ParticipationLogFilter,
    pagination: PaginationParams = DEFAULT_PAGINATION,
  ): Promise<PaginatedResult<ParticipationLogEntity>> {
    const page = await this.store.listParticipationLogs(filter, pagination);
    return formatListResult(page, pagination);
  }

  private async assertParticipationContext(
    teamId: string,
    taskId: string,
    agentId: string,
    userId: string,
  ): Promise<void> {
    await this.assertTeamExists(teamId);
    const task = await this.getTaskById(taskId);
    if (!task) throw new MetadataError("task_not_found", `task not found: ${taskId}`);
    if (task.team_id !== teamId) {
      throw new MetadataError("permission_denied", `task ${taskId} not in team ${teamId}`);
    }
    const agent = await this.getAgentById(agentId);
    if (!agent) throw new MetadataError("agent_not_found", `agent not found: ${agentId}`);
    // if (agent.team_id !== teamId) {
    //   throw new MetadataError("agent_team_mismatch", `agent ${agentId} not in team ${teamId}`);
    // }
    const member = await this.getTeamMember(teamId, userId);
    if (!member || member.status !== "active") {
      throw new MetadataError("member_not_found", `member not found: ${teamId}/${userId}`);
    }
    // const links = await this.store.listTaskAgents(taskId, { limit: 1000, offset: 0 });
    // if (!links.items.some((l) => l.agent_id === agentId)) {
    //   throw new MetadataError("task_agent_not_linked", `task ${taskId} not linked to agent ${agentId}`);
    // }
  }

  // ============================================================
  // Asset（仅主表）
  // ============================================================
  async createAsset(input: CreateAssetInput): Promise<AssetEntity> {
    await this.assertTeamExists(input.team_id);
    return this.store.createAsset(input);
  }

  async getAssetById(assetId: string): Promise<AssetEntity | null> {
    return this.store.getAssetById(assetId);
  }

  async updateAsset(assetId: string, patch: Partial<AssetEntity>): Promise<AssetEntity> {
    if (!(await this.getAssetById(assetId))) throw new MetadataError("asset_not_found", `asset not found: ${assetId}`);
    const updated = await this.store.updateAsset(assetId, patch);
    if (!updated) throw new MetadataError("asset_not_found", `asset not found: ${assetId}`);
    return updated;
  }

  async deleteAssets(assetIds: string[]): Promise<BatchDeleteResult> {
    const result = await this.store.deleteAssets(assetIds);
    // 清 ensure 缓存，避免删后短路径误判「仍存在」
    for (const id of result.deleted_ids) {
      this.ensuredSkillAssets.delete(id);
      this.ensuredChatMemoryAssets.delete(id);
    }
    return result;
  }

  async listAssetsByTeam(
    teamId: string,
    pagination: PaginationParams = DEFAULT_PAGINATION,
    filter?: AssetFilter,
  ): Promise<PaginatedResult<AssetEntity>> {
    const page = await this.store.listAssetsByTeam(teamId, pagination, filter);
    const items = page.items;
    return formatListResult({ items, total: page.total }, pagination);
  }

  async touchAssetUsage(assetId: string): Promise<void> {
    if (!(await this.getAssetById(assetId))) throw new MetadataError("asset_not_found", `asset not found: ${assetId}`);
    await this.store.touchAssetUsage(assetId);
  }

  async listAgentFixedAssets(
    agentId: string,
    pagination: PaginationParams = DEFAULT_PAGINATION,
  ): Promise<PaginatedResult<FixedAssetBindingEntity>> {
    const page = await this.store.listAgentFixedAssets(agentId, pagination);
    return formatListResult(page, pagination);
  }

  /**
   * 多 agent 固定资产分配汇总。缺失 agent / type 补 0；items 顺序与请求 agent_ids 一致。
   * 可选 asset_id：只统计绑定了该资产的行（用于 bound_agent_count）。
   */
  async summarizeAgentFixedAssetsByAgents(
    params: SummarizeAgentFixedAssetsParams,
  ): Promise<AgentFixedAssetSummaryResult> {
    const agentIds = [...new Set(params.agent_ids.filter((id) => id.length > 0))];
    if (agentIds.length === 0) {
      return { items: [], total: 0 };
    }

    const emptyCounts = (): FixedAssetTypeCounts => ({
      skill: 0,
      code_graph: 0,
      llm_wiki: 0,
      chat_memory: 0,
    });

    const rows = await this.store.summarizeAgentFixedAssetsByAgents(agentIds, {
      assetId: params.asset_id,
    });

    const byAgent = new Map<string, FixedAssetTypeCounts>();
    const totals = new Map<string, number>();
    for (const id of agentIds) {
      byAgent.set(id, emptyCounts());
      totals.set(id, 0);
    }
    for (const row of rows) {
      const counts = byAgent.get(row.agent_id);
      if (!counts) continue;
      if (row.asset_type in counts) {
        counts[row.asset_type] = row.cnt;
      }
      totals.set(row.agent_id, (totals.get(row.agent_id) ?? 0) + row.cnt);
    }

    const items: AgentFixedAssetSummary[] = agentIds.map((agent_id) => ({
      agent_id,
      counts: byAgent.get(agent_id) ?? emptyCounts(),
      total: totals.get(agent_id) ?? 0,
    }));

    return { items, total: items.length };
  }

  // ============================================================
  // AgentFixedAsset（canBindAsset 校验 + 详情聚合）
  // ============================================================
  async setAgentFixedAssets(agentId: string, bindings: FixedAssetBindingInput[]): Promise<void> {
    const agent = await this.getAgentById(agentId);
    if (!agent) throw new MetadataError("agent_not_found", `agent not found: ${agentId}`);

    for (const b of bindings) {
      const asset = await this.getAssetById(b.asset_id);
      if (!asset) {
        throw new MetadataError("asset_not_found", `asset not found: ${b.asset_id}`);
      }
      if (!canBindAsset(agent, asset)) {
        throw new MetadataError(
          "asset_not_bindable",
          `asset ${b.asset_id} (visibility=${asset.visibility}) cannot bind to agent ${agentId}`,
        );
      }
    }
    await this.store.setAgentFixedAssets(agentId, bindings);
  }

  /**
   * 追加一条 agent 绑定（保留已有绑定）。用于自动登记 chat_memory 资产等
   * 增量场景；不同于 setAgentFixedAssets 的全量替换。
   *
   * 校验：agent / asset 必须都在当前 instance；canBindAsset 必须通过。
   * 幂等：store 层靠 (agent_id, asset_id) unique 约束，重复调用无副作用。
   */
  async addAgentFixedAsset(agentId: string, b: FixedAssetBindingInput): Promise<void> {
    const agent = await this.getAgentById(agentId);
    if (!agent) throw new MetadataError("agent_not_found", `agent not found: ${agentId}`);
    const asset = await this.getAssetById(b.asset_id);
    if (!asset) throw new MetadataError("asset_not_found", `asset not found: ${b.asset_id}`);
    if (!canBindAsset(agent, asset)) {
      throw new MetadataError(
        "asset_not_bindable",
        `asset ${b.asset_id} (visibility=${asset.visibility}) cannot bind to agent ${agentId}`,
      );
    }
    await this.store.addAgentFixedAsset(agentId, b);
  }

  /**
   * 幂等确保 (team, agent) 对应的 chat_memory 资产存在并已绑定到 agent。
   *
   * 首次调用时会同步完成三件事（严格顺序）：
   *   1. createAsset({asset_type:'chat_memory', visibility:'private',
   *      owner_user_id: agent.owner_user_id})
   *   2. store.addAgentFixedAsset(agent, {asset_id, injection_mode:'summary'})
   *   3. 写入进程内 LRU 缓存，后续同 (team, agent) 请求直接短路
   *
   * 幂等保证：
   *   - asset_id = chat_memory-{team}-{agent} 是稳定确定的
   *   - meta_assets.asset_id 是主键 → 并发 create 撞冲突后回读
   *   - meta_agent_fixed_assets (agent_id, asset_id) 是 unique → 重复
   *     addAgentFixedAsset 由 store 层吸收为 no-op
   *
   * 失败策略：本方法**会抛错**（agent_not_found / team_mismatch / DB 故障）。
   * 调用方（v2-router 里 handleConversationAdd）负责 catch + 只打 warn，
   * 不阻塞主流程 conversation 写入。
   */
  async ensureChatMemoryAsset(params: {
    team_id: string;
    agent_id: string;
  }): Promise<AssetEntity> {
    const assetId = buildChatMemoryAssetId(params.team_id, params.agent_id);

    // 1. 缓存短路：已确认存在直接返回轻量占位（如果调用方需要实体，才回 store）
    //    实践中调用方并不消费返回值（fire-and-forget），命中缓存时不再查 store。
    if (this.ensuredChatMemoryAssets.has(assetId)) {
      const cached = await this.getAssetById(assetId);
      if (cached) return cached;
      // 缓存脏了（被外部删除）—— 清掉重来
      this.ensuredChatMemoryAssets.delete(assetId);
    }

    // 2. 拿 agent，取 owner + team 用于 create + canBindAsset
    //    先拉 agent 是为了在任何路径下都能校验 team_mismatch，同时后续 bind
    //    要用到 owner_user_id 作 created_by。
    const agent = await this.getAgentById(params.agent_id);
    if (!agent) {
      throw new MetadataError(
        "agent_not_found",
        `cannot ensure chat_memory asset: agent ${params.agent_id} not found`,
      );
    }
    if (agent.team_id !== params.team_id) {
      throw new MetadataError(
        "team_mismatch",
        `cannot ensure chat_memory asset: agent ${params.agent_id} belongs to team ` +
        `${agent.team_id}, not ${params.team_id}`,
      );
    }

    // 3. 拿或建 asset：先看是否已在 store（冷启动 / 其他 pod 已建），否则新建。
    //    createAsset 遇主键冲突 = 并发 race，回读兜底。
    let asset = await this.getAssetById(assetId);
    if (!asset) {
      try {
        asset = await this.createAsset({
          asset_id: assetId,
          team_id: params.team_id,
          asset_type: "chat_memory",
          name: `Memory of ${agent.name}`,
          owner_user_id: agent.owner_user_id,
          source_type: "auto",
          visibility: "private",
          status: "active",
        });
      } catch (err) {
        const raced = await this.getAssetById(assetId);
        if (raced) {
          asset = raced;
        } else {
          throw err;
        }
      }
    }

    // 4. 无论 asset 是新建还是已存在，都要**幂等**补一次绑定。
    //    上一次可能只完成 create、bind 阶段失败；bind 有 UNIQUE 约束，重复
    //    调用无副作用。这里直接调 store 跳过 addAgentFixedAsset 的重复校验
    //    —— 我们上面已经查过 agent / asset。
    await this.store.addAgentFixedAsset(params.agent_id, {
      asset_id: assetId,
      asset_type: "chat_memory",
      injection_mode: "summary",
      priority: 50,
      created_by: agent.owner_user_id,
    });

    this.rememberEnsuredChatMemoryAsset(assetId);
    return asset;
  }

  /** LRU-ish 记录：达到上限时淘汰最早写入的条目。 */
  private rememberEnsuredChatMemoryAsset(assetId: string): void {
    if (this.ensuredChatMemoryAssets.has(assetId)) return;
    if (this.ensuredChatMemoryAssets.size >= MetadataService.CHAT_MEMORY_ENSURE_CACHE_SIZE) {
      const oldest = this.ensuredChatMemoryAssets.keys().next().value;
      if (oldest !== undefined) this.ensuredChatMemoryAssets.delete(oldest);
    }
    this.ensuredChatMemoryAssets.set(assetId, true);
  }

  //  ============================================================
  //  Skill Asset — 同款 ensure 模式
  //  ============================================================

  /**
   * 登记 skill 资产并绑定到 agent。与 ensureChatMemoryAsset 同款 5 步结构：
   *
   *   1. LRU 短路（key = skill_id，即 asset_id）
   *   2. 查 agent 取 owner + 校验 team
   *   3. 幂等 createAsset（asset_id = skill_id）
   *   4. 幂等 addAgentFixedAsset（injection_mode = reference）
   *   5. 记入 LRU
   *
   * 幂等保证：
   *   - asset_id 即为外部 skill_id（core 层生成 skl-xxxx），稳定唯一
   *   - meta_assets.asset_id 主键 → 并发 create 冲突时回读
   *   - meta_agent_fixed_assets (agent_id, asset_id) UNIQUE → bind 幂等
   *
   * 失败策略：
   *   - v1 创建路径（onSkillCreated context）：抛出异常以中断 create，
   *     避免 "skill 落库但前端看不到" 的不可自愈状态
   *   - 读时自愈路径（onSkillAccessed context）：由调用方 try/catch，
   *     不影响 skill 返回
   */
  async ensureSkillAsset(params: {
    skill_id: string;
    team_id: string;
    agent_id: string;
    name: string;
  }): Promise<AssetEntity> {
    const assetId = params.skill_id; // skill_id === asset_id（约定）

    // 1. LRU 短路
    if (this.ensuredSkillAssets.has(assetId)) {
      const cached = await this.getAssetById(assetId);
      if (cached) return cached;
      this.ensuredSkillAssets.delete(assetId);
    }

    // 2. 拿 agent 取 owner + 校验 team
    const agent = await this.getAgentById(params.agent_id);
    if (!agent) {
      throw new MetadataError(
        "agent_not_found",
        `cannot ensure skill asset: agent ${params.agent_id} not found`,
      );
    }
    if (agent.team_id !== params.team_id) {
      throw new MetadataError(
        "team_mismatch",
        `cannot ensure skill asset: agent ${params.agent_id} belongs to team ` +
        `${agent.team_id}, not ${params.team_id}`,
      );
    }

    // 3. 幂等 createAsset
    //
    // 默认 visibility = "private"（2026-07 变更）：
    //   - 新建的 skill 默认只有 owner 和 team admin 能看到（严格私密）。
    //   - 想让 team 内所有人可读 → 用户显式在管控页切成"共享"（asset/update visibility=team）。
    //   - 想让特定 user/agent 可读 → 用 acl/grant + visibility=restricted。
    //
    // 为什么不是 "team"：Skill 内容常包含内部知识、脚本、凭证注释等，
    // "默认对整个 team 可见"对隐私敏感场景（例如个人调试用的 skill）不够安全。
    // 私密 → 主动共享的心智更符合直觉。
    let asset = await this.getAssetById(assetId);
    if (!asset) {
      try {
        asset = await this.createAsset({
          asset_id: assetId,
          team_id: params.team_id,
          asset_type: "skill",
          name: params.name,
          owner_user_id: agent.owner_user_id,
          source_type: "extracted",
          visibility: "private",
          status: "active",
        });
      } catch (err) {
        const raced = await this.getAssetById(assetId);
        if (raced) {
          asset = raced;
        } else {
          throw err;
        }
      }
    }

    // 4. 幂等 addAgentFixedAsset
    await this.store.addAgentFixedAsset(params.agent_id, {
      asset_id: assetId,
      asset_type: "skill",
      injection_mode: "reference",
      priority: 50,
      created_by: agent.owner_user_id,
    });

    this.rememberEnsuredSkillAsset(assetId);
    return asset;
  }

  /** LRU-ish 记录：达到上限时淘汰最早写入的条目。 */
  private rememberEnsuredSkillAsset(assetId: string): void {
    if (this.ensuredSkillAssets.has(assetId)) return;
    if (this.ensuredSkillAssets.size >= MetadataService.SKILL_ENSURE_CACHE_SIZE) {
      const oldest = this.ensuredSkillAssets.keys().next().value;
      if (oldest !== undefined) this.ensuredSkillAssets.delete(oldest);
    }
    this.ensuredSkillAssets.set(assetId, true);
  }

  async listAgentFixedAssetsWithDetail(
    params: ListWithDetailParams,
  ): Promise<AgentFixedAssetDetailResult> {
    const agent = await this.getAgentById(params.agent_id);
    if (!agent) throw new MetadataError("agent_not_found", `agent not found: ${params.agent_id}`);

    const pagination = this.pag(params);
    const bindingPage = await this.store.listAgentFixedAssets(params.agent_id, pagination);
    const items: AgentAssetView[] = [];

    for (const b of bindingPage.items) {
      const asset = await this.getAssetById(b.asset_id);
      if (!asset) continue;

      if (FILTERED_STATUSES.includes(asset.status)) continue;

      if (params.apply_visibility_filter && !canBindAsset(agent, asset)) continue;

      if (params.touch_usage) {
        await this.store.touchAssetUsage(asset.asset_id);
      }

      items.push({
        asset_id: asset.asset_id,
        asset_type: asset.asset_type,
        name: asset.name,
        description: asset.description ?? null,
        status: asset.status,
        visibility: asset.visibility,
        injection_mode: b.injection_mode,
        priority: b.priority,
        created_at: asset.created_at,
      });
    }

    return {
      agent: {
        agent_id: agent.agent_id,
        team_id: agent.team_id,
        owner_user_id: agent.owner_user_id,
        prompt: agent.prompt ?? null,
        visibility: agent.visibility,
        status: agent.status,
      },
      items,
      total: bindingPage.total,
      limit: pagination.limit,
      offset: pagination.offset,
    };
  }

  // ============================================================
  // ACL
  // ============================================================
  async grantAcl(input: GrantAclInput): Promise<AclEntity> {
    const asset = await this.getAssetById(input.asset_id);
    if (!asset) throw new MetadataError("asset_not_found", `asset not found: ${input.asset_id}`);
    return this.store.grantAcl(input);
  }

  async revokeAcl(id: string): Promise<void> {
    await this.store.revokeAcl(id);
  }

  async listAclByAsset(assetId: string, pagination: PaginationParams = DEFAULT_PAGINATION): Promise<PaginatedResult<AclEntity>> {
    const page = await this.store.listAclByAsset(assetId, pagination);
    return formatListResult(page, pagination);
  }

  // ============================================================
  // 权限判定（懒加载 ACL）
  // ============================================================
  async checkAssetPermission(params: CheckPermissionParams): Promise<PermCheckResult> {
    const userId = await resolveUserId(this, params);
    const asset = await this.getAssetById(params.asset_id);
    if (!asset || asset.status === "archived") {
      return { allowed: false, reason: "asset_not_available" };
    }

    // owner 短路，无需查成员/ACL
    if (asset.owner_user_id === userId) {
      return { allowed: true, reason: "owner" };
    }

    const membership = await this.store.getTeamMember(asset.team_id, userId);

    // 先用空 ACL 跑一遍：命中角色默认即放行，无需查表
    const action = params.action;
    const fast = checkPermission({
      user: { user_id: userId },
      asset,
      membership,
      action,
      aclRecords: [],
      agentId: params.agent_id,
      logger: this.logger,
    });
    if (fast.allowed) return fast;

    // 只有「通过了前置门但角色默认未覆盖」(no_permission) 才需懒加载 ACL 重判
    if (fast.reason !== "no_permission") return fast;
    if (membership && roleDefaultCovers(membership.role, action)) return fast;

    const aclRecords = await this.allAclRecords(params.asset_id);
    return checkPermission({
      user: { user_id: userId },
      asset,
      membership,
      action,
      aclRecords,
      agentId: params.agent_id,
      logger: this.logger,
    });
  }

  /** 按用户过滤其有权限访问的资产列表（权限聚合后 offset 分页）。 */
  async listAccessibleAssets(params: ListAccessibleAssetsParams): Promise<PaginatedResult<AssetEntity>> {
    const userId = await resolveUserId(this, params);
    const action = params.action ?? "read";
    const pagination = this.pag(params);

    // visibility 白名单（服务端过滤，避免前端拿到不该看到的数据）
    const visFilter: Set<AssetEntity["visibility"]> | null = params.visibility
      ? new Set(Array.isArray(params.visibility) ? params.visibility : [params.visibility])
      : null;

    let teamIds: string[];
    if (params.team_id) {
      const member = await this.store.getTeamMember(params.team_id, userId);
      if (!member || member.status !== "active") {
        return paginateArray([], pagination);
      }
      teamIds = [params.team_id];
    } else {
      const allTeams: TeamEntity[] = [];
      let offset = 0;
      const limit = 100;
      while (true) {
        const page = await this.store.listTeamsByUser(userId, { limit, offset });
        allTeams.push(...page.items);
        if (offset + page.items.length >= page.total) break;
        offset += limit;
      }
      teamIds = allTeams.map((t) => t.team_id);
    }

    const result: AssetEntity[] = [];
    const seen = new Set<string>();

    for (const teamId of teamIds) {
      let offset = 0;
      const limit = 100;
      while (true) {
        const page = await this.store.listAssetsByTeam(
          teamId,
          { limit, offset },
          { asset_type: params.asset_type },
        );
        for (const asset of page.items) {
          if (seen.has(asset.asset_id)) continue;
          if (FILTERED_STATUSES.includes(asset.status)) continue;
          // visibility 白名单过滤（在权限判定前先剔除，节省 checkAssetPermission 开销）
          if (visFilter && !visFilter.has(asset.visibility)) continue;
          const perm = await this.checkAssetPermission({
            user_id: userId,
            asset_id: asset.asset_id,
            action,
            agent_id: params.agent_id,
          });
          if (perm.allowed) {
            seen.add(asset.asset_id);
            result.push(asset);
          }
        }
        if (offset + page.items.length >= page.total) break;
        offset += limit;
      }
    }

    result.sort((a, b) => b.created_at.localeCompare(a.created_at));
    return paginateArray(result, pagination);
  }

  // ============================================================
  // Helpers
  // ============================================================
  private async assertTeamExists(teamId: string): Promise<void> {
    const team = await this.store.getTeamById(teamId);
    if (!team) throw new MetadataError("team_not_found", `team not found: ${teamId}`);
  }

  private requireCallerId(ctx: V3AuthContext): string {
    if (!ctx.userId) {
      throw new MetadataError("permission_denied", "authentication required");
    }
    return ctx.userId;
  }

  private assertCallerIsResourceOwner(ctx: V3AuthContext, ownerUserId: string): void {
    const callerId = this.requireCallerId(ctx);
    if (callerId !== ownerUserId) {
      throw new MetadataError("permission_denied", "caller is not resource owner");
    }
  }

  private async requireActiveTeamMember(ctx: V3AuthContext, teamId: string): Promise<TeamMemberEntity> {
    const callerId = this.requireCallerId(ctx);
    const member = await this.store.getTeamMember(teamId, callerId);
    if (!member || member.status !== "active") {
      throw new MetadataError("permission_denied", "not a team member");
    }
    return member;
  }

  private async assertCallerIsTeamAdmin(ctx: V3AuthContext, teamId: string): Promise<void> {
    const member = await this.requireActiveTeamMember(ctx, teamId);
    if (member.role !== "admin") {
      throw new MetadataError("permission_denied", "caller is not team admin");
    }
  }

  private async assertCallerIsTeamOwnerOrAdmin(ctx: V3AuthContext, teamId: string): Promise<TeamEntity> {
    const callerId = this.requireCallerId(ctx);
    const team = await this.getTeamById(teamId);
    if (!team) throw new MetadataError("team_not_found", `team not found: ${teamId}`);
    if (team.owner_user_id === callerId) return team;
    await this.assertCallerIsTeamAdmin(ctx, teamId);
    return team;
  }

  private async assertCallerIsAgentOwner(ctx: V3AuthContext, agentId: string): Promise<AgentEntity> {
    const agent = await this.getAgentById(agentId);
    if (!agent) throw new MetadataError("agent_not_found", `agent not found: ${agentId}`);
    this.assertCallerIsResourceOwner(ctx, agent.owner_user_id);
    return agent;
  }

  private async assertCallerIsTaskCreator(ctx: V3AuthContext, taskId: string): Promise<TaskEntity> {
    const task = await this.getTaskById(taskId);
    if (!task) throw new MetadataError("task_not_found", `task not found: ${taskId}`);
    this.assertCallerIsResourceOwner(ctx, task.creator_user_id);
    return task;
  }

  private async assertCallerIsAssetOwner(ctx: V3AuthContext, assetId: string): Promise<AssetEntity> {
    const asset = await this.getAssetById(assetId);
    if (!asset) throw new MetadataError("asset_not_found", `asset not found: ${assetId}`);
    this.assertCallerIsResourceOwner(ctx, asset.owner_user_id);
    return asset;
  }

  private async assertCallerIsAssetOwnerOrTeamAdmin(ctx: V3AuthContext, assetId: string): Promise<AssetEntity> {
    const asset = await this.getAssetById(assetId);
    if (!asset) throw new MetadataError("asset_not_found", `asset not found: ${assetId}`);
    const callerId = this.requireCallerId(ctx);
    if (asset.owner_user_id === callerId) return asset;
    await this.assertCallerIsTeamAdmin(ctx, asset.team_id);
    return asset;
  }

  // ============================================================
  // Caller-scoped mutations（L-12 / L-14）
  // ============================================================
  async createTeamForCaller(input: CreateTeamInput, ctx: V3AuthContext): Promise<TeamEntity> {
    this.assertCallerIsResourceOwner(ctx, input.owner_user_id);
    return this.createTeam(input);
  }

  async updateTeamForCaller(
    teamId: string,
    patch: Partial<TeamEntity>,
    ctx: V3AuthContext,
  ): Promise<TeamEntity> {
    await this.assertCallerIsTeamOwnerOrAdmin(ctx, teamId);
    return this.updateTeam(teamId, patch);
  }

  async deleteTeamsForCaller(teamIds: string[], ctx: V3AuthContext): Promise<BatchDeleteResult> {
    for (const teamId of teamIds) {
      await this.assertCallerIsTeamOwnerOrAdmin(ctx, teamId);
    }
    return this.deleteTeams(teamIds);
  }

  async addTeamMemberForCaller(input: AddTeamMemberInput, ctx: V3AuthContext): Promise<TeamMemberEntity> {
    await this.assertCallerIsTeamAdmin(ctx, input.team_id);
    const callerId = this.requireCallerId(ctx);
    // 「添加成员」不应用来改自己的角色；选自己 + role=member 会把 admin 降级。
    if (input.user_id === callerId) {
      throw new MetadataError("permission_denied", "cannot add yourself as a team member");
    }
    return this.addTeamMember(input);
  }

  async removeTeamMemberForCaller(teamId: string, userId: string, ctx: V3AuthContext): Promise<void> {
    await this.assertCallerIsTeamAdmin(ctx, teamId);
    const team = await this.getTeamById(teamId);
    if (!team) throw new MetadataError("team_not_found", `team not found: ${teamId}`);
    if (userId === team.owner_user_id) {
      throw new MetadataError("permission_denied", "cannot remove team owner");
    }
    return this.removeTeamMember(teamId, userId);
  }

  async listTeamMembersForCaller(
    teamId: string,
    ctx: V3AuthContext,
    pagination: PaginationParams = DEFAULT_PAGINATION,
  ): Promise<PaginatedResult<TeamMemberView>> {
    await this.requireActiveTeamMember(ctx, teamId);
    const page = await this.store.listTeamMembersWithProfile(teamId, pagination);
    return formatListResult(page, pagination);
  }

  async getTeamMemberForCaller(
    teamId: string,
    userId: string,
    ctx: V3AuthContext,
  ): Promise<TeamMemberView> {
    await this.requireActiveTeamMember(ctx, teamId);
    const member = await this.store.getTeamMemberWithProfile(teamId, userId);
    if (!member) {
      throw new MetadataError("member_not_found", `member not found: ${teamId}/${userId}`);
    }
    return member;
  }

  async createAgentForCaller(input: CreateAgentInput, ctx: V3AuthContext): Promise<AgentEntity> {
    await this.assertTeamExists(input.team_id);
    await this.requireActiveTeamMember(ctx, input.team_id);
    this.assertCallerIsResourceOwner(ctx, input.owner_user_id);
    return this.createAgent(input);
  }

  async updateAgentForCaller(
    agentId: string,
    patch: Partial<AgentEntity>,
    ctx: V3AuthContext,
  ): Promise<AgentEntity> {
    await this.assertCallerIsAgentOwner(ctx, agentId);
    return this.updateAgent(agentId, patch);
  }

  async deleteAgentsForCaller(agentIds: string[], ctx: V3AuthContext): Promise<BatchDeleteResult> {
    for (const agentId of agentIds) {
      await this.assertCallerIsAgentOwner(ctx, agentId);
    }
    return this.deleteAgents(agentIds);
  }

  async archiveAgentForCaller(agentId: string, ctx: V3AuthContext): Promise<AgentEntity> {
    await this.assertCallerIsAgentOwner(ctx, agentId);
    return this.archiveAgent(agentId);
  }

  async createTaskForCaller(input: CreateTaskInput, ctx: V3AuthContext): Promise<TaskEntity> {
    await this.assertTeamExists(input.team_id);
    await this.requireActiveTeamMember(ctx, input.team_id);
    this.assertCallerIsResourceOwner(ctx, input.creator_user_id);
    return this.createTask(input);
  }

  async updateTaskForCaller(
    taskId: string,
    patch: Partial<TaskEntity>,
    ctx: V3AuthContext,
  ): Promise<TaskEntity> {
    await this.assertCallerIsTaskCreator(ctx, taskId);
    return this.updateTask(taskId, patch);
  }

  async deleteTasksForCaller(taskIds: string[], ctx: V3AuthContext): Promise<BatchDeleteResult> {
    for (const taskId of taskIds) {
      await this.assertCallerIsTaskCreator(ctx, taskId);
    }
    return this.deleteTasks(taskIds);
  }

  async archiveTaskForCaller(taskId: string, ctx: V3AuthContext): Promise<TaskEntity> {
    await this.assertCallerIsTaskCreator(ctx, taskId);
    return this.archiveTask(taskId);
  }

  async linkTaskAgentForCaller(
    taskId: string,
    agentId: string,
    roleInTask: string | undefined,
    ctx: V3AuthContext,
  ): Promise<TaskAgentEntity> {
    await this.assertCallerIsTaskCreator(ctx, taskId);
    return this.linkTaskAgent(taskId, agentId, roleInTask);
  }

  async unlinkTaskAgentForCaller(taskId: string, agentId: string, ctx: V3AuthContext): Promise<void> {
    await this.assertCallerIsTaskCreator(ctx, taskId);
    return this.unlinkTaskAgent(taskId, agentId);
  }

  async appendParticipationLogForCaller(
    input: AppendParticipationLogInput,
    ctx: V3AuthContext,
  ): Promise<ParticipationLogEntity> {
    await this.requireActiveTeamMember(ctx, input.team_id);
    const callerId = this.requireCallerId(ctx);
    if (input.user_id !== callerId) {
      await this.assertCallerIsTeamAdmin(ctx, input.team_id);
    }
    return this.appendParticipationLog(input);
  }

  async listParticipationLogsForCaller(
    filter: ParticipationLogFilter,
    ctx: V3AuthContext,
    pagination: PaginationParams = DEFAULT_PAGINATION,
  ): Promise<PaginatedResult<ParticipationLogEntity>> {
    await this.requireActiveTeamMember(ctx, filter.team_id);
    return this.listParticipationLogs(filter, pagination);
  }

  async createAssetForCaller(input: CreateAssetInput, ctx: V3AuthContext): Promise<AssetEntity> {
    await this.assertTeamExists(input.team_id);
    await this.requireActiveTeamMember(ctx, input.team_id);
    this.assertCallerIsResourceOwner(ctx, input.owner_user_id);
    return this.createAsset(input);
  }

  async updateAssetForCaller(
    assetId: string,
    patch: Partial<AssetEntity>,
    ctx: V3AuthContext,
  ): Promise<AssetEntity> {
    await this.assertCallerIsAssetOwner(ctx, assetId);
    return this.updateAsset(assetId, patch);
  }

  async deleteAssetsForCaller(assetIds: string[], ctx: V3AuthContext): Promise<BatchDeleteResult> {
    // 已不存在的 id 跳过 owner 校验（与 store 层幂等成功对齐）；仍存在的须为 owner。
    for (const assetId of assetIds) {
      const existing = await this.getAssetById(assetId);
      if (!existing) continue;
      await this.assertCallerIsAssetOwner(ctx, assetId);
    }
    return this.deleteAssets(assetIds);
  }

  async touchAssetUsageForCaller(assetId: string, ctx: V3AuthContext): Promise<void> {
    await this.assertCallerIsAssetOwner(ctx, assetId);
    return this.touchAssetUsage(assetId);
  }

  async setAgentFixedAssetsForCaller(
    agentId: string,
    bindings: FixedAssetBindingInput[],
    ctx: V3AuthContext,
  ): Promise<void> {
    await this.assertCallerIsAgentOwner(ctx, agentId);
    return this.setAgentFixedAssets(agentId, bindings);
  }

  async grantAclForCaller(input: GrantAclInput, ctx: V3AuthContext): Promise<AclEntity> {
    await this.assertCallerIsAssetOwner(ctx, input.asset_id);
    const callerId = this.requireCallerId(ctx);
    if (input.granted_by !== callerId) {
      throw new MetadataError("permission_denied", "granted_by must match caller");
    }
    return this.grantAcl(input);
  }

  async revokeAclForCaller(id: string, ctx: V3AuthContext): Promise<void> {
    const acl = await this.store.getAclById(id);
    if (!acl) throw new MetadataError("acl_not_found", `acl not found: ${id}`);
    await this.assertCallerIsAssetOwner(ctx, acl.asset_id);
    return this.revokeAcl(id);
  }

  async listAclByAssetForCaller(
    assetId: string,
    ctx: V3AuthContext,
    pagination: PaginationParams = DEFAULT_PAGINATION,
  ): Promise<PaginatedResult<AclEntity>> {
    await this.assertCallerIsAssetOwnerOrTeamAdmin(ctx, assetId);
    return this.listAclByAsset(assetId, pagination);
  }
}
