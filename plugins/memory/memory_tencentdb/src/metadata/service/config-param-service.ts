/**
 * ConfigParamService — 配置参数业务逻辑层。
 *
 * 职责：
 *  - initDefaults：幂等种子化 global 默认参数
 *  - 生效值解析（user 覆盖 global）
 *  - 1 分钟 TTL 进程内读缓存
 *  - 面向 Router 的 Caller 方法（getUserConfigForCaller / setUserConfigForCaller）
 */

import type { IMetadataStore } from "../store/interface.js";
import type { ConfigParamEntity, UpsertConfigParamInput, AssetType } from "../types.js";
import {
  type ConfigParamRegistry,
  type ModuleDef,
  type ParamDef,
  getModuleDef,
  getParamDef,
  isModuleGlobalOnly,
  isUserWritable,
} from "../config/param-registry.js";
import { MetadataError } from "./metadata-service.js";

// ── 缓存 ──

interface CacheEntry {
  value: string;
  expireAt: number;
}

const CACHE_TTL_MS = 60_000;

// ── 响应类型 ──

export interface UserConfigViewItem {
  module: string;
  param_name: string;
  param_key: string;
  description: string;
  effective_value: string;
}

export interface UserConfigView {
  user_id: string;
  module: string;
  module_description: string;
  items: UserConfigViewItem[];
}

// ── Service 接口 ──

export interface IConfigParamService {
  initDefaults(registry: ConfigParamRegistry, quotaOverrides?: { maxUsersPerInstance?: number; maxTeamsPerInstance?: number }): Promise<void>;
  getEffectiveParam(module: string, paramName: string, userId?: string): Promise<string>;
  getEffectiveInt(module: string, paramName: string, userId?: string): Promise<number>;
  setUserParam(userId: string, module: string, paramName: string, value: string): Promise<ConfigParamEntity>;
  getInstanceQuotaLimits(): Promise<{ max_users_per_instance: number; max_teams_per_instance: number }>;
  getUserConfigView(userId: string, module: string, paramNames?: string[]): Promise<UserConfigView>;
  isAssetTypeEnabledForUser(userId: string, assetType: AssetType): Promise<boolean>;
  getUserConfigForCaller(data: { user_id: string; module: string; param_name?: string }): Promise<UserConfigView>;
  setUserConfigForCaller(data: { user_id: string; module: string; params: Record<string, string> }): Promise<{ ok: true }>;
}

// ── 实现 ──

export class ConfigParamService implements IConfigParamService {
  private readonly cache = new Map<string, CacheEntry>();

  constructor(
    private readonly store: IMetadataStore,
    private readonly registry: ConfigParamRegistry,
  ) {}

  // ── initDefaults ──

  async initDefaults(
    registry: ConfigParamRegistry,
    quotaOverrides?: { maxUsersPerInstance?: number; maxTeamsPerInstance?: number },
  ): Promise<void> {
    for (const [, moduleDef] of registry) {
      for (const param of moduleDef.params) {
        const existing = await this.store.getConfigParam(
          "global", null, moduleDef.module, param.param_name,
        );
        if (existing) continue;

        let value = param.param_value;
        if (quotaOverrides && moduleDef.module === "quota") {
          if (param.param_name === "max_users_per_instance" && quotaOverrides.maxUsersPerInstance) {
            value = String(quotaOverrides.maxUsersPerInstance);
          }
          if (param.param_name === "max_teams_per_instance" && quotaOverrides.maxTeamsPerInstance) {
            value = String(quotaOverrides.maxTeamsPerInstance);
          }
        }

        await this.store.upsertConfigParam({
          scope: "global",
          user_id: null,
          module: moduleDef.module,
          param_name: param.param_name,
          param_value: value,
          description: param.description,
        });
      }
    }
  }

  // ── 生效值解析 ──

  async getEffectiveParam(module: string, paramName: string, userId?: string): Promise<string> {
    const paramDef = getParamDef(this.registry, module, paramName);
    if (!paramDef) {
      throw new MetadataError("invalid_param_key", `unknown param: ${module}.${paramName}`);
    }

    if (!paramDef.allowed_scopes.includes("user") || !userId) {
      return this.getGlobalValue(module, paramName, paramDef);
    }

    const cacheKey = `user:${userId}:${module}:${paramName}`;
    const cached = this.getCached(cacheKey);
    if (cached !== undefined) return cached;

    const userRow = await this.store.getConfigParam("user", userId, module, paramName);
    if (userRow) {
      this.setCache(cacheKey, userRow.param_value);
      return userRow.param_value;
    }

    const globalValue = await this.getGlobalValue(module, paramName, paramDef);
    this.setCache(cacheKey, globalValue);
    return globalValue;
  }

  async getEffectiveInt(module: string, paramName: string, userId?: string): Promise<number> {
    const str = await this.getEffectiveParam(module, paramName, userId);
    const num = parseInt(str, 10);
    if (Number.isNaN(num)) {
      throw new MetadataError("invalid_param_value", `param ${module}.${paramName} value '${str}' is not a valid integer`);
    }
    return num;
  }

  // ── 用户写入 ──

  async setUserParam(userId: string, module: string, paramName: string, value: string): Promise<ConfigParamEntity> {
    const moduleDef = getModuleDef(this.registry, module);
    if (!moduleDef) {
      throw new MetadataError("unknown_module", `unknown module: ${module}`);
    }

    const paramDef = getParamDef(this.registry, module, paramName);
    if (!paramDef) {
      throw new MetadataError("invalid_param_key", `unknown param: ${module}.${paramName}`);
    }

    if (!isUserWritable(this.registry, module, paramName)) {
      throw new MetadataError("invalid_param_scope", `module '${module}' param '${paramName}' is global-only, cannot set for user scope`);
    }

    this.validateParamValue(moduleDef, paramDef, value);

    const result = await this.store.upsertConfigParam({
      scope: "user",
      user_id: userId,
      module,
      param_name: paramName,
      param_value: value,
      description: paramDef.description,
    });

    this.invalidateUserCache(userId, module);
    return result;
  }

  // ── 配额 ──

  async getInstanceQuotaLimits(): Promise<{ max_users_per_instance: number; max_teams_per_instance: number }> {
    const maxUsers = await this.getEffectiveInt("quota", "max_users_per_instance");
    const maxTeams = await this.getEffectiveInt("quota", "max_teams_per_instance");
    return { max_users_per_instance: maxUsers, max_teams_per_instance: maxTeams };
  }

  // ── 用户配置视图 ──

  async getUserConfigView(userId: string, module: string, paramNames?: string[]): Promise<UserConfigView> {
    const moduleDef = getModuleDef(this.registry, module);
    if (!moduleDef) {
      throw new MetadataError("unknown_module", `unknown module: ${module}`);
    }

    if (paramNames && paramNames.length > 0) {
      for (const pn of paramNames) {
        if (!getParamDef(this.registry, module, pn)) {
          throw new MetadataError("invalid_param_key", `unknown param: ${module}.${pn}`);
        }
      }
    }

    const targetParams = paramNames && paramNames.length > 0
      ? moduleDef.params.filter((p) => paramNames.includes(p.param_name))
      : moduleDef.params;

    const items: UserConfigViewItem[] = [];

    for (const paramDef of targetParams) {
      const effectiveValue = await this.getEffectiveParam(module, paramDef.param_name, userId);
      items.push({
        module,
        param_name: paramDef.param_name,
        param_key: `${module}.${paramDef.param_name}`,
        description: paramDef.description,
        effective_value: effectiveValue,
      });
    }

    return {
      user_id: userId,
      module,
      module_description: moduleDef.description,
      items,
    };
  }

  // ── AssetType 快捷方法 ──

  async isAssetTypeEnabledForUser(userId: string, assetType: AssetType): Promise<boolean> {
    const paramName = `${assetType}.enabled`;
    const value = await this.getEffectiveParam("asset_type", paramName, userId);
    return value === "1";
  }

  // ── Caller 方法（面向 Router） ──

  async getUserConfigForCaller(
    data: { user_id: string; module: string; param_name?: string },
  ): Promise<UserConfigView> {
    return this.getUserConfigView(
      data.user_id,
      data.module,
      data.param_name ? [data.param_name] : undefined,
    );
  }

  async setUserConfigForCaller(
    data: { user_id: string; module: string; params: Record<string, string> },
  ): Promise<{ ok: true }> {
    const moduleDef = getModuleDef(this.registry, data.module);
    if (!moduleDef) {
      throw new MetadataError("unknown_module", `unknown module: ${data.module}`);
    }

    if (isModuleGlobalOnly(this.registry, data.module)) {
      throw new MetadataError(
        "invalid_param_scope",
        `module '${data.module}' is global-only, cannot set for user scope`,
      );
    }

    for (const [paramName, value] of Object.entries(data.params)) {
      await this.setUserParam(data.user_id, data.module, paramName, value);
    }

    return { ok: true };
  }

  // ── 内部辅助 ──

  private async getGlobalValue(module: string, paramName: string, paramDef: ParamDef): Promise<string> {
    const cacheKey = `global:${module}:${paramName}`;
    const cached = this.getCached(cacheKey);
    if (cached !== undefined) return cached;

    const row = await this.store.getConfigParam("global", null, module, paramName);
    const value = row ? row.param_value : paramDef.param_value;
    this.setCache(cacheKey, value);
    return value;
  }

  private validateParamValue(_moduleDef: ModuleDef, paramDef: ParamDef, value: string): void {
    const isBooleanParam = paramDef.param_value === "0" || paramDef.param_value === "1";
    if (paramDef.allowed_scopes.includes("user") && isBooleanParam) {
      if (value !== "0" && value !== "1") {
        throw new MetadataError(
          "invalid_param_value",
          `param '${paramDef.param_name}' only accepts '0' or '1', got '${value}'`,
        );
      }
    }
  }

  private getCached(key: string): string | undefined {
    const entry = this.cache.get(key);
    if (!entry) return undefined;
    if (Date.now() > entry.expireAt) {
      this.cache.delete(key);
      return undefined;
    }
    return entry.value;
  }

  private setCache(key: string, value: string): void {
    this.cache.set(key, { value, expireAt: Date.now() + CACHE_TTL_MS });
  }

  private invalidateUserCache(userId: string, module: string): void {
    const prefix = `user:${userId}:${module}:`;
    for (const key of this.cache.keys()) {
      if (key.startsWith(prefix)) {
        this.cache.delete(key);
      }
    }
  }
}
