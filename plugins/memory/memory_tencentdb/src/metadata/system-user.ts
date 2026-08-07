/**
 * YAML 静态 memory 系统用户（仅 auth/verify 路径，不落库、不接入 authenticateV3）。
 */

import { timingSafeEqual } from "node:crypto";
import type { GatewayMetadataConfig } from "../gateway/config.js";
import type { MetadataDeployMode } from "./store/factory.js";
import { MetadataStartupValidationError } from "./store/factory.js";
import type { UserEntity } from "./types.js";
import { USER_KEY_PREFIX } from "./utils/user-key.js";

export const MEMORY_SYSTEM_USER_ID_PREFIX = "usr-sys-";
const MEMORY_SYSTEM_USERNAME = "memory";
const MEMORY_SYSTEM_AUTH_PROVIDER = "system";
const MEMORY_SYSTEM_EXTERNAL_ID = "memory";

export interface MemorySystemUserConfig {
  userId: string;
  displayName: string;
  userKey: string;
}

function env(key: string): string | undefined {
  const v = process.env[key]?.trim();
  return v || undefined;
}

function safeEqual(a: string, b: string): boolean {
  if (!a || !b) return false;
  const ab = Buffer.from(a);
  const bb = Buffer.from(b);
  if (ab.length !== bb.length) return false;
  return timingSafeEqual(ab, bb);
}

/** user_key 格式：sk-mem- + 32 位 base64url。 */
export function isValidMemorySystemUserKey(userKey: string): boolean {
  return /^sk-mem-[A-Za-z0-9_-]{32}$/.test(userKey);
}

/** 从 Gateway metadata + process.env 解析（env 优先）。 */
export function resolveMemorySystemUserConfig(
  metadata: GatewayMetadataConfig,
): MemorySystemUserConfig | undefined {
  const yaml = metadata.systemUser?.memory;
  const userId = env("TDAI_MEMORY_SYSTEM_USER_ID") ?? yaml?.userId?.trim();
  const displayName = env("TDAI_MEMORY_SYSTEM_USER_NAME") ?? yaml?.displayName?.trim() ?? MEMORY_SYSTEM_USERNAME;
  const userKey = env("TDAI_MEMORY_SYSTEM_USER_KEY") ?? yaml?.userKey?.trim();

  if (!userId && !userKey) return undefined;
  if (!userId || !userKey) return undefined;

  return { userId, displayName, userKey };
}

export function validateMemorySystemUserConfig(
  deployMode: MetadataDeployMode,
  config: MemorySystemUserConfig | undefined,
): void {
  if (deployMode !== "service") return;

  if (!config) {
    throw new MetadataStartupValidationError(
      "metadata.systemUser.memory is required when deployMode=service " +
      "(set yaml metadata.systemUser.memory or TDAI_MEMORY_SYSTEM_USER_ID / TDAI_MEMORY_SYSTEM_USER_KEY)",
    );
  }

  const errors: string[] = [];
  if (!config.userId.startsWith(MEMORY_SYSTEM_USER_ID_PREFIX)) {
    errors.push(
      `memory system userId must start with '${MEMORY_SYSTEM_USER_ID_PREFIX}' (got: '${config.userId}')`,
    );
  }
  if (!isValidMemorySystemUserKey(config.userKey)) {
    errors.push("memory system userKey must match sk-mem-[A-Za-z0-9_-]{32}");
  }
  if (!config.displayName.trim()) {
    errors.push("memory system displayName must be non-empty");
  }

  if (errors.length > 0) {
    throw new MetadataStartupValidationError(
      `Memory system user validation failed: ${errors.join("; ")}`,
    );
  }
}

function buildSyntheticUser(
  config: MemorySystemUserConfig,
): UserEntity {
  const now = new Date().toISOString();
  return {
    user_id: config.userId,
    password: null,
    auth_provider: MEMORY_SYSTEM_AUTH_PROVIDER,
    external_id: MEMORY_SYSTEM_EXTERNAL_ID,
    username: MEMORY_SYSTEM_USERNAME,
    display_name: config.displayName,
    email: null,
    raw_profile_json: "{}",
    status: "active",
    user_type: "normal",
    created_at: now,
    updated_at: now,
    metadata_json: "{}",
  };
}

/** 常量时间比较 userKey；命中则返回合成 UserEntity。 */
export function lookupMemorySystemUser(
  userKey: string,
  instanceId: string,
  config: MemorySystemUserConfig | undefined,
): UserEntity | null {
  void instanceId;
  if (!config || !userKey) return null;
  if (!safeEqual(userKey, config.userKey)) return null;
  return buildSyntheticUser(config);
}

/** Header API 鉴权用：配置中的 memory key 不可作 x-tdai-user-key。 */
export function isMemorySystemUserKey(
  userKey: string,
  config: MemorySystemUserConfig | undefined,
): boolean {
  if (!config || !userKey) return false;
  return safeEqual(userKey, config.userKey);
}

/** auth/verify 对外响应：隐藏内部 system 标记。 */
export function toMemorySystemVerifyUser(user: UserEntity): UserEntity {
  const { password: _pw, auth_provider: _ap, external_id: _ei, ...rest } = user;
  return {
    ...rest,
    auth_provider: "local",
    external_id: user.user_id,
    email: null,
    raw_profile_json: "{}",
    metadata_json: "{}",
  };
}

/** 启动日志用：脱敏 key 前缀。 */
export function maskMemorySystemUserKeyForLog(userKey: string): string {
  if (!userKey.startsWith(USER_KEY_PREFIX)) return "****";
  const tail = userKey.slice(-4);
  return `${USER_KEY_PREFIX}****${tail}`;
}
