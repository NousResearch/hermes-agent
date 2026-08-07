/**
 * 实体引用存在性校验工具。
 *
 * 供 v3-meta-router handler 在业务调用前校验输入的带前缀元数据 ID
 * 是否在 store 中存在；不存在则抛 MetadataError（→ 404）。
 *
 * 使用方式：
 *   await requireEntity(svc, EntityType.User, d.user_id);
 */

import { MetadataService, MetadataError } from "../service/metadata-service.js";

/** 元数据实体类型枚举，用于 requireEntity 调用。 */
export const enum EntityType {
  User = "user",
  Team = "team",
  Agent = "agent",
  Task = "task",
  Asset = "asset",
  UserKey = "userKey",
  Acl = "acl",
}

const LOOKUP: Record<EntityType, (svc: MetadataService, id: string) => Promise<unknown>> = {
  [EntityType.User]: (svc, id) => svc.getUserById(id),
  [EntityType.Team]: (svc, id) => svc.getTeamById(id),
  [EntityType.Agent]: (svc, id) => svc.getAgentById(id),
  [EntityType.Task]: (svc, id) => svc.getTaskById(id),
  [EntityType.Asset]: (svc, id) => svc.getAssetById(id),
  [EntityType.UserKey]: (svc, id) => svc.rawStore.getUserKeyById(id),
  [EntityType.Acl]: (svc, id) => svc.rawStore.getAclById(id),
};

const ERROR_CODE: Record<EntityType, string> = {
  [EntityType.User]: "user_not_found",
  [EntityType.Team]: "team_not_found",
  [EntityType.Agent]: "agent_not_found",
  [EntityType.Task]: "task_not_found",
  [EntityType.Asset]: "asset_not_found",
  [EntityType.UserKey]: "user_key_not_found",
  [EntityType.Acl]: "acl_not_found",
};

/**
 * 校验单个实体 ID 存在性，不存在则抛 MetadataError（映射为 404）。
 *
 * @param svc  当前实例的 MetadataService
 * @param type 实体类型枚举
 * @param id   带前缀的实体 ID
 */
export async function requireEntity(
  svc: MetadataService,
  type: EntityType,
  id: string,
): Promise<void> {
  const entity = await LOOKUP[type](svc, id);
  if (!entity) {
    throw new MetadataError(ERROR_CODE[type], `not found: ${id}`);
  }
}
