/**
 * Skill 模块入口 — v2 redesign (2026-06-17)
 *
 * 设计文档：docs/design/2026-06-17-skill-redesign-v2.md
 *
 * 单表多版本（skill_id, version 联合唯一），DB 是 SKILL.md 与 manifest 的权威
 * 源；storage 只存资源字节。绑定语义全部上交管控面，数据面按
 * (user_id, owner_agent_id, team_id, task_id, skill_id) 五元组身份记录。
 */

// 类型
export type {
  IdFields,
  SkillStatus,
  SkillManifestEntry,
  Skill,
  AppendVersionInput,
  ListSkillsOptions,
  SearchSkillsOptions,
  ExtractMessage,
} from "./types.js";

// 配置解析
export type {
  SkillConfigInput,
  ResolvedSkillConfig,
  SkillDegradation,
  SkillEnvProbe,
  SkillSimilarityResult,
  SkillProposeResult,
} from "./types.js";
export { resolveSkillConfig } from "./skill-config.js";
export type { ResolverLogger } from "./skill-config.js";

// SKILL.md format helpers
export {
  parseSkillFile,
  validateSkillFile,
  formatSkillFile,
} from "./skill-format.js";

// DDL 常量
export {
  SKILLS_DDL,
  SKILL_FTS_DDL,
  SKILL_VEC_DDL_TEMPLATE,
  FTS_CONTENT_MAX,
} from "./skill-store-ddl.js";

// 存储接口抽象
export type {
  ISkillStore,
  SkillStoreCapabilities,
  SkillSearchResult,
  ExpiredVersionMeta,
} from "./skill-store.interface.js";

// 数据访问层
export {
  SqliteSkillStore,
  SkillStoreError,
  IdempotentNoOpError,
  type SkillErrorCode as SkillStoreErrorCode,
  type SqliteSkillStoreOptions,
} from "./skill-store.js";

// 资源层
export {
  SkillResourceStore,
  SkillResourceError,
  type SkillResourcePayload,
  type SkillResourceReadResult,
  type ResourceErrorCode,
} from "./skill-resource-store.js";

// 版本编排
export {
  SkillVersioning,
  type SkillVersioningOptions,
  type AppendVersionContext,
  type AppendVersionMutation,
} from "./skill-versioning.js";

// 权限工具
export {
  SkillPermissionError,
  assertOwner,
  assertTeamMatch,
  assertVersionFresh,
  type SkillPermissionErrorCode,
} from "./skill-permission.js";

// 核心门面
export {
  SkillCore,
  SkillCoreError,
  type SkillCoreErrorCode,
  type SkillCoreOptions,
  type CreateInput,
  type UpdateInput,
  type PatchInput,
  type DeleteInput,
  type GetInput,
  type WriteFilesInput,
  type RemoveFilesInput,
  type ReadFileInput,
  type ListInput,
  type SearchInput,
  type ListVersionsInput,
} from "./skill-core.js";

// 抽取链路
export {
  SkillExtractor,
  createExtractorAdapter,
  type ExtractorRunner,
  type ExtractorOptions,
  type ExtractInput,
  type ExtractResult,
} from "./skill-extractor.js";

export {
  createSkillTools,
  type ExtractedAction,
  type ExtractedSkillCandidate,
  type CreateSkillToolsOptions,
} from "./skill-tools.js";

// Listing prompt 常量
export {
  SKILL_LISTING_HEADER,
  SKILL_LISTING_FOOTER,
  SKILLS_GUIDANCE,
} from "./prompts/skill-listing-prompt.js";

// 抽取 prompt
export { SKILL_REVIEW_PROMPT } from "./prompts/skill-review-prompt.js";

// 抽取链路里 worker / dedupe 共用的 ExtractorLLMRunner（与 v2 ExtractorRunner 形状兼容）。
export type { ExtractorLLMRunner } from "./types.js";
