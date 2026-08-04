/**
 * Zod schemas for `/skill/*` (v2 redesign, 2026-06-17).
 *
 * 与 docs/design/2026-06-17-skill-redesign-v2.md §3.4 / §3.5 对齐。
 *
 * team_id / agent_id 约束：
 *   - agent_id 必须以 team_id 为命名空间（有 agent 必须有 team）
 *   - 可以只传 team_id 不传 agent_id（团队维度查询）
 *   - 可以都不传（全局查询，不限制 scope）
 *   - 写接口额外要求 user_id 也必须提供
 *   - user_id / task_id 独立可选
 */

import { z } from "zod";

// ═════════════════════════════════════════════════════════════════════
//  公共片段
// ═════════════════════════════════════════════════════════════════════

const idFieldsShape = {
  user_id: z.string().min(1).optional(),
  team_id: z.string().min(1).optional(),
  agent_id: z.string().min(1).optional(),
  task_id: z.string().min(1).optional(),
};

/** agent_id 必须以 team_id 为命名空间：有 agent 必须有 team；team 可以单独存在。 */
function refineAgentNeedsTeam(data: { team_id?: string; agent_id?: string }, ctx: z.RefinementCtx) {
  const hasAgent = !!data.agent_id;
  const hasTeam = !!data.team_id;
  if (hasAgent && !hasTeam) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "agent_id requires team_id — agent is scoped under a team",
      path: ["agent_id"],
    });
  }
}

export const idFieldsReadSchema = z.object(idFieldsShape).superRefine(refineAgentNeedsTeam);
export const idFieldsWriteSchema = z.object(idFieldsShape).superRefine(refineAgentNeedsTeam);

export const skillResourcePayloadSchema = z.object({
  path: z.string().min(1).max(512),
  content: z.string(),
  encoding: z.enum(["utf-8", "base64"]),
  mime_type: z.string().max(128).optional(),
  is_executable: z.boolean().optional(),
});

export const extractMessageSchema = z.object({
  // 五种 role, 对齐 conversation/add 侧的 conversationMessageSchema
  role: z.enum(["user", "assistant", "tool_call", "tool_result", "system"]),
  content: z.string().min(1),
  timestamp: z.string().datetime().optional(),
  // tool_call/tool_result 携带的锚点字段。tool_name 可选（OpenAI 协议无 tool_name
  // 字段, 见 conversationMessageSchema 上的注释）；tool_call_id 也 optional（schema 层不强制，
  // 具体校验由 direct-trigger handler 决定；本 schema 只做形状约束）。
  tool_name: z.string().min(1).max(128).optional(),
  tool_call_id: z.string().min(1).max(128).optional(),
});

/**
 * 与 extractMessageSchema 类似但允许 system role（对齐 /v3/skill/conversation/add
 * §11.1 请求体的 5 种 role）。
 *
 * tool_name / tool_call_id / timestamp（数值）用于 tool_call/tool_result:
 *   - tool_call_id: schema 上 optional, 但 conversation-add handler 会强制
 *     tool_call/tool_result 必须携带（配对锚点）
 *   - tool_name: schema 和 handler 都 optional —— OpenAI 协议 role=tool 消息
 *     没有 tool_name 字段, 强制反查是绕圈; 对 skill 抽取而言, content 才是关键
 */
export const conversationMessageSchema = z.object({
  role: z.enum(["user", "assistant", "tool_call", "tool_result", "system"]),
  content: z.string(),
  timestamp: z.union([z.number(), z.string()]).optional(),
  tool_name: z.string().min(1).max(128).optional(),
  tool_call_id: z.string().min(1).max(128).optional(),
});

export const paginationSchema = z.object({
  limit: z.number().int().min(1).max(1000).optional(),
  offset: z.number().int().min(0).optional(),
});

// ═════════════════════════════════════════════════════════════════════
//  请求 schemas
// ═════════════════════════════════════════════════════════════════════

export const createRequestSchema = z.object({
  ...idFieldsShape,
  name: z.string().min(1).max(64),
  content: z.string().min(1),
  resources: z.array(skillResourcePayloadSchema).max(100).optional(),
  metadata: z.record(z.string(), z.any()).optional(),
}).superRefine(refineAgentNeedsTeam);

export const updateRequestSchema = z.object({
  ...idFieldsShape,
  skill_id: z.string().min(1),
  expected_version: z.number().int().min(1),
  content: z.string().min(1),
}).superRefine(refineAgentNeedsTeam);

export const patchRequestSchema = z.object({
  ...idFieldsShape,
  skill_id: z.string().min(1),
  expected_version: z.number().int().min(1),
  old_string: z.string().min(1),
  new_string: z.string(),
  replace_all: z.boolean().optional(),
}).superRefine(refineAgentNeedsTeam);

export const deleteRequestSchema = z.object({
  ...idFieldsShape,
  skill_id: z.string().min(1),
  expected_version: z.number().int().min(1),
}).superRefine(refineAgentNeedsTeam);

export const getRequestSchema = z.object({
  ...idFieldsShape,
  skill_id: z.string().min(1),
  version: z.number().int().min(1).optional(),
  include_content: z.boolean().optional(),
  include_manifest: z.boolean().optional(),
}).superRefine(refineAgentNeedsTeam);

export const listRequestSchema = z.object({
  ...idFieldsShape,
  filters: z.object({
    owner_agent_id: z.string().min(1).optional(),
    name_prefix: z.string().min(1).max(64).optional(),
    status: z.array(z.enum(["active", "archived"])).optional(),
  }).optional(),
  pagination: paginationSchema.optional(),
}).superRefine(refineAgentNeedsTeam);

export const searchRequestSchema = z.object({
  ...idFieldsShape,
  query: z.string().min(1).max(2048),
  top_k: z.number().int().min(1).max(50).optional(),
  mode: z.enum(["bm25", "embedding", "hybrid"]).optional(),
  /** When "team", the handler strips agent_id before passing to core (team-wide search, no owner filter). */
  scope: z.enum(["team"]).optional(),
}).superRefine(refineAgentNeedsTeam);

export const versionsRequestSchema = z.object({
  ...idFieldsShape,
  skill_id: z.string().min(1),
  pagination: paginationSchema.optional(),
}).superRefine(refineAgentNeedsTeam);

export const filesWriteRequestSchema = z.object({
  ...idFieldsShape,
  skill_id: z.string().min(1),
  expected_version: z.number().int().min(1),
  files: z.array(skillResourcePayloadSchema).min(1).max(100),
}).superRefine(refineAgentNeedsTeam);

export const filesRemoveRequestSchema = z.object({
  ...idFieldsShape,
  skill_id: z.string().min(1),
  expected_version: z.number().int().min(1),
  paths: z.array(z.string().min(1)).min(1).max(100),
}).superRefine(refineAgentNeedsTeam);

export const filesReadRequestSchema = z.object({
  ...idFieldsShape,
  skill_id: z.string().min(1),
  version: z.number().int().min(1).optional(),
  path: z.string().min(1),
  encoding: z.enum(["utf-8", "base64"]).optional(),
}).superRefine(refineAgentNeedsTeam);

export const listingRequestSchema = z.object({
  ...idFieldsShape,
  query: z.string().max(2048).optional(),
  char_budget: z.number().int().min(0).max(64_000).optional(),
}).superRefine(refineAgentNeedsTeam);

/**
 * `POST /v3/skill/extract` — direct-trigger 归档一次会话切片，语义等价于
 * 手动触发一次 skill 抽取。契约照搬 `/v3/skill/conversation/add`：
 * 只是不做累计 / 阈值判定, 一次调用产生一个独立 archive + task。
 *
 * `space_id`：跟其他 12 个 skill 接口对齐 —— 从 `x-tdai-service-id` header 得到
 * (gateway 解析为 `auth.serviceId`)；body 里也接受，handler 优先用 body 值，
 * 缺省时回退到 `auth.serviceId`。两个值在设计上就该相等（都是"当前登录实例"），
 * 不等表明调用方传错实例。
 *
 * 详见 `docs/design/2026-07-17-skill-extract-direct-trigger-plan.md`。
 */
export const extractRequestSchema = z.object({
  space_id: z.string().min(1)
    .refine((v) => !v.includes("|"), "space_id must not contain '|'")
    .optional(),
  user_id: z.string().min(1).refine((v) => !v.includes("|"), "user_id must not contain '|'"),
  team_id: z.string().min(1).refine((v) => !v.includes("|"), "team_id must not contain '|'"),
  agent_id: z.string().min(1).refine((v) => !v.includes("|"), "agent_id must not contain '|'"),
  session_id: z.string().min(1).optional()
    .refine((v) => !v || !v.includes("|"), { message: "session_id must not contain '|'" }),
  task_id: z.string().min(1).max(128).optional(),   // 业务 task_ref_id, 透传到 SkillTaskEntry
  messages: z.array(extractMessageSchema).min(1).max(500),
  reason: z.string().min(1).max(500).optional(),
  options: z.object({
    max_iterations: z.number().int().min(1).max(64).optional(),
  }).optional(),
});

/**
 * `POST /v3/skill/conversation/add` — 每轮对话结束后, Client 传本轮增量 messages。
 *
 * 强约束（对齐 `docs/design/2026-07-15-skill-trigger-in-core-design.md` §11.1 & §13）：
 *   - session_id / user_id / team_id / agent_id 必填
 *   - 上述 4 个 ID 字段都不能包含 `|`（跟 Redis 队列元素分隔符冲突）
 *   - `space_id` 可选：跟其他 skill 接口对齐, 从 `x-tdai-service-id` header 得到
 *     (`auth.serviceId`); body 里传了也接受, handler 优先 body、缺省回落 auth
 *   - messages 非空; 每条 role 合法; tool_call/tool_result 必须带 tool_name + tool_call_id
 *   - 单次 messages 数量最多 500（防单请求过大, 上层还有字节兜底）
 */
export const conversationAddRequestSchema = z.object({
  session_id: z.string().min(1).refine((v) => !v.includes("|"), "session_id must not contain '|'"),
  space_id: z.string().min(1)
    .refine((v) => !v.includes("|"), "space_id must not contain '|'")
    .optional(),
  user_id: z.string().min(1).refine((v) => !v.includes("|"), "user_id must not contain '|'"),
  team_id: z.string().min(1).refine((v) => !v.includes("|"), "team_id must not contain '|'"),
  agent_id: z.string().min(1).refine((v) => !v.includes("|"), "agent_id must not contain '|'"),
  task_id: z.string().min(1).max(128).optional(),
  messages: z.array(conversationMessageSchema).min(1).max(500),
});

// ═════════════════════════════════════════════════════════════════════
//  类型导出
// ═════════════════════════════════════════════════════════════════════

export type CreateRequest = z.infer<typeof createRequestSchema>;
export type UpdateRequest = z.infer<typeof updateRequestSchema>;
export type PatchRequest = z.infer<typeof patchRequestSchema>;
export type DeleteRequest = z.infer<typeof deleteRequestSchema>;
export type GetRequest = z.infer<typeof getRequestSchema>;
export type ListRequest = z.infer<typeof listRequestSchema>;
export type SearchRequest = z.infer<typeof searchRequestSchema>;
export type VersionsRequest = z.infer<typeof versionsRequestSchema>;
export type FilesWriteRequest = z.infer<typeof filesWriteRequestSchema>;
export type FilesRemoveRequest = z.infer<typeof filesRemoveRequestSchema>;
export type FilesReadRequest = z.infer<typeof filesReadRequestSchema>;
export type ListingRequest = z.infer<typeof listingRequestSchema>;
export type ExtractRequest = z.infer<typeof extractRequestSchema>;
export type ConversationAddRequest = z.infer<typeof conversationAddRequestSchema>;

// ═════════════════════════════════════════════════════════════════════
//  force-archive (手动强制归档，跳过阈值)
// ═════════════════════════════════════════════════════════════════════

export const forceArchiveRequestSchema = z.object({
  space_id: z.string().min(1),
  user_id: z.string().min(1),
  team_id: z.string().min(1),
  agent_id: z.string().min(1),
  session_id: z.string().min(1),
  reason: z.string().max(2000).optional(),
  task_id: z.string().optional(),
});
export type ForceArchiveRequest = z.infer<typeof forceArchiveRequestSchema>;
