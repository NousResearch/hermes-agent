/**
 * TDAI Memory Gateway — v2 REST Router.
 *
 * Implements POST routes defined in `01-api-spec.yaml`:
 *
 *   L0 Conversation: add / query / search / delete
 *   L1 Atomic:       update / query / search / delete
 *   L2 Scenario:     ls / read / write / rm
 *   L3 Core:         read / write
 *
 * All routes are prefixed with `/v2/`.
 * Authentication: Authorization Bearer + x-tdai-service-id.
 * Request validation: Zod v4 safeParse → 400 on failure.
 * Response envelope: { code, message, request_id, data }.
 */

import { createHash, randomUUID } from "node:crypto";
import type http from "node:http";
import { classifyError } from "./error-handler.js";
import type { IMemoryStore, L0Record, ProfileSyncRecord } from "../core/store/types.js";
import type { EmbeddingService } from "../core/store/embedding.js";
import { createScopedStorageAdapter, type StorageAdapter } from "../core/storage/adapter.js";
import { StoragePaths } from "../core/storage/types.js";
import type { Logger } from "../core/types.js";
import type { IStateBackend } from "../core/state/types.js";
import type { PipelineWorker } from "../services/pipeline-worker.js";
import { executeMemorySearch } from "../core/tools/memory-search.js";
import { executeConversationSearch } from "../core/tools/conversation-search.js";
import type { MemoryRecord } from "../core/record/l1-writer.js";
import { reportRecallMetrics } from "../core/report/metric-tracking-recall.js";

// ── Zod schemas (validated types + defaults) ──
import {
  conversationAddRequestSchema,
  conversationQueryRequestSchema,
  conversationSearchRequestSchema,
  conversationDeleteRequestSchema,
  conversationCountRequestSchema,
  atomicUpdateRequestSchema,
  atomicQueryRequestSchema,
  atomicSearchRequestSchema,
  atomicDeleteRequestSchema,
  atomicCountRequestSchema,
  scenarioListRequestSchema,
  scenarioReadRequestSchema,
  scenarioWriteRequestSchema,
  scenarioRmRequestSchema,
  scenarioCountRequestSchema,
  coreWriteRequestSchema,
  coreCountRequestSchema,
  teamCreateRequestSchema,
  teamGetRequestSchema,
  teamUpdateRequestSchema,
  teamBatchDeleteRequestSchema,
  userCreateRequestSchema,
  userGetRequestSchema,
  userUpdateRequestSchema,
  userBatchDeleteRequestSchema,
  agentCreateRequestSchema,
  agentGetRequestSchema,
  agentUpdateRequestSchema,
  agentBatchDeleteRequestSchema,
  taskCreateRequestSchema,
  taskGetRequestSchema,
  taskUpdateRequestSchema,
  taskBatchDeleteRequestSchema,
  formatZodError,
  resolveIsolation,
  type ApiResponseEnvelope,
  type V2AuthContext,
  type ConversationItem,
  type ConversationSearchHit,
  type ConversationAddData,
  type ConversationQueryData,
  type ConversationSearchData,
  type ConversationDeleteData,
  type CountData,
  type AtomicDetail,
  type AtomicUpdateData,
  type AtomicQueryData,
  type AtomicSearchData,
  type AtomicSearchHit,
  type AtomicDeleteData,
  type ScenarioEntry,
  type ScenarioFile,
  type ScenarioWriteData,
  type CoreFile,
  type CoreWriteData,
  type BatchDeleteResult,
  type TeamData,
  type UserData,
  type AgentData,
  type TaskData,
} from "./v2-schemas.js";
import { stripSceneNavigation } from "../core/scene/scene-navigation.js";
import { buildProfileIsolationScope, buildProfileStableId, DEFAULT_PROFILE_SCOPE } from "../core/profile/profile-sync.js";

const TAG = "[tdai-gateway][v2]";
const V2_PREFIX = "/v2";

/**
 * /v3 是 L0–L3 数据面接口的"严格 isolation 版本"：
 *
 *   - 必填 team_id + agent_id + user_id（缺一即 422）
 *   - session_id 可选：传入则按 session 收敛；不传则按 (team, agent, user) 维度聚合
 *     —— 满足"agent 跨 session 聚合视图"（如治理面板的 L0/L1 总数）和
 *     "L2/L3 团队级聚合"（profile scope 公式忽略 session）两类场景
 *   - 不接受 legacy_compat_mode 退化
 *   - 与 /v2 共享 handler 实现，仅在 dispatch 层换一套 isolation 校验
 *
 * 同名 /v2 路径保持现有行为（team_id 可选、user_id fallback、可走 legacyCompatMode），
 * 这样调用方按需选 v2/v3，互不影响。
 *
 * L0–L3 数据面 endpoint（含 count）上 v3；team/user/agent/task/pipeline 等管理面接口保留 v2 唯一入口。
 */
const V3_PREFIX = "/v3";

/**
 * /v3 严格 isolation 必填字段。从 request body 或 x-tdai-* header 任一来源取到即可。
 *
 * v3 全部 L0–L3 接口仅强制 team + agent + user 三元组。
 * session_id 一律可选：
 *   - L0 conversation/* 与 L1 atomic/*：传 session 走 session 内收敛；不传按
 *     (team, agent, user) 跨 session 聚合（便于"agent 维度全量视图"，如
 *     team-memory-control 治理面的 layer-counts 展示总量）
 *   - L2 scenario/* 与 L3 core/*：原本就是 team+agent 级 profile 聚合，session 忽略
 *
 * 返回缺失字段列表（空数组表示全齐）。
 */
function collectV3Missing(
  _subpath: string,
  body: Record<string, unknown> | undefined,
  headers: Record<string, string | string[] | undefined>,
): string[] {
  const headerStr = (k: string): string | undefined => {
    const raw = headers[k] ?? headers[k.toLowerCase()];
    if (Array.isArray(raw)) return raw[0];
    return typeof raw === "string" ? raw : undefined;
  };
  const get = (bodyKey: string, headerKey: string): string => {
    const v = (body?.[bodyKey] as string | undefined) ?? headerStr(headerKey) ?? "";
    return typeof v === "string" ? v.trim() : "";
  };
  const missing: string[] = [];
  if (!get("team_id", "x-tdai-team-id")) missing.push("team_id");
  if (!get("agent_id", "x-tdai-agent-id")) missing.push("agent_id");
  if (!get("user_id", "x-tdai-user-id")) missing.push("user_id");
  // session_id 不再强制：底层 handler 在缺 session 时按 (team,agent,user) 聚合查询。
  return missing;
}

/** /v3 强 isolation 覆盖的 L0–L3 子路径（去掉前缀后的 path）。 */
const V3_ALLOWED_SUBPATHS = new Set<string>([
  "/conversation/add",
  "/conversation/query",
  "/conversation/search",
  "/conversation/delete",
  "/conversation/count",
  "/atomic/update",
  "/atomic/query",
  "/atomic/search",
  "/atomic/delete",
  "/atomic/count",
  "/scenario/ls",
  "/scenario/read",
  "/scenario/write",
  "/scenario/rm",
  "/scenario/count",
  "/core/read",
  "/core/write",
  "/core/count",
]);

/**
 * 写一条审计事件到 store.appendAudit。失败不阻塞主请求（容忍 audit 丢失）。
 *
 * 调用约定（per user 决策）：
 *   - 原始 L0/L1/L2/L3 表完全不动，本函数只追加事件
 *   - team/agent/user/task 来自外部请求 IdFields（resolveIsolation 后的 ctx）
 *   - L0 不参与（不可变流水）
 *   - 5 个 mutation handler 各调一次：
 *     atomic/update + atomic/delete + scenario/write + scenario/rm + core/write
 */
async function recordAudit(
  store: IMemoryStore | undefined,
  args: {
    record_id: string;
    layer: "L1" | "L2" | "L3";
    action: "update" | "delete";
    iso?: { teamId?: string; userId?: string; agentId?: string; sessionId?: string; taskId?: string };
    version: number;
    requestId: string;
    logger?: { warn?: (msg: string) => void };
  },
): Promise<void> {
  if (!store?.appendAudit) return; // store 不支持 audit → 跳过
  try {
    await store.appendAudit({
      audit_id: `audit-${randomUUID().replace(/-/g, "").slice(0, 16)}`,
      record_id: args.record_id,
      layer: args.layer,
      action: args.action,
      team_id: args.iso?.teamId,
      agent_id: args.iso?.agentId,
      user_id: args.iso?.userId,
      task_id: args.iso?.taskId,
      version: args.version,
      updated_at_ms: Date.now(),
      request_id: args.requestId,
    });
  } catch (err) {
    args.logger?.warn?.(
      `${TAG} audit append failed (${args.layer}/${args.action} record=${args.record_id}): ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

// ============================
// Dependencies injected at mount time
// ============================

export interface V2RouterDeps {
  /** Get the default IMemoryStore (standalone fallback). */
  getStore: () => IMemoryStore | undefined;
  /** Get the default EmbeddingService (standalone fallback). */
  getEmbedding: () => EmbeddingService | undefined;
  /** Get the default StorageAdapter (standalone fallback). */
  getStorage: () => StorageAdapter | undefined;
  logger: Logger;

  /**
   * Deploy mode of the gateway. Controls behaviors that diverge between
   * single-node open-source ("standalone") and cloud multi-tenant ("service"):
   *   - standalone: mirror v2 conversation/add L0 to <dataDir>/conversations/<date>.jsonl
   *                 (parity with v1 capture path; useful for human inspection / seed verify)
   *   - service:    skip the JSONL mirror — service stores authoritative L0 in TCVDB +
   *                 COS via its own pathway; mirroring to local FS would write to
   *                 ephemeral pod disk and is operationally meaningless.
   */
  deployMode: "standalone" | "service";

  // ── Service-mode per-instance resolvers (optional) ──
  // When provided, v2 handlers resolve store/storage per-request using
  // auth.serviceId as the instanceId key, falling back to the static getters above.

  /** Resolve IMemoryStore + EmbeddingService for a given instanceId (service mode). */
  resolveStore?: (instanceId: string) => Promise<{ store: IMemoryStore; embedding: EmbeddingService | undefined }>;
  /** Resolve per-instance StorageAdapter for a given instanceId (service mode). */
  resolveStorage?: (instanceId: string) => Promise<StorageAdapter | undefined>;

  /**
   * Notify pipeline that new L0 messages were added for a session.
   * Triggers async L1 extraction via state-backend Buffer → Scanner → Worker.
   *
   * Wired in both modes:
   *   - service mode: remote state backend
   *   - standalone: LocalStateBackend (single-process, default)
   * When absent (misconfiguration), v2 add writes L0 only — pipeline is not triggered.
   */
  notifyPipeline?: (instanceId: string, sessionId: string, messageCount: number, teamId?: string, agentId?: string) => Promise<void>;

  /** Quota manager for memory/credit limit checks and usage reporting (service mode). */
  quotaManager?: import("../core/quota/quota-manager.js").QuotaManager;

  /**
   * 拿到 (per instance) 的 MetadataService。仅当 `/v2/conversation/add` 首次
   * 写入某 (team, agent) 时用来自动登记 chat_memory 资产并绑定到 agent。
   * 未注入时该功能优雅降级：conversation 依旧正常写入，只是资产不会被自动
   * 创建 —— 老部署完全兼容。
   */
  getMetadataService?: (instanceId: string) => Promise<import("../metadata/service/metadata-service.js").MetadataService>;

  /**
   * State backend handle, used by /v2/pipeline/status to call listQueuedTasks().
   * Wired in standalone and service modes, but the status endpoint itself is
   * standalone-only. The handler returns 404 in service mode before touching
   * this field, so remote backends do not need to implement listQueuedTasks().
   */
  stateBackend?: IStateBackend;

  /**
   * Pipeline worker handle, used by /v2/pipeline/status to call getRunningTasks()
   * for per-L-type in-flight stats. Service mode never invokes this getter
   * (status endpoint returns 404 in service mode).
   */
  pipelineWorker?: PipelineWorker;

  // ── Tenancy isolation (three-dim) ──
  //
  // `isolationConfig` is set once at gateway start.  `requestIsolation` and
  // `requestIsolationMissing` are filled per-request by dispatchV2Request so
  // each handler can persist (user_id, agent_id, session_id) on writes
  // without changing handler signatures.

  /** Static config for isolation enforcement (set at gateway start). */
  isolationConfig?: {
    enforce: boolean;
    legacyCompatMode: boolean;
    legacyPlaceholder: string;
  };
  /**
   * Whether `/v3` L0–L3 enforces the team+agent+user triple.
   * Undefined defaults to strict to preserve direct router unit-test semantics;
   * server.ts injects the env-backed value, whose runtime default is OFF.
   * `/v3/skill/*` is always exempt.
   */
  v3StrictIsolation?: boolean;
  /** Resolved isolation context for the current request (set by dispatch). */
  requestIsolation?: { teamId?: string; userId: string; agentId: string; sessionId: string; taskId?: string };
  /** When isolation could not be resolved AND legacy_compat_mode is off, the missing fields. */
  requestIsolationMissing?: string[];
}

// ============================
// Envelope helpers
// ============================

export function makeRequestId(): string {
  return `req-${randomUUID().replace(/-/g, "").slice(0, 16)}`;
}

/** 优先使用上游 x-request-id / x-qcloud-transaction-id，否则本地生成。 */
export function resolveRequestId(
  headers: Record<string, string | string[] | undefined>,
): string {
  const raw = headers["x-qcloud-transaction-id"] ?? headers["x-request-id"];
  const id = Array.isArray(raw) ? raw[0] : raw;
  if (typeof id === "string" && id.trim()) return id.trim();
  return makeRequestId();
}

export function successEnvelope<T>(data: T, requestId: string): ApiResponseEnvelope<T> {
  return { code: 0, message: "ok", request_id: requestId, data };
}

export function errorEnvelope(code: number, message: string, requestId: string, extra?: Record<string, unknown>): ApiResponseEnvelope {
  return { code, message, request_id: requestId, ...(extra ? { data: extra } : {}) };
}

// ============================
// Auth middleware
// ============================

export function parseV2Auth(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  requestId: string,
  sendJsonFn: (res: http.ServerResponse, status: number, body: unknown) => void,
): V2AuthContext | null {
  const authHeader = req.headers["authorization"] ?? "";
  const serviceId = (req.headers["x-tdai-service-id"] as string) ?? "";

  if (!authHeader.startsWith("Bearer ") || !authHeader.slice(7).trim()) {
    sendJsonFn(res, 401, errorEnvelope(401, "Missing or invalid Authorization header. Expected: Bearer {api_key}", requestId));
    return null;
  }
  if (!serviceId.trim()) {
    sendJsonFn(res, 401, errorEnvelope(401, "Missing x-tdai-service-id header", requestId));
    return null;
  }

  return { apiKey: authHeader.slice(7).trim(), serviceId: serviceId.trim() };
}

// ============================
// Per-request resolution helpers
// ============================

/** Resolve store + embedding for a v2 request. Service mode → per-instance; standalone → core singleton. */
async function resolveStoreForRequest(
  auth: V2AuthContext,
  deps: V2RouterDeps,
): Promise<{ store: IMemoryStore | undefined; embedding: EmbeddingService | undefined }> {
  if (deps.resolveStore) {
    // Service mode: per-instance VDB store is mandatory. Do NOT fallback to local SQLite.
    return await deps.resolveStore(auth.serviceId);
  }
  // Standalone mode: use core singleton store
  return { store: deps.getStore(), embedding: deps.getEmbedding() };
}

/** Resolve storage adapter for a v2 request. Service mode → per-instance COS; standalone → core local. */
async function resolveStorageForRequest(
  auth: V2AuthContext,
  deps: V2RouterDeps,
): Promise<StorageAdapter | undefined> {
  if (deps.resolveStorage) {
    // Service mode: per-instance COS storage is mandatory. Do NOT fallback to local filesystem.
    return await deps.resolveStorage(auth.serviceId);
  }
  // Standalone mode: use core local storage
  return deps.getStorage();
}

// ============================
// Route table
// ============================

type RouteHandler = (
  body: unknown,
  auth: V2AuthContext,
  requestId: string,
  deps: V2RouterDeps,
) => Promise<ApiResponseEnvelope>;

/**
 * L0–L3 数据面 handler 映射（子路径 → handler）。历史接口同时挂载 /v2/* 与 /v3/*；
 * count 接口按 sdk-v3.yaml 仅挂载 /v3/*。/v3 走严格 isolation 校验；/v2 沿用现有 enforce/legacyCompat 配置。
 */
const DATAPLANE_HANDLERS: Record<string, RouteHandler> = {
  "/conversation/add": handleConversationAdd,
  "/conversation/query": handleConversationQuery,
  "/conversation/search": handleConversationSearch,
  "/conversation/delete": handleConversationDelete,
  "/conversation/count": handleConversationCount,
  "/atomic/update": handleAtomicUpdate,
  "/atomic/query": handleAtomicQuery,
  "/atomic/search": handleAtomicSearch,
  "/atomic/delete": handleAtomicDelete,
  "/atomic/count": handleAtomicCount,
  "/scenario/ls": handleScenarioLs,
  "/scenario/read": handleScenarioRead,
  "/scenario/write": handleScenarioWrite,
  "/scenario/rm": handleScenarioRm,
  "/scenario/count": handleScenarioCount,
  "/core/read": handleCoreRead,
  "/core/write": handleCoreWrite,
  "/core/count": handleCoreCount,
};

const routeTable: Record<string, RouteHandler> = {
  // L0–L3 数据面：历史读写接口保留 /v2 与 /v3 双入口；count 仅按 sdk-v3.yaml 暴露 /v3。
  ...Object.fromEntries(
    Object.entries(DATAPLANE_HANDLERS).flatMap(([sub, h]) => {
      const v3Route = [`${V3_PREFIX}${sub}`, h] as const;
      if (sub.endsWith("/count")) return [v3Route];
      return [[`${V2_PREFIX}${sub}`, h] as const, v3Route];
    }),
  ),
  // ──────────────────────────────────────────────────────────────────────────
  // @deprecated v2 entity 路由（team/user/agent/task，16 条）。仅 /v2。
  // 元数据已由 v3 `/v3/meta/*`（规范化 meta_* 表 + MetadataService）接管，control
  // 面板 remote 模式直接走 v3。以下路由仍走旧 `entity_*` 聚合数组模型，**仅为兼容现网
  // 保留、行为不变**，不再演进，新接入方一律改用 v3。计划在确认无外部调用方后整体删除。
  // 决策（2026-06-26，per user）：标记废弃、代码暂留，不适配 MetadataService。
  // 见 team-memory-control/docs/architecture/08-metadata-migration-and-permission-design.md §8.5。
  // ──────────────────────────────────────────────────────────────────────────
  [`${V2_PREFIX}/team/create`]: handleTeamCreate, // @deprecated 改用 /v3/meta/team/create
  [`${V2_PREFIX}/team/get`]: handleTeamGet, // @deprecated 改用 /v3/meta/team/get
  [`${V2_PREFIX}/team/update`]: handleTeamUpdate, // @deprecated 改用 /v3/meta/team/update
  [`${V2_PREFIX}/team/delete`]: handleTeamDelete, // @deprecated 改用 /v3/meta/team/delete
  [`${V2_PREFIX}/user/create`]: handleUserCreate, // @deprecated 改用 /v3/meta/user/create
  [`${V2_PREFIX}/user/get`]: handleUserGet, // @deprecated 改用 /v3/meta/user/get
  [`${V2_PREFIX}/user/update`]: handleUserUpdate, // @deprecated 改用 /v3/meta/user/update
  [`${V2_PREFIX}/user/delete`]: handleUserDelete, // @deprecated 改用 /v3/meta/user/delete
  [`${V2_PREFIX}/agent/create`]: handleAgentCreate, // @deprecated 改用 /v3/meta/agent/create
  [`${V2_PREFIX}/agent/get`]: handleAgentGet, // @deprecated 改用 /v3/meta/agent/get
  [`${V2_PREFIX}/agent/update`]: handleAgentUpdate, // @deprecated 改用 /v3/meta/agent/update
  [`${V2_PREFIX}/agent/delete`]: handleAgentDelete, // @deprecated 改用 /v3/meta/agent/delete
  [`${V2_PREFIX}/task/create`]: handleTaskCreate, // @deprecated 改用 /v3/meta/task/create
  [`${V2_PREFIX}/task/get`]: handleTaskGet, // @deprecated 改用 /v3/meta/task/get
  [`${V2_PREFIX}/task/update`]: handleTaskUpdate, // @deprecated 改用 /v3/meta/task/update
  [`${V2_PREFIX}/task/delete`]: handleTaskDelete, // @deprecated 改用 /v3/meta/task/delete
  // ── end @deprecated v2 entity 路由 ──
  [`${V2_PREFIX}/pipeline/status`]: handlePipelineStatus,
};

export async function handleV2Route(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  pathname: string,
  method: string,
  parseJsonBody: <T>(req: http.IncomingMessage) => Promise<T>,
  sendJson: (res: http.ServerResponse, status: number, body: unknown) => void,
  deps: V2RouterDeps,
  /**
   * Optional: extra routes contributed by other modules (e.g.
   * /v3/skill/* from `makeSkillRouteTable()`). Looked up only when the
   * built-in `routeTable` doesn't contain the pathname, so module-level
   * routes always win on collision. Supports `/v2/*`, `/v3/*` (L0–L3
   * data-plane + extraRouteTable for /v3/skill/* and /v3/knowledge/*).
   *
   * The handler's `deps` parameter is intentionally typed `unknown` here:
   * extra modules (skill/*, future namespaces) declare their own deps
   * shape (e.g. `SkillRouterDeps`). The caller is responsible for passing
   * a `deps` object that satisfies the union of every handler set's
   * requirements; v2-router just forwards it verbatim.
   */
  extraRouteTable?: Record<
    string,
    (body: unknown, auth: V2AuthContext, requestId: string, deps: unknown) => Promise<ApiResponseEnvelope>
  >,
): Promise<boolean> {
  if (method !== "POST") return false;
  const isV3 = pathname.startsWith(`${V3_PREFIX}/`);
  const isV2 = pathname.startsWith(`${V2_PREFIX}/`);
  // /v3/skill/* and /v3/knowledge/* are provided by extraRouteTable
  // (makeSkillRouteTable / makeKnowledgeRouteTable), NOT in the built-in
  // V3_ALLOWED_SUBPATHS. Both are management-plane, bypass strict isolation.
  const isV3Extra = !!extraRouteTable && (
    pathname.startsWith("/v3/skill/") || pathname.startsWith("/v3/knowledge/")
  );
  if (!isV2 && !isV3) return false;

  // /v3 暴露 L0–L3 数据面 14 条（V3_ALLOWED_SUBPATHS）+ /v3/skill/* + /v3/knowledge/*（extraRouteTable）；
  // 其他 /v3 子路径直接走 404
  if (isV3 && !isV3Extra) {
    const sub = pathname.slice(V3_PREFIX.length);
    if (!V3_ALLOWED_SUBPATHS.has(sub)) return false;
  }

  const handler = routeTable[pathname];
  const extra = extraRouteTable?.[pathname];
  if (!handler && !extra) return false;

  const requestId = makeRequestId();
  // [skill-perf 2026-07-21] 只对 /v3/skill/ 打分段耗时，避免污染其他链路 log。
  // T0 从 server.ts socket-level 埋点取；没有则用 dispatch 进入时刻兜底。
  const isSkillPerf = pathname.startsWith("/v3/skill/");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const perfT0 = isSkillPerf ? ((req as any).__skillPerfT0 as number | undefined) ?? Date.now() : 0;
  // 把 request_id 挂到 res 上，让 server.ts 的 res.on('finish') 也能带上
  if (isSkillPerf) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (res as any).__skillReqId = requestId;
    deps.logger.info(
      `[skill-perf] dispatch.enter req_id=${requestId} path=${pathname} t=${Date.now()} t0_offset=${Date.now() - perfT0}ms`,
    );
  }
  const perfMark = (phase: string, extraFields?: string) => {
    if (!isSkillPerf) return;
    const now = Date.now();
    deps.logger.info(
      `[skill-perf] phase=${phase} req_id=${requestId} elapsed=${now - perfT0}ms${extraFields ? " " + extraFields : ""}`,
    );
  };

  const authStart = Date.now();
  const auth = parseV2Auth(req, res, requestId, sendJson);
  if (isSkillPerf) {
    perfMark("parseAuth", `dur=${Date.now() - authStart}ms ok=${auth ? "true" : "false"}`);
  }
  if (!auth) return true;

  try {
    // Pre-resolve per-request store/storage (service mode → per-instance, standalone → core singleton)
    const resolveStart = Date.now();
    const resolved = await resolveStoreForRequest(auth, deps);
    const resolvedStorage = await resolveStorageForRequest(auth, deps);
    perfMark("resolveStoreAndStorage", `dur=${Date.now() - resolveStart}ms serviceId=${auth.serviceId}`);

    // Wrap deps so handlers use the resolved per-instance resources
    const resolvedDeps: V2RouterDeps = {
      ...deps,
      getStore: () => resolved.store,
      getEmbedding: () => resolved.embedding,
      getStorage: () => resolvedStorage,
    };

    const bodyStart = Date.now();
    const body = await parseJsonBody(req);
    perfMark("parseJsonBody", `dur=${Date.now() - bodyStart}ms len=${req.headers["content-length"] ?? "?"}`);

    // Tenancy isolation — pulled from body (preferred) or x-tdai-* headers
    // and attached to the per-request deps so handlers can persist
    // (user_id, agent_id, session_id) on every L0/L1 write without
    // changing every handler signature.
    //
    // We only attempt resolution; whether missing fields are fatal is up
    // to each handler (some endpoints don't need isolation at all, e.g.
    // /v2/pipeline/status). See resolveIsolation() in v2-schemas.
    //
    // /v3 走严格校验：必须同时提供 team_id + agent_id + user_id + session_id，
    // 缺任意一个直接 422，且不走 legacyCompatMode 退化。
    const headers = (req.headers ?? {}) as Record<string, string | string[] | undefined>;
    const isoLegacyCompat = isV3 ? false : (deps.isolationConfig?.legacyCompatMode ?? false);
    const isoResolved = resolveIsolation(body as Record<string, unknown> | undefined, headers, {
      legacyCompatMode: isoLegacyCompat,
      legacyPlaceholder: deps.isolationConfig?.legacyPlaceholder,
    });

    // /v3 strict isolation is for L0–L3 memory data-plane only.
    // Skill and knowledge endpoints are team-scoped management-plane
    // and must not be blocked by per-agent memory isolation.
    // Runtime default comes from server.ts/env and is OFF;
    // undefined keeps strict in direct router tests for backward compatibility.
    const v3StrictEnabled = deps.v3StrictIsolation ?? true;
    if (isV3 && !isV3Extra && v3StrictEnabled) {
      const v3Subpath = pathname.slice(V3_PREFIX.length);
      const v3Missing = collectV3Missing(v3Subpath, body as Record<string, unknown> | undefined, headers);
      if (v3Missing.length > 0) {
        sendJson(res, 422, errorEnvelope(
          422,
          `/v3 requires strict isolation: missing ${v3Missing.join(", ")}. ` +
          `Provide via request body or x-tdai-{team-id,agent-id,user-id,session-id} headers.`,
          requestId,
        ));
        return true;
      }
    }

    const depsWithIsolation: V2RouterDeps = {
      ...resolvedDeps,
      // /v3 路径强制覆盖 isolationConfig.enforce，确保 handler 内部一致地走严格分支
      isolationConfig: isV3
        ? { enforce: true, legacyCompatMode: false, legacyPlaceholder: resolvedDeps.isolationConfig?.legacyPlaceholder ?? "" }
        : resolvedDeps.isolationConfig,
      requestIsolation: isoResolved.ctx,
      // resolveIsolation always returns { ok: true } — missing fields are filled with defaults.
      // requestIsolationMissing is only set when the caller explicitly needs to reject incomplete
      // isolation (e.g. /v3 strict mode), which is handled separately above via collectV3Missing.
      requestIsolationMissing: undefined,
    };

    const handlerStart = Date.now();
    const envelope = handler
      ? await handler(body, auth, requestId, depsWithIsolation)
      : await extra!(body, auth, requestId, depsWithIsolation as unknown);
    perfMark("handler", `dur=${Date.now() - handlerStart}ms envelope_code=${envelope.code}`);
    const httpStatus = envelope.code === 0 ? 200 : envelope.code >= 400 && envelope.code < 600 ? envelope.code : 200;
    const sendStart = Date.now();
    sendJson(res, httpStatus, envelope);
    perfMark("sendJson", `dur=${Date.now() - sendStart}ms status=${httpStatus}`);
  } catch (err) {
    // H-13: use classifyError so 5xx leaves no err.message leak; PayloadTooLargeError
    // and RecallFailure already carry safe messages but go through the same path for uniformity.
    const classified = classifyError(err);
    if (classified.status >= 500) {
      deps.logger.error(`${TAG} [${pathname}] ${classified.logLine}`);
    } else {
      deps.logger.warn(`${TAG} [${pathname}] ${classified.logLine}`);
    }
    sendJson(res, classified.status, {
      ...errorEnvelope(classified.client.code, classified.client.message, requestId),
      trace_id: classified.client.trace_id,
      retryable: classified.client.retryable,
    });
  }

  return true;
}

// ============================
// L0 Conversation Handlers
// ============================

async function handleConversationAdd(body: unknown, auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = conversationAddRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { session_id, messages } = parsed.data;

  // Enforce three-dim isolation. user_id / agent_id come from request body
  // or x-tdai-* headers (resolved in dispatchV2Request).  When the gateway's
  // isolationConfig.enforce is on AND legacy_compat_mode is off, missing
  // fields are a 422.
  if (deps.isolationConfig?.enforce && deps.requestIsolationMissing && deps.requestIsolationMissing.length > 0) {
    return errorEnvelope(
      422,
      `Tenancy isolation required: missing ${deps.requestIsolationMissing.join(", ")}. ` +
      `Provide via request body or x-tdai-{user-id,agent-id,session-id} headers.`,
      requestId,
    );
  }
  const iso = deps.requestIsolation;

  const store = deps.getStore();
  if (!store) return errorEnvelope(503, "Store not available", requestId);

  // Quota check: memory limit
  if (deps.quotaManager) {
    const check = await deps.quotaManager.checkMemoryQuota(auth.serviceId, messages.length);
    if (!check.allowed) {
      return errorEnvelope(4291, `Memory limit exceeded (current=${check.current}, limit=${check.limit})`, requestId);
    }
  }

  // 自动登记 chat_memory 资产（team+agent 粒度）并绑定到 agent。首次触发
  // 会 create asset + append 绑定；后续同 (team, agent) 走进程内 LRU 短路。
  // 失败降级：只打 warn，不阻塞 conversation 写入 —— 记忆数据的可用性优先
  // 于资产登记的一致性（asset 登记失败时下次调用会自动重试）。
  if (deps.getMetadataService && iso?.teamId && iso?.agentId) {
    try {
      const metaSvc = await deps.getMetadataService(auth.serviceId);
      await metaSvc.ensureChatMemoryAsset({
        team_id: iso.teamId,
        agent_id: iso.agentId,
      });
    } catch (err) {
      deps.logger.warn(
        `${TAG} ensureChatMemoryAsset failed (team=${iso.teamId} agent=${iso.agentId}): ` +
        `${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  const embedding = deps.getEmbedding();
  const acceptedIds: string[] = [];
  const ingestBaseMs = Date.now();

  for (const [index, msg] of messages.entries()) {
    const id = `msg-${randomUUID().replace(/-/g, "").slice(0, 12)}`;
    const recordedAtMs = ingestBaseMs + index;
    const record: L0Record = {
      id,
      sessionKey: session_id,
      sessionId: session_id,
      taskId: iso?.taskId,
      // Tenancy isolation: resolveIsolation() defaults missing fields to the default bucket.
      teamId: iso?.teamId,
      userId: iso?.userId,
      agentId: iso?.agentId,
      role: msg.role,
      messageText: msg.content,
      recordedAt: new Date(recordedAtMs).toISOString(),
      timestamp: msg.timestamp ? new Date(msg.timestamp).getTime() : recordedAtMs,
    };

    let emb: Float32Array | undefined;
    if (embedding) {
      try { emb = await embedding.embed(msg.content); } catch (e) { console.warn(`[v2-router] L0 embedding failed:`, e); }
    }

    await store.upsertL0(record, emb);
    acceptedIds.push(id);
  }

  // Notify pipeline: trigger async L1 extraction (service mode).
  // Each role=user message counts as one conversation round for threshold/timer logic.
  // teamId/agentId 透传给 captureAtomic 决定 hash slot 与锁粒度。
  if (deps.notifyPipeline) {
    const rounds = messages.filter((m) => m.role === "user").length;
    if (rounds > 0) {
      try {
        await deps.notifyPipeline(auth.serviceId, session_id, rounds, iso?.teamId, iso?.agentId);
      } catch (err) {
        // Non-fatal: L0 is already persisted, pipeline will catch up later
        deps.logger.warn(`${TAG} Pipeline notify failed for ${session_id}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }

  // Standalone-only: mirror L0 to <dataDir>/conversations/<date>.jsonl.
  // Parity with v1 /capture (l0-recorder) path — gives humans a grep-able audit
  // log alongside SQLite. Service mode skips: COS is the authoritative store,
  // and writing to local FS in a multi-replica pod would be ephemeral + useless.
  // Failure is non-fatal: SQLite is the source of truth.
  if (deps.deployMode === "standalone") {
    const storage = deps.getStorage();
    if (storage) {
      try {
        const recordKey = StoragePaths.conversation(formatLocalDateForJsonl(new Date(ingestBaseMs)));
        const lines = messages.map((msg, idx) => JSON.stringify({
          id: acceptedIds[idx],
          sessionKey: session_id,
          sessionId: session_id,
          taskId: iso?.taskId,
          teamId: iso?.teamId,
          userId: iso?.userId,
          agentId: iso?.agentId,
          role: msg.role,
          content: msg.content,
          recordedAt: new Date(ingestBaseMs + idx).toISOString(),
          timestamp: msg.timestamp ? new Date(msg.timestamp).getTime() : ingestBaseMs + idx,
        })).join("\n") + "\n";
        await storage.appendFile(recordKey, lines);
      } catch (err) {
        deps.logger.warn(`${TAG} JSONL mirror failed for ${session_id}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }

  // Report memory usage (non-fatal)
  if (deps.quotaManager && acceptedIds.length > 0) {
    deps.quotaManager.reportMemoryAdded(auth.serviceId, acceptedIds.length).catch(() => {});
  }

  return successEnvelope<ConversationAddData>(
    { accepted_ids: acceptedIds, accepted_versions: acceptedIds.map(() => "v1"), total_count: acceptedIds.length },
    requestId,
  );
}

async function handleConversationQuery(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = conversationQueryRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { session_id, time_start, time_end } = parsed.data;
  const limit = parsed.data.limit ?? 20;
  const offset = parsed.data.offset ?? 0;

  const store = deps.getStore();
  if (!store) return errorEnvelope(503, "Store not available", requestId);

  // Tenancy isolation — narrow the query when caller supplied user_id /
  // agent_id (via body or headers). session_id, if present, comes from the
  // request body and is already in `session_id`.
  const iso = deps.requestIsolation;

  // Use paginated query if available (AR-3), else fallback
  if (store.queryL0Paginated) {
    const result = await store.queryL0Paginated({
      sessionId: session_id,
      teamId: iso?.teamId,
      userId: iso?.userId,
      agentId: iso?.agentId,
      taskId: iso?.taskId,
      timeStartMs: time_start ? new Date(time_start).getTime() : undefined,
      timeEndMs: time_end ? new Date(time_end).getTime() : undefined,
      limit,
      offset,
    });

    const messages: ConversationItem[] = result.rows.map((r) => ({
      id: r.record_id,
      session_id: r.session_id,
      team_id: r.team_id,
      user_id: r.user_id,
      agent_id: r.agent_id,
      task_id: r.task_id,
      role: r.role as ConversationItem["role"],
      content: r.message_text,
      timestamp: r.recorded_at,
    }));

    return successEnvelope<ConversationQueryData>({ messages, total: result.total }, requestId);
  }

  // Fallback: legacy path (capped at 1000 for safety)
  const allRows = await store.queryL0ForL1(session_id ?? "", undefined, 1000);
  let filtered = session_id ? allRows.filter((r) => r.session_key === session_id || r.session_id === session_id) : allRows;
  // Tenancy isolation post-filter for the legacy path.
  if (iso?.teamId) filtered = filtered.filter((r) => r.team_id === iso.teamId);
  if (iso?.userId) filtered = filtered.filter((r) => r.user_id === iso.userId);
  if (iso?.agentId) filtered = filtered.filter((r) => r.agent_id === iso.agentId);
  if (iso?.taskId) filtered = filtered.filter((r) => r.task_id === iso.taskId);
  if (time_start) { const ms = new Date(time_start).getTime(); filtered = filtered.filter((r) => r.timestamp >= ms); }
  if (time_end) { const ms = new Date(time_end).getTime(); filtered = filtered.filter((r) => r.timestamp <= ms); }
  const total = filtered.length;
  const page = filtered.slice(offset, offset + limit);
  const messages: ConversationItem[] = page.map((r) => ({
    id: r.record_id,
    session_id: r.session_id,
    team_id: r.team_id,
    user_id: r.user_id,
    agent_id: r.agent_id,
    task_id: r.task_id,
    role: r.role as ConversationItem["role"],
    content: r.message_text,
    timestamp: r.recorded_at,
  }));

  return successEnvelope<ConversationQueryData>({ messages, total }, requestId);
}

async function handleConversationCount(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = conversationCountRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { session_id, time_start, time_end } = parsed.data;

  const store = deps.getStore();
  if (!store) return errorEnvelope(503, "Store not available", requestId);
  const iso = deps.requestIsolation;

  const countFilter = {
    sessionId: session_id,
    teamId: iso?.teamId,
    userId: iso?.userId,
    agentId: iso?.agentId,
    taskId: iso?.taskId,
    timeStartMs: time_start ? new Date(time_start).getTime() : undefined,
    timeEndMs: time_end ? new Date(time_end).getTime() : undefined,
  };
  const total = await store.countL0(countFilter);
  return successEnvelope<CountData>({ total }, requestId);

  const allRows = await store.queryL0ForL1(session_id ?? "", undefined, 10000);
  let filtered = session_id ? allRows.filter((r) => r.session_key === session_id || r.session_id === session_id) : allRows;
  if (iso?.teamId) filtered = filtered.filter((r) => r.team_id === iso.teamId);
  if (iso?.userId) filtered = filtered.filter((r) => r.user_id === iso.userId);
  if (iso?.agentId) filtered = filtered.filter((r) => r.agent_id === iso.agentId);
  if (iso?.taskId) filtered = filtered.filter((r) => r.task_id === iso.taskId);
  if (time_start) { const ms = new Date(time_start).getTime(); filtered = filtered.filter((r) => r.timestamp >= ms); }
  if (time_end) { const ms = new Date(time_end).getTime(); filtered = filtered.filter((r) => r.timestamp <= ms); }
  return successEnvelope<CountData>({ total: filtered.length }, requestId);
}

async function handleConversationSearch(body: unknown, auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = conversationSearchRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { query, session_id } = parsed.data;
  const limit = parsed.data.limit ?? 5;

  const tStart = performance.now();
  // 搜索场景：只使用显式传入的 isolation 维度作为 filter，避免默认 sessionId="default" 错误过滤真实 session 数据。
  // 当请求未传 session_id 时，搜索应跨 session；当传了 session_id 时，由 sessionKey 参数单独过滤。
  const iso = deps.requestIsolation;
  const searchFilter = iso ? {
    ...(iso.teamId ? { teamId: iso.teamId } : {}),
    ...(iso.userId ? { userId: iso.userId } : {}),
    ...(iso.agentId ? { agentId: iso.agentId } : {}),
    ...(iso.taskId ? { taskId: iso.taskId } : {}),
    // 不传 sessionId：全局搜索不应被默认 sessionId 限制
  } : undefined;
  const result = await executeConversationSearch({
    query,
    limit,
    sessionKey: session_id,
    filter: searchFilter,
    vectorStore: deps.getStore(),
    embeddingService: deps.getEmbedding(),
    logger: deps.logger,
  });
  const recallLatencyMs = performance.now() - tStart;

  // 非侵入式上报召回指标（service 模式，静默失败，绝不影响业务返回）
  // L0 conversation search 同样属于"召回"行为，strategy 映射逻辑与 L1 相同
  try {
    reportRecallMetrics({
      instanceId: auth.serviceId,
      recalledL1Memories: result.results.map((r) => ({ content: r.content, score: r.score, type: "conversation" })),
      recallStrategy: result.strategy === "fts" ? "keyword" : result.strategy === "none" ? "skipped" : result.strategy,
      recallLatencyMs,
      hasError: false,
    });
  } catch {
    // 静默失败
  }

  // 非侵入式在当前 Span 上记录 recall query 和 results
  try {
    const otelApi = await import("@opentelemetry/api");
    const activeSpan = otelApi.trace.getSpan(otelApi.context.active());
    if (activeSpan) {
      activeSpan.setAttribute("tdai.recall.query", query);
      activeSpan.setAttribute("tdai.recall.hitCount", result.results.length);
      activeSpan.setAttribute("tdai.recall.strategy", result.strategy || "unknown");
      activeSpan.setAttribute("tdai.recall.level", "l0");
      if (result.results.length > 0) {
        activeSpan.setAttribute("tdai.recall.topScore", Math.max(...result.results.map(r => r.score)));
        const truncatedResults = result.results.slice(0, 5).map(r => ({
          content: r.content.substring(0, 200),
          score: r.score,
        }));
        activeSpan.setAttribute("tdai.recall.results", JSON.stringify(truncatedResults));
      } else {
        activeSpan.setAttribute("tdai.recall.results", "[]");
      }
    }
  } catch {
    // 静默失败
  }

  const messages: ConversationSearchHit[] = result.results.map((r) => ({
    id: r.id, role: r.role as ConversationSearchHit["role"], content: r.content, timestamp: r.recorded_at, score: r.score,
  }));

  return successEnvelope<ConversationSearchData>({ messages }, requestId);
}

async function handleConversationDelete(body: unknown, auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = conversationDeleteRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { message_ids, session_id } = parsed.data;

  const store = deps.getStore();
  if (!store) return errorEnvelope(503, "Store not available", requestId);

  let deletedCount = 0;

  if (message_ids && message_ids.length > 0) {
    for (const id of message_ids) {
      const ok = await store.deleteL0(id, deps.requestIsolation);
      if (ok) deletedCount++;
    }
  } else if (session_id) {
    // Use deleteL0BySession if available, else fallback
    if (store.deleteL0BySession) {
      deletedCount = await store.deleteL0BySession(session_id, deps.requestIsolation);
    } else {
      const rows = await store.queryL0ForL1(session_id, undefined, 10000);
      const sessionRows = rows.filter((r) => r.session_key === session_id || r.session_id === session_id);
      for (const row of sessionRows) {
        const ok = await store.deleteL0(row.record_id, deps.requestIsolation);
        if (ok) deletedCount++;
      }
    }
  }

  // Report memory deletion (non-fatal)
  if (deps.quotaManager && deletedCount > 0) {
    deps.quotaManager.reportMemoryDeleted(auth.serviceId, deletedCount).catch(() => {});
  }

  return successEnvelope<ConversationDeleteData>({ deleted_count: deletedCount }, requestId);
}

async function handleAtomicUpdate(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = atomicUpdateRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { id, content, background } = parsed.data;

  const store = deps.getStore();
  if (!store) return errorEnvelope(503, "Store not available", requestId);

  // Read existing record by primary key
  const existing = await store.queryL1Records({ recordIds: [id] });
  if (!existing || existing.length === 0) {
    return errorEnvelope(404, `Atomic note not found: ${id}`, requestId);
  }

  const now = new Date().toISOString();
  const record = existing[0];

  // Build update: content is always overwritten; background (scene_name) only if provided.
  // user_id / agent_id are preserved from the existing row — updates don't
  // re-derive them. If the caller supplied an isolation triple that does NOT
  // match the existing row, we treat it as a permission denial.
  const iso = deps.requestIsolation;
  if (iso?.userId && record.user_id && record.user_id !== iso.userId) {
    return errorEnvelope(403, `Atomic note ${id} belongs to a different user`, requestId);
  }
  if (iso?.agentId && record.agent_id && record.agent_id !== iso.agentId) {
    return errorEnvelope(403, `Atomic note ${id} belongs to a different agent`, requestId);
  }
  const updatedVersion = (record.version ?? 0) + 1;
  const updated: MemoryRecord = {
    id,
    content,
    type: record.type as any,
    priority: record.priority ?? 50,
    scene_name: background !== undefined ? background : (record.scene_name ?? ""),
    source_message_ids: [],
    metadata: parseMetadataJson(record.metadata_json),
    timestamps: record.timestamp_str ? [record.timestamp_str] : [],
    createdAt: record.created_time,
    updatedAt: now,
    version: updatedVersion,
    sessionKey: record.session_key ?? "",
    sessionId: record.session_id ?? iso?.sessionId ?? "",
    taskId: record.task_id ?? iso?.taskId,
    teamId: record.team_id ?? iso?.teamId,
    userId: record.user_id ?? iso?.userId,
    agentId: record.agent_id ?? iso?.agentId,
  };

  const embedding = deps.getEmbedding();
  let emb: Float32Array | undefined;
  if (embedding) { try { emb = await embedding.embed(content); } catch (e) { console.warn(`[v2-router] L1 embedding failed:`, e); } }

  await store.upsertL1(updated, emb);

  // 审计：L1 update — 用外部请求的 IdFields 而非 record 原值（per user 决策）
  await recordAudit(store, {
    record_id: id,
    layer: "L1",
    action: "update",
    iso,
    version: updatedVersion,
    requestId,
    logger: deps.logger,
  });

  return successEnvelope<AtomicUpdateData>({ id, version: `v${updatedVersion}`, updated_at: now }, requestId);
}

async function handleAtomicQuery(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = atomicQueryRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { type, time_start, time_end } = parsed.data;
  const limit = parsed.data.limit ?? 20;
  const offset = parsed.data.offset ?? 0;

  const store = deps.getStore();
  if (!store) return errorEnvelope(503, "Store not available", requestId);

  // Tenancy isolation — narrow the query by user_id / agent_id when supplied.
  const iso = deps.requestIsolation;

  // Use paginated query if available
  if (store.queryL1Paginated) {
    const result = await store.queryL1Paginated({
      type, timeStart: time_start, timeEnd: time_end, limit, offset,
      teamId: iso?.teamId, userId: iso?.userId, agentId: iso?.agentId, taskId: iso?.taskId,
    });
    const items: AtomicDetail[] = result.rows.map((r) => ({
      id: r.record_id, type: r.type, content: r.content,
      background: r.scene_name || undefined,
      version: r.version ?? 0,
      team_id: r.team_id,
      user_id: r.user_id,
      agent_id: r.agent_id,
      task_id: r.task_id,
      created_at: r.created_time, updated_at: r.updated_time,
    }));
    return successEnvelope<AtomicQueryData>({ items, total: result.total }, requestId);
  }

  // Fallback: legacy
  const allRecords = await store.queryL1Records();
  let filtered = allRecords;
  if (type) filtered = filtered.filter((r) => r.type === type);
  if (iso?.teamId) filtered = filtered.filter((r) => r.team_id === iso.teamId);
  if (iso?.userId) filtered = filtered.filter((r) => r.user_id === iso.userId);
  if (iso?.agentId) filtered = filtered.filter((r) => r.agent_id === iso.agentId);
  if (iso?.taskId) filtered = filtered.filter((r) => r.task_id === iso.taskId);
  if (time_start) filtered = filtered.filter((r) => r.updated_time >= time_start);
  if (time_end) filtered = filtered.filter((r) => r.updated_time <= time_end);
  const total = filtered.length;
  const page = filtered.slice(offset, offset + limit);
  const items: AtomicDetail[] = page.map((r) => ({
    id: r.record_id, type: r.type, content: r.content,
    background: r.scene_name || undefined,
    version: r.version ?? 0,
    team_id: r.team_id,
    user_id: r.user_id,
    agent_id: r.agent_id,
    task_id: r.task_id,
    created_at: r.created_time, updated_at: r.updated_time,
  }));

  return successEnvelope<AtomicQueryData>({ items, total }, requestId);
}

async function handleAtomicCount(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = atomicCountRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { type, time_start, time_end } = parsed.data;

  const store = deps.getStore();
  if (!store) return errorEnvelope(503, "Store not available", requestId);
  const iso = deps.requestIsolation;

  const total = await store.countL1({
    type,
    timeStart: time_start,
    timeEnd: time_end,
    teamId: iso?.teamId,
    userId: iso?.userId,
    agentId: iso?.agentId,
    taskId: iso?.taskId,
  });
  return successEnvelope<CountData>({ total }, requestId);
}

async function handleAtomicSearch(body: unknown, auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = atomicSearchRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { query, type } = parsed.data;
  const limit = parsed.data.limit ?? 5;

  const tStart = performance.now();
  // L1 召回为 agent 维度（跨 session）：filter 只取 team/user/agent/task，
  // **不带 sessionId**，否则会把其它 session 写入的 L1 记忆过滤掉，导致
  // 新会话里召不回历史记忆（与 conversation/search 的处理保持一致）。
  const iso = deps.requestIsolation;
  const searchFilter = iso ? {
    ...(iso.teamId ? { teamId: iso.teamId } : {}),
    ...(iso.userId ? { userId: iso.userId } : {}),
    ...(iso.agentId ? { agentId: iso.agentId } : {}),
    ...(iso.taskId ? { taskId: iso.taskId } : {}),
    // 不传 sessionId：L1 召回应跨 session（agent 维度）
  } : undefined;
  const result = await executeMemorySearch({
    query, limit, type,
    filter: searchFilter,
    vectorStore: deps.getStore(),
    embeddingService: deps.getEmbedding(),
    logger: deps.logger,
  });
  const recallLatencyMs = performance.now() - tStart;

  // 非侵入式上报召回指标（service 模式，静默失败，绝不影响业务返回）
  try {
    reportRecallMetrics({
      instanceId: auth.serviceId,
      recalledL1Memories: result.results.map((r) => ({ content: r.content, score: r.score, type: r.type })),
      recallStrategy: result.strategy === "fts" ? "keyword" : result.strategy === "none" ? "skipped" : result.strategy,
      recallLatencyMs,
      hasError: false,
    });
  } catch {
    // 静默失败
  }

  // 非侵入式在当前 Span 上记录 recall query 和 results，供在线评测系统消费
  try {
    const { getObservabilityBackend } = await import("../core/report/factory.js");
    const ctx = getObservabilityBackend().tracePropagation.serializeTraceContext();
    if (ctx && (ctx as any)._traceId) {
      // 通过 OTel API 在当前 span 上添加属性
      try {
        const otelApi = await import("@opentelemetry/api");
        const activeSpan = otelApi.trace.getSpan(otelApi.context.active());
        if (activeSpan) {
          activeSpan.setAttribute("tdai.recall.query", query);
          activeSpan.setAttribute("tdai.recall.hitCount", result.results.length);
          activeSpan.setAttribute("tdai.recall.strategy", result.strategy || "unknown");
          if (result.results.length > 0) {
            activeSpan.setAttribute("tdai.recall.topScore", Math.max(...result.results.map(r => r.score)));
            // 限制 results 属性长度（OTel 属性不宜过长），最多前 5 条
            const truncatedResults = result.results.slice(0, 5).map(r => ({
              content: r.content.substring(0, 200),
              score: r.score,
              type: r.type,
            }));
            activeSpan.setAttribute("tdai.recall.results", JSON.stringify(truncatedResults));
          } else {
            activeSpan.setAttribute("tdai.recall.results", "[]");
          }
          activeSpan.setAttribute("tdai.recall.level", type === "l0" ? "l0" : "l1");
        }
      } catch {
        // OTel API 不可用时静默降级
      }
    }
  } catch {
    // 静默失败
  }

  const items: AtomicSearchHit[] = result.results.map((r) => ({
    id: r.id, type: r.type, content: r.content,
    background: r.scene_name || undefined,
    version: r.version ?? 0,
    team_id: r.team_id,
    user_id: r.user_id,
    agent_id: r.agent_id,
    task_id: r.task_id,
    created_at: r.created_at, updated_at: r.updated_at, score: r.score,
  }));

  return successEnvelope<AtomicSearchData>({ items }, requestId);
}

async function handleAtomicDelete(body: unknown, auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = atomicDeleteRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { ids } = parsed.data;

  const store = deps.getStore();
  if (!store) return errorEnvelope(503, "Store not available", requestId);

  // deleteL1Batch returns bool, but we need actual count
  // Fall back to per-id deletion for accurate counting
  // 按 id 删除时，只使用显式传入的 isolation 维度作为 filter。
  // 避免默认 sessionId="default" 导致真实 session 下的 L1 记录无法删除。
  const iso = deps.requestIsolation;
  const deleteFilter = iso ? {
    ...(iso.teamId ? { teamId: iso.teamId } : {}),
    ...(iso.userId ? { userId: iso.userId } : {}),
    ...(iso.agentId ? { agentId: iso.agentId } : {}),
    ...(iso.taskId ? { taskId: iso.taskId } : {}),
    // 不传 sessionId：按 id 删除不应被默认 sessionId 限制
  } : undefined;
  let deletedCount = 0;
  const deletedIds: string[] = [];
  for (const id of ids) {
    const ok = await store.deleteL1(id, deleteFilter);
    if (ok) {
      deletedCount++;
      deletedIds.push(id);
    }
  }

  // 审计：L1 delete — 每条删除一行 audit
  for (const id of deletedIds) {
    await recordAudit(store, {
      record_id: id,
      layer: "L1",
      action: "delete",
      iso: deps.requestIsolation,
      version: 0, // 已删除，无新版本
      requestId,
      logger: deps.logger,
    });
  }

  // Report memory deletion (non-fatal)
  if (deps.quotaManager && deletedCount > 0) {
    deps.quotaManager.reportMemoryDeleted(auth.serviceId, deletedCount).catch(() => {});
  }

  return successEnvelope<AtomicDeleteData>({ deleted_count: deletedCount }, requestId);
}

// ============================
// Entity Metadata Handlers (Team / User / Agent / Task)
// ============================

type EntityStore = IMemoryStore & {
  createTeam?: (input: any) => TeamData | Promise<TeamData>;
  getTeam?: (id: string) => TeamData | null | Promise<TeamData | null>;
  updateTeam?: (id: string, patch: any) => TeamData | null | Promise<TeamData | null>;
  deleteTeams?: (ids: string[]) => BatchDeleteResult | Promise<BatchDeleteResult>;
  createUser?: (input: any) => UserData | Promise<UserData>;
  getUser?: (id: string) => UserData | null | Promise<UserData | null>;
  updateUser?: (id: string, patch: any) => UserData | null | Promise<UserData | null>;
  deleteUsers?: (ids: string[]) => BatchDeleteResult | Promise<BatchDeleteResult>;
  createAgent?: (input: any) => AgentData | Promise<AgentData>;
  getAgent?: (id: string) => AgentData | null | Promise<AgentData | null>;
  updateAgent?: (id: string, patch: any) => AgentData | null | Promise<AgentData | null>;
  deleteAgents?: (ids: string[]) => BatchDeleteResult | Promise<BatchDeleteResult>;
  createTask?: (input: any) => TaskData | Promise<TaskData>;
  getTask?: (id: string) => TaskData | null | Promise<TaskData | null>;
  updateTask?: (id: string, patch: any) => TaskData | null | Promise<TaskData | null>;
  deleteTasks?: (ids: string[]) => BatchDeleteResult | Promise<BatchDeleteResult>;
};

function getEntityStore(deps: V2RouterDeps): EntityStore | undefined {
  return deps.getStore() as EntityStore | undefined;
}

function missingEntityStore(requestId: string): ApiResponseEnvelope {
  return errorEnvelope(503, "Entity metadata store not available", requestId);
}

/* ============================================================================
 * @deprecated v2 entity handlers（team/user/agent/task，下方 16 个）。
 *
 * 这些 handler 走旧 `entity_*` 聚合数组模型（EntityStore 可选方法）。元数据已迁移到
 * v3 规范化模型（`/v3/meta/*` + MetadataService），control 面板 remote 模式直连 v3。
 * 保留这些 handler 仅为兼容现网既有调用方，**行为冻结、不再演进**；新接入方一律用 v3。
 * 计划：确认无外部调用方后，连同 routeTable 中对应条目与 `entity_*` 表一并删除。
 * 决策（2026-06-26，per user）：标记废弃 + 代码暂留，不适配 MetadataService。
 * ========================================================================== */

/** @deprecated 改用 `/v3/meta/team/create`（MetadataService.createTeam）。 */
async function handleTeamCreate(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = teamCreateRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const store = getEntityStore(deps);
  if (!store?.createTeam) return missingEntityStore(requestId);
  return successEnvelope<TeamData>(await store.createTeam(parsed.data), requestId);
}

/** @deprecated 改用 `/v3/meta/team/get`。 */
async function handleTeamGet(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = teamGetRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const store = getEntityStore(deps);
  if (!store?.getTeam) return missingEntityStore(requestId);
  const data = await store.getTeam(parsed.data.team_id);
  return data ? successEnvelope<TeamData>(data, requestId) : errorEnvelope(404, "Team not found", requestId);
}

/** @deprecated 改用 `/v3/meta/team/update`。 */
async function handleTeamUpdate(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = teamUpdateRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { team_id, ...patch } = parsed.data;
  const store = getEntityStore(deps);
  if (!store?.updateTeam) return missingEntityStore(requestId);
  const data = await store.updateTeam(team_id, patch);
  return data ? successEnvelope<TeamData>(data, requestId) : errorEnvelope(404, "Team not found", requestId);
}

/** @deprecated 改用 `/v3/meta/team/delete`。 */
async function handleTeamDelete(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = teamBatchDeleteRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const store = getEntityStore(deps);
  if (!store?.deleteTeams) return missingEntityStore(requestId);
  return successEnvelope<BatchDeleteResult>(await store.deleteTeams(parsed.data.team_ids), requestId);
}

/** @deprecated 改用 `/v3/meta/user/create`。 */
async function handleUserCreate(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = userCreateRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const store = getEntityStore(deps);
  if (!store?.createUser) return missingEntityStore(requestId);
  return successEnvelope<UserData>(await store.createUser(parsed.data), requestId);
}

/** @deprecated 改用 `/v3/meta/user/get`。 */
async function handleUserGet(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = userGetRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const store = getEntityStore(deps);
  if (!store?.getUser) return missingEntityStore(requestId);
  const data = await store.getUser(parsed.data.user_id);
  return data ? successEnvelope<UserData>(data, requestId) : errorEnvelope(404, "User not found", requestId);
}

/** @deprecated 改用 `/v3/meta/user/update`。 */
async function handleUserUpdate(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = userUpdateRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { user_id, ...patch } = parsed.data;
  const store = getEntityStore(deps);
  if (!store?.updateUser) return missingEntityStore(requestId);
  const data = await store.updateUser(user_id, patch);
  return data ? successEnvelope<UserData>(data, requestId) : errorEnvelope(404, "User not found", requestId);
}

/** @deprecated 改用 `/v3/meta/user/delete`。 */
async function handleUserDelete(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = userBatchDeleteRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const store = getEntityStore(deps);
  if (!store?.deleteUsers) return missingEntityStore(requestId);
  return successEnvelope<BatchDeleteResult>(await store.deleteUsers(parsed.data.user_ids), requestId);
}

/** @deprecated 改用 `/v3/meta/agent/create`。 */
async function handleAgentCreate(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = agentCreateRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const store = getEntityStore(deps);
  if (!store?.createAgent) return missingEntityStore(requestId);
  return successEnvelope<AgentData>(await store.createAgent(parsed.data), requestId);
}

/** @deprecated 改用 `/v3/meta/agent/get`。 */
async function handleAgentGet(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = agentGetRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const store = getEntityStore(deps);
  if (!store?.getAgent) return missingEntityStore(requestId);
  const data = await store.getAgent(parsed.data.agent_id);
  if (!data) return errorEnvelope(404, "Agent not found", requestId);
  if (parsed.data.team_id && data.team_id !== parsed.data.team_id) return errorEnvelope(403, "Agent team_id mismatch", requestId);
  return successEnvelope<AgentData>(data, requestId);
}

/** @deprecated 改用 `/v3/meta/agent/update`。 */
async function handleAgentUpdate(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = agentUpdateRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { agent_id, team_id, ...patch } = parsed.data;
  const store = getEntityStore(deps);
  if (!store?.updateAgent) return missingEntityStore(requestId);
  const current = store.getAgent ? await store.getAgent(agent_id) : null;
  if (!current) return errorEnvelope(404, "Agent not found", requestId);
  if (team_id && current.team_id !== team_id) return errorEnvelope(403, "Agent team_id mismatch", requestId);
  const data = await store.updateAgent(agent_id, patch);
  return data ? successEnvelope<AgentData>(data, requestId) : errorEnvelope(404, "Agent not found", requestId);
}

/** @deprecated 改用 `/v3/meta/agent/delete`。 */
async function handleAgentDelete(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = agentBatchDeleteRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const store = getEntityStore(deps);
  if (!store?.deleteAgents) return missingEntityStore(requestId);
  return successEnvelope<BatchDeleteResult>(await store.deleteAgents(parsed.data.agent_ids), requestId);
}

/** @deprecated 改用 `/v3/meta/task/create`。 */
async function handleTaskCreate(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = taskCreateRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const store = getEntityStore(deps);
  if (!store?.createTask) return missingEntityStore(requestId);
  return successEnvelope<TaskData>(await store.createTask(parsed.data), requestId);
}

/** @deprecated 改用 `/v3/meta/task/get`。 */
async function handleTaskGet(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = taskGetRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const store = getEntityStore(deps);
  if (!store?.getTask) return missingEntityStore(requestId);
  const data = await store.getTask(parsed.data.task_id);
  return data ? successEnvelope<TaskData>(data, requestId) : errorEnvelope(404, "Task not found", requestId);
}

/** @deprecated 改用 `/v3/meta/task/update`。 */
async function handleTaskUpdate(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = taskUpdateRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { task_id, ...patch } = parsed.data;
  const store = getEntityStore(deps);
  if (!store?.updateTask) return missingEntityStore(requestId);
  const data = await store.updateTask(task_id, patch);
  return data ? successEnvelope<TaskData>(data, requestId) : errorEnvelope(404, "Task not found", requestId);
}

/** @deprecated 改用 `/v3/meta/task/delete`。 */
async function handleTaskDelete(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = taskBatchDeleteRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const store = getEntityStore(deps);
  if (!store?.deleteTasks) return missingEntityStore(requestId);
  return successEnvelope<BatchDeleteResult>(await store.deleteTasks(parsed.data.task_ids), requestId);
}

// ============================
// L2/L3 Profile Sync Helpers (write-through to VDB)
// ============================

type RequestIsolation = { teamId?: string; userId: string; agentId: string; sessionId: string; taskId?: string };

function buildIsolationScope(isolation?: RequestIsolation): string {
  return isolation ? buildProfileIsolationScope(isolation) : DEFAULT_PROFILE_SCOPE;
}

function buildIsolationStoragePrefix(isolation: RequestIsolation): string {
  return `profiles/${encodeURIComponent(buildIsolationScope(isolation))}/`;
}

function scopedProfileStorage(storage: StorageAdapter, isolation?: RequestIsolation): StorageAdapter {
  // Direct unit callers may not go through handleV2Route and therefore do not
  // have requestIsolation attached. Keep that legacy path at root; real HTTP
  // requests always resolve to either explicit ids or the `default` bucket.
  if (!isolation) return storage;
  return createScopedStorageAdapter(storage, buildIsolationStoragePrefix(isolation));
}

function md5Hex(text: string): string {
  return createHash("md5").update(text).digest("hex");
}

function parseMetadataJson(raw: string | undefined): MemoryRecord["metadata"] {
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed as MemoryRecord["metadata"] : {};
  } catch {
    return {};
  }
}

async function getProfileVersion(
  store: IMemoryStore | undefined,
  type: "l2" | "l3",
  filename: string,
  isolation?: RequestIsolation,
): Promise<number> {
  if (!store) return 0;
  const scope = buildIsolationScope(isolation);
  const id = buildProfileStableId(scope, type, filename);
  try {
    // 优先使用轻量按 id 查询，避免全量 pullProfiles()
    if (typeof store.queryProfilesByIds === "function") {
      const results = await store.queryProfilesByIds([id]);
      return results[0]?.version ?? 0;
    }
    if (typeof store.pullProfiles === "function") {
      const existing = (await store.pullProfiles()).find((r) => r.id === id);
      return existing?.version ?? 0;
    }
  } catch {
    // fall through
  }
  return 0;
}

/**
 * 批量获取多个 profile 的 version，一次查询。
 * 优先使用 queryProfilesByIds（轻量），fallback 到 pullProfiles（全量）。
 */
async function getProfileVersionBatch(
  store: IMemoryStore | undefined,
  type: "l2" | "l3",
  filenames: string[],
  isolation?: RequestIsolation,
): Promise<Map<string, number>> {
  const result = new Map<string, number>();
  if (!store || filenames.length === 0) return result;
  const scope = buildIsolationScope(isolation);
  const ids = filenames.map((fn) => buildProfileStableId(scope, type, fn));
  try {
    if (typeof store.queryProfilesByIds === "function") {
      const records = await store.queryProfilesByIds(ids);
      const versionMap = new Map(records.map((r) => [r.id, r.version]));
      for (let i = 0; i < filenames.length; i++) {
        result.set(filenames[i], versionMap.get(ids[i]) ?? 0);
      }
      return result;
    }
    if (typeof store.pullProfiles === "function") {
      const all = await store.pullProfiles();
      const versionMap = new Map(all.map((r) => [r.id, r.version]));
      for (let i = 0; i < filenames.length; i++) {
        result.set(filenames[i], versionMap.get(ids[i]) ?? 0);
      }
      return result;
    }
  } catch {
    // fall through
  }
  return result;
}

/** Best-effort write-through L2/L3 profile to VDB. Failure is logged but does not break the API. */
async function syncProfileToVdb(
  store: IMemoryStore | undefined,
  type: "l2" | "l3",
  filename: string,
  content: string,
  logger: Logger,
  createdAtOverride?: number,
  isolation?: RequestIsolation,
): Promise<number> {
  if (!store || typeof store.syncProfiles !== "function") return 0;
  try {
    const now = Date.now();

    // Try to extract created time from META in content
    let createdAtMs = createdAtOverride ?? 0;
    if (!createdAtMs) {
      const metaMatch = content.match(/^-----META-START-----\n([\s\S]*?)\n-----META-END-----/);
      if (metaMatch) {
        for (const line of metaMatch[1].split("\n")) {
          if (line.startsWith("created: ")) {
            const ts = Date.parse(line.slice(9));
            if (!isNaN(ts)) createdAtMs = ts;
            break;
          }
        }
      }
    }
    if (!createdAtMs) createdAtMs = now;

    const scope = buildIsolationScope(isolation);
    const id = buildProfileStableId(scope, type, filename);

    // Probe current VDB version to satisfy the optimistic-lock check in
    // TcvdbMemoryStore.syncProfiles (which compares baselineVersion against
    // the remote version). Without this, the second and subsequent writes
    // to the same profile would be silently skipped as a version conflict.
    // Best-effort: if queryProfilesByIds/pullProfiles is unavailable or fails,
    // fall back to undefined and let syncProfiles decide (insert if remote missing,
    // otherwise log + skip — which preserves the previous behaviour).
    let baselineVersion: number | undefined;
    let currentMd5: string | undefined;
    try {
      if (typeof store.queryProfilesByIds === "function") {
        const results = await store.queryProfilesByIds([id]);
        const existing = results[0];
        if (existing) {
          baselineVersion = existing.version;
          currentMd5 = existing.contentMd5;
        }
      } else if (typeof store.pullProfiles === "function") {
        const remote = await store.pullProfiles();
        const existing = remote.find((r) => r.id === id);
        if (existing) {
          baselineVersion = existing.version;
          currentMd5 = existing.contentMd5;
        }
      }
    } catch (err) {
      logger.warn(`${TAG} [profile-sync] probe failed for ${filename}: ${err instanceof Error ? err.message : String(err)}`);
    }

    const contentMd5 = md5Hex(content);
    const nextVersion = currentMd5 === contentMd5
      ? (baselineVersion ?? 0)
      : (baselineVersion === undefined ? 0 : baselineVersion + 1);

    const record: ProfileSyncRecord = {
      id,
      type,
      filename,
      content,
      contentMd5,
      version: nextVersion,
      createdAtMs,
      updatedAtMs: now,
      baselineVersion,
      teamId: isolation?.teamId,
      userId: isolation?.userId,
      agentId: isolation?.agentId,
      // L2/L3 profiles are team+agent scoped; session_id/task_id are intentionally not written.
      sessionId: undefined,
    };
    await store.syncProfiles([record]);
    logger.debug?.(`${TAG} [profile-sync] ${type} upserted to VDB: ${filename} (baselineVersion=${baselineVersion ?? "new"}, version=${nextVersion})`);
    return nextVersion;
  } catch (err) {
    logger.warn(`${TAG} [profile-sync] FAILED to sync ${type} profile ${filename} to VDB: ${err instanceof Error ? err.message : String(err)}`);
    return 0;
  }
}

/** Best-effort delete L2 profiles from VDB. */
async function deleteProfilesFromVdb(
  store: IMemoryStore | undefined,
  type: "l2" | "l3",
  filenames: string[],
  logger: Logger,
  isolation?: RequestIsolation,
): Promise<void> {
  if (!store || typeof store.deleteProfiles !== "function" || filenames.length === 0) return;
  try {
    const scope = buildIsolationScope(isolation);
    const ids = filenames.map((fn) => buildProfileStableId(scope, type, fn));
    await store.deleteProfiles(ids);
    logger.debug?.(`${TAG} [profile-sync] ${type} deleted from VDB: ${filenames.length} files`);
  } catch (err) {
    logger.warn(`${TAG} [profile-sync] FAILED to delete ${type} profiles from VDB: ${err instanceof Error ? err.message : String(err)}`);
  }
}

/** Best-effort refresh scene_index.json so pipeline sees the user-written L2 files. */
async function refreshSceneIndex(storage: StorageAdapter, logger: Logger): Promise<void> {
  try {
    const { syncSceneIndex } = await import("../core/scene/scene-index.js");
    // Pass empty dataDir; we only use storage in service mode.
    await syncSceneIndex("", storage);
  } catch (err) {
    logger.warn(`${TAG} [scene-index] FAILED to refresh scene index: ${err instanceof Error ? err.message : String(err)}`);
  }
}

// ============================
// L2 Scenario Handlers
// ============================

async function handleScenarioLs(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = scenarioListRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { path_prefix } = parsed.data;

  const baseStorage = deps.getStorage();
  if (!baseStorage) return errorEnvelope(503, "Storage not available", requestId);
  const storage = scopedProfileStorage(baseStorage, deps.requestIsolation);

  const prefix = path_prefix
    ? `${StoragePaths.sceneBlocksDir}${path_prefix}`
    : StoragePaths.sceneBlocksDir;

  deps.logger.debug?.(`${TAG} [scenario/ls] storage.type=${storage.type}, prefix="${prefix}"`);

  // One-shot full listing (no pagination; marker-based pagination planned for phase 2)
  const backend = storage.getBackend();
  const result = await backend.listObjects(prefix, { recursive: true });
  deps.logger.debug?.(`${TAG} [scenario/ls] listObjects returned ${result.entries.length} entries`);
  const allEntries = result.entries;

  // Read scene_index.json for summary + created/updated lookup
  const { readSceneIndex } = await import("../core/scene/scene-index.js");
  const sceneIndex = await readSceneIndex("", storage);
  const indexMap = new Map(sceneIndex.map((e) => [e.filename, e]));

  // 批量获取所有 L2 文件的 profile version（一次查询，避免 N+1）
  const l2Filenames = allEntries.filter((e) => !e.isDirectory).map((e) => {
    return e.key.startsWith(StoragePaths.sceneBlocksDir)
      ? e.key.slice(StoragePaths.sceneBlocksDir.length)
      : e.key;
  });
  const versionMap = await getProfileVersionBatch(deps.getStore(), "l2", l2Filenames, deps.requestIsolation);

  const entries: ScenarioEntry[] = allEntries.map((e) => {
    const externalPath = e.key.startsWith(StoragePaths.sceneBlocksDir)
      ? e.key.slice(StoragePaths.sceneBlocksDir.length)
      : e.key;
    const displayPath = e.isDirectory && !externalPath.endsWith("/") ? `${externalPath}/` : externalPath;
    const indexEntry = indexMap.get(externalPath);
    const fallbackTime = e.lastModified.toISOString();
    return {
      path: displayPath,
      summary: indexEntry?.summary || undefined,
      version: e.isDirectory ? 0 : (versionMap.get(externalPath) ?? 0),
      team_id: deps.requestIsolation?.teamId,
      agent_id: deps.requestIsolation?.agentId,
      created_at: indexEntry?.created || fallbackTime,
      updated_at: indexEntry?.updated || fallbackTime,
    };
  });

  return successEnvelope({ entries, total: entries.length }, requestId);
}

async function handleScenarioCount(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = scenarioCountRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { path_prefix } = parsed.data;

  const store = deps.getStore();
  if (typeof store?.countProfiles === "function") {
    const total = await store.countProfiles({
      type: "l2",
      pathPrefix: path_prefix,
      teamId: deps.requestIsolation?.teamId,
      userId: deps.requestIsolation?.teamId ? undefined : deps.requestIsolation?.userId,
      agentId: deps.requestIsolation?.agentId,
    });
    return successEnvelope<CountData>({ total }, requestId);
  }

  const baseStorage = deps.getStorage();
  if (!baseStorage) return errorEnvelope(503, "Storage not available", requestId);
  const storage = scopedProfileStorage(baseStorage, deps.requestIsolation);
  const prefix = path_prefix
    ? `${StoragePaths.sceneBlocksDir}${path_prefix}`
    : StoragePaths.sceneBlocksDir;
  const result = await storage.getBackend().listObjects(prefix, { recursive: true });
  const total = result.entries.filter((e) => !e.isDirectory).length;
  return successEnvelope<CountData>({ total }, requestId);
}

async function handleScenarioRead(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = scenarioReadRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { path } = parsed.data;

  const baseStorage = deps.getStorage();
  if (!baseStorage) return errorEnvelope(503, "Storage not available", requestId);
  const storage = scopedProfileStorage(baseStorage, deps.requestIsolation);

  const key = `${StoragePaths.sceneBlocksDir}${path}`;
  const content = await storage.readFile(key);

  // File not found → return 200 with null content (not 404)
  if (content === null) {
    return successEnvelope<ScenarioFile>({
      path,
      content: null as unknown as string,
      created_at: null as unknown as string,
      updated_at: null as unknown as string,
    }, requestId);
  }

  // Parse META for created/updated
  const now = new Date().toISOString();
  let createdAt = now;
  let updatedAt = now;

  const metaMatch = content.match(/^-----META-START-----\n([\s\S]*?)\n-----META-END-----/);
  if (metaMatch) {
    for (const line of metaMatch[1].split("\n")) {
      const idx = line.indexOf(": ");
      if (idx > 0) {
        const k = line.slice(0, idx);
        const v = line.slice(idx + 2);
        if (k === "created") createdAt = v;
        if (k === "updated") updatedAt = v;
      }
    }
  } else {
    // Fallback: try scene_index
    const { readSceneIndex } = await import("../core/scene/scene-index.js");
    const sceneIndex = await readSceneIndex("", storage);
    const entry = sceneIndex.find((e) => e.filename === path);
    if (entry) {
      createdAt = entry.created || now;
      updatedAt = entry.updated || now;
    } else {
      const stat = await storage.stat(key);
      if (stat) {
        createdAt = new Date(stat.lastModified).toISOString();
        updatedAt = createdAt;
      }
    }
  }

  return successEnvelope<ScenarioFile>({
    path, content,
    version: await getProfileVersion(deps.getStore(), "l2", path, deps.requestIsolation),
    team_id: deps.requestIsolation?.teamId,
    agent_id: deps.requestIsolation?.agentId,
    created_at: createdAt,
    updated_at: updatedAt,
  }, requestId);
}

async function handleScenarioWrite(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = scenarioWriteRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { path, content, summary } = parsed.data;

  const baseStorage = deps.getStorage();
  if (!baseStorage) return errorEnvelope(503, "Storage not available", requestId);
  const storage = scopedProfileStorage(baseStorage, deps.requestIsolation);

  const key = `${StoragePaths.sceneBlocksDir}${path}`;

  // Existence check: path must already exist (no upsert/create)
  const existing = await storage.readFile(key);
  if (existing === null) return errorEnvelope(404, `Scenario file not found: ${path}`, requestId);

  // Parse existing META to preserve created + update updated/summary
  const now = new Date().toISOString();
  let finalContent: string;

  const metaMatch = existing.match(/^-----META-START-----\n([\s\S]*?)\n-----META-END-----\n?/);
  if (metaMatch) {
    // Parse existing META fields
    const metaBlock = metaMatch[1];
    const metaFields: Record<string, string> = {};
    for (const line of metaBlock.split("\n")) {
      const idx = line.indexOf(": ");
      if (idx > 0) metaFields[line.slice(0, idx)] = line.slice(idx + 2);
    }

    // Update fields
    metaFields["updated"] = now;
    if (summary !== undefined) metaFields["summary"] = summary;

    const newMeta = Object.entries(metaFields).map(([k, v]) => `${k}: ${v}`).join("\n");
    finalContent = `-----META-START-----\n${newMeta}\n-----META-END-----\n\n${content}`;
  } else {
    // META missing or corrupted — rebuild
    const metaLines = [
      `created: ${now}`,
      `updated: ${now}`,
    ];
    if (summary !== undefined) metaLines.push(`summary: ${summary}`);
    finalContent = `-----META-START-----\n${metaLines.join("\n")}\n-----META-END-----\n\n${content}`;
  }

  await storage.writeFile(key, finalContent);

  // Sync L2 to VDB profiles + refresh scene index (best-effort)
  const store = deps.getStore();
  const version = await syncProfileToVdb(store, "l2", path, finalContent, deps.logger, undefined, deps.requestIsolation);
  await refreshSceneIndex(storage, deps.logger);

  // 审计：L2 update — record_id 用 path（L2 主键 = 文件路径）
  await recordAudit(store, {
    record_id: path,
    layer: "L2",
    action: "update",
    iso: deps.requestIsolation,
    version,
    requestId,
    logger: deps.logger,
  });

  return successEnvelope<ScenarioWriteData>({
    path,
    updated_at: now,
    version,
    team_id: deps.requestIsolation?.teamId,
    agent_id: deps.requestIsolation?.agentId,
  }, requestId);
}

async function handleScenarioRm(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = scenarioRmRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { path } = parsed.data;

  const baseStorage = deps.getStorage();
  if (!baseStorage) return errorEnvelope(503, "Storage not available", requestId);
  const storage = scopedProfileStorage(baseStorage, deps.requestIsolation);

  const key = `${StoragePaths.sceneBlocksDir}${path}`;
  // Collect filenames to delete from VDB (single file or all files under a directory)
  let removedFilenames: string[] = [];
  if (path.endsWith("/")) {
    try {
      const names = await storage.readdirNames(key, ".md");
      removedFilenames = names.map((name) => `${path}${name}`);
    } catch { /* ignore */ }
    await storage.rmdir(key);
  } else {
    removedFilenames = [path];
    await storage.unlink(key);
  }

  // Delete L2 profiles from VDB (best-effort)
  const store = deps.getStore();
  await deleteProfilesFromVdb(store, "l2", removedFilenames, deps.logger, deps.requestIsolation);
  await refreshSceneIndex(storage, deps.logger);

  // 审计：L2 delete — 每个被删的 path 一行
  for (const fname of removedFilenames) {
    await recordAudit(store, {
      record_id: fname,
      layer: "L2",
      action: "delete",
      iso: deps.requestIsolation,
      version: 0,
      requestId,
      logger: deps.logger,
    });
  }

  return successEnvelope(undefined, requestId);
}

// ============================
// L3 Core Handlers
// ============================

async function handleCoreRead(_body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const baseStorage = deps.getStorage();
  if (!baseStorage) return errorEnvelope(503, "Storage not available", requestId);
  const storage = scopedProfileStorage(baseStorage, deps.requestIsolation);

  deps.logger.debug?.(`${TAG} [core/read] storage.type=${storage.type}, key="${StoragePaths.persona}"`);
  const content = await storage.readFile(StoragePaths.persona);
  deps.logger.debug?.(`${TAG} [core/read] readFile result: ${content === null ? "null (not found)" : `${content.length} chars`}`);

  // File not found → return 200 with null content (not 404)
  if (content === null) {
    return successEnvelope<CoreFile>({
      content: null as unknown as string,
      created_at: null as unknown as string,
      updated_at: null as unknown as string,
    }, requestId);
  }

  const stat = await storage.stat(StoragePaths.persona);
  const now = new Date().toISOString();

  return successEnvelope<CoreFile>({
    content,
    version: await getProfileVersion(deps.getStore(), "l3", StoragePaths.persona, deps.requestIsolation),
    team_id: deps.requestIsolation?.teamId,
    agent_id: deps.requestIsolation?.agentId,
    created_at: stat ? new Date(stat.createdAt).toISOString() : now,
    updated_at: stat ? new Date(stat.lastModified).toISOString() : now,
  }, requestId);
}

async function handleCoreCount(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = coreCountRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const store = deps.getStore();
  if (typeof store?.countProfiles === "function") {
    const total = await store.countProfiles({
      type: "l3",
      teamId: deps.requestIsolation?.teamId,
      userId: deps.requestIsolation?.teamId ? undefined : deps.requestIsolation?.userId,
      agentId: deps.requestIsolation?.agentId,
    });
    return successEnvelope<CountData>({ total }, requestId);
  }

  const baseStorage = deps.getStorage();
  if (!baseStorage) return errorEnvelope(503, "Storage not available", requestId);
  const storage = scopedProfileStorage(baseStorage, deps.requestIsolation);
  const content = await storage.readFile(StoragePaths.persona);
  return successEnvelope<CountData>({ total: content ? 1 : 0 }, requestId);
}

async function handleCoreWrite(body: unknown, _auth: V2AuthContext, requestId: string, deps: V2RouterDeps): Promise<ApiResponseEnvelope> {
  const parsed = coreWriteRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodError(parsed.error), requestId);
  const { content } = parsed.data;

  const baseStorage = deps.getStorage();
  if (!baseStorage) return errorEnvelope(503, "Storage not available", requestId);
  const storage = scopedProfileStorage(baseStorage, deps.requestIsolation);

  // Normalize before persistence: persona body must NOT contain Scene Navigation
  // (a derived section rebuilt from scene_index.json) or stray surrounding
  // whitespace. Both COS and VDB get the *exact* same bytes so md5(content) is
  // a stable identity across stores. Without this, /v2/core/write callers that
  // post the raw round-tripped body (which includes the navigation footer and
  // a trailing newline appended by refreshPersonaNavigation) would write a
  // mismatched copy to each store, and pullProfilesToLocal would later treat
  // the persona as corrupted and delete the COS copy.
  const personaBody = stripSceneNavigation(content).trim();

  await storage.writeFile(StoragePaths.persona, personaBody);

  // Sync L3 persona to VDB profiles (best-effort)
  const store = deps.getStore();
  const version = await syncProfileToVdb(store, "l3", StoragePaths.persona, personaBody, deps.logger, undefined, deps.requestIsolation);

  // 审计：L3 update — record_id 用 persona 的 storage path
  await recordAudit(store, {
    record_id: StoragePaths.persona,
    layer: "L3",
    action: "update",
    iso: deps.requestIsolation,
    version,
    requestId,
    logger: deps.logger,
  });

  return successEnvelope<CoreWriteData>({
    updated_at: new Date().toISOString(),
    version,
    team_id: deps.requestIsolation?.teamId,
    agent_id: deps.requestIsolation?.agentId,
  }, requestId);
}

// ─────────────────────────────────────────────────────────────────────────
// /v2/pipeline/status — standalone-only introspection.
// Returns per-L-type queue/in-flight stats by reading the in-memory task
// queue (LocalStateBackend.listQueuedTasks) and worker's running set
// (PipelineWorker.getRunningTasks). idle = queued===0 && running===0.
// Mirrors the old MemoryPipelineManager.getQueueSizes() {l1Idle,l2Idle,l3Idle}
// semantics so seed clients can wait specifically for L1 to drain (without
// being blocked by slow L2/L3 cascades).
// Service mode returns 404 (route not exposed).
// ─────────────────────────────────────────────────────────────────────────

interface LayerStatus {
  /** Tasks waiting to be consumed (in queue). */
  queued: number;
  /** Tasks consumed by worker but not yet completed/failed. */
  running: number;
  /** Distinct sessionIds of queued tasks (for diagnostics). */
  queued_sessions: string[];
  /** Distinct sessionIds of running tasks (for diagnostics). */
  running_sessions: string[];
  /** True iff queued===0 && running===0. */
  idle: boolean;
}

interface PipelineStatusData {
  l1: LayerStatus;
  l2: LayerStatus;
  l3: LayerStatus;
}

function emptyLayer(): LayerStatus {
  return { queued: 0, running: 0, queued_sessions: [], running_sessions: [], idle: true };
}

async function handlePipelineStatus(
  _body: unknown,
  _auth: V2AuthContext,
  requestId: string,
  deps: V2RouterDeps,
): Promise<ApiResponseEnvelope> {
  // Service mode does not expose this endpoint — pretend it's not routed.
  if (deps.deployMode !== "standalone") {
    return errorEnvelope(404, "Not found", requestId);
  }

  // Legacy standalone (no stateBackend / no worker) — pipeline isn't running.
  if (!deps.stateBackend || !deps.pipelineWorker) {
    return errorEnvelope(503, "Pipeline not running (legacy standalone mode)", requestId);
  }

  // listQueuedTasks is optional on IStateBackend; LocalStateBackend implements
  // it, remote backends may not. Service mode never reaches here anyway.
  if (!deps.stateBackend.listQueuedTasks) {
    return errorEnvelope(
      503,
      "stateBackend does not support listQueuedTasks (status endpoint requires LocalStateBackend)",
      requestId,
    );
  }

  const queued = await deps.stateBackend.listQueuedTasks();
  const running = deps.pipelineWorker.getRunningTasks();

  const layers: Record<"L1" | "L2" | "L3", LayerStatus> = {
    L1: emptyLayer(),
    L2: emptyLayer(),
    L3: emptyLayer(),
  };
  // Track sessionIds in a Set per layer/category for de-dup, then materialize.
  const queuedSessionSets: Record<"L1" | "L2" | "L3", Set<string>> = {
    L1: new Set(),
    L2: new Set(),
    L3: new Set(),
  };
  const runningSessionSets: Record<"L1" | "L2" | "L3", Set<string>> = {
    L1: new Set(),
    L2: new Set(),
    L3: new Set(),
  };

  for (const t of queued) {
    if (t.type === "L1" || t.type === "L2" || t.type === "L3") {
      layers[t.type].queued++;
      queuedSessionSets[t.type].add(t.sessionId);
    }
    // "flush" tasks behave like L1 work (see executor.executeFlush fallback);
    // tally them under L1 so the seed-v2 idle wait doesn't miss them.
    if (t.type === "flush") {
      layers.L1.queued++;
      queuedSessionSets.L1.add(t.sessionId);
    }
  }
  for (const t of running) {
    if (t.type === "L1" || t.type === "L2" || t.type === "L3") {
      layers[t.type].running++;
      runningSessionSets[t.type].add(t.sessionId);
    }
    if (t.type === "flush") {
      layers.L1.running++;
      runningSessionSets.L1.add(t.sessionId);
    }
  }
  for (const k of ["L1", "L2", "L3"] as const) {
    layers[k].queued_sessions = Array.from(queuedSessionSets[k]).sort();
    layers[k].running_sessions = Array.from(runningSessionSets[k]).sort();
    layers[k].idle = layers[k].queued === 0 && layers[k].running === 0;
  }

  const data: PipelineStatusData = {
    l1: layers.L1,
    l2: layers.L2,
    l3: layers.L3,
  };

  return successEnvelope<PipelineStatusData>(data, requestId);
}

// ============================
// Helpers
// ============================

/**
 * Format a Date as YYYY-MM-DD in local timezone, matching the convention used by
 * v1 l0-recorder and l1-writer for daily JSONL shard names. Local copy to keep
 * v2-router self-contained (avoids exporting a util just for one call site).
 */
function formatLocalDateForJsonl(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

// ============================
// Exported for testing
// ============================

export {
  handleConversationAdd,
  handleConversationQuery,
  handleConversationSearch,
  handleConversationDelete,
  handleConversationCount,
  handleAtomicUpdate,
  handleAtomicQuery,
  handleAtomicSearch,
  handleAtomicDelete,
  handleAtomicCount,
  handleScenarioLs,
  handleScenarioRead,
  handleScenarioWrite,
  handleScenarioRm,
  handleScenarioCount,
  handleCoreRead,
  handleCoreWrite,
  handleCoreCount,
  handleTeamCreate,
  handleTeamGet,
  handleTeamUpdate,
  handleTeamDelete,
  handleUserCreate,
  handleUserGet,
  handleUserUpdate,
  handleUserDelete,
  handleAgentCreate,
  handleAgentGet,
  handleAgentUpdate,
  handleAgentDelete,
  handleTaskCreate,
  handleTaskGet,
  handleTaskUpdate,
  handleTaskDelete,
  handlePipelineStatus,
};
