/**
 * TDAI Memory Gateway — v2 API Schemas.
 *
 * Generated baseline: `generated/schemas.ts` (Kubb, do not edit).
 * This file re-exports everything from the generated baseline, then adds
 * hand-written overrides that OpenAPI cannot express:
 *   - safePath refine (path traversal prevention)
 *   - conversationDelete mutual exclusion refine
 *   - Generic ApiResponseEnvelope<T> interface
 *   - formatZodError utility
 */

import { z } from "zod";
import { DEFAULT_ISOLATION_ID } from "../core/store/types.js";

// ============================
// Re-export all generated schemas as-is
// ============================

export {
  apiResponseEnvelopeSchema,
  conversationRoleSchema,
  paginationSchema,
  conversationItemSchema,
  conversationAddDataSchema,
  conversationQueryRequestSchema,
  conversationQueryDataSchema,
  conversationDeleteDataSchema,
  atomicDetailSchema,
  atomicUpdateRequestSchema,
  atomicUpdateDataSchema,
  atomicDeleteRequestSchema,
  atomicDeleteDataSchema,
  atomicQueryRequestSchema,
  atomicQueryDataSchema,
  scenarioListRequestSchema,
  scenarioEntrySchema,
  scenarioListDataSchema,
  scenarioFileSchema,
  scenarioWriteDataSchema,
  coreFileSchema,
  coreReadRequestSchema,
  coreWriteRequestSchema,
  coreWriteDataSchema,
  conversationSearchRequestSchema,
  conversationSearchHitSchema,
  conversationSearchDataSchema,
  atomicSearchRequestSchema,
  atomicSearchHitSchema,
  atomicSearchDataSchema,
} from "./generated/schemas.js";

// Re-export generated types
export type {
  ConversationRole,
  Pagination,
  ConversationAddRequest,
  ConversationAddData,
  ConversationQueryRequest,
  ConversationQueryData,
  ConversationDeleteData,
  AtomicUpdateRequest,
  AtomicUpdateData,
  AtomicDeleteRequest,
  AtomicDeleteData,
  AtomicQueryRequest,
  ScenarioListRequest,
  CoreReadRequest,
  CoreWriteRequest,
  ConversationSearchRequest,
  ConversationSearchData,
  AtomicSearchRequest,
} from "./generated/types.js";

// Import schemas we need to override
import type {
  AtomicDetail as GeneratedAtomicDetail,
  ConversationItem as GeneratedConversationItem,
  ScenarioEntry as GeneratedScenarioEntry,
  ScenarioFile as GeneratedScenarioFile,
  ScenarioWriteData as GeneratedScenarioWriteData,
  CoreFile as GeneratedCoreFile,
  CoreWriteData as GeneratedCoreWriteData,
} from "./generated/types.js";

export interface ConversationItem extends GeneratedConversationItem {
  /** L0 session isolation dimension returned by query/search responses. */
  session_id?: string;
  /** Team ownership dimension. */
  team_id?: string;
  /** L0 user isolation dimension returned by query/search responses. */
  user_id?: string;
  /** L0 agent isolation dimension returned by query/search responses. */
  agent_id?: string;
  /** Optional task ownership dimension for L0/L1 filtering. */
  task_id?: string;
}

export type ConversationSearchHit = ConversationItem & { score: number };
import {
  conversationItemSchema as _conversationItemSchema,
  scenarioReadRequestSchema as _scenarioReadRequestSchema,
  scenarioWriteRequestSchema as _scenarioWriteRequestSchema,
  scenarioRmRequestSchema as _scenarioRmRequestSchema,
  conversationDeleteRequestSchema as _conversationDeleteRequestSchema,
} from "./generated/schemas.js";

// ============================
// Override: conversation add default session
// ============================

/** conversationAdd with session_id defaulting to compatibility bucket. */
export const conversationAddRequestSchema = z.object({
  session_id: z.string().min(1).default(DEFAULT_ISOLATION_ID),
  messages: z.array(_conversationItemSchema).min(1).max(100),
});
export type ConversationAddRequest = z.infer<typeof conversationAddRequestSchema>;

// ============================
// Count endpoints (sdk-v3.yaml)
// ============================

export interface CountData {
  total: number;
}

export const conversationCountRequestSchema = z.object({
  session_id: z.string().min(1).optional(),
  time_start: z.string().optional(),
  time_end: z.string().optional(),
});
export type ConversationCountRequest = z.infer<typeof conversationCountRequestSchema>;

export const atomicCountRequestSchema = z.object({
  type: z.string().optional(),
  time_start: z.string().optional(),
  time_end: z.string().optional(),
});
export type AtomicCountRequest = z.infer<typeof atomicCountRequestSchema>;

export const scenarioCountRequestSchema = z.object({
  path_prefix: z.string().optional(),
});
export type ScenarioCountRequest = z.infer<typeof scenarioCountRequestSchema>;

export const coreCountRequestSchema = z.object({});
export type CoreCountRequest = z.infer<typeof coreCountRequestSchema>;

// ============================
// Override: atomic response version exposure
// ============================

export interface AtomicDetail extends GeneratedAtomicDetail {
  /** Monotonic L1 memory version, starts from 0 and increments on update/merge. */
  version: number;
  team_id?: string;
  user_id?: string;
  agent_id?: string;
  task_id?: string;
}

export interface AtomicQueryData {
  items: AtomicDetail[];
  total: number;
}

export type AtomicSearchHit = AtomicDetail & { score: number };

export interface AtomicSearchData {
  items: AtomicSearchHit[];
}

export interface ScenarioEntry extends Omit<GeneratedScenarioEntry, "version"> {
  version?: number;
  summary?: string;
  team_id?: string;
  agent_id?: string;
}

export interface ScenarioListData {
  entries: ScenarioEntry[];
  total: number;
}

export interface ScenarioFile extends Omit<GeneratedScenarioFile, "version"> {
  version?: number;
  team_id?: string;
  agent_id?: string;
}

export interface ScenarioWriteData extends Omit<GeneratedScenarioWriteData, "version"> {
  version?: number;
  team_id?: string;
  agent_id?: string;
}

export interface CoreFile extends Omit<GeneratedCoreFile, "version"> {
  version?: number;
  team_id?: string;
  agent_id?: string;
}

export interface CoreWriteData extends Omit<GeneratedCoreWriteData, "version"> {
  version?: number;
  team_id?: string;
  agent_id?: string;
}

// ============================
// Entity metadata schemas (Team / User / Agent / Task)
// ============================

const stringArray = z.array(z.string()).max(100).default([]);
const teamStatusSchema = z.enum(["active", "archived"]);
const userStatusSchema = z.enum(["active", "inactive"]);
const agentStatusSchema = z.enum(["active", "inactive"]);
const agentVisibilitySchema = z.enum(["team", "restricted"]);
const taskSourceTypeSchema = z.enum(["manual", "github", "tapd", "other"]);

export interface BatchDeleteResult { deleted_ids: string[]; failed: Array<{ id: string; reason: string }> }

export interface TeamData {
  team_id: string; name: string; description?: string; owner_user_id: string; status: "active" | "archived";
  user_ids?: string[]; agent_ids?: string[]; task_ids?: string[]; created_at: string; updated_at: string;
}
export const teamCreateRequestSchema = z.object({ name: z.string().min(1), description: z.string().optional(), owner_user_id: z.string().min(1) });
export const teamGetRequestSchema = z.object({ team_id: z.string().min(1) });
export const teamUpdateRequestSchema = z.object({ team_id: z.string().min(1), name: z.string().min(1).optional(), description: z.string().optional(), owner_user_id: z.string().min(1).optional(), user_ids: z.array(z.string()).max(200).optional(), agent_ids: z.array(z.string()).max(200).optional(), status: teamStatusSchema.optional() });
export const teamBatchDeleteRequestSchema = z.object({ team_ids: z.array(z.string().min(1)).min(1).max(100) });

export interface UserData {
  user_id: string; name: string; job_description?: string; team_ids: string[]; task_ids: string[]; owned_agent_ids: string[]; task_agent_ids?: string[]; status: "active" | "inactive"; created_at: string; updated_at?: string;
}
export const userCreateRequestSchema = z.object({ name: z.string().min(1), job_description: z.string().optional() });
export const userGetRequestSchema = z.object({ user_id: z.string().min(1) });
export const userUpdateRequestSchema = z.object({ user_id: z.string().min(1), name: z.string().min(1).optional(), job_description: z.string().optional(), status: userStatusSchema.optional() });
export const userBatchDeleteRequestSchema = z.object({ user_ids: z.array(z.string().min(1)).min(1).max(100) });

export interface AgentData {
  agent_id: string; team_id: string; name: string; description?: string; prompt?: string; owner_user_id?: string; visibility: "team" | "restricted"; status: "active" | "inactive"; task_ids?: string[]; created_at: string; updated_at: string;
}
export const agentCreateRequestSchema = z.object({ team_id: z.string().min(1), name: z.string().min(1), description: z.string().optional(), prompt: z.string().optional(), owner_user_id: z.string().optional(), visibility: agentVisibilitySchema.optional() });
export const agentGetRequestSchema = z.object({ agent_id: z.string().min(1), team_id: z.string().min(1).optional() });
export const agentUpdateRequestSchema = z.object({ agent_id: z.string().min(1), team_id: z.string().min(1).optional(), name: z.string().min(1).optional(), description: z.string().optional(), prompt: z.string().optional(), owner_user_id: z.string().optional(), visibility: agentVisibilitySchema.optional(), status: agentStatusSchema.optional() });
export const agentBatchDeleteRequestSchema = z.object({ agent_ids: z.array(z.string().min(1)).min(1).max(100) });

export interface TaskData {
  task_id: string; team_id: string; creator_user_id: string; title?: string; description?: string; source_type: "manual" | "github" | "tapd" | "other"; source_url?: string; agent_ids: string[]; user_ids: string[]; created_at: string; updated_at: string;
}
export const taskCreateRequestSchema = z.object({ team_id: z.string().min(1), creator_user_id: z.string().min(1), title: z.string().optional(), description: z.string().optional(), source_type: taskSourceTypeSchema.optional(), source_url: z.string().optional(), agent_ids: stringArray.optional(), user_ids: stringArray.optional() });
export const taskGetRequestSchema = z.object({ task_id: z.string().min(1) });
export const taskUpdateRequestSchema = z.object({ task_id: z.string().min(1), title: z.string().optional(), description: z.string().optional(), source_type: taskSourceTypeSchema.optional(), source_url: z.string().optional(), agent_ids: stringArray.optional(), user_ids: stringArray.optional() });
export const taskBatchDeleteRequestSchema = z.object({ task_ids: z.array(z.string().min(1)).min(1).max(100) });

// ============================
// Override: safe path (prevent path traversal)
// ============================

const safePath = z.string().min(1).refine(
  (p) => !p.includes("\0")
    && !p.includes("\\")
    && !p.startsWith("/")
    && !p.split("/").some((part) => part === ".."),
  { message: "Path must be relative (no '..', no leading '/', no backslash/NUL)" },
);

/** scenarioRead with path traversal prevention. */
export const scenarioReadRequestSchema = z.object({ path: safePath });
export type ScenarioReadRequest = z.infer<typeof scenarioReadRequestSchema>;

/** scenarioWrite with path traversal prevention + summary. */
export const scenarioWriteRequestSchema = z.object({
  path: safePath,
  content: z.string(),
  summary: z.string().optional(),
});
export type ScenarioWriteRequest = z.infer<typeof scenarioWriteRequestSchema>;

/** scenarioRm with path traversal prevention. */
export const scenarioRmRequestSchema = z.object({ path: safePath });
export type ScenarioRmRequest = z.infer<typeof scenarioRmRequestSchema>;

// ============================
// Override: conversation delete mutual exclusion
// ============================

/** conversationDelete with mutual exclusion refine. */
export const conversationDeleteRequestSchema = z.object({
  message_ids: z.array(z.string()).min(1).max(100).optional(),
  session_id: z.string().optional(),
}).refine(
  (data) => {
    const hasIds = data.message_ids !== undefined && data.message_ids.length > 0;
    const hasSession = data.session_id !== undefined && data.session_id.trim().length > 0;
    return hasIds || hasSession;
  },
  { message: "At least one of message_ids or session_id must be provided" },
);
export type ConversationDeleteRequest = z.infer<typeof conversationDeleteRequestSchema>;

// ============================
// Tenancy isolation extension
// ============================
//
// L0/L1 carry (user_id, agent_id, session_id); L2/L3 Profile is agent-level
// and uses (user_id, agent_id). The generated request schemas only validate `session_id`; user_id / agent_id arrive
// either in the request body or in HTTP headers. This block keeps the
// generated schemas untouched and adds a separate schema for the new fields,
// so the router can validate body and headers independently.

/** Headers / body fields used to carry the three-dim isolation context. */
export const isolationFieldsSchema = z.object({
  user_id: z.string().min(1).default(DEFAULT_ISOLATION_ID),
  agent_id: z.string().min(1).default(DEFAULT_ISOLATION_ID),
  session_id: z.string().min(1).default(DEFAULT_ISOLATION_ID),
}).passthrough();
export type IsolationFields = z.infer<typeof isolationFieldsSchema>;

/**
 * Pull the three-dim isolation triple from the parsed body or from request
 * headers. Body wins (consistent with how `session_id` is currently sourced).
 *
 * Missing fields are filled with the default isolation bucket. This keeps the
 * API backward-compatible while still making ownership explicit in storage.
 */
export function resolveIsolation(
  body: Record<string, unknown> | undefined,
  headers: Record<string, string | string[] | undefined>,
  opts: { legacyCompatMode?: boolean; legacyPlaceholder?: string } = {},
): { ok: true; ctx: { userId: string; agentId: string; sessionId: string; taskId?: string } } {
  const headerStr = (k: string): string | undefined => {
    const raw = headers[k] ?? headers[k.toLowerCase()];
    if (Array.isArray(raw)) return raw[0];
    return typeof raw === "string" ? raw : undefined;
  };
  const teamId = (body?.team_id as string | undefined) ?? headerStr("x-tdai-team-id") ?? "";
  const userId = (body?.user_id as string | undefined) ?? headerStr("x-tdai-user-id") ?? "";
  const agentId = (body?.agent_id as string | undefined) ?? headerStr("x-tdai-agent-id") ?? "";
  const sessionId =
    (body?.session_id as string | undefined)
    ?? headerStr("x-tdai-session-id")
    ?? "";
  const taskId =
    (body?.task_id as string | undefined)
    ?? headerStr("x-tdai-task-id")
    ?? undefined;
  const missing: string[] = [];
  if (!userId) missing.push("user_id");
  if (!agentId) missing.push("agent_id");
  if (!sessionId) missing.push("session_id");

  const ph = opts.legacyCompatMode ? (opts.legacyPlaceholder ?? DEFAULT_ISOLATION_ID) : DEFAULT_ISOLATION_ID;
  const ctx = { ...(teamId ? { teamId } : {}), userId: userId || ph, agentId: agentId || ph, sessionId: sessionId || ph, ...(taskId ? { taskId } : {}) };
  return { ok: true, ctx };
}

// ============================
// Generic API Response Envelope
// ============================

export interface ApiResponseEnvelope<T = unknown> {
  code: number;
  message: string;
  request_id: string;
  data?: T;
}

// ============================
// Zod error formatter
// ============================

export function formatZodError(error: z.ZodError): string {
  return error.issues
    .map((issue) => {
      const path = issue.path.length > 0 ? issue.path.join(".") : "(root)";
      return `${path}: ${issue.message}`;
    })
    .join("; ");
}

// ============================
// Auth context
// ============================

export const v2AuthContextSchema = z.object({
  apiKey: z.string().min(1),
  serviceId: z.string().min(1),
});
export type V2AuthContext = z.infer<typeof v2AuthContextSchema>;
