/**
 * /v3/knowledge/* HTTP handlers — Knowledge entity CRUD.
 *
 * Pattern mirrors team/agent entity handlers in v2-router.ts:
 *   Zod safeParse → getEntityStore → store.method → successEnvelope/errorEnvelope
 *
 * 5 endpoints:
 *   POST /v3/knowledge/create — upsert knowledge metadata (idempotent)
 *   POST /v3/knowledge/get    — get single knowledge by id
 *   POST /v3/knowledge/update — partial update (name, summary, etc.)
 *   POST /v3/knowledge/delete — batch delete
 *   POST /v3/knowledge/list   — list by team_id (with optional type filter)
 *
 * No binding endpoints — binding is TODO (see design doc 04-kernel-knowledge-api.md §5).
 *
 * Registered as extraRouteTable (same as /v3/skill/*), bypasses V3
 * strict-isolation triad check since knowledge is management-plane.
 */

import { ZodError } from "zod";

import { errorEnvelope, successEnvelope } from "./v2-router.js";
import {
  knowledgeCreateRequestSchema,
  knowledgeGetRequestSchema,
  knowledgeUpdateRequestSchema,
  knowledgeBatchDeleteRequestSchema,
  knowledgeListRequestSchema,
} from "./knowledge-schemas.js";
import type { ApiResponseEnvelope, V2AuthContext } from "./v2-schemas.js";
import type { KnowledgeEntity, KnowledgeListResult, BatchDeleteResult } from "../core/store/types.js";

function formatZodErr(err: ZodError): string {
  return err.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; ");
}

// ── EntityStore access (same pattern as v2-router.ts) ──

type EntityStore = {
  createKnowledge?(input: Omit<KnowledgeEntity, "created_at" | "updated_at">): KnowledgeEntity | Promise<KnowledgeEntity>;
  getKnowledge?(knowledgeId: string): KnowledgeEntity | null | Promise<KnowledgeEntity | null>;
  updateKnowledge?(knowledgeId: string, patch: Partial<Pick<KnowledgeEntity, "name" | "summary" | "service_url" | "repo_url" | "branch">>): KnowledgeEntity | null | Promise<KnowledgeEntity | null>;
  deleteKnowledge?(knowledgeIds: string[], teamId?: string): BatchDeleteResult | Promise<BatchDeleteResult>;
  listKnowledge?(input: { team_id: string; type?: "wiki" | "code-graph"; knowledge_ids?: string[]; limit?: number; offset?: number }): KnowledgeListResult | Promise<KnowledgeListResult>;
};

function getEntityStore(deps: unknown): EntityStore | undefined {
  const d = deps as { getStore?: () => unknown };
  return d?.getStore?.() as EntityStore | undefined;
}

function missingEntityStore(requestId: string): ApiResponseEnvelope {
  return errorEnvelope(503, "Entity metadata store not available", requestId);
}

// ── Handlers ──

async function handleKnowledgeCreate(
  body: unknown, _auth: V2AuthContext, requestId: string, deps: unknown,
): Promise<ApiResponseEnvelope> {
  const parsed = knowledgeCreateRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodErr(parsed.error), requestId);
  const store = getEntityStore(deps);
  if (!store?.createKnowledge) return missingEntityStore(requestId);
  return successEnvelope<KnowledgeEntity>(await store.createKnowledge(parsed.data), requestId);
}

async function handleKnowledgeGet(
  body: unknown, _auth: V2AuthContext, requestId: string, deps: unknown,
): Promise<ApiResponseEnvelope> {
  const parsed = knowledgeGetRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodErr(parsed.error), requestId);
  const store = getEntityStore(deps);
  if (!store?.getKnowledge) return missingEntityStore(requestId);
  const data = await store.getKnowledge(parsed.data.knowledge_id);
  if (!data) return errorEnvelope(404, "Knowledge not found", requestId);
  if (parsed.data.team_id && data.team_id !== parsed.data.team_id) {
    return errorEnvelope(403, "Knowledge team_id mismatch", requestId);
  }
  return successEnvelope<KnowledgeEntity>(data, requestId);
}

async function handleKnowledgeUpdate(
  body: unknown, _auth: V2AuthContext, requestId: string, deps: unknown,
): Promise<ApiResponseEnvelope> {
  const parsed = knowledgeUpdateRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodErr(parsed.error), requestId);
  const { knowledge_id, team_id, ...patch } = parsed.data;
  const store = getEntityStore(deps);
  if (!store?.updateKnowledge) return missingEntityStore(requestId);

  // team_id ownership check
  if (team_id) {
    const current = store.getKnowledge ? await store.getKnowledge(knowledge_id) : null;
    if (!current) return errorEnvelope(404, "Knowledge not found", requestId);
    if (current.team_id !== team_id) return errorEnvelope(403, "Knowledge team_id mismatch", requestId);
  }

  const data = await store.updateKnowledge(knowledge_id, patch);
  return data
    ? successEnvelope<KnowledgeEntity>(data, requestId)
    : errorEnvelope(404, "Knowledge not found", requestId);
}

async function handleKnowledgeDelete(
  body: unknown, _auth: V2AuthContext, requestId: string, deps: unknown,
): Promise<ApiResponseEnvelope> {
  const parsed = knowledgeBatchDeleteRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodErr(parsed.error), requestId);
  const store = getEntityStore(deps);
  if (!store?.deleteKnowledge) return missingEntityStore(requestId);
  return successEnvelope<BatchDeleteResult>(
    await store.deleteKnowledge(parsed.data.knowledge_ids, parsed.data.team_id),
    requestId,
  );
}

async function handleKnowledgeList(
  body: unknown, _auth: V2AuthContext, requestId: string, deps: unknown,
): Promise<ApiResponseEnvelope> {
  const parsed = knowledgeListRequestSchema.safeParse(body);
  if (!parsed.success) return errorEnvelope(400, formatZodErr(parsed.error), requestId);
  const store = getEntityStore(deps);
  if (!store?.listKnowledge) return missingEntityStore(requestId);
  const result = await store.listKnowledge({
    team_id: parsed.data.team_id,
    type: parsed.data.type,
    knowledge_ids: parsed.data.knowledge_ids,
    limit: parsed.data.pagination?.limit,
    offset: parsed.data.pagination?.offset,
  });
  return successEnvelope<KnowledgeListResult>(result, requestId);
}

// ── Route table factory (same pattern as makeSkillRouteTable) ──

type RouteHandler = (
  body: unknown,
  auth: V2AuthContext,
  requestId: string,
  deps: unknown,
) => Promise<ApiResponseEnvelope>;

export function makeKnowledgeRouteTable(): Record<string, RouteHandler> {
  // Mirrors makeSkillRouteTable: /v3/* only, no /v2 dual-registration.
  // Knowledge entity CRUD is team-scoped management-plane; callers are
  // TMC (kernel-sync) + proxy prewarm, all migrated to /v3 together.
  return {
    "/v3/knowledge/create": handleKnowledgeCreate,
    "/v3/knowledge/get": handleKnowledgeGet,
    "/v3/knowledge/update": handleKnowledgeUpdate,
    "/v3/knowledge/delete": handleKnowledgeDelete,
    "/v3/knowledge/list": handleKnowledgeList,
  };
}
