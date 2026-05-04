export interface Env {
  MODEL_KV: KVNamespace;
  DB: D1Database;

  HF_TOKEN?: string;
  HF_WEBHOOK_SECRET?: string;
  ADMIN_SECRET?: string;
  PRIVATE_JWK: string;

  WATCHED_AUTHORS: string;
  INCLUDE_PATTERNS: string;
  EXCLUDE_PATTERNS: string;
  UNSLOTH_SEARCH_TERMS: string;
  POLL_LIMIT: string;
  INDEX_TTL_SECONDS: string;
  SIGNING_KEY_ID: string;
  MAX_MODEL_BYTES: string;
}

type HfModel = {
  id?: string;
  modelId?: string;
  author?: string;
  sha?: string;
  lastModified?: string;
  last_modified?: string;
  tags?: string[];
  gated?: string | boolean | null;
  private?: boolean;
  siblings?: HfSibling[];
  [key: string]: unknown;
};

type HfSibling = {
  rfilename?: string;
  path?: string;
  size?: number;
  lfs?: { size?: number };
};

type ModelRow = {
  repo_id: string;
  author: string;
  sha: string | null;
  last_modified: string | null;
  gated: string | null;
  private: number;
  source: string | null;
  tags_json: string | null;
  raw_json: string | null;
  updated_at: string;
};

const LATEST_INDEX_KEY = "model-index:latest";

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    try {
      if (request.method === "OPTIONS") {
        return corsResponse("", 204);
      }

      if (url.pathname === "/health" && request.method === "GET") {
        return jsonResponse({ ok: true, service: "hf-model-index-worker" });
      }

      if (url.pathname === "/models.json" && request.method === "GET") {
        return await serveModelsJson(env);
      }

      if (url.pathname === "/hf-webhook" && request.method === "POST") {
        return await handleHuggingFaceWebhook(request, env, ctx);
      }

      if (url.pathname === "/admin/scan" && (request.method === "POST" || request.method === "GET")) {
        requireAdmin(request, env);
        const result = await scanAndRebuild(env, request.method === "GET" ? "http-get-fallback" : "manual");
        return jsonResponse(result);
      }

      if (url.pathname === "/admin/add" && request.method === "POST") {
        requireAdmin(request, env);
        const body = await request.json<{ repo: string }>();
        if (!body.repo || !body.repo.includes("/")) {
          return jsonResponse({ error: "Expected JSON body like {\"repo\":\"unsloth/example-GGUF\"}" }, 400);
        }
        const changed = await validateAndUpsertRepo(body.repo, env, "manual-add");
        if (changed) await rebuildSignedIndex(env);
        return jsonResponse({ ok: true, repo: body.repo, changed });
      }

      if (url.pathname === "/admin/status" && request.method === "GET") {
        requireAdmin(request, env);
        return jsonResponse(await getStatus(env));
      }

      return jsonResponse({ error: "not_found" }, 404);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await logEvent(env, "error", null, "fetch", message);
      return jsonResponse({ error: message }, 500);
    }
  },

  async scheduled(event: ScheduledEvent, env: Env, ctx: ExecutionContext): Promise<void> {
    ctx.waitUntil(scanAndRebuild(env, "cloudflare-cron"));
  }
};

async function serveModelsJson(env: Env): Promise<Response> {
  let signed = await env.MODEL_KV.get(LATEST_INDEX_KEY);
  if (!signed) {
    signed = JSON.stringify(await rebuildSignedIndex(env));
  }
  return new Response(signed, {
    status: 200,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "public, max-age=300",
      "access-control-allow-origin": "*"
    }
  });
}

async function handleHuggingFaceWebhook(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
  const expectedSecret = env.HF_WEBHOOK_SECRET || "";
  if (!expectedSecret) {
    return jsonResponse({ error: "webhook_secret_not_configured" }, 503);
  }
  const receivedSecret = request.headers.get("X-Webhook-Secret") ?? "";
  if (!constantTimeEqual(receivedSecret, expectedSecret)) {
    return jsonResponse({ error: "forbidden" }, 403);
  }

  const payload = await request.json<any>();
  const repoType = payload?.repo?.type;
  const repoName = payload?.repo?.name;
  if (repoType !== "model" || typeof repoName !== "string") {
    return jsonResponse({ ok: true, ignored: true, reason: "not_model_repo" });
  }

  await env.MODEL_KV.put(
    pendingKey(repoName),
    JSON.stringify({ repo_id: repoName, source: "webhook", received_at: new Date().toISOString(), payload }),
    { expirationTtl: 60 * 60 * 24 * 3 }
  );

  ctx.waitUntil(
    validateAndUpsertRepo(repoName, env, "webhook")
      .then(async (changed) => {
        if (changed) await rebuildSignedIndex(env);
        await env.MODEL_KV.delete(pendingKey(repoName));
      })
      .catch(async (err) => {
        await logEvent(env, "webhook_validation_failed", repoName, "webhook", String(err));
      })
  );

  return jsonResponse({ ok: true, queued: repoName });
}

async function scanAndRebuild(env: Env, source: string) {
  const startedAt = new Date().toISOString();
  const authors = parseStringArray(env.WATCHED_AUTHORS);
  const pollLimit = Number(env.POLL_LIMIT || "24");

  let seen = 0;
  let changed = 0;
  let errors = 0;

  const pending = await env.MODEL_KV.list({ prefix: "pending:", limit: 100 });
  for (const key of pending.keys) {
    const repoId = key.name.slice("pending:".length);
    try {
      const didChange = await validateAndUpsertRepo(repoId, env, "pending-retry");
      if (didChange) changed++;
      await env.MODEL_KV.delete(key.name);
    } catch (err) {
      errors++;
      await logEvent(env, "pending_retry_failed", repoId, source, String(err));
    }
  }

  for (const author of authors) {
    try {
      const models = await listRecentModels(author, pollLimit, env);
      for (const model of models) {
        const repoId = getRepoId(model);
        if (!repoId) continue;
        seen++;
        if (!isAllowedRepo(repoId, model, env)) continue;
        const didChange = await upsertModelFromInfo(model, author, env, source);
        if (didChange) changed++;
      }
    } catch (err) {
      errors++;
      await logEvent(env, "poll_failed", null, source, `${author}: ${String(err)}`);
    }
  }

  for (const term of parseStringArray(env.UNSLOTH_SEARCH_TERMS)) {
    try {
      const models = await searchUnslothGguf(term, pollLimit, env);
      for (const model of models) {
        const repoId = getRepoId(model);
        if (!repoId) continue;
        seen++;
        if (!isAllowedRepo(repoId, model, env)) continue;
        const didChange = await upsertModelFromInfo(model, "unsloth", env, "unsloth-search");
        if (didChange) changed++;
      }
    } catch (err) {
      errors++;
      await logEvent(env, "unsloth_search_failed", null, source, `${term}: ${String(err)}`);
    }
  }

  const signed = await rebuildSignedIndex(env);

  await setState(env, "last_scan_started_at", startedAt);
  await setState(env, "last_scan_finished_at", new Date().toISOString());
  await setState(env, "last_scan_source", source);
  await setState(env, "last_scan_summary", JSON.stringify({ seen, changed, errors }));

  return {
    ok: true,
    source,
    seen,
    changed,
    errors,
    index_version: signed.payload.version,
    model_count: signed.payload.models.length
  };
}

async function listRecentModels(author: string, limit: number, env: Env): Promise<HfModel[]> {
  const url = new URL("https://huggingface.co/api/models");
  url.searchParams.set("author", author);
  url.searchParams.set("sort", "lastModified");
  url.searchParams.set("direction", "-1");
  url.searchParams.set("limit", String(limit));
  url.searchParams.set("full", "true");
  return await hfGet<HfModel[]>(url.toString(), env);
}

async function searchUnslothGguf(term: string, limit: number, env: Env): Promise<HfModel[]> {
  const url = new URL("https://huggingface.co/api/models");
  url.searchParams.set("author", "unsloth");
  url.searchParams.set("search", `${term} GGUF`);
  url.searchParams.set("sort", "lastModified");
  url.searchParams.set("direction", "-1");
  url.searchParams.set("limit", String(Math.min(limit, 20)));
  url.searchParams.set("full", "true");
  return await hfGet<HfModel[]>(url.toString(), env);
}

async function getModelInfo(repoId: string, env: Env): Promise<HfModel> {
  return await hfGet<HfModel>(`https://huggingface.co/api/models/${encodeURIComponent(repoId)}?full=true`, env);
}

async function hfGet<T>(url: string, env: Env): Promise<T> {
  const headers: Record<string, string> = {
    accept: "application/json",
    "user-agent": "hermes-agent-model-index/0.1"
  };
  if (env.HF_TOKEN) {
    headers.authorization = `Bearer ${env.HF_TOKEN}`;
  }
  const res = await fetch(url, { headers });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Hugging Face API error ${res.status}: ${text.slice(0, 300)}`);
  }
  return await res.json<T>();
}

async function validateAndUpsertRepo(repoId: string, env: Env, source: string): Promise<boolean> {
  const info = await getModelInfo(repoId, env);
  const resolvedRepoId = getRepoId(info) ?? repoId;
  const author = resolvedRepoId.split("/")[0];
  if (!isAllowedRepo(resolvedRepoId, info, env)) {
    await logEvent(env, "repo_excluded", resolvedRepoId, source, "Repo did not match filters");
    return false;
  }
  return await upsertModelFromInfo(info, author, env, source);
}

async function upsertModelFromInfo(model: HfModel, fallbackAuthor: string, env: Env, source: string): Promise<boolean> {
  const repoId = getRepoId(model);
  if (!repoId) return false;

  const existing = await env.DB.prepare(
    "SELECT sha, last_modified, status FROM models WHERE repo_id = ?"
  ).bind(repoId).first<{ sha: string | null; last_modified: string | null; status: string }>();

  const sha = typeof model.sha === "string" ? model.sha : null;
  const lastModified =
    typeof model.lastModified === "string"
      ? model.lastModified
      : typeof model.last_modified === "string"
        ? model.last_modified
        : null;
  const changed = !existing || existing.sha !== sha || existing.last_modified !== lastModified || existing.status !== "active";
  if (!changed) return false;

  const now = new Date().toISOString();
  const author = repoId.split("/")[0] || fallbackAuthor;
  const gated = model.gated == null ? null : String(model.gated);
  const isPrivate = model.private === true ? 1 : 0;
  const tagsJson = JSON.stringify(model.tags ?? []);
  const rawJson = JSON.stringify(compactRawModel(model));

  await env.DB.prepare(
    `
    INSERT INTO models (
      repo_id, author, sha, last_modified, gated, private, status, source,
      tags_json, raw_json, discovered_at, updated_at
    )
    VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)
    ON CONFLICT(repo_id) DO UPDATE SET
      author = excluded.author,
      sha = excluded.sha,
      last_modified = excluded.last_modified,
      gated = excluded.gated,
      private = excluded.private,
      status = 'active',
      source = excluded.source,
      tags_json = excluded.tags_json,
      raw_json = excluded.raw_json,
      updated_at = excluded.updated_at
    `
  )
    .bind(repoId, author, sha, lastModified, gated, isPrivate, source, tagsJson, rawJson, now, now)
    .run();

  await logEvent(env, "model_upserted", repoId, source, sha ?? lastModified ?? "updated");
  return true;
}

async function rebuildSignedIndex(env: Env) {
  const rows = await env.DB.prepare(
    `
    SELECT repo_id, author, sha, last_modified, gated, private, source, tags_json, raw_json, updated_at
    FROM models
    WHERE status = 'active'
    ORDER BY author ASC, repo_id ASC
    `
  ).all<ModelRow>();

  const nowMs = Date.now();
  const ttlSeconds = Number(env.INDEX_TTL_SECONDS || "21600");
  const maxBytes = Number(env.MAX_MODEL_BYTES || "5368709120");
  const models = rows.results
    .map((row) => catalogEntry(row, maxBytes))
    .filter((entry) => entry !== null);

  const payload = {
    schema: "hf-model-index/v1",
    version: nowMs,
    issued_at: new Date(nowMs).toISOString(),
    expires_at: new Date(nowMs + ttlSeconds * 1000).toISOString(),
    revision_policy: "mutable-main-with-current-sha",
    source: "cloudflare-d1-kv",
    models
  };

  const payloadCanonical = stableStringify(payload);
  const signature = await signPayloadCanonical(payloadCanonical, env);
  const signed = {
    payload,
    payload_canonical: payloadCanonical,
    signature: {
      alg: "ES256",
      key_id: env.SIGNING_KEY_ID || "model-index-key-v1",
      value: signature
    }
  };

  await env.MODEL_KV.put(LATEST_INDEX_KEY, JSON.stringify(signed));
  await setState(env, "last_index_rebuilt_at", payload.issued_at);
  await setState(env, "last_index_model_count", String(payload.models.length));
  return signed;
}

function catalogEntry(row: ModelRow, maxBytes: number) {
  const tags = parseStringArray(row.tags_json || "[]");
  const raw = safeJson(row.raw_json);
  const repoId = row.repo_id;
  const bestFile = selectBestFile(raw?.siblings, tags, maxBytes);
  if (!bestFile) return null;
  const runtimeFlavor = inferRuntimeFlavor(repoId, tags, bestFile?.path);
  const title = titleForRepo(repoId, bestFile?.path);
  return {
    id: sanitizeId(`${repoId}:${bestFile?.path || runtimeFlavor}`),
    title,
    repo: repoId,
    author: row.author,
    revision: "main",
    current_sha: row.sha,
    last_modified: row.last_modified,
    gated: row.gated,
    private: Boolean(row.private),
    updated_at: row.updated_at,
    source: row.source,
    runtime_flavor: runtimeFlavor,
    tags,
    summary: summaryForRepo(repoId, runtimeFlavor, bestFile?.path),
    download: {
      repo_or_url: repoId,
      file_path: bestFile?.path || "",
      revision: "main",
      runtime_flavor: runtimeFlavor,
      max_model_bytes: maxBytes
    }
  };
}

function selectBestFile(siblings: unknown, tags: string[], maxBytes: number): { path: string; size: number | null } | null {
  if (!Array.isArray(siblings)) return null;
  const candidates = siblings
    .map((item) => {
      const sibling = item as HfSibling;
      const path = sibling.rfilename || sibling.path || "";
      const size = typeof sibling.size === "number" ? sibling.size : typeof sibling.lfs?.size === "number" ? sibling.lfs.size : null;
      return { path, size };
    })
    .filter((item) => item.path && isDownloadableModelFile(item.path))
    .filter((item) => !isShardedModelFile(item.path))
    .filter((item) => item.size == null || item.size <= maxBytes)
    .sort((a, b) => fileRank(a.path, tags) - fileRank(b.path, tags) || a.path.localeCompare(b.path));
  return candidates[0] ?? null;
}

function isDownloadableModelFile(path: string): boolean {
  const lower = path.toLowerCase();
  return lower.endsWith(".gguf") || lower.endsWith(".litertlm") || (lower.endsWith(".task") && !lower.includes("-web."));
}

function isShardedModelFile(path: string): boolean {
  const lower = path.toLowerCase();
  return /-\d{5}-of-\d{5}/.test(lower) || /\.part\d+of\d+/.test(lower) || /\.part\d+\./.test(lower);
}

function fileRank(path: string, tags: string[]): number {
  const lower = path.toLowerCase();
  if (lower.endsWith(".litertlm")) return 0;
  if (lower.includes("q4_k_m")) return 1;
  if (lower.includes("q4") || lower.includes("iq4") || lower.includes("int4")) return 2;
  if (lower.includes("q5")) return 3;
  if (lower.includes("q6")) return 4;
  if (lower.includes("q8") || lower.includes("int8")) return 5;
  if (lower.endsWith(".gguf")) return tags.includes("gguf") ? 6 : 7;
  if (lower.endsWith(".task")) return 8;
  return 99;
}

function inferRuntimeFlavor(repoId: string, tags: string[], filePath?: string): string {
  const lower = `${repoId} ${tags.join(" ")} ${filePath || ""}`.toLowerCase();
  if (lower.includes("litert") || lower.endsWith(".litertlm") || lower.endsWith(".task")) {
    return "LiteRT-LM";
  }
  return "GGUF";
}

function summaryForRepo(repoId: string, runtimeFlavor: string, filePath?: string): string {
  if (repoId.toLowerCase().startsWith("unsloth/")) {
    return `Unsloth ${runtimeFlavor} quantization detected by the Hermes Cloudflare catalog.`;
  }
  if (filePath) {
    return `${runtimeFlavor} artifact detected: ${filePath.substring(filePath.lastIndexOf("/") + 1)}`;
  }
  return `${runtimeFlavor} repo detected by the Hermes Cloudflare catalog.`;
}

function titleForRepo(repoId: string, filePath?: string): string {
  if (filePath) {
    return filePath.substring(filePath.lastIndexOf("/") + 1).replace(/[-_]+/g, " ");
  }
  return repoId.split("/").pop()?.replace(/[-_]+/g, " ") || repoId;
}

function getRepoId(model: HfModel): string | null {
  if (typeof model.id === "string") return model.id;
  if (typeof model.modelId === "string") return model.modelId;
  return null;
}

function isAllowedRepo(repoId: string, model: HfModel, env: Env): boolean {
  if (model.private === true) return false;
  const includePatterns = parseRegexArray(env.INCLUDE_PATTERNS);
  const excludePatterns = parseRegexArray(env.EXCLUDE_PATTERNS);
  if (includePatterns.length > 0 && !includePatterns.some((regex) => regex.test(repoId))) return false;
  if (excludePatterns.some((regex) => regex.test(repoId))) return false;
  if (repoId.toLowerCase().startsWith("unsloth/")) {
    const tags = model.tags ?? [];
    return repoId.toLowerCase().includes("gguf") || tags.some((tag) => tag.toLowerCase() === "gguf");
  }
  return true;
}

function compactRawModel(model: HfModel) {
  return {
    id: model.id,
    modelId: model.modelId,
    author: model.author,
    sha: model.sha,
    lastModified: model.lastModified,
    last_modified: model.last_modified,
    tags: model.tags,
    gated: model.gated,
    private: model.private,
    siblings: model.siblings
  };
}

async function signPayloadCanonical(canonical: string, env: Env): Promise<string> {
  const privateJwk = JSON.parse(env.PRIVATE_JWK);
  const key = await crypto.subtle.importKey(
    "jwk",
    privateJwk,
    { name: "ECDSA", namedCurve: "P-256" },
    false,
    ["sign"]
  );
  const data = new TextEncoder().encode(canonical);
  const signature = await crypto.subtle.sign({ name: "ECDSA", hash: "SHA-256" }, key, data);
  return base64url(new Uint8Array(signature));
}

function stableStringify(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "number" || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map((item) => stableStringify(item)).join(",")}]`;
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    const body = Object.keys(record)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableStringify(record[key])}`)
      .join(",");
    return `{${body}}`;
  }
  return JSON.stringify(String(value));
}

function jsonResponse(body: unknown, status = 200): Response {
  return corsResponse(JSON.stringify(body), status, "application/json; charset=utf-8");
}

function corsResponse(body: string, status = 200, contentType = "text/plain; charset=utf-8"): Response {
  return new Response(body, {
    status,
    headers: {
      "content-type": contentType,
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "GET, POST, OPTIONS",
      "access-control-allow-headers": "authorization, content-type, x-admin-secret, x-webhook-secret"
    }
  });
}

function requireAdmin(request: Request, env: Env) {
  const expected = env.ADMIN_SECRET || "";
  if (!expected) throw new Error("ADMIN_SECRET is not configured");
  const url = new URL(request.url);
  const bearer = request.headers.get("authorization")?.replace(/^Bearer\s+/i, "") ?? "";
  const header = request.headers.get("x-admin-secret") ?? "";
  const query = url.searchParams.get("token") ?? "";
  if (![bearer, header, query].some((value) => constantTimeEqual(value, expected))) {
    throw new Error("forbidden");
  }
}

function constantTimeEqual(left: string, right: string): boolean {
  if (!left || !right || left.length !== right.length) return false;
  let diff = 0;
  for (let i = 0; i < left.length; i++) {
    diff |= left.charCodeAt(i) ^ right.charCodeAt(i);
  }
  return diff === 0;
}

function parseStringArray(value: string): string[] {
  try {
    const parsed = JSON.parse(value || "[]");
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : [];
  } catch {
    return [];
  }
}

function parseRegexArray(value: string): RegExp[] {
  return parseStringArray(value).map((pattern) => new RegExp(pattern, "i"));
}

function safeJson(value: string | null): any | null {
  if (!value) return null;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function sanitizeId(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
}

function pendingKey(repoId: string): string {
  return `pending:${repoId}`;
}

async function setState(env: Env, key: string, value: string) {
  await env.DB.prepare(
    `
    INSERT INTO worker_state (key, value, updated_at)
    VALUES (?, ?, ?)
    ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
    `
  ).bind(key, value, new Date().toISOString()).run();
}

async function getStatus(env: Env) {
  const rows = await env.DB.prepare("SELECT key, value, updated_at FROM worker_state ORDER BY key ASC").all<{
    key: string;
    value: string;
    updated_at: string;
  }>();
  return {
    ok: true,
    state: rows.results,
    cached_index: Boolean(await env.MODEL_KV.get(LATEST_INDEX_KEY))
  };
}

async function logEvent(env: Env, kind: string, repoId: string | null, source: string, message: string) {
  await env.DB.prepare(
    "INSERT INTO events (kind, repo_id, source, message, created_at) VALUES (?, ?, ?, ?, ?)"
  ).bind(kind, repoId, source, message.slice(0, 800), new Date().toISOString()).run();
}

function base64url(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}
