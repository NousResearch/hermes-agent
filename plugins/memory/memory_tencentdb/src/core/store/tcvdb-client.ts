/**
 * Tencent Cloud VectorDB HTTP Client.
 *
 * Thin wrapper around the VectorDB HTTP API. Handles authentication, timeouts,
 * retries (5xx / timeout), and error normalization.
 *
 * API docs: https://cloud.tencent.com/document/product/1709
 */

import fs from "node:fs";
import { request as undiciRequest, Agent as UndiciAgent } from "undici";
import type { Dispatcher } from "undici";
import type { StoreLogger } from "./types.js";

// ============================
// Types
// ============================

export interface TcvdbClientConfig {
  /** Instance URL (e.g. "http://10.0.1.1:80") */
  url: string;
  /** Account name (default: "root") */
  username: string;
  /** API Key */
  apiKey: string;
  /** Database name */
  database: string;
  /** Request timeout in ms (default: 10000) */
  timeout: number;
  /** Path to CA certificate PEM file (for HTTPS connections) */
  caPemPath?: string;
}

/** Standard VectorDB API response envelope. */
interface ApiResponse {
  code: number;
  msg: string;
  [key: string]: unknown;
}

/** Search/hybridSearch response shape. */
export interface SearchResponse {
  documents: Array<Array<Record<string, unknown>>>;
}

/** Query response shape. */
export interface QueryResponse {
  documents: Array<Record<string, unknown>>;
  count?: number;
}

/** Collection info from describeCollection. */
export interface CollectionInfo {
  collection: string;
  database: string;
  documentCount?: number;
  embedding?: {
    field: string;
    vectorField: string;
    model: string;
  };
  indexes?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

export class TcvdbApiError extends Error {
  readonly apiCode: number;
  constructor(path: string, code: number, msg: string) {
    super(`VectorDB ${path}: code=${code}, msg=${msg}`);
    this.name = "TcvdbApiError";
    this.apiCode = code;
  }
}

// ============================
// Client
// ============================

const TAG = "[memory-tdai][tcvdb-client]";
const MAX_RETRIES = 2;

export class TcvdbClient {
  private readonly baseUrl: string;
  private readonly authHeader: string;
  private readonly database: string;
  private readonly timeout: number;
  private readonly logger?: StoreLogger;
  /** undici dispatcher for HTTPS + custom CA. */
  private readonly dispatcher?: Dispatcher;

  constructor(config: TcvdbClientConfig, logger?: StoreLogger) {
    // 防御性 normalize：Shark 某些实例的 VdbUrl 可能只返回 `host:port`（缺 scheme），
    // 此时 undici.request() 会抛 Invalid URL。这里补齐 http://（默认 VDB 内网走 HTTP）。
    let url = config.url.replace(/\/+$/, "");
    if (!/^https?:\/\//i.test(url)) {
      url = `http://${url}`;
      this.logger?.debug?.(`${TAG} normalized url: "${config.url}" → "${url}" (no scheme, prepended http://)`);
    }
    this.baseUrl = url;
    this.authHeader = `Bearer account=${config.username}&api_key=${config.apiKey}`;
    this.database = config.database;
    this.timeout = config.timeout;
    this.logger = logger;

    // Log connection info at construction time.
    this.logger?.debug?.(`${TAG} url=${this.baseUrl} db=${this.database} timeout=${this.timeout}${this.baseUrl.startsWith("https://") ? ` https=true caPemPath=${config.caPemPath ?? "(none)"}` : ""}`);

    // For HTTPS with a custom CA certificate, create a dedicated undici Agent.
    // We use undici.request() instead of global fetch because fetch's
    // `dispatcher` option is unreliable across Node versions.
    if (this.baseUrl.startsWith("https://") && config.caPemPath) {
      try {
        const ca = fs.readFileSync(config.caPemPath, "utf-8");
        this.dispatcher = new UndiciAgent({ connect: { ca } });
        this.logger?.debug?.(`${TAG} HTTPS enabled with CA from ${config.caPemPath}`);
      } catch (err) {
        this.logger?.error(`${TAG} Failed to load CA PEM from ${config.caPemPath}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }

  // ── Generic request ─────────────────────────────────────

  /**
   * Send a POST request to VectorDB API.
   * Handles auth, timeout, retries (5xx/timeout), and error unwrapping.
   */
  async request<T = ApiResponse>(path: string, body: Record<string, unknown>): Promise<T> {
    let lastError: Error | undefined;
    const t0 = performance.now();

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      const tAttempt = performance.now();
      try {
        this.logger?.debug?.(`${TAG} → ${path} attempt=${attempt} body=${JSON.stringify(body).slice(0, 500)}`);
        const { statusCode, body: respBody } = await undiciRequest(`${this.baseUrl}${path}`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": this.authHeader,
          },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(this.timeout),
          ...(this.dispatcher ? { dispatcher: this.dispatcher } : {}),
        });

        const text = await respBody.text();
        const json = JSON.parse(text) as ApiResponse;
        const attemptMs = Math.round(performance.now() - tAttempt);
        this.logger?.debug?.(`${TAG} ← ${path} status=${statusCode} code=${json.code} attemptMs=${attemptMs} attempt=${attempt}`);

        if (json.code !== 0) {
          const err = new TcvdbApiError(path, json.code, json.msg);
          if (statusCode !== undefined && statusCode >= 400 && statusCode < 500) throw err;
          lastError = err;
          continue;
        }

        // Always log completion at info level (one line per request)
        const totalMs = Math.round(performance.now() - t0);
        this.logger?.info(`${TAG} ${path} ${totalMs}ms${attempt > 0 ? ` (${attempt + 1} attempts)` : ""}`);

        return json as unknown as T;
      } catch (err) {
        const attemptMs = Math.round(performance.now() - tAttempt);
        if (err instanceof TcvdbApiError && err.apiCode !== 0) throw err;
        lastError = err instanceof Error ? err : new Error(String(err));
        if (attempt < MAX_RETRIES) {
          const delay = 500 * (attempt + 1);
          this.logger?.debug?.(`${TAG} ${path} retry ${attempt + 1}/${MAX_RETRIES} in ${delay}ms (lastAttemptMs=${attemptMs}, error=${lastError.message})`);
          await new Promise((r) => setTimeout(r, delay));
        }
      }
    }

    const totalMs = Math.round(performance.now() - t0);
    this.logger?.debug?.(`${TAG} ✗ ${path} totalMs=${totalMs} attempts=${MAX_RETRIES + 1} error=${lastError?.message}`);
    throw lastError ?? new Error(`${TAG} ${path} failed after retries`);
  }

  // ── Database operations ─────────────────────────────────

  async createDatabase(dbName?: string): Promise<boolean> {
    const name = dbName ?? this.database;
    // SDK pattern: list first, create only if not found
    const listResp = await this.request<{ databases: string[] }>("/database/list", {});
    const exists = (listResp.databases ?? []).includes(name);
    if (exists) {
      this.logger?.debug?.(`${TAG} Database already exists: ${name}`);
      return false;
    }
    await this.request("/database/create", { database: name });
    this.logger?.info(`${TAG} Database created: ${name}`);
    return true;
  }

  // ── Collection operations ───────────────────────────────

  async createCollection(params: Record<string, unknown>): Promise<void> {
    const name = String(params.collection ?? "");
    // SDK pattern: try describe first, create only if not found (code 15302)
    try {
      await this.describeCollection(name);
      this.logger?.debug?.(`${TAG} Collection already exists: ${name}`);
      return;
    } catch (err) {
      if (!(err instanceof TcvdbApiError && err.apiCode === 15302)) {
        throw err; // unexpected error
      }
      // 15302 = collection not found → proceed to create
    }
    try {
      await this.request("/collection/create", {
        database: this.database,
        ...params,
      });
      this.logger?.info(`${TAG} Collection created: ${name}`);
    } catch (err) {
      // 15202 = collection already exists — race between describe and create.
      // Semantically identical to "describe found it", so treat as success.
      if (err instanceof TcvdbApiError && err.apiCode === 15202) {
        this.logger?.debug?.(`${TAG} Collection already exists (race): ${name}`);
        return;
      }
      throw err;
    }
  }

  async describeCollection(collection: string): Promise<CollectionInfo> {
    const resp = await this.request<{ collection: CollectionInfo }>("/collection/describe", {
      database: this.database,
      collection,
    });
    return resp.collection;
  }

  // ── Document operations ─────────────────────────────────

  async upsert(collection: string, documents: Record<string, unknown>[]): Promise<void> {
    await this.request("/document/upsert", {
      database: this.database,
      collection,
      buildIndex: true,
      documents,
    });
  }

  async search(collection: string, searchParams: Record<string, unknown>): Promise<SearchResponse> {
    return this.request<SearchResponse>("/document/search", {
      database: this.database,
      collection,
      readConsistency: "strongConsistency",
      search: searchParams,
    });
  }

  async hybridSearch(collection: string, searchParams: Record<string, unknown>): Promise<SearchResponse> {
    return this.request<SearchResponse>("/document/hybridSearch", {
      database: this.database,
      collection,
      readConsistency: "strongConsistency",
      search: searchParams,
    });
  }

  async query(collection: string, queryParams: Record<string, unknown>): Promise<QueryResponse> {
    return this.request<QueryResponse>("/document/query", {
      database: this.database,
      collection,
      readConsistency: "strongConsistency",
      query: queryParams,
    });
  }

  async deleteDoc(collection: string, params: Record<string, unknown>): Promise<number> {
    const resp = await this.request<{ affectedCount: number }>("/document/delete", {
      database: this.database,
      collection,
      ...params,
    });
    return resp.affectedCount ?? 0;
  }

  /**
   * Count documents matching an optional filter.
   * Uses the dedicated /document/count endpoint.
   */
  async count(collection: string, filter?: string): Promise<number> {
    const query: Record<string, unknown> = {};
    if (filter) query.filter = filter;
    const resp = await this.request<{ count: number }>("/document/count", {
      database: this.database,
      collection,
      readConsistency: "strongConsistency",
      query,
    });
    return resp.count ?? 0;
  }

  // ── Convenience getters ─────────────────────────────────

  getDatabase(): string {
    return this.database;
  }
}
