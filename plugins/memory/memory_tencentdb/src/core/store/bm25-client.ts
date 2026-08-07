/**
 * BM25 Sparse Vector Encoding Client.
 *
 * HTTP client for the BM25 Python sidecar service (bm25_server.py).
 * Used by TCVDB backend to generate sparse vectors for hybridSearch.
 *
 * Two operations:
 * - `encodeTexts(texts)` — encode documents for upsert (TF-based)
 * - `encodeQueries(texts)` — encode queries for search (IDF-based)
 *
 * Graceful degradation: if the sidecar is unreachable, all methods
 * return empty arrays and `isHealthy()` returns false. Callers can
 * check health to dynamically downgrade to pure semantic search.
 */

import type { Logger } from "../types.js";

// ============================
// Types
// ============================

/** Sparse vector: array of [token_hash, weight] pairs. */
export type SparseVector = Array<[number, number]>;

export interface BM25ClientConfig {
  /** Sidecar service URL (default: "http://127.0.0.1:8084") */
  serviceUrl: string;
  /** Request timeout in ms (default: 5000) */
  timeout: number;
}

interface EncodeResponse {
  vectors: SparseVector[];
}

// ============================
// Implementation
// ============================

const TAG = "[memory-tdai][bm25-client]";

export class BM25Client {
  private readonly baseUrl: string;
  private readonly timeout: number;
  private readonly logger?: Logger;

  /** Cached health status to avoid repeated checks on every call. */
  private _healthy: boolean | undefined;
  private _lastHealthCheck = 0;
  private static readonly HEALTH_CHECK_INTERVAL_MS = 30_000; // re-check every 30s

  constructor(config: BM25ClientConfig, logger?: Logger) {
    this.baseUrl = config.serviceUrl.replace(/\/+$/, "");
    this.timeout = config.timeout;
    this.logger = logger;
  }

  /**
   * Encode document texts for upsert (TF-based BM25 scoring).
   * Returns one SparseVector per input text.
   * Returns empty array on error (non-throwing).
   */
  async encodeTexts(texts: string[]): Promise<SparseVector[]> {
    if (texts.length === 0) return [];
    return this._encode("/encode_texts", texts);
  }

  /**
   * Encode query texts for search (IDF-based BM25 scoring).
   * Returns one SparseVector per input text.
   * Returns empty array on error (non-throwing).
   */
  async encodeQueries(texts: string[]): Promise<SparseVector[]> {
    if (texts.length === 0) return [];
    return this._encode("/encode_queries", texts);
  }

  /**
   * Check if the BM25 sidecar is reachable.
   * Result is cached for 30 seconds to avoid spamming health checks.
   */
  async isHealthy(): Promise<boolean> {
    const now = Date.now();
    if (
      this._healthy !== undefined &&
      now - this._lastHealthCheck < BM25Client.HEALTH_CHECK_INTERVAL_MS
    ) {
      return this._healthy;
    }

    try {
      const resp = await fetch(`${this.baseUrl}/health`, {
        signal: AbortSignal.timeout(3000),
      });
      this._healthy = resp.ok;
    } catch {
      this._healthy = false;
    }
    this._lastHealthCheck = now;

    if (!this._healthy) {
      this.logger?.warn(`${TAG} BM25 sidecar health check failed (${this.baseUrl})`);
    }

    return this._healthy;
  }

  // ── Internal ──────────────────────────────────────────────────

  private async _encode(path: string, texts: string[]): Promise<SparseVector[]> {
    try {
      const resp = await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts }),
        signal: AbortSignal.timeout(this.timeout),
      });

      if (!resp.ok) {
        const errBody = await resp.text().catch(() => "(unreadable)");
        this.logger?.warn(
          `${TAG} ${path} HTTP ${resp.status}: ${errBody.slice(0, 200)}`,
        );
        return [];
      }

      const json = (await resp.json()) as EncodeResponse;
      return json.vectors ?? [];
    } catch (err) {
      // Mark unhealthy on connection errors
      this._healthy = false;
      this._lastHealthCheck = Date.now();

      this.logger?.warn(
        `${TAG} ${path} failed: ${err instanceof Error ? err.message : String(err)}`,
      );
      return [];
    }
  }
}

// ============================
// Factory
// ============================

/**
 * Create a BM25Client if BM25 is enabled in config.
 * Returns undefined if disabled — callers should check before using.
 */
export function createBM25Client(
  config: { enabled: boolean; serviceUrl: string; timeout: number },
  logger?: Logger,
): BM25Client | undefined {
  if (!config.enabled) {
    logger?.info(`${TAG} BM25 sparse encoding disabled`);
    return undefined;
  }
  logger?.info(`${TAG} BM25 client → ${config.serviceUrl}`);
  return new BM25Client(
    { serviceUrl: config.serviceUrl, timeout: config.timeout },
    logger,
  );
}
