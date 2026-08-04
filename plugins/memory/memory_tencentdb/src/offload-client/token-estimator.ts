/**
 * offload-client — Token estimation using tiktoken (precise).
 * Aligned with server-side preciseMessageTokens: only counts LLM-visible content.
 *
 * Strategy:
 *   - Primary: tiktoken BPE encoding (o200k_base) on role + content
 *   - Fallback: CJK-aware heuristic if tiktoken fails
 *
 * `estimateAllTokens` still supports optional calibration from framework-reported
 * totalTokens, but with tiktoken the drift should be minimal.
 */
import { getEncoding, type Tiktoken } from "js-tiktoken";

// ─── Tiktoken Encoder (lazy singleton) ──────────────────────────────────────

let _encoder: Tiktoken | null = null;
function getEncoder(): Tiktoken {
  if (!_encoder) _encoder = getEncoding("o200k_base");
  return _encoder;
}

// ─── Calibration constants ──────────────────────────────────────────────────

/** If heuristic drifts > 15% from known total, apply linear scaling. */
const CALIBRATION_THRESHOLD = 0.15;
/** Clamp calibration factor to prevent extreme scaling from noisy estimates. */
const CALIBRATION_FACTOR_MIN = 0.5;
const CALIBRATION_FACTOR_MAX = 3.0;

// ─── LLM-visible text extraction (must match server-side extractLlmVisibleText) ──

/**
 * Extract the LLM-visible portion of a message (role + content only).
 * Matches server-side preciseMessageTokens logic exactly.
 */
function extractLlmVisibleText(msg: any): string {
  const role: string = msg?.role ?? msg?.message?.role ?? "";
  const rawContent = msg?.content ?? msg?.message?.content ?? "";

  let contentStr: string;
  if (typeof rawContent === "string") {
    contentStr = rawContent;
  } else if (Array.isArray(rawContent)) {
    const parts: string[] = [];
    for (const block of rawContent) {
      if (typeof block === "string") {
        parts.push(block);
      } else if (block?.type === "text" && typeof block.text === "string") {
        parts.push(block.text);
      } else if (block?.type === "tool_use" || block?.type === "toolCall") {
        parts.push(block.name ?? block.toolName ?? "");
        if (block.arguments) {
          parts.push(typeof block.arguments === "string" ? block.arguments : JSON.stringify(block.arguments));
        }
        if (block.input) {
          parts.push(typeof block.input === "string" ? block.input : JSON.stringify(block.input));
        }
      } else if (block?.type === "tool_result") {
        if (typeof block.content === "string") parts.push(block.content);
        else if (block.content) parts.push(JSON.stringify(block.content));
      } else {
        parts.push(JSON.stringify(block));
      }
    }
    contentStr = parts.join("\n");
  } else {
    contentStr = JSON.stringify(rawContent);
  }

  return `${role}\n${contentStr}`;
}

// ─── Per-message token cache (WeakMap — auto GC when msg object is released) ──

const _tokenCache = new WeakMap<object, number>();

/** Max messages to tiktoken precisely per call. Beyond this, old messages use heuristic. */
const PRECISE_BUDGET = 200;

// ─── Core estimation ────────────────────────────────────────────────────────

/**
 * Estimate tokens for a text string using tiktoken.
 * Falls back to CJK-aware heuristic on error.
 */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  try {
    return getEncoder().encode(text).length;
  } catch {
    return heuristicTokens(text);
  }
}

/**
 * Estimate tokens for a single message using tiktoken (with WeakMap cache).
 * Counts only LLM-visible content (role + content), aligned with server.
 */
export function estimateMessageTokens(msg: any): number {
  if (msg == null) return 0;
  // Cache hit
  if (typeof msg === "object" && _tokenCache.has(msg)) {
    return _tokenCache.get(msg)!;
  }
  const tokens = _computeMessageTokens(msg);
  // Cache store (only for objects)
  if (typeof msg === "object" && msg !== null) {
    _tokenCache.set(msg, tokens);
  }
  return tokens;
}

function _computeMessageTokens(msg: any): number {
  try {
    const text = extractLlmVisibleText(msg);
    return getEncoder().encode(text).length + 4; // +4 for message framing overhead
  } catch {
    return heuristicTokens(extractLlmVisibleText(msg));
  }
}

/**
 * Fast heuristic estimate for a message (no tiktoken, ~0.01ms per msg).
 * Used for old messages when total count exceeds PRECISE_BUDGET.
 */
function heuristicMessageTokens(msg: any): number {
  if (msg == null) return 0;
  return heuristicTokens(extractLlmVisibleText(msg));
}

// ─── Calibrated batch estimation ────────────────────────────────────────────

/**
 * Estimate total tokens and per-message tokens for a message array.
 *
 * Performance strategy:
 *   - If messages.length <= PRECISE_BUDGET (200): tiktoken all (with cache)
 *   - If messages.length > PRECISE_BUDGET: heuristic for old, tiktoken for recent N
 *     Then calibrate old estimates using recent tiktoken as reference.
 *
 * @param knownTotalTokens  Optional authoritative total from the framework.
 */
export function estimateAllTokens(
  messages: any[],
  knownTotalTokens?: number,
): { total: number; perMessage: number[] } {
  const n = messages.length;
  if (n === 0) return { total: 0, perMessage: [] };

  let raw: number[];

  if (n <= PRECISE_BUDGET) {
    // Small batch: tiktoken all (cache makes repeat calls fast)
    raw = messages.map((msg) => estimateMessageTokens(msg));
  } else {
    // Large batch: tiktoken recent, heuristic old, calibrate
    const preciseStart = n - PRECISE_BUDGET;
    raw = new Array(n);

    // Recent messages: precise (also populates cache for next call)
    let preciseSum = 0;
    let heuristicSumForRecent = 0;
    for (let i = preciseStart; i < n; i++) {
      raw[i] = estimateMessageTokens(messages[i]);
      preciseSum += raw[i];
      heuristicSumForRecent += heuristicMessageTokens(messages[i]);
    }

    // Calibration factor from recent messages
    const calibFactor = heuristicSumForRecent > 0 ? preciseSum / heuristicSumForRecent : 1;

    // Old messages: heuristic × calibration factor (or cache hit if available)
    for (let i = 0; i < preciseStart; i++) {
      if (typeof messages[i] === "object" && _tokenCache.has(messages[i])) {
        raw[i] = _tokenCache.get(messages[i])!;
      } else {
        raw[i] = Math.max(1, Math.round(heuristicMessageTokens(messages[i]) * calibFactor));
      }
    }
  }

  let rawTotal = raw.reduce((s, v) => s + v, 0);

  // External calibration (from framework totalTokens)
  if (!knownTotalTokens || knownTotalTokens <= 0 || rawTotal <= 0) {
    return { total: rawTotal, perMessage: raw };
  }

  const drift = Math.abs(knownTotalTokens - rawTotal) / knownTotalTokens;
  if (drift <= CALIBRATION_THRESHOLD) {
    return { total: rawTotal, perMessage: raw };
  }

  // Apply linear calibration factor, clamped
  const factor = Math.max(
    CALIBRATION_FACTOR_MIN,
    Math.min(CALIBRATION_FACTOR_MAX, knownTotalTokens / rawTotal),
  );
  const calibrated = raw.map((v) => Math.max(1, Math.round(v * factor)));
  const calibratedTotal = calibrated.reduce((s, v) => s + v, 0);

  return { total: calibratedTotal, perMessage: calibrated };
}

// ─── Heuristic fallback ─────────────────────────────────────────────────────

function countCjkChars(text: string): number {
  let n = 0;
  for (const ch of text) {
    const c = ch.codePointAt(0)!;
    if (
      (c >= 0x4e00 && c <= 0x9fff) ||
      (c >= 0x3400 && c <= 0x4dbf) ||
      (c >= 0xf900 && c <= 0xfaff)
    ) {
      n++;
    }
  }
  return n;
}

function heuristicTokens(text: string): number {
  if (!text) return 0;
  const cjk = countCjkChars(text);
  const rest = Math.max(0, text.length - cjk);
  return Math.max(1, Math.ceil(cjk / 1.7 + rest / 4));
}
