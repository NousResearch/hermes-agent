"use client";

import type { Step } from "../data/catalog";

/**
 * Live AI plumbing — the one thing carried over from the first Aro prototype.
 * Scripted tool steps still demo the harness, but the closing reply is a real
 * model answer streamed word-by-word. The backend is switchable: "demo" POSTs
 * this sandbox's /api/chat (z-ai model), "live" targets any OpenAI-compatible
 * endpoint such as the Aro agent's api_server on :8642.
 */

export type HistoryTurn = { role: "user" | "assistant"; content: string };

/* ---------------- backend config ---------------- */
export type BackendMode = "demo" | "live";
export type BackendConfig = { mode: BackendMode; baseUrl: string; model: string; key: string };
export const DEFAULT_BACKEND: BackendConfig = { mode: "demo", baseUrl: "http://localhost:8642/v1", model: "aro-4-70b", key: "" };

/** Persisted backend selection ("aro.backend" in localStorage, merged over the defaults). */
export function getBackend(): BackendConfig {
  try {
    const saved = JSON.parse(localStorage.getItem("aro.backend") ?? "{}") as Partial<BackendConfig>;
    return { ...DEFAULT_BACKEND, ...saved, mode: saved.mode === "live" ? "live" : "demo" };
  } catch {
    return { ...DEFAULT_BACKEND };
  }
}

export function setBackend(config: BackendConfig): void {
  try { localStorage.setItem("aro.backend", JSON.stringify(config)); } catch { /* storage unavailable — config stays for this session only */ }
}

/** Probe a backend without ever throwing. Live: GET /models with Bearer; demo: POST /api/chat ping. */
export async function testBackend(config: BackendConfig): Promise<{ ok: boolean; detail: string; ms: number }> {
  const t0 = performance.now();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);
  try {
    let res: Response;
    if (config.mode === "live") {
      res = await fetch(`${config.baseUrl.replace(/\/+$/, "")}/models`, {
        signal: controller.signal,
        headers: config.key ? { Authorization: `Bearer ${config.key}` } : undefined,
      });
    } else {
      res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: [{ role: "user", content: "ping" }] }),
        signal: controller.signal,
      });
    }
    const ms = Math.round(performance.now() - t0);
    return { ok: res.ok, detail: `HTTP ${res.status}`, ms };
  } catch (error) {
    const ms = Math.round(performance.now() - t0);
    const detail = error instanceof Error ? (error.name === "AbortError" ? "timed out (5s)" : error.message) : "network error";
    return { ok: false, detail, ms };
  } finally {
    clearTimeout(timer);
  }
}

/** Flatten a transcript into chat history (users + agent text turns). */
export function historyFromSteps(steps: Step[]): HistoryTurn[] {
  const out: HistoryTurn[] = [];
  for (const s of steps) {
    if (s.type === "user" && s.text.trim()) out.push({ role: "user", content: s.text });
    else if (s.type === "text" && s.text.trim()) out.push({ role: "assistant", content: s.text });
  }
  return out.slice(-10);
}

/** Ask the active backend; falls back to the scripted text if it is down. */
export async function fetchLiveReply(history: HistoryTurn[], fallback: string): Promise<string> {
  const backend = getBackend();
  if (backend.mode === "live" && backend.baseUrl.trim()) {
    const live = await liveCompletion(backend, history);
    if (live !== null) return live;
  }
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });
    if (!res.ok) throw new Error(`api ${res.status}`);
    const data = (await res.json()) as { reply?: string };
    return data.reply && data.reply.trim().length > 0 ? data.reply : fallback;
  } catch {
    return fallback;
  }
}

/** POST an OpenAI-compatible /chat/completions; null means "unavailable — fall back to the demo endpoint". */
async function liveCompletion(backend: BackendConfig, history: HistoryTurn[]): Promise<string | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 12000);
  try {
    const res = await fetch(`${backend.baseUrl.replace(/\/+$/, "")}/chat/completions`, {
      method: "POST",
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...(backend.key ? { Authorization: `Bearer ${backend.key}` } : {}) },
      body: JSON.stringify({ model: backend.model, messages: history, max_tokens: 1024, stream: false }),
    });
    if (!res.ok) return null;
    const data = (await res.json()) as { choices?: { message?: { content?: string } }[] };
    const content = data.choices?.[0]?.message?.content;
    return content && content.trim().length > 0 ? content : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

export const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
