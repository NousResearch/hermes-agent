"use client";

import type { Step } from "../data/catalog";

/**
 * Live AI plumbing — the one thing carried over from the first Aro prototype.
 * Scripted tool steps still demo the harness, but the closing reply is a real
 * model answer streamed word-by-word from the server-side z-ai endpoint.
 */

export type HistoryTurn = { role: "user" | "assistant"; content: string };

/** Flatten a transcript into chat history (users + agent text turns). */
export function historyFromSteps(steps: Step[]): HistoryTurn[] {
  const out: HistoryTurn[] = [];
  for (const s of steps) {
    if (s.type === "user" && s.text.trim()) out.push({ role: "user", content: s.text });
    else if (s.type === "text" && s.text.trim()) out.push({ role: "assistant", content: s.text });
  }
  return out.slice(-10);
}

/** Ask /api/chat; fall back to the scripted text if the endpoint is down. */
export async function fetchLiveReply(history: HistoryTurn[], fallback: string): Promise<string> {
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

export const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
