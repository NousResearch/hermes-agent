// Translate the Hermes gateway's agent event stream into Boardstate `AgentStreamEvent`s
// (SPEC §14), so the board's `builtin:chat` widget renders the REAL Hermes agent's turn
// live. This is the correctness core of the flagship "chat → agent builds the board" loop.
//
// Hermes events (over `/api/ws`, JSON-RPC `{method:"event", params:{type,...}}`) come as:
//   turn.start {request_id, session_key, text}
//   message.start                          → text stream begins
//   message.delta {delta}                  → a token
//   message.complete {text, status}        → text stream ends (status "error" ⇒ error)
//   tool.start {name, id?}                 → a tool call begins
//   tool.complete {name, id?, ...}         → a tool call finished
//   error {message, code?}                 → turn error
//   turn.end / turn.error                  → turn boundary
//   reasoning.delta / thinking.delta       → model's private reasoning (not shown, v1)
//
// The translator is stateful per session: it tracks the current turn id, the text-part
// id, whether the text part is open, and open tool calls — so it can synthesize the
// start/delta/end triads §14 requires (exactly one `turn-end` last, paired tool events).

import type { AgentStreamEvent, ChatStopReason } from "@boardstate/schema";

type HermesEvent = { type: string } & Record<string, unknown>;

const str = (v: unknown): string => (typeof v === "string" ? v : "");
const asRecord = (v: unknown): Record<string, unknown> =>
  typeof v === "object" && v !== null ? (v as Record<string, unknown>) : {};

export type HermesChatTranslator = {
  /** Map one Hermes event to zero or more AgentStreamEvents. */
  translate: (event: HermesEvent) => AgentStreamEvent[];
};

export function createHermesChatTranslator(sessionKey: string): HermesChatTranslator {
  let turnId = "";
  let textId = "";
  let textOpen = false;
  let turnLive = false;
  let toolSeq = 0;
  const openTools = new Map<string, string>(); // hermes tool key → boardstate callId

  const base = () => ({ sessionKey, turnId });

  /** Close an open text part if one is streaming (idempotent). */
  function endText(out: AgentStreamEvent[]): void {
    if (textOpen) {
      out.push({ type: "text-end", ...base(), id: textId });
      textOpen = false;
    }
  }

  /** End the turn exactly once, closing any open text first. */
  function endTurn(out: AgentStreamEvent[], stopReason: ChatStopReason): void {
    if (!turnLive) {
      return;
    }
    endText(out);
    out.push({ type: "turn-end", ...base(), stopReason });
    turnLive = false;
    openTools.clear();
  }

  function toolKey(p: Record<string, unknown>): string {
    return str(p.id ?? p.call_id ?? p.tool_call_id ?? p.name);
  }

  return {
    translate(event) {
      const out: AgentStreamEvent[] = [];
      const p = asRecord(event);
      switch (event.type) {
        case "turn.start": {
          // A new turn — end any dangling prior turn defensively.
          endTurn(out, "end");
          turnId = str(p.request_id ?? p.turn_id ?? p.rid) || `turn-${Date.now()}`;
          textId = `${turnId}:text`;
          textOpen = false;
          turnLive = true;
          out.push({ type: "turn-start", ...base() });
          break;
        }
        case "message.start": {
          if (!turnLive) break;
          if (!textOpen) {
            out.push({ type: "text-start", ...base(), id: textId });
            textOpen = true;
          }
          break;
        }
        case "message.delta": {
          if (!turnLive) break;
          if (!textOpen) {
            out.push({ type: "text-start", ...base(), id: textId });
            textOpen = true;
          }
          out.push({ type: "text-delta", ...base(), id: textId, delta: str(p.delta ?? p.text) });
          break;
        }
        case "message.complete": {
          if (!turnLive) break;
          if (str(p.status) === "error") {
            endText(out);
            out.push({
              type: "error",
              sessionKey,
              turnId,
              code: "agent_error",
              message: str(p.text) || "agent error",
              retryable: false,
            });
            endTurn(out, "end");
          } else {
            endTurn(out, "end");
          }
          break;
        }
        case "tool.start": {
          if (!turnLive) break;
          const callId = `${turnId}:tool:${toolSeq++}`;
          openTools.set(toolKey(p), callId);
          out.push({ type: "tool-call-start", ...base(), callId, name: str(p.name) });
          out.push({ type: "tool-call-ready", ...base(), callId, name: str(p.name), args: p.args ?? p.input ?? {} });
          break;
        }
        case "tool.complete": {
          if (!turnLive) break;
          const key = toolKey(p);
          const callId = openTools.get(key) ?? `${turnId}:tool:${toolSeq++}`;
          openTools.delete(key);
          out.push({
            type: "tool-result",
            ...base(),
            callId,
            ok: str(p.status) !== "error",
            result: p.result ?? p.output ?? undefined,
          });
          break;
        }
        case "error": {
          out.push({
            type: "error",
            sessionKey,
            turnId: turnId || undefined,
            code: str(p.code) || "gateway_error",
            message: str(asRecord(p.payload).message ?? p.message) || "gateway error",
            retryable: false,
          });
          endTurn(out, "end");
          break;
        }
        case "turn.error": {
          endTurn(out, "end");
          break;
        }
        case "turn.end": {
          endTurn(out, "end");
          break;
        }
        // reasoning.delta / thinking.delta — the model's private reasoning; not surfaced
        // in the chat answer stream in v1.
        default:
          break;
      }
      return out;
    },
  };
}
