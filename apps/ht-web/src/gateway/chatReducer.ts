// Pure reducer that turns the gateway event stream into chat render state.
// Kept free of React and DOM so it can be unit-tested in isolation — this is
// the correctness core of ht-web (see chatReducer.test.ts).

import type {
  ApprovalRequestPayload,
  ClarifyRequestPayload,
  GatewayTranscriptMessage,
  LiveSessionStatus,
  MessageCompletePayload,
  MessageDeltaPayload,
  StatusUpdatePayload,
  ToolCompletePayload,
  ToolStartPayload,
} from "./types";

export type MessageRole = "user" | "assistant" | "system";

export interface ToolCall {
  toolId: string;
  name: string;
  argsText?: string;
  preview?: string;
  status: "running" | "done" | "error";
  summary?: string;
  error?: string;
  durationS?: number;
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  /** Final/committed text. */
  text: string;
  /** True while assistant deltas are still streaming into this bubble. */
  streaming: boolean;
  tools: ToolCall[];
}

export interface PendingClarify {
  requestId: string;
  question: string;
  choices: string[] | null;
}

export interface PendingApproval {
  command: string;
  description: string;
  allowPermanent: boolean;
}

export interface ChatState {
  messages: ChatMessage[];
  status: LiveSessionStatus;
  statusText: string;
  clarify: PendingClarify | null;
  approval: PendingApproval | null;
  error: string | null;
  /** Monotonic counter used to mint stable message ids without Date/random. */
  seq: number;
}

export const initialChatState: ChatState = {
  messages: [],
  status: "idle",
  statusText: "",
  clarify: null,
  approval: null,
  error: null,
  seq: 0,
};

export type ChatAction =
  | { type: "reset" }
  | { type: "seedTranscript"; messages: GatewayTranscriptMessage[] }
  | { type: "userSubmitted"; text: string }
  | { type: "event"; name: string; payload: unknown };

/** Build a ChatState seeded from a resumed transcript. */
export function stateFromTranscript(
  messages: GatewayTranscriptMessage[],
): ChatState {
  return chatReducer(initialChatState, { type: "seedTranscript", messages });
}

function mintId(seq: number, role: string): string {
  return `${role}-${seq}`;
}

/** Index of the last assistant message that is still streaming, or -1. */
function streamingIndex(messages: ChatMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]!.role === "assistant" && messages[i]!.streaming) {
      return i;
    }
  }
  return -1;
}

function replaceAt(
  messages: ChatMessage[],
  index: number,
  patch: Partial<ChatMessage>,
): ChatMessage[] {
  const next = messages.slice();
  next[index] = { ...next[index]!, ...patch };
  return next;
}

/**
 * Ensure there is an open streaming assistant bubble and return its index plus
 * the (possibly extended) message list and seq. Callers append/finalize into it.
 */
function ensureStreamingBubble(state: ChatState): {
  messages: ChatMessage[];
  index: number;
  seq: number;
} {
  const existing = streamingIndex(state.messages);
  if (existing >= 0) {
    return { messages: state.messages, index: existing, seq: state.seq };
  }
  const seq = state.seq + 1;
  const bubble: ChatMessage = {
    id: mintId(seq, "assistant"),
    role: "assistant",
    text: "",
    streaming: true,
    tools: [],
  };
  return { messages: [...state.messages, bubble], index: state.messages.length, seq };
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "reset":
      return initialChatState;

    case "seedTranscript": {
      let seq = 0;
      const messages: ChatMessage[] = [];
      for (const m of action.messages) {
        // Tool transcript rows are folded into the prior assistant turn in the
        // live stream; on resume we surface only the conversational roles.
        if (m.role !== "user" && m.role !== "assistant" && m.role !== "system") {
          continue;
        }
        seq += 1;
        messages.push({
          id: mintId(seq, m.role),
          role: m.role,
          text: m.text ?? "",
          streaming: false,
          tools: [],
        });
      }
      return { ...initialChatState, messages, seq };
    }

    case "userSubmitted": {
      const seq = state.seq + 1;
      const msg: ChatMessage = {
        id: mintId(seq, "user"),
        role: "user",
        text: action.text,
        streaming: false,
        tools: [],
      };
      return {
        ...state,
        seq,
        messages: [...state.messages, msg],
        status: "working",
        error: null,
      };
    }

    case "event":
      return applyEvent(state, action.name, action.payload);

    default:
      return state;
  }
}

function applyEvent(state: ChatState, name: string, payload: unknown): ChatState {
  switch (name) {
    case "message.start": {
      // Open a fresh bubble even if a prior one is somehow still open.
      const seq = state.seq + 1;
      const bubble: ChatMessage = {
        id: mintId(seq, "assistant"),
        role: "assistant",
        text: "",
        streaming: true,
        tools: [],
      };
      return { ...state, seq, status: "working", messages: [...state.messages, bubble] };
    }

    case "message.delta": {
      const p = payload as MessageDeltaPayload;
      if (!p?.text) return state;
      const { messages, index, seq } = ensureStreamingBubble(state);
      const bubble = messages[index]!;
      return {
        ...state,
        seq,
        messages: replaceAt(messages, index, { text: bubble.text + p.text }),
      };
    }

    case "message.complete": {
      const p = (payload ?? {}) as MessageCompletePayload;
      const { messages, index, seq } = ensureStreamingBubble(state);
      const bubble = messages[index]!;
      // message.complete.text is authoritative; fall back to the streamed text.
      const finalText = typeof p.text === "string" && p.text.length > 0 ? p.text : bubble.text;
      return {
        ...state,
        seq,
        status: "idle",
        statusText: "",
        messages: replaceAt(messages, index, { text: finalText, streaming: false }),
      };
    }

    case "status.update": {
      const p = payload as StatusUpdatePayload;
      return { ...state, statusText: p?.text ?? "" };
    }

    case "tool.start": {
      const p = payload as ToolStartPayload;
      if (!p?.tool_id) return state;
      const { messages, index, seq } = ensureStreamingBubble(state);
      const bubble = messages[index]!;
      const tool: ToolCall = {
        toolId: p.tool_id,
        name: p.name ?? "tool",
        argsText: p.args_text,
        status: "running",
      };
      return {
        ...state,
        seq,
        messages: replaceAt(messages, index, { tools: [...bubble.tools, tool] }),
      };
    }

    case "tool.progress": {
      const p = payload as { name?: string; preview?: string };
      const index = streamingIndex(state.messages);
      if (index < 0) return state;
      const bubble = state.messages[index]!;
      if (bubble.tools.length === 0) return state;
      const tools = bubble.tools.slice();
      // Update the most recent running tool.
      for (let i = tools.length - 1; i >= 0; i--) {
        if (tools[i]!.status === "running") {
          tools[i] = { ...tools[i]!, preview: p?.preview ?? tools[i]!.preview };
          break;
        }
      }
      return { ...state, messages: replaceAt(state.messages, index, { tools }) };
    }

    case "tool.complete": {
      const p = payload as ToolCompletePayload;
      if (!p?.tool_id) return state;
      const index = streamingIndex(state.messages);
      if (index < 0) return state;
      const bubble = state.messages[index]!;
      const tools = bubble.tools.map((t) =>
        t.toolId === p.tool_id
          ? {
              ...t,
              status: p.error ? ("error" as const) : ("done" as const),
              summary: p.summary ?? p.result_text,
              error: p.error,
              durationS: p.duration_s,
            }
          : t,
      );
      return { ...state, messages: replaceAt(state.messages, index, { tools }) };
    }

    case "clarify.request": {
      const p = payload as ClarifyRequestPayload;
      if (!p?.request_id) return state;
      return {
        ...state,
        status: "waiting",
        clarify: { requestId: p.request_id, question: p.question, choices: p.choices ?? null },
      };
    }

    case "approval.request": {
      const p = payload as ApprovalRequestPayload;
      if (!p?.command) return state;
      return {
        ...state,
        status: "waiting",
        approval: {
          command: p.command,
          description: p.description,
          allowPermanent: Boolean(p.allow_permanent),
        },
      };
    }

    case "clarify.resolved":
      return { ...state, clarify: null, status: "working" };

    case "approval.resolved":
      return { ...state, approval: null, status: "working" };

    case "error": {
      const p = payload as { message?: string };
      return { ...state, error: p?.message ?? "Unknown error", status: "idle" };
    }

    default:
      return state;
  }
}
