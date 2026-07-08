import { describe, it, expect } from "vitest";
import {
  chatReducer,
  initialChatState,
  stateFromTranscript,
  type ChatState,
} from "./chatReducer";

// Helper: feed a sequence of gateway events through the reducer.
function feed(state: ChatState, events: [string, unknown][]): ChatState {
  return events.reduce(
    (s, [name, payload]) => chatReducer(s, { type: "event", name, payload }),
    state,
  );
}

describe("chatReducer — streaming", () => {
  it("opens a bubble on message.start and appends deltas in order", () => {
    const s = feed(initialChatState, [
      ["message.start", undefined],
      ["message.delta", { text: "Hel" }],
      ["message.delta", { text: "lo" }],
    ]);
    expect(s.messages).toHaveLength(1);
    expect(s.messages[0]!.role).toBe("assistant");
    expect(s.messages[0]!.text).toBe("Hello");
    expect(s.messages[0]!.streaming).toBe(true);
    expect(s.status).toBe("working");
  });

  it("auto-opens a bubble when a delta arrives without message.start", () => {
    const s = feed(initialChatState, [["message.delta", { text: "hi" }]]);
    expect(s.messages).toHaveLength(1);
    expect(s.messages[0]!.text).toBe("hi");
  });

  it("finalizes with the authoritative message.complete text", () => {
    const s = feed(initialChatState, [
      ["message.start", undefined],
      ["message.delta", { text: "partial" }],
      ["message.complete", { text: "the full corrected answer" }],
    ]);
    expect(s.messages[0]!.text).toBe("the full corrected answer");
    expect(s.messages[0]!.streaming).toBe(false);
    expect(s.status).toBe("idle");
  });

  it("keeps streamed text when message.complete omits text", () => {
    const s = feed(initialChatState, [
      ["message.delta", { text: "streamed only" }],
      ["message.complete", {}],
    ]);
    expect(s.messages[0]!.text).toBe("streamed only");
    expect(s.messages[0]!.streaming).toBe(false);
  });

  it("ignores the terminal-only `rendered` field", () => {
    const s = feed(initialChatState, [
      ["message.delta", { text: "clean", rendered: "[1mbold[0m" }],
    ]);
    expect(s.messages[0]!.text).toBe("clean");
  });

  it("mints stable, unique message ids without Date/random", () => {
    let s = chatReducer(initialChatState, { type: "userSubmitted", text: "q1" });
    s = feed(s, [["message.start", undefined], ["message.complete", { text: "a1" }]]);
    s = chatReducer(s, { type: "userSubmitted", text: "q2" });
    s = feed(s, [["message.start", undefined], ["message.complete", { text: "a2" }]]);
    const ids = s.messages.map((m) => m.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(s.messages.map((m) => m.text)).toEqual(["q1", "a1", "q2", "a2"]);
  });
});

describe("chatReducer — tools", () => {
  it("correlates tool.start and tool.complete by tool_id", () => {
    const s = feed(initialChatState, [
      ["message.start", undefined],
      ["tool.start", { tool_id: "t1", name: "read_file" }],
      ["tool.progress", { preview: "reading…" }],
      ["tool.complete", { tool_id: "t1", summary: "read 40 lines" }],
    ]);
    const tools = s.messages[0]!.tools;
    expect(tools).toHaveLength(1);
    expect(tools[0]!.name).toBe("read_file");
    expect(tools[0]!.preview).toBe("reading…");
    expect(tools[0]!.status).toBe("done");
    expect(tools[0]!.summary).toBe("read 40 lines");
  });

  it("marks a tool errored when tool.complete carries an error", () => {
    const s = feed(initialChatState, [
      ["message.start", undefined],
      ["tool.start", { tool_id: "t9", name: "terminal" }],
      ["tool.complete", { tool_id: "t9", error: "exit 1" }],
    ]);
    expect(s.messages[0]!.tools[0]!.status).toBe("error");
    expect(s.messages[0]!.tools[0]!.error).toBe("exit 1");
  });

  it("ignores tool.complete for an unknown tool_id", () => {
    const s = feed(initialChatState, [
      ["message.start", undefined],
      ["tool.start", { tool_id: "t1", name: "a" }],
      ["tool.complete", { tool_id: "nope", summary: "x" }],
    ]);
    expect(s.messages[0]!.tools[0]!.status).toBe("running");
  });
});

describe("chatReducer — interactive requests", () => {
  it("surfaces and clears a clarify request", () => {
    let s = feed(initialChatState, [
      ["clarify.request", { request_id: "c1", question: "Which file?", choices: ["a", "b"] }],
    ]);
    expect(s.clarify).toEqual({ requestId: "c1", question: "Which file?", choices: ["a", "b"] });
    expect(s.status).toBe("waiting");
    s = chatReducer(s, { type: "event", name: "clarify.resolved", payload: {} });
    expect(s.clarify).toBeNull();
    expect(s.status).toBe("working");
  });

  it("surfaces and clears an approval request", () => {
    let s = feed(initialChatState, [
      ["approval.request", { command: "rm -rf x", description: "delete x", allow_permanent: true }],
    ]);
    expect(s.approval).toEqual({ command: "rm -rf x", description: "delete x", allowPermanent: true });
    s = chatReducer(s, { type: "event", name: "approval.resolved", payload: {} });
    expect(s.approval).toBeNull();
  });
});

describe("chatReducer — transcript + misc", () => {
  it("seeds conversational roles from a resumed transcript and drops tool rows", () => {
    const s = stateFromTranscript([
      { role: "user", text: "hi" },
      { role: "assistant", text: "hello" },
      { role: "tool", text: "{...}" },
      { role: "system", text: "note" },
    ]);
    expect(s.messages.map((m) => m.role)).toEqual(["user", "assistant", "system"]);
    expect(s.messages.every((m) => !m.streaming)).toBe(true);
  });

  it("records an error event and returns to idle", () => {
    const s = feed(initialChatState, [["error", { message: "boom" }]]);
    expect(s.error).toBe("boom");
    expect(s.status).toBe("idle");
  });

  it("userSubmitted appends a user bubble and sets working", () => {
    const s = chatReducer(initialChatState, { type: "userSubmitted", text: "hey" });
    expect(s.messages).toEqual([
      expect.objectContaining({ role: "user", text: "hey", streaming: false }),
    ]);
    expect(s.status).toBe("working");
  });

  it("reset returns to the initial state", () => {
    let s = chatReducer(initialChatState, { type: "userSubmitted", text: "x" });
    s = chatReducer(s, { type: "reset" });
    expect(s).toEqual(initialChatState);
  });
});
