import { describe, it, expect, afterEach } from "vitest";
import { render, cleanup, screen } from "@testing-library/react";
import { MessageList } from "./MessageList";
import type { ChatMessage } from "@/gateway/chatReducer";

afterEach(cleanup);

function msg(partial: Partial<ChatMessage> & Pick<ChatMessage, "id" | "role">): ChatMessage {
  return { text: "", streaming: false, tools: [], ...partial };
}

describe("MessageList", () => {
  it("shows the empty state when there are no messages", () => {
    render(<MessageList messages={[]} />);
    expect(screen.getByText(/get started/i)).toBeTruthy();
  });

  it("renders user text and assistant markdown", () => {
    render(
      <MessageList
        messages={[
          msg({ id: "u1", role: "user", text: "hello there" }),
          msg({ id: "a1", role: "assistant", text: "**bold** reply" }),
        ]}
      />,
    );
    expect(screen.getByText("hello there")).toBeTruthy();
    // react-markdown renders **bold** as a <strong>
    const strong = screen.getByText("bold");
    expect(strong.tagName).toBe("STRONG");
  });

  it("renders a tool activity row with its status", () => {
    render(
      <MessageList
        messages={[
          msg({
            id: "a1",
            role: "assistant",
            text: "working",
            tools: [{ toolId: "t1", name: "read_file", status: "done", summary: "ok" }],
          }),
        ]}
      />,
    );
    expect(screen.getByText("read_file")).toBeTruthy();
    expect(screen.getByText("ok")).toBeTruthy();
  });
});
