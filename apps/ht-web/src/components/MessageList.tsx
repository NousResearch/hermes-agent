import { useEffect, useRef } from "react";
import type { ChatMessage } from "@/gateway/chatReducer";
import { Markdown } from "./Markdown";
import { ToolActivity } from "./ToolActivity";

export function MessageList({ messages }: { messages: ChatMessage[] }) {
  const endRef = useRef<HTMLDivElement>(null);

  // Keep the newest content in view as it streams. Guarded because
  // scrollIntoView is absent in jsdom (tests) and older embedders.
  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: "end" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="ht-empty">
        <p>Ask anything to get started.</p>
      </div>
    );
  }

  return (
    <div className="ht-messages">
      {messages.map((m) => (
        <article key={m.id} className={`ht-msg ht-msg--${m.role}`}>
          <div className="ht-msg__role">{m.role}</div>
          <div className="ht-msg__body">
            {m.role === "assistant" ? <Markdown text={m.text} /> : <p>{m.text}</p>}
            {m.streaming && m.text.length === 0 && m.tools.length === 0 && (
              <span className="ht-cursor" aria-label="thinking">
                ▍
              </span>
            )}
            <ToolActivity tools={m.tools} />
          </div>
        </article>
      ))}
      <div ref={endRef} />
    </div>
  );
}
