import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Client-side Markdown rendering. Per the protocol spec we render the agent's
// `text` (never the terminal-only `rendered` field).
export function Markdown({ text }: { text: string }) {
  return (
    <div className="ht-md">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}
