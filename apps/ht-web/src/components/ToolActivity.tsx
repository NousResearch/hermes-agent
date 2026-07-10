import type { ToolCall } from "@/gateway/chatReducer";

const STATUS_GLYPH: Record<ToolCall["status"], string> = {
  running: "◆",
  done: "✓",
  error: "✗",
};

export function ToolActivity({ tools }: { tools: ToolCall[] }) {
  if (tools.length === 0) return null;
  return (
    <ul className="ht-tools">
      {tools.map((t) => (
        <li key={t.toolId} className={`ht-tool ht-tool--${t.status}`}>
          <span className="ht-tool__glyph" aria-hidden>
            {STATUS_GLYPH[t.status]}
          </span>
          <span className="ht-tool__name">{t.name}</span>
          {t.durationS != null && <span className="ht-tool__dur">{t.durationS.toFixed(1)}s</span>}
          {(t.error || t.summary || t.preview) && (
            <span className="ht-tool__detail">{t.error ?? t.summary ?? t.preview}</span>
          )}
        </li>
      ))}
    </ul>
  );
}
