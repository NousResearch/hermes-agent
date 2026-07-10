import type { SessionListItem } from "@/gateway/types";

interface SidebarProps {
  sessions: SessionListItem[];
  activeId: string | null;
  agentName: string;
  onNew: () => void;
  onResume: (id: string) => void;
}

export function SessionSidebar({ sessions, activeId, agentName, onNew, onResume }: SidebarProps) {
  return (
    <aside className="ht-sidebar">
      <div className="ht-sidebar__head">
        <span className="ht-wordmark">{agentName}</span>
        <button type="button" className="ht-btn ht-btn--new" onClick={onNew}>
          + New
        </button>
      </div>
      <nav className="ht-sidebar__list" aria-label="Conversations">
        {sessions.length === 0 && <p className="ht-sidebar__empty">No conversations yet.</p>}
        {sessions.map((s) => (
          <button
            key={s.id}
            type="button"
            className={`ht-sidebar__item${s.id === activeId ? " is-active" : ""}`}
            onClick={() => onResume(s.id)}
            title={s.preview}
          >
            <span className="ht-sidebar__title">{s.title || "Untitled"}</span>
            <span className="ht-sidebar__meta">{s.message_count} msgs</span>
          </button>
        ))}
      </nav>
    </aside>
  );
}
