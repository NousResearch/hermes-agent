import { useState } from "react";
import { deleteSession, getSessions, renameSession, type SessionInfo } from "@/api/endpoints";
import { ManagementPage, ResourceView, useResource } from "@/components/PageScaffold";

function relTime(epochSeconds: number): string {
  if (!epochSeconds) return "—";
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - epochSeconds));
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

export default function SessionsPage() {
  const [order, setOrder] = useState<"recent" | "created">("recent");
  const sessions = useResource(() => getSessions(50, 0, order), [order]);

  return (
    <ManagementPage
      title="Sessions"
      actions={
        <>
          <select
            className="ht-select"
            value={order}
            onChange={(e) => setOrder(e.target.value as "recent" | "created")}
            aria-label="Sort order"
          >
            <option value="recent">Recently active</option>
            <option value="created">Newest</option>
          </select>
          <button type="button" className="ht-btn ht-btn--sm" onClick={sessions.reload}>
            Refresh
          </button>
        </>
      }
    >
      <ResourceView resource={sessions} empty={(d) => d.sessions.length === 0}>
        {(d) => (
          <>
            <p className="ht-muted">{d.total} total</p>
            <table className="ht-table">
              <thead>
                <tr>
                  <th>Title</th>
                  <th>Source</th>
                  <th>Msgs</th>
                  <th>Active</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {d.sessions.map((s) => (
                  <SessionRow key={s.id} s={s} onChanged={sessions.reload} />
                ))}
              </tbody>
            </table>
          </>
        )}
      </ResourceView>
    </ManagementPage>
  );
}

function SessionRow({ s, onChanged }: { s: SessionInfo; onChanged: () => void }) {
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState(s.title ?? "");
  const [busy, setBusy] = useState(false);

  const rename = async () => {
    setBusy(true);
    try {
      await renameSession(s.id, title.trim());
      setEditing(false);
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    setBusy(true);
    try {
      await deleteSession(s.id);
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  return (
    <tr>
      <td>
        {editing ? (
          <input
            className="ht-input ht-input--sm"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            autoFocus
          />
        ) : (
          <>
            <div>{s.title || <span className="ht-dim">Untitled</span>}</div>
            {s.preview && <div className="ht-muted ht-sm">{s.preview}</div>}
          </>
        )}
      </td>
      <td>
        <span className="ht-chip">{s.source ?? "—"}</span>
      </td>
      <td>{s.message_count}</td>
      <td>
        {s.is_active ? <span className="ht-ok">live</span> : relTime(s.last_active)}
      </td>
      <td className="ht-row-actions">
        {editing ? (
          <>
            <button type="button" className="ht-btn ht-btn--sm" disabled={busy} onClick={rename}>
              Save
            </button>
            <button
              type="button"
              className="ht-btn ht-btn--sm ht-btn--ghost"
              onClick={() => setEditing(false)}
            >
              Cancel
            </button>
          </>
        ) : (
          <>
            <button type="button" className="ht-btn ht-btn--sm" onClick={() => setEditing(true)}>
              Rename
            </button>
            <button
              type="button"
              className="ht-btn ht-btn--sm ht-btn--stop"
              disabled={busy}
              onClick={remove}
            >
              Delete
            </button>
          </>
        )}
      </td>
    </tr>
  );
}
