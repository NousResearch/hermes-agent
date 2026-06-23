import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { listHandoffs, listLogs, runAgainHandoff, type HandoffRecord, type LogRecord } from "../lib/controlLayer";

const cardStyle: CSSProperties = {
  borderRadius: 16,
  background: "rgba(15, 23, 42, 0.72)",
  border: "1px solid rgba(148, 163, 184, 0.15)",
  padding: 16,
};

function statusColor(status: string): string {
  switch (status) {
    case "completed":
      return "#22c55e";
    case "failed":
      return "#ef4444";
    case "running":
      return "#f59e0b";
    default:
      return "#64748b";
  }
}

export default function HandoffLogs() {
  const [handoffs, setHandoffs] = useState<HandoffRecord[]>([]);
  const [logs, setLogs] = useState<LogRecord[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [handoffData, logData] = await Promise.all([listHandoffs(), listLogs()]);
      const sortedHandoffs = [...handoffData].sort((left, right) => right.created_at.localeCompare(left.created_at));
      setHandoffs(sortedHandoffs);
      setLogs([...logData].sort((left, right) => left.timestamp.localeCompare(right.timestamp)));
      setSelectedId((current) => current ?? sortedHandoffs[0]?.id ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load handoffs");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const selected = useMemo(() => handoffs.find((item) => item.id === selectedId) ?? null, [handoffs, selectedId]);
  const relatedLogs = useMemo(
    () => (selected ? logs.filter((log) => log.metadata?.handoff_id === selected.id) : []),
    [logs, selected],
  );

  const runAgain = async (handoffId: string) => {
    setBusyId(handoffId);
    setError(null);
    try {
      await runAgainHandoff(handoffId);
      await load();
      setSelectedId(handoffId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to run handoff again");
    } finally {
      setBusyId(null);
    }
  };

  return (
    <section style={{ display: "grid", gap: 20 }}>
      <div>
        <h2 style={{ marginTop: 0 }}>Handoff Logs</h2>
        <p style={{ color: "#94a3b8" }}>Inspect handoff result state, linked timeline, and rerun the same payload when needed.</p>
      </div>

      {error ? <div style={{ ...cardStyle, color: "#fecaca", borderColor: "rgba(239, 68, 68, 0.45)" }}>{error}</div> : null}

      <div style={{ display: "grid", gridTemplateColumns: "minmax(320px, 1fr) minmax(340px, 0.95fr)", gap: 20 }}>
        <div style={{ display: "grid", gap: 12 }}>
          {loading ? (
            <div style={cardStyle}>Loading handoffs...</div>
          ) : handoffs.length === 0 ? (
            <div style={cardStyle}>No handoffs available yet.</div>
          ) : (
            handoffs.map((handoff) => (
              <div key={handoff.id} style={cardStyle}>
                <button
                  type="button"
                  onClick={() => setSelectedId(handoff.id)}
                  style={{ background: "transparent", border: 0, color: "inherit", width: "100%", textAlign: "left", padding: 0, cursor: "pointer" }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
                    <strong>
                      {handoff.from_agent} -&gt; {handoff.to_agent}
                    </strong>
                    <span style={{ color: statusColor(handoff.status), fontWeight: 700 }}>{handoff.status}</span>
                  </div>
                  <div style={{ marginTop: 8, color: "#94a3b8", fontSize: 13 }}>{handoff.created_at}</div>
                  <div style={{ marginTop: 8, color: "#cbd5e1" }}>
                    {typeof handoff.payload?.prompt === "string" ? handoff.payload.prompt : JSON.stringify(handoff.payload || {}, null, 2)}
                  </div>
                </button>
                <div style={{ marginTop: 12 }}>
                  <button
                    type="button"
                    onClick={() => void runAgain(handoff.id)}
                    disabled={busyId === handoff.id}
                    style={{ border: 0, borderRadius: 12, padding: "10px 14px", background: "#22c55e", color: "#052e16", fontWeight: 700, cursor: "pointer" }}
                  >
                    {busyId === handoff.id ? "Running..." : "Run Again"}
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        <aside style={{ ...cardStyle, alignSelf: "start", display: "grid", gap: 12 }}>
          <h3 style={{ margin: 0 }}>Handoff Detail</h3>
          {!selected ? (
            <div style={{ color: "#94a3b8" }}>Select a handoff to inspect its result and timeline.</div>
          ) : (
            <>
              <div style={{ color: statusColor(selected.status), fontWeight: 700 }}>{selected.status}</div>
              <div style={{ color: "#94a3b8", fontSize: 13 }}>
                Started {selected.created_at}
                {selected.completed_at ? ` | Completed ${selected.completed_at}` : ""}
              </div>
              <pre style={{ margin: 0, padding: 12, borderRadius: 12, background: "rgba(2, 6, 23, 0.9)", overflowX: "auto" }}>
                {JSON.stringify(selected.result || selected.payload || {}, null, 2)}
              </pre>
              <div>
                <strong style={{ display: "block", marginBottom: 8 }}>Timeline</strong>
                {relatedLogs.length === 0 ? (
                  <div style={{ color: "#94a3b8" }}>No linked logs.</div>
                ) : (
                  <div style={{ display: "grid", gap: 10 }}>
                    {relatedLogs.map((log) => (
                      <div key={log.id} style={{ paddingLeft: 12, borderLeft: `3px solid ${statusColor(selected.status)}` }}>
                        <div>{log.message}</div>
                        <div style={{ color: "#94a3b8", fontSize: 12 }}>{log.timestamp}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </aside>
      </div>
    </section>
  );
}
