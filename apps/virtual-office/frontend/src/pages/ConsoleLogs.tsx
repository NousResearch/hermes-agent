import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { getLog, listLogs, type LogRecord } from "../lib/controlLayer";

type LogLevel = "INFO" | "WARN" | "ERROR";
type AgentName = "all" | "system" | "hermes" | "codex" | "chez";

const badgeColors: Record<LogLevel, string> = {
  INFO: "#2563eb",
  WARN: "#d97706",
  ERROR: "#dc2626",
};

const cardStyle: CSSProperties = {
  borderRadius: 16,
  background: "rgba(15, 23, 42, 0.72)",
  border: "1px solid rgba(148, 163, 184, 0.15)",
  padding: 16,
};

export default function ConsoleLogs() {
  const [logs, setLogs] = useState<LogRecord[]>([]);
  const [levelFilter, setLevelFilter] = useState<LogLevel | "all">("all");
  const [agentFilter, setAgentFilter] = useState<AgentName>("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedLog, setSelectedLog] = useState<LogRecord | null>(null);

  const loadLogs = async () => {
    setLoading(true);
    setError(null);
    try {
      const items = await listLogs({
        level: levelFilter === "all" ? undefined : levelFilter,
        agent: agentFilter === "all" ? undefined : agentFilter,
      });
      setLogs([...items].sort((left, right) => right.timestamp.localeCompare(left.timestamp)));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load logs");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadLogs();
  }, [agentFilter, levelFilter]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      void loadLogs();
    }, 4000);
    return () => window.clearInterval(interval);
  }, [agentFilter, levelFilter]);

  const filteredLogs = useMemo(() => logs, [logs]);

  const openLog = async (logId: string) => {
    try {
      const detail = await getLog(logId);
      setSelectedLog(detail);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load log detail");
    }
  };

  return (
    <section style={{ display: "grid", gap: 20 }}>
      <div>
        <h2 style={{ marginTop: 0 }}>Console Logs</h2>
        <p style={{ color: "#94a3b8" }}>Real backend logs with operator-friendly detail view.</p>
      </div>

      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "end" }}>
        <label style={{ display: "grid", gap: 6 }}>
          <span style={{ color: "#94a3b8", fontSize: 13 }}>Agent</span>
          <select value={agentFilter} onChange={(event) => setAgentFilter(event.target.value as AgentName)}>
            <option value="all">All agents</option>
            <option value="system">System</option>
            <option value="hermes">Hermes</option>
            <option value="codex">Codex</option>
            <option value="chez">Chez</option>
          </select>
        </label>
        <label style={{ display: "grid", gap: 6 }}>
          <span style={{ color: "#94a3b8", fontSize: 13 }}>Level</span>
          <select value={levelFilter} onChange={(event) => setLevelFilter(event.target.value as LogLevel | "all")}>
            <option value="all">All levels</option>
            <option value="INFO">INFO</option>
            <option value="WARN">WARN</option>
            <option value="ERROR">ERROR</option>
          </select>
        </label>
        <button
          type="button"
          onClick={() => void loadLogs()}
          style={{ border: 0, borderRadius: 12, padding: "10px 14px", background: "#1d4ed8", color: "#eff6ff", fontWeight: 700, cursor: "pointer" }}
        >
          Refresh Logs
        </button>
      </div>

      {error ? <div style={{ ...cardStyle, color: "#fecaca", borderColor: "rgba(239, 68, 68, 0.45)" }}>{error}</div> : null}

      <div style={{ display: "grid", gridTemplateColumns: "minmax(320px, 1fr) minmax(320px, 0.9fr)", gap: 20 }}>
        <div style={{ display: "grid", gap: 12 }}>
          {loading ? (
            <div style={cardStyle}>Loading logs...</div>
          ) : filteredLogs.length === 0 ? (
            <div style={cardStyle}>No matching logs.</div>
          ) : (
            filteredLogs.map((log) => {
              const agent = log.metadata?.agent ?? "system";
              return (
                <button
                  key={log.id}
                  type="button"
                  onClick={() => void openLog(log.id)}
                  style={{
                    ...cardStyle,
                    display: "grid",
                    gap: 8,
                    textAlign: "left",
                    color: "inherit",
                    cursor: "pointer",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                    <span
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        justifyContent: "center",
                        minWidth: 64,
                        padding: "4px 10px",
                        borderRadius: 999,
                        background: badgeColors[log.level],
                        color: "#fff",
                        fontSize: 12,
                        fontWeight: 700,
                      }}
                    >
                      {log.level}
                    </span>
                    <span style={{ color: "#34d399", fontSize: 12, fontWeight: 700, textTransform: "uppercase" }}>{agent}</span>
                    <span style={{ color: "#94a3b8", fontSize: 13 }}>{log.timestamp}</span>
                  </div>
                  <div>{log.message}</div>
                </button>
              );
            })
          )}
        </div>

        <aside style={{ ...cardStyle, alignSelf: "start", display: "grid", gap: 10 }}>
          <h3 style={{ margin: 0 }}>Log Detail</h3>
          {!selectedLog ? (
            <div style={{ color: "#94a3b8" }}>Select a log entry to inspect metadata and linked task/handoff IDs.</div>
          ) : (
            <>
              <div style={{ color: badgeColors[selectedLog.level], fontWeight: 700 }}>{selectedLog.level}</div>
              <div style={{ color: "#94a3b8", fontSize: 13 }}>{selectedLog.timestamp}</div>
              <div style={{ whiteSpace: "pre-wrap" }}>{selectedLog.message}</div>
              <pre style={{ margin: 0, padding: 12, borderRadius: 12, background: "rgba(2, 6, 23, 0.9)", overflowX: "auto" }}>
                {JSON.stringify(selectedLog.metadata || {}, null, 2)}
              </pre>
            </>
          )}
        </aside>
      </div>
    </section>
  );
}
