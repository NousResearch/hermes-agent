import { useEffect, useMemo, useState, type CSSProperties } from "react";
import {
  getHealth,
  getSettings,
  listAdapters,
  listHandoffs,
  listSessions,
  listTasks,
  type AdapterRecord,
  type HandoffRecord,
  type HealthRecord,
  type SessionRecord,
  type SettingsRecord,
  type TaskRecord,
} from "../lib/controlLayer";

const cardStyle: CSSProperties = {
  borderRadius: 18,
  background: "rgba(15, 23, 42, 0.78)",
  border: "1px solid rgba(148, 163, 184, 0.16)",
  padding: 18,
};

export default function MainOffice() {
  const [health, setHealth] = useState<HealthRecord | null>(null);
  const [settings, setSettings] = useState<SettingsRecord | null>(null);
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [adapters, setAdapters] = useState<AdapterRecord[]>([]);
  const [tasks, setTasks] = useState<TaskRecord[]>([]);
  const [handoffs, setHandoffs] = useState<HandoffRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [healthData, settingsData, sessionData, adapterData, taskData, handoffData] = await Promise.allSettled([
        getHealth(),
        getSettings(),
        listSessions(),
        listAdapters(),
        listTasks(),
        listHandoffs(),
      ]);
      setHealth(healthData.status === "fulfilled" ? healthData.value : { status: "offline" });
      setSettings(settingsData.status === "fulfilled" ? settingsData.value : null);
      setSessions(sessionData.status === "fulfilled" ? sessionData.value : []);
      setAdapters(adapterData.status === "fulfilled" ? adapterData.value : []);
      setTasks(taskData.status === "fulfilled" ? taskData.value : []);
      setHandoffs(handoffData.status === "fulfilled" ? handoffData.value : []);
      const failures = [healthData, settingsData, sessionData, adapterData, taskData, handoffData].filter(
        (result): result is PromiseRejectedResult => result.status === "rejected",
      );
      if (failures.length > 0) {
        const first = failures[0]?.reason;
        setError(first instanceof Error ? first.message : "Some dashboard data could not be loaded.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load main office data");
      setHealth({ status: "offline" });
      setSettings(null);
      setSessions([]);
      setAdapters([]);
      setTasks([]);
      setHandoffs([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const pendingTasks = useMemo(() => tasks.filter((task) => task.status === "pending" || task.status === "in_progress").length, [tasks]);
  const completedToday = useMemo(() => tasks.filter((task) => task.status === "completed").length, [tasks]);
  const recentTasks = useMemo(() => [...tasks].sort((a, b) => b.updated_at.localeCompare(a.updated_at)).slice(0, 5), [tasks]);
  const recentHandoffs = useMemo(
    () => [...handoffs].sort((a, b) => String(b.completed_at || b.created_at).localeCompare(String(a.completed_at || a.created_at))).slice(0, 5),
    [handoffs],
  );

  const stats = [
    { label: "Sessions", value: String(sessions.length), accent: "#60a5fa" },
    { label: "Agents Online", value: String(adapters.filter((adapter) => adapter.status === "online").length), accent: "#34d399" },
    { label: "Open Tasks", value: String(pendingTasks), accent: "#fbbf24" },
    { label: "Completed", value: String(completedToday), accent: "#a78bfa" },
  ];

  return (
    <section style={{ display: "grid", gap: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", gap: 16, flexWrap: "wrap" }}>
        <div>
          <h2 style={{ margin: 0 }}>Main Office Dashboard</h2>
          <p style={{ color: "#94a3b8", marginTop: 8 }}>Live operator overview for sessions, adapters, task volume, and current Codex workspace wiring.</p>
        </div>
        <button
          type="button"
          onClick={() => void load()}
          style={{ border: 0, borderRadius: 12, padding: "10px 14px", background: "#1d4ed8", color: "#eff6ff", fontWeight: 700, cursor: "pointer" }}
        >
          Refresh Overview
        </button>
      </div>

      {error ? <div style={{ ...cardStyle, color: "#fecaca", borderColor: "rgba(239, 68, 68, 0.45)" }}>{error}</div> : null}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 16 }}>
        {stats.map((item) => (
          <article key={item.label} style={cardStyle}>
            <div style={{ color: item.accent, fontWeight: 700, fontSize: 13, textTransform: "uppercase", letterSpacing: "0.05em" }}>{item.label}</div>
            <div style={{ fontSize: 32, fontWeight: 800, marginTop: 8 }}>{loading ? "..." : item.value}</div>
          </article>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(320px, 1.05fr) minmax(320px, 1fr)", gap: 20 }}>
        <article style={{ ...cardStyle, display: "grid", gap: 12 }}>
          <h3 style={{ margin: 0 }}>Runtime Status</h3>
          <div style={{ color: health?.status === "ok" ? "#34d399" : "#fca5a5", fontWeight: 700 }}>
            Backend {health?.status === "ok" ? "online" : "offline"}
          </div>
          <div style={{ color: "#cbd5e1" }}>Codex workdir: {settings?.codex_workdir || "unknown"}</div>
          <div style={{ color: "#cbd5e1" }}>Backend port: {settings?.backend_port ?? "unknown"}</div>
          <div style={{ color: "#cbd5e1" }}>Frontend port: {settings?.frontend_port ?? "unknown"}</div>
        </article>

        <article style={{ ...cardStyle, display: "grid", gap: 12 }}>
          <h3 style={{ margin: 0 }}>Adapters</h3>
          {adapters.length === 0 ? (
            <div style={{ color: "#94a3b8" }}>{loading ? "Loading adapters..." : "No adapters detected."}</div>
          ) : (
            adapters.map((adapter) => (
              <div key={adapter.name} style={{ display: "flex", justifyContent: "space-between", gap: 12, padding: 12, borderRadius: 12, background: "rgba(30, 41, 59, 0.75)" }}>
                <div>
                  <strong style={{ textTransform: "capitalize" }}>{adapter.name}</strong>
                  <div style={{ color: adapter.status === "online" ? "#34d399" : "#fca5a5", fontSize: 13 }}>{adapter.status}</div>
                </div>
                <div style={{ textAlign: "right", color: "#94a3b8", fontSize: 13 }}>
                  <div>{adapter.version}</div>
                  <div>{adapter.model}</div>
                </div>
              </div>
            ))
          )}
        </article>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 20 }}>
        <article style={{ ...cardStyle, display: "grid", gap: 12 }}>
          <h3 style={{ margin: 0 }}>Recent Tasks</h3>
          {recentTasks.length === 0 ? (
            <div style={{ color: "#94a3b8" }}>{loading ? "Loading tasks..." : "No tasks recorded yet."}</div>
          ) : (
            recentTasks.map((task) => (
              <div key={task.id} style={{ padding: 12, borderRadius: 12, background: "rgba(30, 41, 59, 0.75)" }}>
                <strong>{task.title}</strong>
                <div style={{ marginTop: 6, color: "#cbd5e1" }}>{task.status} | {task.agent} | {task.room || "main-office"}</div>
                <div style={{ marginTop: 6, color: "#94a3b8", fontSize: 13 }}>{task.updated_at}</div>
              </div>
            ))
          )}
        </article>

        <article style={{ ...cardStyle, display: "grid", gap: 12 }}>
          <h3 style={{ margin: 0 }}>Recent Handoffs</h3>
          {recentHandoffs.length === 0 ? (
            <div style={{ color: "#94a3b8" }}>{loading ? "Loading handoffs..." : "No handoffs recorded yet."}</div>
          ) : (
            recentHandoffs.map((handoff) => (
              <div key={handoff.id} style={{ padding: 12, borderRadius: 12, background: "rgba(30, 41, 59, 0.75)" }}>
                <strong>{handoff.from_agent} → {handoff.to_agent}</strong>
                <div style={{ marginTop: 6, color: "#cbd5e1" }}>{handoff.status}</div>
                <div style={{ marginTop: 6, color: "#94a3b8", fontSize: 13 }}>{handoff.completed_at || handoff.created_at}</div>
              </div>
            ))
          )}
        </article>
      </div>
    </section>
  );
}
