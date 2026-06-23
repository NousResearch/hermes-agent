import { useEffect, useState, type CSSProperties } from "react";
import { listAdapters, type AdapterRecord } from "../lib/controlLayer";

const cardStyle: CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: 16,
  padding: 18,
  borderRadius: 18,
  background: "rgba(15, 23, 42, 0.92)",
  border: "1px solid rgba(148, 163, 184, 0.14)",
};

export default function AgentsPanel() {
  const [adapters, setAdapters] = useState<AdapterRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      setAdapters(await listAdapters());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load adapters");
      setAdapters([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  return (
    <section style={{ display: "grid", gap: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", gap: 16, flexWrap: "wrap" }}>
        <div>
          <h2 style={{ margin: 0 }}>Agents Panel</h2>
          <p style={{ margin: "10px 0 0", color: "#94a3b8" }}>Current adapter availability and runtime identity.</p>
        </div>
        <button
          type="button"
          onClick={() => void load()}
          style={{ border: 0, borderRadius: 12, padding: "10px 14px", background: "#1d4ed8", color: "#eff6ff", fontWeight: 700, cursor: "pointer" }}
        >
          Refresh Adapters
        </button>
      </div>

      {error ? <div style={{ ...cardStyle, color: "#fecaca", borderColor: "rgba(239, 68, 68, 0.45)" }}>{error}</div> : null}

      <section style={{ display: "grid", gap: 16 }}>
        {loading ? (
          <div style={cardStyle}>Loading adapters...</div>
        ) : adapters.length === 0 ? (
          <div style={cardStyle}>No adapters available.</div>
        ) : (
          adapters.map((adapter) => {
            const isOnline = adapter.status === "online";
            return (
              <article key={adapter.name} style={cardStyle}>
                <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                  <span
                    style={{
                      width: 14,
                      height: 14,
                      borderRadius: 999,
                      background: isOnline ? "#22c55e" : "#ef4444",
                      boxShadow: `0 0 18px ${isOnline ? "rgba(34, 197, 94, 0.5)" : "rgba(239, 68, 68, 0.45)"}`,
                    }}
                  />
                  <div>
                    <strong style={{ display: "block", fontSize: "1.05rem", textTransform: "capitalize" }}>{adapter.name}</strong>
                    <span style={{ color: "#94a3b8", fontSize: "0.92rem" }}>{isOnline ? "Online" : "Offline"}</span>
                  </div>
                </div>

                <div style={{ textAlign: "right", color: "#cbd5e1" }}>
                  <div>Version {adapter.version}</div>
                  <div>{adapter.model}</div>
                  <div style={{ color: "#94a3b8", fontSize: "0.9rem" }}>{isOnline ? "Adapter responding" : "Adapter unavailable"}</div>
                </div>
              </article>
            );
          })
        )}
      </section>
    </section>
  );
}
