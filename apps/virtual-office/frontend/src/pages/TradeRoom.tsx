import { useEffect, useState, type CSSProperties } from "react";
import { execCodex, getSettings, type CodexExecResult, type SettingsRecord } from "../lib/controlLayer";

const watchlist = [
  { symbol: "XAUUSD", price: "2348.12", change: "+0.42%" },
  { symbol: "EURUSD", price: "1.0738", change: "-0.11%" },
  { symbol: "BTCUSD", price: "64210.55", change: "+1.87%" },
];

const panelStyle: CSSProperties = {
  borderRadius: 18,
  background: "rgba(15, 23, 42, 0.72)",
  border: "1px solid rgba(148, 163, 184, 0.16)",
  padding: 20,
};

export default function TradeRoom() {
  const [prompt, setPrompt] = useState("");
  const [settings, setSettings] = useState<SettingsRecord | null>(null);
  const [result, setResult] = useState<CodexExecResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void getSettings()
      .then(setSettings)
      .catch(() => setSettings(null));
  }, []);

  const runAnalysis = async () => {
    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt) {
      setError("Prompt is required.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const payload = await execCodex({
        prompt: trimmedPrompt,
        workdir: settings?.codex_workdir,
        timeout: 180,
      });
      setResult(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section style={{ display: "grid", gap: 20 }}>
      <div>
        <h2 style={{ margin: 0 }}>Trade Room</h2>
        <p style={{ color: "#94a3b8", marginTop: 8 }}>Market operator panel with direct Codex execution using the persisted project workdir.</p>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 20 }}>
        <aside style={panelStyle}>
          <h3 style={{ marginTop: 0 }}>Watchlist</h3>
          <div style={{ display: "grid", gap: 12 }}>
            {watchlist.map((item) => (
              <div key={item.symbol} style={{ display: "grid", gap: 6, padding: 14, borderRadius: 14, background: "rgba(30, 41, 59, 0.75)" }}>
                <strong>{item.symbol}</strong>
                <span style={{ fontSize: 20 }}>{item.price}</span>
                <span style={{ color: item.change.startsWith("-") ? "#f87171" : "#34d399" }}>{item.change}</span>
              </div>
            ))}
          </div>
        </aside>
        <div style={{ display: "grid", gap: 20 }}>
          <section style={panelStyle}>
            <h3 style={{ marginTop: 0 }}>Codex Analysis</h3>
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="Ask Codex to review market context, summarize risk, or suggest next steps."
              rows={10}
              style={{ width: "100%", resize: "vertical", borderRadius: 14, border: "1px solid rgba(148, 163, 184, 0.2)", padding: 14, background: "rgba(15, 23, 42, 0.9)", color: "#e5eef7" }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 14, gap: 12, flexWrap: "wrap" }}>
              <span style={{ color: "#94a3b8", fontSize: 13 }}>Workdir: {settings?.codex_workdir || "loading..."}</span>
              <button
                type="button"
                onClick={() => void runAnalysis()}
                disabled={loading}
                style={{ border: 0, borderRadius: 12, padding: "10px 16px", background: loading ? "#475569" : "#34d399", color: "#0f172a", fontWeight: 700, cursor: loading ? "wait" : "pointer" }}
              >
                {loading ? "Running..." : "Run Codex Analysis"}
              </button>
            </div>
            {error ? <p style={{ color: "#fca5a5", marginBottom: 0 }}>{error}</p> : null}
          </section>
          <section style={panelStyle}>
            <h3 style={{ marginTop: 0 }}>Result</h3>
            {result ? (
              <div style={{ display: "grid", gap: 12 }}>
                <div style={{ color: "#94a3b8", fontSize: 13 }}>
                  Session {result.session_id} | Exit {result.exit_code} | Workdir {result.workdir}
                </div>
                <div style={{ whiteSpace: "pre-wrap", padding: 14, borderRadius: 14, background: "rgba(30, 41, 59, 0.75)", lineHeight: 1.5 }}>
                  {result.output}
                </div>
              </div>
            ) : (
              <div style={{ color: "#94a3b8" }}>{loading ? "Waiting for adapter response..." : "No analysis yet."}</div>
            )}
          </section>
        </div>
      </div>
    </section>
  );
}
