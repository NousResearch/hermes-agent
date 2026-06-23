import { useEffect, useState, type CSSProperties } from "react";
import { getSettings, updateSettings, type SettingsRecord } from "../lib/controlLayer";

const sectionStyle: CSSProperties = {
  marginTop: 24,
  padding: 20,
  borderRadius: 18,
  background: "rgba(15, 23, 42, 0.92)",
  border: "1px solid rgba(148, 163, 184, 0.14)",
};

const inputStyle: CSSProperties = {
  width: "100%",
  marginTop: 8,
  padding: "12px 14px",
  borderRadius: 12,
  border: "1px solid rgba(148, 163, 184, 0.2)",
  background: "#020617",
  color: "#e2e8f0",
};

const defaults: SettingsRecord = {
  hermes_adapter_path: "adapters/hermes-adapter/main.py",
  codex_adapter_path: "adapters/codex-adapter/main.py",
  backend_port: 8647,
  frontend_port: 5173,
  codex_workdir: "D:\\Codex",
};

export default function Settings() {
  const [form, setForm] = useState<SettingsRecord>(defaults);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<string | null>(null);

  useEffect(() => {
    void getSettings()
      .then((data) => setForm(data))
      .catch((err) => setError(err instanceof Error ? err.message : "Unable to load settings"))
      .finally(() => setLoading(false));
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const saved = await updateSettings(form);
      setForm(saved);
      setSavedAt(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to save settings");
    } finally {
      setSaving(false);
    }
  };

  return (
    <section style={{ display: "grid", gap: 20 }}>
      <div>
        <h2 style={{ margin: 0, fontSize: "2rem" }}>Settings</h2>
        <p style={{ margin: "10px 0 0", color: "#94a3b8" }}>Persist adapter wiring, runtime ports, and the default Codex working directory for the operator UI.</p>
      </div>

      {error ? <div style={{ ...sectionStyle, color: "#fecaca", borderColor: "rgba(239, 68, 68, 0.45)" }}>{error}</div> : null}
      {savedAt ? <div style={{ ...sectionStyle, color: "#bbf7d0", borderColor: "rgba(34, 197, 94, 0.35)" }}>Saved at {savedAt}</div> : null}

      <section style={sectionStyle}>
        <h3 style={{ marginTop: 0 }}>Adapter Configuration</h3>
        <label>
          Hermes adapter path
          <input style={inputStyle} type="text" value={form.hermes_adapter_path} disabled={loading} onChange={(event) => setForm((current) => ({ ...current, hermes_adapter_path: event.target.value }))} />
        </label>
        <label style={{ display: "block", marginTop: 16 }}>
          Codex adapter path
          <input style={inputStyle} type="text" value={form.codex_adapter_path} disabled={loading} onChange={(event) => setForm((current) => ({ ...current, codex_adapter_path: event.target.value }))} />
        </label>
      </section>

      <section style={sectionStyle}>
        <h3 style={{ marginTop: 0 }}>Server</h3>
        <label>
          Backend port
          <input style={inputStyle} type="number" value={form.backend_port} disabled={loading} onChange={(event) => setForm((current) => ({ ...current, backend_port: Number(event.target.value) }))} />
        </label>
        <label style={{ display: "block", marginTop: 16 }}>
          Frontend port
          <input style={inputStyle} type="number" value={form.frontend_port} disabled={loading} onChange={(event) => setForm((current) => ({ ...current, frontend_port: Number(event.target.value) }))} />
        </label>
      </section>

      <section style={sectionStyle}>
        <h3 style={{ marginTop: 0 }}>Working Directory</h3>
        <label>
          Codex workdir
          <input style={inputStyle} type="text" value={form.codex_workdir} disabled={loading} onChange={(event) => setForm((current) => ({ ...current, codex_workdir: event.target.value }))} />
        </label>
      </section>

      <button
        type="button"
        onClick={() => void handleSave()}
        disabled={loading || saving}
        style={{ width: "fit-content", padding: "12px 18px", borderRadius: 12, border: "none", background: saving ? "#475569" : "#2563eb", color: "#eff6ff", cursor: saving ? "wait" : "pointer", fontWeight: 600 }}
      >
        {saving ? "Saving..." : "Save Settings"}
      </button>
    </section>
  );
}
