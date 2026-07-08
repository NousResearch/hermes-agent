import { useCallback, useEffect, useState } from "react";
import {
  disableAgentPlugin,
  enableAgentPlugin,
  getPlugins,
  getPluginsHub,
  installAgentPlugin,
  removeAgentPlugin,
  rescanPlugins,
  updateAgentPlugin,
  type HubAgentPluginRow,
  type PluginManifestResponse,
  type PluginsHubResponse,
} from "@/api/plugins";
import { ManagementPage, ResourceView, useResource } from "@/components/PageScaffold";
import { ApiError, pollAction } from "@/api/client";

type Note = { kind: "ok" | "err"; msg: string } | null;

function errMsg(e: unknown): string {
  return e instanceof ApiError ? e.message : String(e);
}

export default function PluginsPage() {
  const plugins = useResource(getPlugins);
  const [note, setNote] = useState<Note>(null);
  const [rescanning, setRescanning] = useState(false);

  const rescan = async () => {
    setRescanning(true);
    setNote(null);
    try {
      const res = await rescanPlugins();
      setNote({ kind: "ok", msg: `Rescanned dashboard plugins (${res.count}).` });
      plugins.reload();
    } catch (e) {
      setNote({ kind: "err", msg: errMsg(e) });
    } finally {
      setRescanning(false);
    }
  };

  return (
    <ManagementPage
      title="Plugins"
      actions={
        <button
          type="button"
          className="ht-btn ht-btn--sm"
          disabled={rescanning}
          onClick={() => void rescan()}
        >
          {rescanning ? "Rescanning…" : "Rescan"}
        </button>
      }
    >
      {note && (
        <p
          className={note.kind === "ok" ? "ht-ok" : "ht-error-inline"}
          style={{ margin: "0 0 12px" }}
        >
          {note.msg}
        </p>
      )}

      <section className="ht-card">
        <h2 className="ht-card__title">Dashboard plugins</h2>
        <ResourceView resource={plugins} empty={(d) => d.length === 0}>
          {(list) => <DashboardPluginList plugins={list} />}
        </ResourceView>
      </section>

      <HubSection />
    </ManagementPage>
  );
}

// ── Baseline dashboard-plugin manifests (always available) ──────────
function DashboardPluginList({ plugins }: { plugins: PluginManifestResponse[] }) {
  return (
    <table className="ht-table">
      <thead>
        <tr>
          <th>Plugin</th>
          <th>Version</th>
          <th>Source</th>
        </tr>
      </thead>
      <tbody>
        {plugins.map((p) => (
          <tr key={p.name}>
            <td>
              <code>{p.label || p.name}</code>
              {p.description && <div className="ht-muted ht-sm">{p.description}</div>}
            </td>
            <td>{p.version}</td>
            <td>
              <span className="ht-chip">{p.source}</span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ── Agent-plugin hub (collapsible + guarded against hub outages) ─────
function HubSection() {
  const [open, setOpen] = useState(false);
  const [hub, setHub] = useState<PluginsHubResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [hubError, setHubError] = useState<string | null>(null);
  const [note, setNote] = useState<Note>(null);

  const [identifier, setIdentifier] = useState("");
  const [installing, setInstalling] = useState(false);
  const [progress, setProgress] = useState<string[]>([]);

  const load = useCallback(async () => {
    setLoading(true);
    setHubError(null);
    try {
      const res = await getPluginsHub();
      setHub(res);
    } catch (e) {
      // A hub failure stays contained — the dashboard-plugin list is unaffected.
      setHubError(errMsg(e));
      setHub(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open && hub === null && !hubError) void load();
  }, [open, hub, hubError, load]);

  const install = async () => {
    const id = identifier.trim();
    if (!id) return;
    setInstalling(true);
    setProgress([]);
    setNote(null);
    try {
      const res = await installAgentPlugin({ identifier: id, enable: true });
      if (res.error) {
        setNote({ kind: "err", msg: res.error });
        return;
      }
      // Some backends run the install as a background action (name/pid set);
      // poll it to completion. Otherwise the POST already finished the work.
      if (res.name && res.pid != null) {
        const status = await pollAction(res.name, { onProgress: (s) => setProgress(s.lines) });
        if (status.exit_code !== 0) {
          setNote({ kind: "err", msg: `Install failed (exit ${status.exit_code}).` });
          return;
        }
      }
      const warn = (res.warnings ?? []).concat(
        (res.missing_env ?? []).length > 0 ? [`missing env: ${res.missing_env!.join(", ")}`] : [],
      );
      setNote({
        kind: "ok",
        msg: `Installed ${res.plugin_name ?? id}.${warn.length ? ` ${warn.join(" ")}` : ""}`,
      });
      setIdentifier("");
      await load();
    } catch (e) {
      setNote({ kind: "err", msg: errMsg(e) });
    } finally {
      setInstalling(false);
    }
  };

  return (
    <section className="ht-card">
      <button
        type="button"
        className="ht-provider__head"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="ht-provider__name">Agent plugin hub</span>
        <span className="ht-muted ht-sm">install, enable, update &amp; remove agent plugins</span>
        <span aria-hidden>{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <div>
          <div className="ht-row-actions" style={{ margin: "12px 0" }}>
            <input
              className="ht-input ht-input--sm"
              placeholder="Plugin identifier or git URL…"
              value={identifier}
              onChange={(e) => setIdentifier(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void install();
              }}
              aria-label="Plugin identifier"
            />
            <button
              type="button"
              className="ht-btn ht-btn--sm"
              disabled={installing || !identifier.trim()}
              onClick={() => void install()}
            >
              {installing ? "Installing…" : "Install"}
            </button>
          </div>

          {hubError && <p className="ht-error-inline">Hub unavailable: {hubError}</p>}
          {note && (
            <p className={note.kind === "ok" ? "ht-ok" : "ht-error-inline"}>{note.msg}</p>
          )}
          {installing && progress.length > 0 && (
            <pre className="ht-sm ht-muted" style={{ maxHeight: 160, overflow: "auto" }}>
              {progress.join("\n")}
            </pre>
          )}

          {loading && hub === null && <p className="ht-muted">Loading…</p>}
          {hub && hub.plugins.length === 0 && !loading && (
            <p className="ht-muted">No agent plugins installed.</p>
          )}
          {hub && hub.plugins.length > 0 && (
            <ul className="ht-model-list">
              {hub.plugins.map((p) => (
                <PluginRow
                  key={p.name}
                  plugin={p}
                  onChanged={load}
                  onNote={setNote}
                />
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}

function PluginRow({
  plugin,
  onChanged,
  onNote,
}: {
  plugin: HubAgentPluginRow;
  onChanged: () => void | Promise<void>;
  onNote: (n: Note) => void;
}) {
  const [busy, setBusy] = useState(false);
  const enabled = plugin.runtime_status === "enabled";

  const run = async (fn: () => Promise<unknown>, ok: string) => {
    setBusy(true);
    onNote(null);
    try {
      await fn();
      onNote({ kind: "ok", msg: ok });
      await onChanged();
    } catch (e) {
      onNote({ kind: "err", msg: errMsg(e) });
    } finally {
      setBusy(false);
    }
  };

  const toggle = () =>
    run(
      () => (enabled ? disableAgentPlugin(plugin.name) : enableAgentPlugin(plugin.name)),
      `${plugin.name} ${enabled ? "disabled" : "enabled"}.`,
    );

  const update = () =>
    run(async () => {
      const res = await updateAgentPlugin(plugin.name);
      if (res.error) throw new ApiError(res.error, 0);
    }, `${plugin.name} updated.`);

  const remove = () => run(() => removeAgentPlugin(plugin.name), `${plugin.name} removed.`);

  return (
    <li className="ht-model">
      <div style={{ flex: 1 }}>
        <code>{plugin.name}</code>
        {plugin.description && <div className="ht-muted ht-sm">{plugin.description}</div>}
        <div className="ht-chips">
          <span className="ht-chip">v{plugin.version}</span>
          <span className="ht-chip">{plugin.source}</span>
          {plugin.runtime_status === "inactive" && (
            <span className="ht-chip ht-chip--warn">inactive</span>
          )}
        </div>
      </div>
      <span className={enabled ? "ht-chip ht-chip--ok" : "ht-chip"}>
        {enabled ? "enabled" : "disabled"}
      </span>
      <div className="ht-row-actions">
        <button
          type="button"
          className={`ht-btn ht-btn--sm${enabled ? " ht-btn--ghost" : ""}`}
          disabled={busy}
          onClick={() => void toggle()}
        >
          {enabled ? "Disable" : "Enable"}
        </button>
        {plugin.can_update_git && (
          <button
            type="button"
            className="ht-btn ht-btn--sm ht-btn--ghost"
            disabled={busy}
            onClick={() => void update()}
          >
            Update
          </button>
        )}
        {plugin.can_remove && (
          <button
            type="button"
            className="ht-btn ht-btn--sm ht-btn--stop"
            disabled={busy}
            onClick={() => void remove()}
          >
            Remove
          </button>
        )}
      </div>
    </li>
  );
}
