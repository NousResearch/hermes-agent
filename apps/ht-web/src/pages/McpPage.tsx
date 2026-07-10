import { useState } from "react";
import {
  addMcpServer,
  getMcpCatalog,
  getMcpServers,
  installMcpCatalogEntry,
  removeMcpServer,
  setMcpServerEnabled,
  testMcpServer,
  type McpCatalogEntry,
  type McpServer,
  type McpServerCreate,
  type McpTestResult,
} from "@/api/mcp";
import { ManagementPage, ResourceView, useResource } from "@/components/PageScaffold";

export default function McpPage() {
  const servers = useResource(getMcpServers);
  const [adding, setAdding] = useState(false);
  const [showCatalog, setShowCatalog] = useState(false);

  return (
    <ManagementPage
      title="MCP Servers"
      actions={
        <>
          <button type="button" className="ht-btn ht-btn--sm" onClick={() => setAdding((v) => !v)}>
            {adding ? "Close" : "Add server"}
          </button>
          <button type="button" className="ht-btn ht-btn--sm ht-btn--ghost" onClick={servers.reload}>
            Refresh
          </button>
        </>
      }
    >
      {adding && (
        <AddServerForm
          onAdded={() => {
            setAdding(false);
            servers.reload();
          }}
        />
      )}

      <ResourceView resource={servers} empty={(d) => d.servers.length === 0}>
        {(d) => (
          <div className="ht-card">
            <table className="ht-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Transport</th>
                  <th>Status</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {d.servers.map((s) => (
                  <ServerRow key={s.name} server={s} onChanged={servers.reload} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </ResourceView>

      <div className="ht-card">
        <button
          type="button"
          className="ht-btn ht-btn--sm ht-btn--ghost"
          onClick={() => setShowCatalog((v) => !v)}
          aria-expanded={showCatalog}
        >
          {showCatalog ? "Hide catalog" : "Browse catalog"}
        </button>
        {showCatalog && <CatalogSection onInstalled={servers.reload} />}
      </div>
    </ManagementPage>
  );
}

function ServerRow({ server, onChanged }: { server: McpServer; onChanged: () => void }) {
  const [busy, setBusy] = useState(false);
  const [test, setTest] = useState<McpTestResult | null>(null);
  const [testing, setTesting] = useState(false);

  const toggle = async () => {
    setBusy(true);
    try {
      await setMcpServerEnabled(server.name, !server.enabled);
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    setBusy(true);
    try {
      await removeMcpServer(server.name);
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  const runTest = async () => {
    setTesting(true);
    setTest(null);
    try {
      setTest(await testMcpServer(server.name));
    } catch (e) {
      setTest({ ok: false, error: String(e), tools: [] });
    } finally {
      setTesting(false);
    }
  };

  const toolCount = server.tools?.length ?? 0;

  return (
    <>
      <tr>
        <td>
          <code>{server.name}</code>
          {server.url && <div className="ht-muted ht-sm">{server.url}</div>}
          {server.command && (
            <div className="ht-muted ht-sm">
              {server.command} {server.args.join(" ")}
            </div>
          )}
        </td>
        <td>
          <span className="ht-chip">{server.transport}</span>
        </td>
        <td>
          {!server.enabled ? (
            <span className="ht-chip ht-chip--warn">disabled</span>
          ) : server.tools ? (
            <span className="ht-chip ht-chip--ok">
              connected · {toolCount} tool{toolCount === 1 ? "" : "s"}
            </span>
          ) : (
            <span className="ht-chip">enabled</span>
          )}
        </td>
        <td className="ht-row-actions">
          <button
            type="button"
            className="ht-btn ht-btn--sm ht-btn--ghost"
            disabled={testing}
            onClick={runTest}
          >
            {testing ? "Testing…" : "Test"}
          </button>
          <button type="button" className="ht-btn ht-btn--sm" disabled={busy} onClick={toggle}>
            {server.enabled ? "Disable" : "Enable"}
          </button>
          <button
            type="button"
            className="ht-btn ht-btn--sm ht-btn--stop"
            disabled={busy}
            onClick={remove}
          >
            Remove
          </button>
        </td>
      </tr>
      {test && (
        <tr>
          <td colSpan={4}>
            {test.ok ? (
              <div className="ht-ok ht-sm">
                Connected — {test.tools.length} tool{test.tools.length === 1 ? "" : "s"}
                <div className="ht-chips">
                  {test.tools.map((t) => (
                    <span key={t.name} className="ht-chip" title={t.description}>
                      {t.name}
                    </span>
                  ))}
                </div>
              </div>
            ) : (
              <div className="ht-error-inline">Test failed: {test.error ?? "unknown error"}</div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

function AddServerForm({ onAdded }: { onAdded: () => void }) {
  const [transport, setTransport] = useState<"http" | "stdio">("http");
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [command, setCommand] = useState("");
  const [args, setArgs] = useState("");
  const [auth, setAuth] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      const body: McpServerCreate = { name: name.trim() };
      if (transport === "http") {
        body.url = url.trim();
      } else {
        body.command = command.trim();
        const parts = args.trim().split(/\s+/).filter(Boolean);
        if (parts.length) body.args = parts;
      }
      if (auth.trim()) body.auth = auth.trim();
      await addMcpServer(body);
      onAdded();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const valid = name.trim() && (transport === "http" ? url.trim() : command.trim());

  return (
    <div className="ht-card">
      <div className="ht-card__title">Add MCP server</div>
      <dl className="ht-kv">
        <dt>Name</dt>
        <dd>
          <input
            className="ht-input ht-input--sm"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="my-server"
          />
        </dd>
        <dt>Transport</dt>
        <dd>
          <select
            className="ht-select"
            value={transport}
            onChange={(e) => setTransport(e.target.value as "http" | "stdio")}
          >
            <option value="http">http</option>
            <option value="stdio">stdio</option>
          </select>
        </dd>
        {transport === "http" ? (
          <>
            <dt>URL</dt>
            <dd>
              <input
                className="ht-input ht-input--sm"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com/mcp"
              />
            </dd>
          </>
        ) : (
          <>
            <dt>Command</dt>
            <dd>
              <input
                className="ht-input ht-input--sm"
                value={command}
                onChange={(e) => setCommand(e.target.value)}
                placeholder="npx"
              />
            </dd>
            <dt>Args</dt>
            <dd>
              <input
                className="ht-input ht-input--sm"
                value={args}
                onChange={(e) => setArgs(e.target.value)}
                placeholder="-y @scope/server"
              />
            </dd>
          </>
        )}
        <dt>Auth</dt>
        <dd>
          <input
            className="ht-input ht-input--sm"
            value={auth}
            onChange={(e) => setAuth(e.target.value)}
            placeholder="optional bearer token"
          />
        </dd>
      </dl>
      {error && <div className="ht-error-inline">{error}</div>}
      <div className="ht-row-actions">
        <button type="button" className="ht-btn ht-btn--sm" disabled={busy || !valid} onClick={submit}>
          {busy ? "Adding…" : "Add"}
        </button>
      </div>
    </div>
  );
}

function CatalogSection({ onInstalled }: { onInstalled: () => void }) {
  const catalog = useResource(getMcpCatalog);
  return (
    <ResourceView resource={catalog} empty={(d) => d.entries.length === 0}>
      {(d) => (
        <table className="ht-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Description</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {d.entries.map((entry) => (
              <CatalogRow key={entry.name} entry={entry} onInstalled={onInstalled} />
            ))}
          </tbody>
        </table>
      )}
    </ResourceView>
  );
}

function CatalogRow({
  entry,
  onInstalled,
}: {
  entry: McpCatalogEntry;
  onInstalled: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const install = async () => {
    setBusy(true);
    setError(null);
    try {
      await installMcpCatalogEntry(entry.name);
      onInstalled();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <tr>
      <td>
        <code>{entry.name}</code>
        <div className="ht-chips">
          <span className="ht-chip">{entry.transport}</span>
          <span className="ht-chip">{entry.auth_type}</span>
        </div>
      </td>
      <td>
        <span className="ht-sm">{entry.description}</span>
        {error && <div className="ht-error-inline">{error}</div>}
      </td>
      <td className="ht-row-actions">
        {entry.installed ? (
          <span className="ht-dim ht-sm">installed</span>
        ) : (
          <button type="button" className="ht-btn ht-btn--sm" disabled={busy} onClick={install}>
            {busy ? "Installing…" : "Install"}
          </button>
        )}
      </td>
    </tr>
  );
}
