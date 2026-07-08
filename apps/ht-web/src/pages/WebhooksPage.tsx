import { useState } from "react";
import {
  createWebhook,
  deleteWebhook,
  enableWebhooks,
  getWebhooks,
  setWebhookEnabled,
  type WebhookCreate,
  type WebhookRoute,
  type WebhooksResponse,
} from "@/api/webhooks";
import { ManagementPage, ResourceView, useResource } from "@/components/PageScaffold";

export default function WebhooksPage() {
  const webhooks = useResource(getWebhooks);
  const [creating, setCreating] = useState(false);
  const [newSecret, setNewSecret] = useState<{ name: string; secret: string } | null>(null);

  return (
    <ManagementPage
      title="Webhooks"
      actions={
        <button type="button" className="ht-btn ht-btn--sm ht-btn--ghost" onClick={webhooks.reload}>
          Refresh
        </button>
      }
    >
      {newSecret && (
        <div className="ht-card" style={{ borderColor: "var(--ht-ok, #2e7d32)" }}>
          <div className="ht-card__title">Secret for "{newSecret.name}"</div>
          <p className="ht-warn ht-sm">
            Copy this now — it will not be shown again.
          </p>
          <code>{newSecret.secret}</code>
          <div className="ht-row-actions">
            <button
              type="button"
              className="ht-btn ht-btn--sm ht-btn--ghost"
              onClick={() => setNewSecret(null)}
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      <ResourceView resource={webhooks}>
        {(d) => (
          <>
            <WebhookStatus data={d} onChanged={webhooks.reload} />

            <div className="ht-card">
              <div className="ht-card__title">Routes</div>
              {d.subscriptions.length === 0 ? (
                <p className="ht-muted">No webhook routes.</p>
              ) : (
                <table className="ht-table">
                  <thead>
                    <tr>
                      <th>Name / Path</th>
                      <th>Target</th>
                      <th>Enabled</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {d.subscriptions.map((r) => (
                      <WebhookRow key={r.name} route={r} onChanged={webhooks.reload} />
                    ))}
                  </tbody>
                </table>
              )}
              <div className="ht-row-actions">
                <button
                  type="button"
                  className="ht-btn ht-btn--sm"
                  onClick={() => setCreating((v) => !v)}
                >
                  {creating ? "Close" : "New route"}
                </button>
              </div>
              {creating && (
                <CreateForm
                  onCreated={(name, secret) => {
                    setCreating(false);
                    setNewSecret({ name, secret });
                    webhooks.reload();
                  }}
                />
              )}
            </div>
          </>
        )}
      </ResourceView>
    </ManagementPage>
  );
}

function WebhookStatus({
  data,
  onChanged,
}: {
  data: WebhooksResponse;
  onChanged: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const enable = async () => {
    setBusy(true);
    setNote(null);
    try {
      const res = await enableWebhooks();
      if (res.needs_restart) {
        setNote(
          res.restart_started
            ? "Enabled — gateway restart started."
            : "Enabled — a gateway restart is required.",
        );
      }
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="ht-card">
      <dl className="ht-kv">
        <dt>Status</dt>
        <dd>
          {data.enabled ? (
            <span className="ht-chip ht-chip--ok">enabled</span>
          ) : (
            <span className="ht-chip ht-chip--warn">disabled</span>
          )}
        </dd>
        <dt>Base URL</dt>
        <dd>
          <code>{data.base_url || "—"}</code>
        </dd>
      </dl>
      {!data.enabled && (
        <div className="ht-row-actions">
          <button type="button" className="ht-btn ht-btn--sm" disabled={busy} onClick={enable}>
            {busy ? "Enabling…" : "Enable webhooks"}
          </button>
        </div>
      )}
      {note && <div className="ht-warn ht-sm">{note}</div>}
    </div>
  );
}

function WebhookRow({ route, onChanged }: { route: WebhookRoute; onChanged: () => void }) {
  const [busy, setBusy] = useState(false);

  const toggle = async () => {
    setBusy(true);
    try {
      await setWebhookEnabled(route.name, !route.enabled);
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    setBusy(true);
    try {
      await deleteWebhook(route.name);
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  return (
    <tr>
      <td>
        <code>{route.name}</code>
        {route.url && <div className="ht-muted ht-sm">{route.url}</div>}
      </td>
      <td>
        <span className="ht-sm">{route.deliver || "—"}</span>
      </td>
      <td>
        {route.enabled ? (
          <span className="ht-chip ht-chip--ok">on</span>
        ) : (
          <span className="ht-chip ht-chip--warn">off</span>
        )}
      </td>
      <td className="ht-row-actions">
        <button type="button" className="ht-btn ht-btn--sm" disabled={busy} onClick={toggle}>
          {route.enabled ? "Disable" : "Enable"}
        </button>
        <button
          type="button"
          className="ht-btn ht-btn--sm ht-btn--stop"
          disabled={busy}
          onClick={remove}
        >
          Delete
        </button>
      </td>
    </tr>
  );
}

function CreateForm({ onCreated }: { onCreated: (name: string, secret: string) => void }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [prompt, setPrompt] = useState("");
  const [deliver, setDeliver] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      const body: WebhookCreate = { name: name.trim() };
      if (description.trim()) body.description = description.trim();
      if (prompt.trim()) body.prompt = prompt.trim();
      if (deliver.trim()) body.deliver = deliver.trim();
      const res = await createWebhook(body);
      onCreated(res.name, res.secret);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <dl className="ht-kv">
      <dt>Name</dt>
      <dd>
        <input
          className="ht-input ht-input--sm"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="github-push"
        />
      </dd>
      <dt>Description</dt>
      <dd>
        <input
          className="ht-input ht-input--sm"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="optional"
        />
      </dd>
      <dt>Prompt</dt>
      <dd>
        <input
          className="ht-input ht-input--sm"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="optional prompt for the agent"
        />
      </dd>
      <dt>Deliver to</dt>
      <dd>
        <input
          className="ht-input ht-input--sm"
          value={deliver}
          onChange={(e) => setDeliver(e.target.value)}
          placeholder="optional target"
        />
      </dd>
      <dt />
      <dd>
        {error && <div className="ht-error-inline">{error}</div>}
        <div className="ht-row-actions">
          <button
            type="button"
            className="ht-btn ht-btn--sm"
            disabled={busy || !name.trim()}
            onClick={submit}
          >
            {busy ? "Creating…" : "Create route"}
          </button>
        </div>
      </dd>
    </dl>
  );
}
