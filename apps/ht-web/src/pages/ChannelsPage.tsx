import { useState } from "react";
import {
  getMessagingPlatforms,
  testMessagingPlatform,
  updateMessagingPlatform,
  type MessagingPlatform,
  type MessagingPlatformTestResult,
} from "@/api/channels";
import { ApiError } from "@/api/client";
import { ManagementPage, ResourceView, useResource } from "@/components/PageScaffold";

// Messaging platforms (gateway channels): list each platform with its
// enabled/connection state, a credential editor built from the platform's
// env_vars, and a Test button.
//
// DEFERRED: the Telegram / WhatsApp guided onboarding wizards
// (/api/messaging/{telegram,whatsapp}/onboarding/*) are NOT ported here — they
// are multi-step flows that poll a session/state endpoint between prompts. This
// page only exposes direct enable/config/test. The onboarding entry points can
// be added later as a separate wizard component.
export default function ChannelsPage() {
  const platforms = useResource(getMessagingPlatforms);

  return (
    <ManagementPage
      title="Channels"
      actions={
        <button type="button" className="ht-btn ht-btn--sm" onClick={platforms.reload}>
          Refresh
        </button>
      }
    >
      <ResourceView resource={platforms} empty={(d) => d.platforms.length === 0}>
        {(d) => (
          <div className="ht-cards">
            {d.platforms.map((p) => (
              <PlatformCard key={p.id} platform={p} onChanged={platforms.reload} />
            ))}
          </div>
        )}
      </ResourceView>
    </ManagementPage>
  );
}

function stateChipClass(state: string): string {
  if (state === "connected") return "ht-chip ht-chip--ok";
  if (state === "disabled" || state === "not_configured") return "ht-chip";
  // pending_restart / disconnected / startup_failed / fatal → warn
  return "ht-chip ht-chip--warn";
}

function PlatformCard({
  platform,
  onChanged,
}: {
  platform: MessagingPlatform;
  onChanged: () => void;
}) {
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [test, setTest] = useState<MessagingPlatformTestResult | null>(null);

  const guard = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      throw e;
    } finally {
      setBusy(false);
    }
  };

  const toggleEnabled = () =>
    void guard(async () => {
      await updateMessagingPlatform(platform.id, { enabled: !platform.enabled });
      onChanged();
    }).catch(() => {});

  const saveConfig = () =>
    void guard(async () => {
      const env: Record<string, string> = {};
      for (const [k, v] of Object.entries(edits)) {
        if (v.trim() !== "") env[k] = v;
      }
      if (Object.keys(env).length === 0) return;
      await updateMessagingPlatform(platform.id, { env });
      setEdits({});
      onChanged();
    }).catch(() => {});

  const runTest = () =>
    void guard(async () => {
      setTest(null);
      const res = await testMessagingPlatform(platform.id);
      setTest(res);
    }).catch(() => {});

  return (
    <section className="ht-card">
      <h2 className="ht-card__title">
        {platform.name}
        <span className={stateChipClass(platform.state)} style={{ marginLeft: "0.5rem" }}>
          {platform.state}
        </span>
      </h2>
      {platform.description && <p className="ht-muted ht-sm">{platform.description}</p>}

      <dl className="ht-kv">
        <dt>Enabled</dt>
        <dd>
          <label>
            <input
              type="checkbox"
              checked={platform.enabled}
              disabled={busy}
              onChange={toggleEnabled}
            />{" "}
            {platform.enabled ? "on" : "off"}
          </label>
        </dd>
        <dt>Configured</dt>
        <dd>
          {platform.configured ? (
            <span className="ht-ok">yes</span>
          ) : (
            <span className="ht-dim">no</span>
          )}
        </dd>
        {platform.error_message && (
          <>
            <dt>Error</dt>
            <dd>
              <span className="ht-warn">{platform.error_message}</span>
            </dd>
          </>
        )}
        {platform.home_channel && (
          <>
            <dt>Home channel</dt>
            <dd>{platform.home_channel.name || platform.home_channel.chat_id}</dd>
          </>
        )}
      </dl>

      {platform.env_vars.length > 0 && (
        <table className="ht-table">
          <thead>
            <tr>
              <th>Credential</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {platform.env_vars.map((ev) => (
              <tr key={ev.key}>
                <td>
                  <code>{ev.key}</code>
                  {ev.required && <span className="ht-warn ht-sm"> *</span>}
                  {ev.description && (
                    <div className="ht-muted ht-sm">{ev.description}</div>
                  )}
                </td>
                <td>
                  <input
                    className="ht-input ht-input--sm"
                    type={ev.is_password ? "password" : "text"}
                    value={edits[ev.key] ?? ""}
                    placeholder={
                      ev.is_set ? (ev.redacted_value ?? "••••") : "not set"
                    }
                    onChange={(e) =>
                      setEdits((prev) => ({ ...prev, [ev.key]: e.target.value }))
                    }
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {error && <div className="ht-error-inline">{error}</div>}
      {test && (
        <div className={test.ok ? "ht-ok" : "ht-warn"}>
          Test: {test.state} — {test.message}
        </div>
      )}

      <div className="ht-row-actions">
        <button
          type="button"
          className="ht-btn ht-btn--sm"
          disabled={busy || Object.keys(edits).length === 0}
          onClick={saveConfig}
        >
          Save config
        </button>
        <button
          type="button"
          className="ht-btn ht-btn--sm ht-btn--ghost"
          disabled={busy}
          onClick={runTest}
        >
          Test
        </button>
        {platform.docs_url && (
          <a
            className="ht-btn ht-btn--sm ht-btn--ghost"
            href={platform.docs_url}
            target="_blank"
            rel="noreferrer"
          >
            Docs
          </a>
        )}
      </div>
    </section>
  );
}
