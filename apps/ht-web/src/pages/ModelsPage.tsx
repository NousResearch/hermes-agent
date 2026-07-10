import { useState } from "react";
import {
  getModelInfo,
  getModelOptions,
  setModelAssignment,
  type ModelOptionProvider,
} from "@/api/endpoints";
import { ManagementPage, ResourceView, useResource } from "@/components/PageScaffold";
import { ApiError } from "@/api/client";

export default function ModelsPage() {
  const info = useResource(getModelInfo);
  const options = useResource(getModelOptions);
  const [note, setNote] = useState<{ kind: "ok" | "err"; msg: string } | null>(null);
  const [saving, setSaving] = useState(false);

  const assign = async (provider: string, model: string) => {
    setSaving(true);
    setNote(null);
    try {
      const res = await setModelAssignment({ scope: "main", provider, model });
      if (res.error) {
        setNote({ kind: "err", msg: res.error });
      } else {
        setNote({ kind: "ok", msg: `Main model set to ${model} (${provider}).` });
        info.reload();
        options.reload();
      }
    } catch (e) {
      setNote({ kind: "err", msg: e instanceof ApiError ? e.message : String(e) });
    } finally {
      setSaving(false);
    }
  };

  return (
    <ManagementPage
      title="Models"
      actions={
        <button
          type="button"
          className="ht-btn ht-btn--sm"
          onClick={() => {
            info.reload();
            options.reload();
          }}
        >
          Refresh
        </button>
      }
    >
      <ResourceView resource={info}>
        {(m) => (
          <section className="ht-card">
            <h2 className="ht-card__title">Current main model</h2>
            <dl className="ht-kv">
              <dt>Model</dt>
              <dd>
                <code>{m.model}</code>
              </dd>
              <dt>Provider</dt>
              <dd>{m.provider}</dd>
              <dt>Context</dt>
              <dd>{m.effective_context_length.toLocaleString()} tokens</dd>
            </dl>
            <div className="ht-chips">
              {m.capabilities.supports_tools && <span className="ht-chip ht-chip--ok">tools</span>}
              {m.capabilities.supports_vision && <span className="ht-chip ht-chip--ok">vision</span>}
              {m.capabilities.supports_reasoning && (
                <span className="ht-chip ht-chip--ok">reasoning</span>
              )}
              {m.capabilities.model_family && (
                <span className="ht-chip">{m.capabilities.model_family}</span>
              )}
            </div>
          </section>
        )}
      </ResourceView>

      {note && (
        <p className={note.kind === "ok" ? "ht-ok" : "ht-error-inline"} style={{ margin: "12px 0" }}>
          {note.msg}
        </p>
      )}

      <ResourceView resource={options} empty={(d) => (d.providers ?? []).length === 0}>
        {(d) => (
          <div className="ht-providers">
            {(d.providers ?? []).map((p) => (
              <ProviderCard
                key={p.slug}
                provider={p}
                currentModel={d.model}
                disabled={saving}
                onPick={(model) => void assign(p.slug, model)}
              />
            ))}
          </div>
        )}
      </ResourceView>
    </ManagementPage>
  );
}

function ProviderCard({
  provider,
  currentModel,
  disabled,
  onPick,
}: {
  provider: ModelOptionProvider;
  currentModel?: string;
  disabled: boolean;
  onPick: (model: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const models = provider.models ?? [];

  return (
    <section className={`ht-card ht-provider${provider.is_current ? " is-current" : ""}`}>
      <button type="button" className="ht-provider__head" onClick={() => setOpen((v) => !v)}>
        <span className="ht-provider__name">{provider.name}</span>
        {!provider.authenticated && <span className="ht-chip ht-chip--warn">no key</span>}
        {provider.is_current && <span className="ht-chip ht-chip--ok">active</span>}
        <span className="ht-provider__count">
          {provider.total_models ?? models.length} models
        </span>
        <span aria-hidden>{open ? "▾" : "▸"}</span>
      </button>
      {provider.warning && <p className="ht-muted ht-sm">{provider.warning}</p>}
      {open && models.length > 0 && (
        <ul className="ht-model-list">
          {models.map((model) => {
            const active = provider.is_current && model === currentModel;
            return (
              <li key={model}>
                <button
                  type="button"
                  className={`ht-model${active ? " is-active" : ""}`}
                  disabled={disabled || active}
                  onClick={() => onPick(model)}
                >
                  <code>{model}</code>
                  {active && <span className="ht-ok ht-sm"> current</span>}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
