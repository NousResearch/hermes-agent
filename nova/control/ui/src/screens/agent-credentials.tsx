/* Setting an agent's credentials.
 *
 * This is the one screen in the Control Centre that sends a secret, and its design follows
 * from that rather than from convenience.
 *
 * **Values only travel one way.** The read says whether a variable is set, never what it
 * is, so a field always starts empty — it is not showing you a masked version of the stored
 * value, because it does not have one. Clearing the field and saving removes the variable;
 * leaving it empty does nothing.
 *
 * **Only declared names appear.** The list comes from the tenant's own declaration — the
 * channels that grant this agent and its model — and the backend refuses anything outside
 * it. `.env` is loaded into the environment of the process that runs the agent, so a form
 * that accepted arbitrary names would be a way to set PYTHONPATH.
 *
 * **Nothing is remembered here.** No draft is kept after a save, nothing goes to
 * localStorage, and the inputs are `type="password"` with autocomplete off.
 */

import * as React from "react";
import { AlertTriangle, Check, ExternalLink, KeyRound, Loader2, X } from "lucide-react";

import { GlassPanel, SectionHeader, StatusPill } from "@/components/glass";
import { post } from "@/lib/api";
import { usePanel } from "@/lib/hooks";

type Credential = {
  name: string; source: string; required: boolean; label: string;
  description: string; url: string; secret: boolean; set: boolean;
};
type Payload = { agent_id: string; credentials: Credential[]; writable: boolean };

type Result =
  | { kind: "idle" }
  | { kind: "saving" }
  | { kind: "done"; changed: string[]; detail: string }
  | { kind: "error"; message: string };

export function AgentCredentials({ agentId }: { agentId: string }) {
  const [nonce, setNonce] = React.useState(0);
  const loaded = usePanel<Payload>(
    `/agents/${encodeURIComponent(agentId)}/credentials`, 60000, nonce,
  );
  const [draft, setDraft] = React.useState<Record<string, string>>({});
  const [clearing, setClearing] = React.useState<Set<string>>(new Set());
  const [result, setResult] = React.useState<Result>({ kind: "idle" });

  const pendingSets = Object.entries(draft).filter(([, v]) => v.trim());
  const dirty = pendingSets.length > 0 || clearing.size > 0;
  const busy = result.kind === "saving";

  async function save() {
    if (busy || !dirty) return;
    setResult({ kind: "saving" });
    const values: Record<string, string | null> = {};
    for (const [name, value] of pendingSets) values[name] = value;
    for (const name of clearing) values[name] = null;
    try {
      const response: any = await post(`/agents/${encodeURIComponent(agentId)}/credentials`, { values });
      setResult({
        kind: "done",
        changed: response?.changed ?? [],
        detail: String(response?.detail ?? ""),
      });
      // Dropped immediately. A secret that lingers in component state is a secret in a
      // heap snapshot and in the next render's props.
      setDraft({});
      setClearing(new Set());
      setNonce((n) => n + 1);
    } catch (cause) {
      setResult({
        kind: "error",
        message: cause instanceof Error ? cause.message : "the request did not complete",
      });
    }
  }

  if (loaded.state === "loading") {
    return <GlassPanel className="p-5"><p className="text-ink-faint text-[13px]">reading…</p></GlassPanel>;
  }
  if (loaded.state === "forbidden") {
    return (
      <GlassPanel className="p-5">
        <p className="text-ink-muted text-[13px]">
          Credentials are not visible to your role.
        </p>
      </GlassPanel>
    );
  }
  if (loaded.state === "error") {
    return (
      <GlassPanel className="p-5">
        <p className="text-blocked text-[13px]">
          <b>Credentials could not be read.</b> {loaded.message}
        </p>
      </GlassPanel>
    );
  }

  const credentials = loaded.data.credentials;
  const grouped = new Map<string, Credential[]>();
  for (const c of credentials) {
    grouped.set(c.source, [...(grouped.get(c.source) ?? []), c]);
  }

  return (
    <GlassPanel className="p-5">
      <SectionHeader
        icon={KeyRound} title="Credentials"
        detail="What this agent needs in order to run. Set here, stored in its own profile."
        action={
          <span className="text-ink-faint text-[11.5px]">
            {credentials.filter((c) => c.set).length} of {credentials.length} set
          </span>
        }
      />

      {credentials.length === 0 ? (
        <p className="text-ink-muted text-[12.5px] leading-relaxed">
          This agent declares no credentials. Grant it a channel, or give it a model with a
          credential variable, and the fields appear here.
        </p>
      ) : (
        <div className="space-y-5">
          {[...grouped.entries()].map(([source, rows]) => (
            <div key={source}>
              <p className="text-ink-faint mb-2 text-[11px] font-medium tracking-wide uppercase">
                {source === "model" ? "This agent's model"
                  : source === "deployment" ? "Deployment default model"
                  : `Channel · ${source}`}
              </p>
              <div className="space-y-3">
                {rows.map((c) => (
                  <Row
                    key={c.name} credential={c}
                    value={draft[c.name] ?? ""}
                    clearing={clearing.has(c.name)}
                    disabled={busy}
                    onChange={(v) => {
                      setDraft((d) => ({ ...d, [c.name]: v }));
                      if (result.kind !== "idle") setResult({ kind: "idle" });
                    }}
                    onToggleClear={() =>
                      setClearing((s) => {
                        const next = new Set(s);
                        next.has(c.name) ? next.delete(c.name) : next.add(c.name);
                        return next;
                      })
                    }
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {credentials.length ? (
        <>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button
              type="button" onClick={save} disabled={!dirty || busy}
              className="glass-solid text-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium disabled:opacity-40"
            >
              {busy ? <Loader2 className="size-3.5 animate-spin" /> : <KeyRound className="size-3.5" />}
              {busy ? "Saving credentials…" : "Save credentials"}
            </button>
            {dirty && !busy ? (
              <span className="text-waiting text-[11.5px]">
                {pendingSets.length ? `${pendingSets.length} to set` : ""}
                {pendingSets.length && clearing.size ? ", " : ""}
                {clearing.size ? `${clearing.size} to clear` : ""}
              </span>
            ) : null}
          </div>

          {result.kind === "error" ? (
            <div className="border-blocked/30 bg-blocked/5 mt-3 rounded-lg border p-3">
              <p className="text-blocked flex items-start gap-1.5 text-[12.5px]">
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                <span><b>Nothing was stored.</b> {result.message}</span>
              </p>
            </div>
          ) : null}
          {result.kind === "done" ? (
            <div className="border-glass-border mt-3 rounded-lg border p-3">
              <p className="text-running flex items-center gap-1.5 text-[12.5px]">
                <Check className="size-3.5" />
                {result.changed.length
                  ? `Stored: ${result.changed.join(", ")}`
                  : result.detail || "Nothing changed."}
              </p>
              <p className="text-ink-faint mt-1 text-[11.5px]">
                Written to this agent's own profile, owner-readable only. A running agent
                reads its environment at start, so it picks this up on its next run.
              </p>
            </div>
          ) : null}

          <p className="text-ink-faint mt-4 text-[11.5px] leading-relaxed">
            A stored value is never shown again — the control plane can say whether a
            credential is set and nothing more. To replace one, type the new value.
          </p>
        </>
      ) : null}
    </GlassPanel>
  );
}

function Row({
  credential, value, clearing, disabled, onChange, onToggleClear,
}: {
  credential: Credential; value: string; clearing: boolean; disabled: boolean;
  onChange: (v: string) => void; onToggleClear: () => void;
}) {
  const id = `cred-${credential.name}`;
  return (
    <div className="border-glass-border rounded-lg border p-3">
      <div className="flex flex-wrap items-center gap-2">
        <label htmlFor={id} className="text-ink font-mono text-[12px] font-medium">
          {credential.name}
        </label>
        {credential.required ? (
          <span className="text-waiting text-[11px]">required</span>
        ) : (
          <span className="text-ink-faint text-[11px]">optional</span>
        )}
        <StatusPill state={credential.set ? "running" : "waiting"}>
          {credential.set ? "Set" : "Not set"}
        </StatusPill>
        {credential.url ? (
          <a
            href={credential.url} target="_blank" rel="noreferrer noopener"
            className="text-ink-faint hover:text-ink ml-auto inline-flex items-center gap-1 text-[11.5px]"
          >
            Where to get one <ExternalLink className="size-3" />
          </a>
        ) : null}
      </div>

      {credential.description ? (
        <p className="text-ink-faint mt-1.5 text-[11.5px] leading-relaxed">{credential.description}</p>
      ) : null}

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <input
          id={id}
          type={credential.secret ? "password" : "text"}
          value={clearing ? "" : value}
          disabled={disabled || clearing}
          autoComplete="off"
          spellCheck={false}
          placeholder={credential.set ? "Type a new value to replace it" : "Not set"}
          onChange={(e) => onChange(e.target.value)}
          className="border-glass-border bg-glass text-ink focus-visible:ring-info/50 min-w-0 flex-1 rounded-lg border px-3 py-2 font-mono text-[12.5px] outline-none focus-visible:ring-2 disabled:opacity-50"
        />
        {credential.set ? (
          <button
            type="button" onClick={onToggleClear} disabled={disabled}
            className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-[11.5px] transition-colors ${
              clearing ? "border-blocked/40 bg-blocked/10 text-blocked" : "border-glass-border text-ink-muted hover:text-ink"
            }`}
          >
            <X className="size-3" /> {clearing ? "Will be removed" : "Remove"}
          </button>
        ) : null}
      </div>
    </div>
  );
}
