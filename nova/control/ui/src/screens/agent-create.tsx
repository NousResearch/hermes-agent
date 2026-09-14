/* Create Agent.
 *
 * One POST, at the end, after a review step. Not a wizard that writes as it goes: the
 * backend validates the *whole* bundle before committing, so a half-created agent is not a
 * state that can exist — and a form that pretended otherwise would have to invent a way to
 * roll back.
 *
 * Every option list comes from a read of the thing that owns it. Permissions are the
 * tenant's policy, toolsets the runtime's registry, corpora the knowledge catalogue. There
 * is no hard-coded list here, so a choice that disappears upstream disappears from the form
 * rather than failing at save time.
 *
 * Channels are shown and not selected. A grant is declared on the connection — Phase 9 made
 * that the single place it lives — so granting one here would be a second place, and the
 * two would disagree. The step says which connections would reach this agent and where to
 * change that.
 */

import * as React from "react";
import { ArrowLeft, ArrowRight, Check, Loader2, Plus } from "lucide-react";

import { GlassPanel, SectionHeader } from "@/components/glass";
import { ChipSelect, TextArea, TextInput } from "@/components/form";
import { post } from "@/lib/api";
import { SaveResult, type SaveState } from "@/lib/editing";
import { usePanel } from "@/lib/hooks";
import type { AgentConfig } from "@/screens/agent-config";

const STEPS = ["Identity", "Instructions", "Model", "Capabilities", "Channels", "Review"] as const;
type Step = (typeof STEPS)[number];

const ID_RE = /^[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?$/;

type Draft = {
  id: string; name: string; role: string; description: string;
  instructions: string;
  model: Record<string, string>;
  toolsets: string[]; deny: string[]; permissions: string[]; knowledge: string[];
};

const EMPTY: Draft = {
  id: "", name: "", role: "", description: "", instructions: "",
  model: {}, toolsets: [], deny: [], permissions: [], knowledge: [],
};

export function CreateAgent({
  existing, onCancel, onCreated,
}: {
  existing: string[];
  onCancel: () => void;
  /** Called when the operator chooses to open the new agent, not when the POST resolves:
   *  navigating on the response would hide the result the response was for, and a "saved
   *  but not applied" outcome is exactly the one somebody needs to read. */
  onCreated: (agentId: string) => void;
}) {
  const [step, setStep] = React.useState<Step>("Identity");
  const [draft, setDraft] = React.useState<Draft>(EMPTY);
  const [state, setState] = React.useState<SaveState>({ kind: "idle" });
  const [busy, setBusy] = React.useState(false);
  const [created, setCreated] = React.useState<string | null>(null);

  // The choices come from any existing agent's config read — the option lists are the
  // tenant's, not the agent's. Falls back gracefully when a tenant has no agent yet.
  const sample = existing[0];
  const choicesPanel = usePanel<AgentConfig>(
    sample ? `/agents/${encodeURIComponent(sample)}/config` : "/health", 300000,
  );
  const choices =
    sample && choicesPanel.state === "ok"
      ? choicesPanel.data.choices
      : { permissions: [], actions: [], toolsets: [], knowledge: [], channels: [], agents: [] };

  const set = (patch: Partial<Draft>) => {
    setDraft((d) => ({ ...d, ...patch }));
    if (state.kind !== "idle") setState({ kind: "idle" });
  };

  const idError =
    !draft.id.trim() ? "" // not an error until they try to advance
    : !ID_RE.test(draft.id.trim()) ? "Lowercase letters, digits and hyphens, starting and ending alphanumeric."
    : existing.includes(draft.id.trim()) ? "An agent with this id already exists."
    : "";
  const identityReady = ID_RE.test(draft.id.trim()) && !existing.includes(draft.id.trim());

  const index = STEPS.indexOf(step);
  const canAdvance = step !== "Identity" || identityReady;

  async function create() {
    if (busy) return;
    setBusy(true);
    setState({ kind: "saving" });
    try {
      const fields: Record<string, unknown> = {
        name: draft.name.trim() || draft.id.trim(),
        enabled: true,
      };
      if (draft.role.trim()) fields.role = draft.role.trim();
      if (draft.description.trim()) fields.description = draft.description.trim();
      const model = Object.fromEntries(
        Object.entries(draft.model).filter(([, v]) => String(v ?? "").trim()),
      );
      if (Object.keys(model).length) fields.model = model;
      if (draft.toolsets.length || draft.deny.length) {
        fields.tools = {
          ...(draft.toolsets.length ? { toolsets: draft.toolsets } : {}),
          ...(draft.deny.length ? { deny: draft.deny } : {}),
        };
      }
      if (draft.permissions.length) fields.permissions = draft.permissions;
      if (draft.knowledge.length) fields.knowledge = { sources: draft.knowledge };

      const result: any = await post("/agents", {
        id: draft.id.trim(),
        fields,
        instructions: draft.instructions,
      });
      const runtime = result?.runtime;
      const applied = runtime === undefined ? true : Boolean(runtime.applied);
      setState({
        kind: "saved", applied,
        detail: applied ? "" : String(runtime?.error ?? "the runtime was not updated"),
        files: result?.files_changed ?? [],
      });
      setCreated(draft.id.trim());
    } catch (cause) {
      setState({
        kind: "error",
        message: cause instanceof Error ? cause.message : "the request did not complete",
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <button
        type="button" onClick={onCancel}
        className="text-ink-faint hover:text-ink inline-flex items-center gap-1.5 text-[12.5px] transition-colors"
      >
        <ArrowLeft className="size-3.5" /> All agents
      </button>

      <GlassPanel elevated className="p-5">
        <SectionHeader
          icon={Plus} title="Create an agent"
          detail="Declared in the tenant bundle, then applied to the runtime."
        />
        <ol className="flex flex-wrap gap-1.5">
          {STEPS.map((s, i) => (
            <li key={s}>
              <button
                type="button"
                onClick={() => (i <= index || identityReady) && setStep(s)}
                disabled={i > index && !identityReady}
                className={`rounded-lg px-2.5 py-1.5 text-[12px] font-medium transition-colors ${
                  s === step ? "glass-solid text-ink"
                  : i < index ? "text-running"
                  : "text-ink-faint"
                } disabled:opacity-40`}
              >
                {i < index ? <Check className="mr-1 inline size-3" /> : `${i + 1}. `}
                {s}
              </button>
            </li>
          ))}
        </ol>
      </GlassPanel>

      <GlassPanel className="p-5">
        {step === "Identity" ? (
          <div className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <TextInput
                id="new-agent-id" label="Agent id" value={draft.id} mono
                onChange={(id) => set({ id })} placeholder="night-ops"
                error={idError}
                hint="Permanent. It is also the runtime profile's directory name."
              />
              <TextInput
                id="new-agent-name" label="Name" value={draft.name}
                onChange={(name) => set({ name })} placeholder="Night Ops"
                hint="Optional — defaults to the id."
              />
            </div>
            <TextInput
              id="new-agent-role" label="Role" value={draft.role}
              onChange={(role) => set({ role })} placeholder="operations"
              hint="Optional. A short label used as a routing signal."
            />
            <TextArea
              id="new-agent-desc" label="Description" value={draft.description} rows={3}
              onChange={(description) => set({ description })}
              hint="What this agent is for."
            />
          </div>
        ) : null}

        {step === "Instructions" ? (
          <div className="space-y-2">
            <TextArea
              id="new-agent-soul" label="Soul" value={draft.instructions} rows={14} mono
              onChange={(instructions) => set({ instructions })}
              placeholder="You are…"
              hint="Who this agent is. Prepended to every turn it takes, and editable later."
            />
            <p className="text-ink-faint text-[11.5px] leading-relaxed">
              Saved as a prompt file in the bundle. The runtime's SOUL.md is composed from it
              on apply, with the tenant's branding and the agent's knowledge briefing added.
            </p>
          </div>
        ) : null}

        {step === "Model" ? (
          <div className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              {["provider", "name", "endpoint", "api_key_env", "region", "reasoning_effort"].map((key) => (
                <TextInput
                  key={key} id={`new-model-${key}`} mono={key !== "reasoning_effort"}
                  label={key === "name" ? "Model" : key === "api_key_env" ? "Credential variable" : key.replace("_", " ")}
                  value={draft.model[key] ?? ""}
                  onChange={(v) => set({ model: { ...draft.model, [key]: v } })}
                  hint={key === "api_key_env" ? "The NAME of the variable. Never the key itself." : undefined}
                />
              ))}
            </div>
            <p className="text-ink-faint text-[11.5px]">
              Leave everything blank to inherit the deployment's own model settings.
            </p>
          </div>
        ) : null}

        {step === "Capabilities" ? (
          <div className="space-y-6">
            <ChipSelect
              label="Toolsets" options={choices.toolsets.map((t: any) => t.id)}
              selected={draft.toolsets} onChange={(toolsets) => set({ toolsets })}
              describe={(id) => choices.toolsets.find((t: any) => t.id === id)?.description}
              hint={`${choices.toolsets.length} groups, from the runtime's own registry.`}
              emptyNote="This runtime does not publish a toolset registry."
            />
            <ChipSelect
              label="Permissions" options={choices.permissions}
              selected={draft.permissions} onChange={(permissions) => set({ permissions })}
              hint="Business actions, declared in this tenant's policy."
              emptyNote="This tenant's policy declares no permissions."
            />
            <ChipSelect
              label="Knowledge" options={choices.knowledge}
              selected={draft.knowledge} onChange={(knowledge) => set({ knowledge })}
              emptyNote="This tenant declares no knowledge sources."
            />
          </div>
        ) : null}

        {step === "Channels" ? (
          <div className="space-y-3">
            <p className="text-ink-muted text-[12.5px] leading-relaxed">
              A channel grant is declared on the connection, not on the agent — that is the
              single place it lives, so this form does not offer a second one. Create the
              agent, then grant it on the connection in{" "}
              <span className="font-mono">channels.yaml</span>.
            </p>
            <div className="flex flex-wrap gap-1.5">
              {choices.channels.slice(0, 24).map((c: any) => (
                <span key={c.id} className="border-glass-border text-ink-muted rounded-md border px-2 py-1 text-[11.5px]">
                  {c.label}
                </span>
              ))}
            </div>
            <p className="text-ink-faint text-[11.5px]">
              {choices.channels.length} platforms available on this runtime.
            </p>
          </div>
        ) : null}

        {step === "Review" ? (
          <div className="space-y-4">
            <SectionHeader title="Review" detail="Nothing has been created yet." />
            <dl className="grid gap-x-8 gap-y-2 text-[12.5px] sm:grid-cols-2">
              <Item label="Agent">{draft.name.trim() || draft.id} <span className="text-ink-faint font-mono">({draft.id})</span></Item>
              <Item label="Role">{draft.role.trim() || "—"}</Item>
              <Item label="Description">{draft.description.trim() || "—"}</Item>
              <Item label="Model">
                {draft.model.provider || draft.model.name
                  ? `${draft.model.provider ?? ""} ${draft.model.name ?? ""}`.trim()
                  : "inherits the deployment default"}
              </Item>
              <Item label="Soul">{draft.instructions.trim() ? `${draft.instructions.trim().length} characters` : "none — a default will be composed"}</Item>
              <Item label="Toolsets">{draft.toolsets.join(", ") || "none"}</Item>
              <Item label="Permissions">{draft.permissions.join(", ") || "none"}</Item>
              <Item label="Knowledge">{draft.knowledge.join(", ") || "none"}</Item>
              <Item label="Channels">granted on the connection, after creation</Item>
              <Item label="Schedules">declared separately, after creation</Item>
            </dl>

            <div className="flex flex-wrap items-center gap-3 pt-1">
              <button
                type="button" onClick={create} disabled={busy || !identityReady || Boolean(created)}
                className="glass-solid text-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium disabled:opacity-40"
              >
                {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Plus className="size-3.5" />}
                {busy ? "Creating agent…" : "Create agent"}
              </button>
              <span className="text-ink-faint text-[11.5px]">
                Validated against the whole bundle before anything is written.
              </span>
            </div>
            <SaveResult state={state} />

            {created ? (
              <div className="border-glass-border flex flex-wrap items-center gap-3 rounded-lg border p-3">
                <p className="text-ink min-w-0 flex-1 text-[12.5px]">
                  <b>{created}</b> now exists in this tenant.
                </p>
                <button
                  type="button" onClick={() => onCreated(created)}
                  className="glass-solid text-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium"
                >
                  Open {created} <ArrowRight className="size-3.5" />
                </button>
              </div>
            ) : null}
          </div>
        ) : null}

        {step !== "Review" ? (
          <div className="border-glass-border mt-5 flex items-center justify-between border-t pt-4">
            <button
              type="button" disabled={index === 0}
              onClick={() => setStep(STEPS[Math.max(0, index - 1)])}
              className="text-ink-faint hover:text-ink inline-flex items-center gap-1.5 text-[12.5px] disabled:opacity-30"
            >
              <ArrowLeft className="size-3.5" /> Back
            </button>
            <button
              type="button" disabled={!canAdvance}
              onClick={() => setStep(STEPS[Math.min(STEPS.length - 1, index + 1)])}
              className="glass-solid text-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium disabled:opacity-40"
            >
              Next <ArrowRight className="size-3.5" />
            </button>
          </div>
        ) : null}
      </GlassPanel>
    </div>
  );
}

function Item({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-ink-faint">{label}</dt>
      <dd className="text-ink min-w-0 text-right">{children}</dd>
    </div>
  );
}
