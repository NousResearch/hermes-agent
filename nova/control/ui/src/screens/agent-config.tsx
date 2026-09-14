/* Editing an agent's declaration: identity, model, capabilities.
 *
 * Reads `/agents/<id>/config`, which returns exactly the fields the write route accepts
 * plus the option lists a form needs. Writes `/agents/<id>/update` with only the sections
 * that changed. Save behaviour is `useEditable` — the same one the Soul editor uses, so
 * "unsaved", "saving", "saved but not applied" and "failed" mean the same thing here.
 *
 * Option lists are never hard-coded. Permissions come from the tenant's own policy,
 * toolsets from the runtime's registry, corpora from the knowledge catalogue, teammates
 * from the bundle. An option that disappears upstream disappears here, rather than
 * lingering in a dropdown and failing at save time.
 */

import * as React from "react";
import { ShieldCheck, SlidersHorizontal, User, Wrench } from "lucide-react";

import { GlassPanel, SectionHeader } from "@/components/glass";
import { ChipSelect, TextArea, TextInput, Toggle } from "@/components/form";
import { post } from "@/lib/api";
import { SaveBar, useEditable } from "@/lib/editing";
import { usePanel } from "@/lib/hooks";

export type AgentConfig = {
  id: string;
  fields: {
    name: string; role: string; description: string; enabled: boolean;
    model: Record<string, any>;
    tools: { toolsets?: string[]; allow?: string[]; deny?: string[] };
    knowledge: { sources?: string[] };
    permissions: string[];
    approval: Record<string, any>;
    limits: Record<string, any>;
    delegation: Record<string, any>;
  };
  settable: string[];
  choices: {
    permissions: string[];
    actions: string[];
    toolsets: { id: string; description: string; tools: string[]; includes: string[] }[];
    knowledge: string[];
    channels: { id: string; label: string }[];
    agents: string[];
  };
};

type Section = "identity" | "model" | "capabilities";

export function AgentConfigPanel({
  agentId, section, onChanged,
}: { agentId: string; section: Section; onChanged?: () => void }) {
  const [nonce, setNonce] = React.useState(0);
  const loaded = usePanel<AgentConfig>(
    `/agents/${encodeURIComponent(agentId)}/config`, 120000, nonce,
  );

  if (loaded.state === "loading") {
    return <GlassPanel className="p-5"><p className="text-ink-faint text-[13px]">reading…</p></GlassPanel>;
  }
  if (loaded.state === "forbidden") {
    return (
      <GlassPanel className="p-5">
        <p className="text-ink-muted text-[13px]">
          An agent's configuration includes what it is permitted to do, so it is not visible
          to your role.
        </p>
      </GlassPanel>
    );
  }
  if (loaded.state === "error") {
    return (
      <GlassPanel className="p-5">
        <p className="text-blocked text-[13px]">
          <b>This agent's configuration could not be read.</b> {loaded.message}
        </p>
      </GlassPanel>
    );
  }

  const done = () => {
    setNonce((n) => n + 1);
    onChanged?.();
  };

  if (section === "identity") return <IdentitySection agentId={agentId} config={loaded.data} onSaved={done} />;
  if (section === "model") return <ModelSection agentId={agentId} config={loaded.data} onSaved={done} />;
  return <CapabilitySection agentId={agentId} config={loaded.data} onSaved={done} />;
}

function useSection<V>(agentId: string, server: V, key: string, onSaved: () => void) {
  const editor = useEditable<V>(server, async (value) => {
    const result = await post(`/agents/${encodeURIComponent(agentId)}/update`, {
      fields: { [key]: value },
    });
    onSaved();
    return result;
  });
  return editor;
}

/* ── Identity ─────────────────────────────────────────────────────────────── */

function IdentitySection({
  agentId, config, onSaved,
}: { agentId: string; config: AgentConfig; onSaved: () => void }) {
  const server = React.useMemo(
    () => ({
      name: config.fields.name ?? "",
      role: config.fields.role ?? "",
      description: config.fields.description ?? "",
      enabled: Boolean(config.fields.enabled),
    }),
    [config],
  );

  const editor = useEditable(server, async (value) => {
    const result = await post(`/agents/${encodeURIComponent(agentId)}/update`, { fields: value });
    onSaved();
    return result;
  });
  const v = editor.value;

  return (
    <GlassPanel className="p-5">
      <SectionHeader icon={User} title="Identity" detail="How this agent is named and described." />
      <div className="grid gap-4 sm:grid-cols-2">
        <TextInput
          id={`name-${agentId}`} label="Name" value={v.name}
          onChange={(name) => editor.edit((c) => ({ ...c, name }))}
          hint="Shown wherever this agent appears."
        />
        <TextInput
          id={`role-${agentId}`} label="Role" value={v.role}
          onChange={(role) => editor.edit((c) => ({ ...c, role }))}
          hint="A short label, used for routing signals."
        />
      </div>
      <div className="mt-4">
        <TextArea
          id={`desc-${agentId}`} label="Description" value={v.description} rows={3}
          onChange={(description) => editor.edit((c) => ({ ...c, description }))}
          hint="What this agent is for. Read by people, and used when routing work."
        />
      </div>
      <div className="mt-4">
        <Toggle
          id={`enabled-${agentId}`} label="Enabled" checked={v.enabled}
          onChange={(enabled) => editor.edit((c) => ({ ...c, enabled }))}
          hint="A disabled agent is not scheduled and is not routed to. Its declaration and history stay."
        />
      </div>
      <p className="text-ink-faint mt-4 font-mono text-[11.5px]">
        id: {config.id} — an agent's id is fixed. Use Duplicate to start one under a new id.
      </p>
      <SaveBar
        dirty={editor.dirty} busy={editor.busy} state={editor.state}
        onSave={editor.submit} onDiscard={editor.discard}
      />
    </GlassPanel>
  );
}

/* ── Model ────────────────────────────────────────────────────────────────── */

const MODEL_FIELDS: { key: string; label: string; hint: string; mono?: boolean }[] = [
  { key: "provider", label: "Provider", hint: "A name the runtime resolves, or a custom gateway.", mono: true },
  { key: "name", label: "Model", hint: "The model identifier this agent runs on.", mono: true },
  { key: "endpoint", label: "Endpoint", hint: "Only for a custom provider.", mono: true },
  { key: "api_key_env", label: "Credential variable", hint: "The NAME of the variable holding the key. Never the key.", mono: true },
  { key: "region", label: "Region", hint: "Where the provider serves this model." },
  { key: "reasoning_effort", label: "Reasoning effort", hint: "Provider-specific; leave blank to inherit." },
];

function ModelSection({
  agentId, config, onSaved,
}: { agentId: string; config: AgentConfig; onSaved: () => void }) {
  const server = React.useMemo(() => ({ ...(config.fields.model ?? {}) }), [config]);
  const editor = useSection<Record<string, any>>(agentId, server, "model", onSaved);
  const v = editor.value ?? {};

  return (
    <GlassPanel className="p-5">
      <SectionHeader
        icon={SlidersHorizontal} title="Model"
        detail="Blank fields inherit from the deployment's own settings."
      />
      <div className="grid gap-4 sm:grid-cols-2">
        {MODEL_FIELDS.map((f) => (
          <TextInput
            key={f.key} id={`model-${f.key}-${agentId}`} label={f.label} mono={f.mono}
            value={String(v[f.key] ?? "")}
            onChange={(next) =>
              editor.edit((c) => {
                const out = { ...c };
                if (next.trim()) out[f.key] = next;
                else delete out[f.key];
                return out;
              })
            }
            hint={f.hint}
          />
        ))}
      </div>
      <p className="text-ink-faint mt-4 text-[11.5px] leading-relaxed">
        A credential's value never passes through this screen. Set it in the agent's
        <span className="font-mono"> .env</span>, which NOVA does not write.
      </p>
      <SaveBar
        dirty={editor.dirty} busy={editor.busy} state={editor.state}
        onSave={editor.submit} onDiscard={editor.discard}
      />
    </GlassPanel>
  );
}

/* ── Capabilities ─────────────────────────────────────────────────────────── */

function CapabilitySection({
  agentId, config, onSaved,
}: { agentId: string; config: AgentConfig; onSaved: () => void }) {
  const serverTools = React.useMemo(
    () => ({
      toolsets: [...(config.fields.tools?.toolsets ?? [])],
      allow: [...(config.fields.tools?.allow ?? [])],
      deny: [...(config.fields.tools?.deny ?? [])],
    }),
    [config],
  );
  const serverPerms = React.useMemo(() => [...(config.fields.permissions ?? [])], [config]);
  const serverKnowledge = React.useMemo(
    () => ({ sources: [...(config.fields.knowledge?.sources ?? [])] }),
    [config],
  );

  const tools = useSection<typeof serverTools>(agentId, serverTools, "tools", onSaved);
  const perms = useSection<string[]>(agentId, serverPerms, "permissions", onSaved);
  const knowledge = useSection<typeof serverKnowledge>(agentId, serverKnowledge, "knowledge", onSaved);

  const toolsetIds = config.choices.toolsets.map((t) => t.id);
  const describe = (id: string) => {
    const entry = config.choices.toolsets.find((t) => t.id === id);
    if (!entry) return undefined;
    return entry.tools.length ? `${entry.description} — ${entry.tools.join(", ")}` : entry.description;
  };
  const denyOptions = React.useMemo(() => {
    const named = new Set<string>(tools.value?.deny ?? []);
    for (const id of tools.value?.toolsets ?? []) {
      for (const t of config.choices.toolsets.find((x) => x.id === id)?.tools ?? []) named.add(t);
    }
    return [...named].sort();
  }, [tools.value, config]);

  return (
    <div className="space-y-5">
      <GlassPanel className="p-5">
        <SectionHeader
          icon={Wrench} title="Tools"
          detail="Groups the runtime understands. Deny wins over everything else."
        />
        <div className="space-y-5">
          <ChipSelect
            label="Toolsets" options={toolsetIds} selected={tools.value?.toolsets ?? []}
            describe={describe}
            onChange={(toolsets) => tools.edit((c) => ({ ...c, toolsets }))}
            hint={`${toolsetIds.length} groups, read from the runtime's own registry.`}
            emptyNote="This runtime does not publish a toolset registry."
          />
          <ChipSelect
            label="Denied tools" options={denyOptions} selected={tools.value?.deny ?? []}
            onChange={(deny) => tools.edit((c) => ({ ...c, deny }))}
            hint="Removed even when a selected toolset would grant them. This is the half the runtime enforces."
            emptyNote="Select a toolset first; its tools become deniable."
          />
        </div>
        <p className="text-waiting mt-4 text-[11.5px] leading-relaxed">
          Positive scoping is recorded but not enforced by the Hermes adapter — denials are.
        </p>
        <SaveBar dirty={tools.dirty} busy={tools.busy} state={tools.state}
                 onSave={tools.submit} onDiscard={tools.discard} label="Save tools" />
      </GlassPanel>

      <GlassPanel className="p-5">
        <SectionHeader
          icon={ShieldCheck} title="Permissions"
          detail="Business actions this agent may take, from the tenant's policy."
        />
        <ChipSelect
          label="Granted" options={config.choices.permissions} selected={perms.value ?? []}
          onChange={(next) => perms.edit(next)}
          hint="Compiled into the runtime's fail-closed tool hook."
          emptyNote="This tenant's policy declares no permissions."
        />
        <SaveBar dirty={perms.dirty} busy={perms.busy} state={perms.state}
                 onSave={perms.submit} onDiscard={perms.discard} label="Save permissions" />
      </GlassPanel>

      <GlassPanel className="p-5">
        <SectionHeader title="Knowledge" detail="Corpora this agent may read." />
        <ChipSelect
          label="Readable corpora" options={config.choices.knowledge}
          selected={knowledge.value?.sources ?? []}
          onChange={(sources) => knowledge.edit((c) => ({ ...c, sources }))}
          emptyNote="This tenant declares no knowledge sources."
        />
        <SaveBar dirty={knowledge.dirty} busy={knowledge.busy} state={knowledge.state}
                 onSave={knowledge.submit} onDiscard={knowledge.discard} label="Save knowledge" />
      </GlassPanel>
    </div>
  );
}
