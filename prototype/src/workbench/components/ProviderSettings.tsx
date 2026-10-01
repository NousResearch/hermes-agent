"use client";

import { useState } from "react";
import { cn } from "../utils/cn";
import { probeProvider, type ModelProvider, type ProviderStatus } from "../data/providers";
import { useApp } from "../lib/app";
import { IconCheck, IconChevronDown, IconCloud, IconExternal, IconLaptop, IconPlus, IconRefresh, IconSearch, IconShield, IconTerminal, IconWarning } from "./Icons";
import { Badge, Button, Input, Menu, MenuItem } from "./ui";
import { PageHeader } from "./Titlebar";

const statusTone: Record<ProviderStatus, "mint" | "amber" | "rose" | "neutral" | "cyan"> = {
  ready: "mint", offline: "rose", "needs-key": "amber", checking: "cyan", error: "rose",
};
const statusLabel: Record<ProviderStatus, string> = {
  ready: "Available", offline: "Offline", "needs-key": "Key required", checking: "Checking", error: "Error",
};

function ProviderCard({ provider }: { provider: ModelProvider }) {
  const { setProviders, toast } = useApp();
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [endpoint, setEndpoint] = useState(provider.baseUrl ?? "");
  const local = provider.kind === "local";
  const patch = (update: Partial<ModelProvider>) => setProviders((all) => all.map((p) => p.id === provider.id ? { ...p, ...update } : p));
  const check = async () => {
    setBusy(true); setMessage(""); patch({ status: "checking" });
    const result = await probeProvider({ ...provider, baseUrl: endpoint });
    patch({ baseUrl: endpoint, status: result.status, models: result.models });
    setMessage(result.message); setBusy(false);
    if (result.status === "ready") toast(`${provider.name}: ${result.message}`, "mint");
  };

  return (
    <article className={cn("overflow-hidden rounded-[12px] border bg-raise transition-colors", provider.status === "ready" ? "border-mint/25" : "border-line-soft hover:border-line-strong")}>
      <div className="flex items-start gap-3 p-3.5">
        <span className="flex size-[36px] shrink-0 items-center justify-center rounded-[10px] border border-line-soft bg-well font-mono text-[10px] font-bold" style={{ color: provider.accent }}>{provider.glyph}</span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            <h3 className="font-display text-[14px] font-semibold text-ink">{provider.name}</h3>
            <Badge tone={statusTone[provider.status]} mono dot className="text-[9px]">{statusLabel[provider.status]}</Badge>
            <Badge tone={local ? "cyan" : "iris"} mono className="text-[9px]">{local ? "LOCAL" : "CLOUD"}</Badge>
          </div>
          <p className="mt-0.5 text-[11.5px] text-ink-3">{provider.description}</p>
          <p className="mt-1 font-mono text-[9.5px] text-ink-4">{provider.hint}</p>
        </div>
        <Menu align="right" width={190} trigger={() => <button className="flex size-[26px] cursor-pointer items-center justify-center rounded-[6px] text-ink-4 hover:bg-hover hover:text-ink"><IconChevronDown size={12} /></button>}>
          {(close) => <>
            <MenuItem icon={<IconExternal size={12} />} label="Provider documentation" onClick={() => { if (provider.docs) window.open(provider.docs, "_blank", "noopener,noreferrer"); close(); }} />
            <MenuItem icon={<IconTerminal size={12} />} label="Copy connection command" onClick={() => { navigator.clipboard?.writeText(`aro provider set ${provider.id} --base-url ${provider.baseUrl ?? provider.defaultUrl ?? ""}`); toast("Connection command copied", "mint"); close(); }} />
          </>}
        </Menu>
      </div>

      {local ? (
        <div className="border-t border-line-soft px-3.5 py-3">
          <div className="flex items-center gap-2">
            <div className="relative min-w-0 flex-1">
              <span className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 font-mono text-[9px] text-ink-4">URL</span>
              <Input value={endpoint} onChange={(e) => setEndpoint(e.target.value)} className="h-[31px] pl-10 font-mono text-[11px]" aria-label={`${provider.name} endpoint`} />
            </div>
            <Button variant="secondary" size="sm" icon={busy ? IconRefresh : IconSearch} disabled={busy} onClick={check}>{busy ? "Checking" : "Test"}</Button>
          </div>
          {message && <p className={cn("mt-2 flex items-center gap-1.5 text-[10.5px]", provider.status === "ready" ? "text-mint" : "text-amber")}>
            {provider.status === "ready" ? <IconCheck size={11} /> : <IconWarning size={11} />}{message}
          </p>}
          {provider.models.length > 0 ? (
            <div className="mt-2.5 flex flex-wrap gap-1.5">
              {provider.models.map((m) => <span key={m.id} className="inline-flex items-center gap-1 rounded-[5px] border border-line-soft bg-well px-1.5 py-1 font-mono text-[10px] text-ink-2"><i className="size-[5px] rounded-full bg-mint" />{m.label}{m.size && <span className="text-ink-4">· {m.size}</span>}</span>)}
            </div>
          ) : <p className="mt-2 text-[10.5px] text-ink-4">Models appear here after a successful connection check.</p>}
        </div>
      ) : (
        <div className="border-t border-line-soft px-3.5 py-3">
          <div className="flex items-center justify-between gap-2">
            <span className="font-mono text-[9px] font-semibold tracking-[.14em] text-ink-4 uppercase">models</span>
            <span className="font-mono text-[9.5px] text-ink-4">credentials managed by daemon</span>
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {provider.models.map((m) => <span key={m.id} className="rounded-[5px] border border-line-soft bg-well px-1.5 py-1 font-mono text-[10px] text-ink-2">{m.label}<span className="ml-1 text-ink-4">{m.context}</span></span>)}
          </div>
          <p className="mt-2 flex items-start gap-1.5 text-[10.5px] leading-[1.45] text-ink-4"><IconShield size={10} className="mt-[1px] shrink-0" />Add credentials to the local Aro daemon. API keys are never stored in browser storage.</p>
        </div>
      )}
    </article>
  );
}

export function ProviderSettings() {
  const { providers, setProviders, toast } = useApp();
  const [filter, setFilter] = useState("");
  const [kind, setKind] = useState<"all" | "local" | "cloud">("all");
  const visible = providers.filter((p) => (kind === "all" || p.kind === kind) && `${p.name} ${p.description} ${p.models.map((m) => m.label).join(" ")}`.toLowerCase().includes(filter.toLowerCase()));
  const local = visible.filter((p) => p.kind === "local");
  const cloud = visible.filter((p) => p.kind === "cloud");
  const addCompatible = () => {
    const id = `custom-${Date.now()}`;
    const p: ModelProvider = { id, name: "Custom OpenAI-compatible", kind: "local", description: "Custom local endpoint · OpenAI-compatible", baseUrl: "http://localhost:8000/v1", defaultUrl: "http://localhost:8000/v1", status: "offline", glyph: "+", accent: "#86a8ff", models: [], hint: "Configure your base URL and discover models" };
    setProviders((all) => [...all, p]);
    toast("Custom local provider added", "mint");
  };
  return (
    <div className="scroll-thin flex-1 overflow-y-auto">
      <PageHeader eyebrow="inference" title="Providers & models" sub="Bring your own cloud API or keep inference on this device. Local discovery uses the official Ollama and OpenAI-compatible model-list endpoints." right={<Button variant="secondary" icon={IconPlus} onClick={addCompatible}>Add endpoint</Button>} />
      <div className="mx-auto max-w-[940px] space-y-5 p-5">
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative min-w-[220px] flex-1"><IconSearch size={12} className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-ink-4" /><Input placeholder="Filter providers or models…" value={filter} onChange={(e) => setFilter(e.target.value)} className="h-[32px] pl-8" /></div>
          <div className="flex gap-1 rounded-[8px] border border-line bg-sunken p-0.5">
            {([{ id: "all", label: `All ${providers.length}` }, { id: "local", label: "On-device" }, { id: "cloud", label: "Cloud" }] as const).map((t) => <button key={t.id} onClick={() => setKind(t.id)} className={cn("cursor-pointer rounded-[6px] px-2.5 py-1 text-[11.5px]", kind === t.id ? "bg-raise text-ink shadow-e1" : "text-ink-3 hover:text-ink-2")}>{t.label}</button>)}
          </div>
        </div>

        {(kind === "all" || kind === "local") && <section>
          <div className="mb-2.5 flex items-center gap-2"><span className="flex size-[24px] items-center justify-center rounded-[7px] bg-cyan-tint text-cyan"><IconLaptop size={12} /></span><div><h2 className="font-display text-[13.5px] font-semibold text-ink">On-device</h2><p className="text-[10.5px] text-ink-4">Private inference · no per-token cloud cost</p></div><span className="ml-auto font-mono text-[10px] text-ink-4">{local.filter((p) => p.status === "ready").length} reachable</span></div>
          <div className="grid gap-2.5 lg:grid-cols-2">{local.map((p) => <ProviderCard key={p.id} provider={p} />)}</div>
        </section>}

        {(kind === "all" || kind === "cloud") && <section>
          <div className="mb-2.5 flex items-center gap-2"><span className="flex size-[24px] items-center justify-center rounded-[7px] bg-iris-tint text-iris-soft"><IconCloud size={12} /></span><div><h2 className="font-display text-[13.5px] font-semibold text-ink">Cloud APIs</h2><p className="text-[10.5px] text-ink-4">Hosted models · credentials stay in the local daemon</p></div><span className="ml-auto font-mono text-[10px] text-ink-4">{cloud.length} providers</span></div>
          <div className="grid gap-2.5 lg:grid-cols-2">{cloud.map((p) => <ProviderCard key={p.id} provider={p} />)}</div>
        </section>}

        <div className="flex items-start gap-2 rounded-[10px] border border-amber/20 bg-amber-tint/40 px-3 py-2.5 text-[11px] leading-[1.5] text-amber"><IconShield size={12} className="mt-0.5 shrink-0" /><span>Local model discovery runs from this browser. If a request is blocked, enable the app origin in the provider's CORS settings or connect through the Aro daemon. API keys must be stored in the daemon, never localStorage.</span></div>
      </div>
    </div>
  );
}