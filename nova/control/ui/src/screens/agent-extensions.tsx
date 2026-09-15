/* What one agent can reach beyond its own tools: MCP servers and runtime plugins.
 *
 * Both are capabilities the runtime already ships. This screen grants them per agent; it
 * does not invent either, and every row comes from the runtime's own catalogue.
 *
 * **Granting is not the same as working, and the screen says which.** Most catalogue
 * servers authenticate with OAuth — a browser consent on the host that the Control Centre
 * cannot perform. A granted OAuth server is configured and uncallable until somebody runs
 * `hermes mcp login`, so it is labelled that way rather than given a green tick.
 *
 * **Saved is not the same as applied.** The grant is written to the bundle; the profile is
 * written by apply. Both are reported, as everywhere else in this Control Centre.
 *
 * **A plugin has three states, not two.** A bundled backend loads unless it is disabled, so
 * "not listed" and "disabled" are genuinely different answers for it. A checkbox would make
 * one of them unreachable.
 */

import * as React from "react";
import {
  AlertTriangle, Blocks, Check, ExternalLink, Loader2, Plug, ShieldAlert,
} from "lucide-react";

import { GlassPanel, SectionHeader, StatusPill } from "@/components/glass";
import { post } from "@/lib/api";
import { usePanel } from "@/lib/hooks";

type Credential = { name: string; prompt: string; required: boolean; secret: boolean };
type Server = {
  id: string; description: string; source: string; transport: string; auth: string;
  url: string; installs: boolean; toolset: string; needs_interactive_auth: boolean;
  credentials: Credential[]; granted: boolean;
};
type Plugin = {
  id: string; name: string; description: string; kind: string; source: string;
  version: string; tools: string[]; credentials: Credential[];
  state: "enable" | "disable" | "default"; loads_by_default: boolean;
};
type Applied = {
  known: boolean; mcp: string[]; enabled: string[]; disabled: string[]; detail: string;
};
type Payload = {
  agent_id: string; mcp: Server[]; plugins: Plugin[];
  granted: { mcp: string[]; enable: string[]; disable: string[] };
  applied: Applied; detail: string;
};

type Result =
  | { kind: "idle" }
  | { kind: "busy"; what: string }
  | { kind: "done"; message: string; warning: string }
  | { kind: "error"; message: string };

const STATES: { id: Plugin["state"]; label: string }[] = [
  { id: "default", label: "Default" },
  { id: "enable", label: "Enabled" },
  { id: "disable", label: "Disabled" },
];

export function AgentExtensions({ agentId }: { agentId: string }) {
  const [nonce, setNonce] = React.useState(0);
  const loaded = usePanel<Payload>(
    `/agents/${encodeURIComponent(agentId)}/extensions`, 0, nonce,
  );
  const [result, setResult] = React.useState<Result>({ kind: "idle" });
  const [filter, setFilter] = React.useState("");
  const [showAll, setShowAll] = React.useState(false);

  const busy = result.kind === "busy";

  async function act(what: string, path: string, body: Record<string, unknown>) {
    if (busy) return;
    setResult({ kind: "busy", what });
    try {
      const response: any = await post(`/agents/${encodeURIComponent(agentId)}/${path}`, body);
      const runtime = response?.runtime ?? {};
      const warnings: string[] = runtime?.warnings ?? [];
      // Only the warnings about THIS agent, and only the ones about extensions. The apply
      // report covers every agent in the tenant, and showing an operator a note about
      // somebody else's missing model key after they clicked a toggle here is noise.
      const mine = warnings.filter(
        (w) => w.startsWith(`${agentId}:`) && /MCP|plugin|mcp/.test(w),
      );
      setResult({
        kind: "done",
        message: runtime?.applied ? "Saved and applied" : "Saved",
        warning: mine.length ? mine[0].replace(`${agentId}: `, "") : "",
      });
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
    return <GlassPanel className="p-5"><p className="text-ink-muted text-[13px]">Not visible to your role.</p></GlassPanel>;
  }
  if (loaded.state === "error") {
    return (
      <GlassPanel className="p-5">
        <p className="text-blocked text-[13px]"><b>Extensions could not be read.</b> {loaded.message}</p>
      </GlassPanel>
    );
  }

  const data = loaded.data;
  const needle = filter.trim().toLowerCase();
  const matches = (s: Server) =>
    !needle || s.id.toLowerCase().includes(needle) || s.description.toLowerCase().includes(needle);
  // Granted first, always visible; the other sixty are behind a filter or an explicit
  // "show all", because a list of sixty checkboxes is not a decision aid.
  const granted = data.mcp.filter((s) => s.granted);
  const rest = data.mcp.filter((s) => !s.granted && matches(s));
  const visible = showAll || needle ? rest : [];

  return (
    <div className="space-y-5">
      {result.kind === "error" ? (
        <GlassPanel className="border-blocked/30 p-3">
          <p className="text-blocked flex items-start gap-1.5 text-[12.5px]">
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
            <span><b>Nothing changed.</b> {result.message}</span>
          </p>
        </GlassPanel>
      ) : null}
      {result.kind === "done" ? (
        <GlassPanel className="p-3">
          <p className="text-running flex items-center gap-1.5 text-[12.5px]">
            <Check className="size-3.5" /> {result.message}
          </p>
          {result.warning ? (
            <p className="text-waiting mt-1 flex items-start gap-1.5 text-[12px]">
              <AlertTriangle className="mt-0.5 size-3 shrink-0" /> {result.warning}
            </p>
          ) : null}
        </GlassPanel>
      ) : null}

      <GlassPanel className="p-5">
        <SectionHeader
          icon={Plug} title="MCP servers"
          detail={`${granted.length} granted of ${data.mcp.length} in this runtime's catalogue`}
        />

        <p className="text-ink-faint mb-4 text-[11.5px] leading-relaxed">
          Each server's tools arrive as a toolset named <span className="font-mono">mcp-&lt;server&gt;</span>,
          so they are decided by this deployment's policy like every other tool. Only servers
          in the runtime's own catalogue can be granted — an MCP server is a command or a URL
          your agent's data flows through, and NOVA will not accept an arbitrary one.
        </p>

        {granted.length === 0 ? (
          <p className="text-ink-muted mb-4 text-[12.5px]">
            This agent is granted no MCP servers. It still has every tool its toolsets give it.
          </p>
        ) : (
          <ul className="divide-glass-border mb-4 divide-y">
            {granted.map((server) => (
              <ServerRow
                key={server.id} server={server} busy={busy} applied={data.applied}
                onToggle={() => void act(server.id, "mcp", { server: server.id, granted: false })}
              />
            ))}
          </ul>
        )}

        <div className="border-glass-border flex flex-wrap items-center gap-3 rounded-lg border p-3">
          <input
            type="search" value={filter} onChange={(e) => setFilter(e.target.value)}
            placeholder="Search the catalogue…" aria-label="Search MCP servers"
            className="glass-solid text-ink min-w-0 flex-1 rounded-lg px-3 py-1.5 text-[12.5px]"
          />
          <button
            type="button" onClick={() => setShowAll((v) => !v)}
            className="text-ink-faint hover:text-ink text-[11.5px]"
          >
            {showAll ? "Hide" : `Show all ${data.mcp.length}`}
          </button>
        </div>

        {visible.length ? (
          <ul className="divide-glass-border mt-3 divide-y">
            {visible.map((server) => (
              <ServerRow
                key={server.id} server={server} busy={busy} applied={data.applied}
                onToggle={() => void act(server.id, "mcp", { server: server.id, granted: true })}
              />
            ))}
          </ul>
        ) : needle ? (
          <p className="text-ink-muted mt-3 text-[12.5px]">
            Nothing in the catalogue matches “{filter}”.
          </p>
        ) : null}

        {data.detail ? (
          <p className="text-waiting mt-3 flex items-start gap-1.5 text-[11.5px]">
            <AlertTriangle className="mt-0.5 size-3 shrink-0" /> {data.detail}
          </p>
        ) : null}
      </GlassPanel>

      <GlassPanel className="p-5">
        <SectionHeader
          icon={Blocks} title="Plugins"
          detail={`${data.plugins.length} available to this runtime`}
        />
        <p className="text-ink-faint mb-4 text-[11.5px] leading-relaxed">
          Default is what the runtime does on its own: a bundled provider loads, everything
          else stays off. Channels are plugins too, but they are configured on the Channels
          tab and are not listed here.
        </p>
        <ul className="divide-glass-border divide-y">
          {data.plugins.map((plugin) => (
            <PluginRow
              key={plugin.id} plugin={plugin} busy={busy}
              onSet={(state) => void act(plugin.id, "plugins", { plugin: plugin.id, state })}
            />
          ))}
        </ul>
      </GlassPanel>
    </div>
  );
}

function ServerRow({
  server, busy, applied, onToggle,
}: { server: Server; busy: boolean; applied: Applied; onToggle: () => void }) {
  const live = applied.known && applied.mcp.includes(server.id);
  return (
    <li className="flex flex-wrap items-start gap-3 py-3 first:pt-0 last:pb-0">
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-ink font-mono text-[12.5px]">{server.id}</span>
          {server.granted ? (
            live
              ? <StatusPill state="running">Applied</StatusPill>
              : <StatusPill state="waiting">Saved, not applied</StatusPill>
          ) : null}
          {server.granted && server.needs_interactive_auth ? (
            <StatusPill state="waiting">Needs authorization</StatusPill>
          ) : null}
          {server.installs ? <StatusPill state="blocked">Host install required</StatusPill> : null}
        </div>
        <p className="text-ink-muted mt-1 text-[12px] leading-relaxed">{server.description}</p>
        <p className="text-ink-faint mt-1 text-[11.5px]">
          {server.transport === "http" && server.url ? (
            <>Data goes to <span className="font-mono">{server.url}</span>. </>
          ) : null}
          Tools arrive as <span className="font-mono">{server.toolset}</span>.
          {server.granted && server.needs_interactive_auth ? (
            <> Authorize it on the host with{" "}
              <span className="font-mono">hermes mcp login {server.id}</span> — until then it
              is configured but cannot be called.</>
          ) : null}
          {server.granted && server.credentials.length ? (
            <> Set <span className="font-mono">
              {server.credentials.map((c) => c.name).join(", ")}
            </span> under Channels → Credentials.</>
          ) : null}
        </p>
        {server.source ? (
          <a
            href={server.source} target="_blank" rel="noreferrer noopener"
            className="text-ink-faint hover:text-ink mt-1 inline-flex items-center gap-1 text-[11px]"
          >
            <ExternalLink className="size-3" /> Documentation
          </a>
        ) : null}
      </div>
      <button
        type="button" disabled={busy} onClick={onToggle}
        className={
          "shrink-0 rounded-lg px-3 py-1.5 text-[12px] font-medium disabled:opacity-40 " +
          (server.granted ? "border-glass-border text-ink-muted border" : "glass-solid text-ink")
        }
      >
        {busy ? <Loader2 className="size-3.5 animate-spin" /> : server.granted ? "Revoke" : "Grant"}
      </button>
    </li>
  );
}

function PluginRow({
  plugin, busy, onSet,
}: { plugin: Plugin; busy: boolean; onSet: (state: Plugin["state"]) => void }) {
  return (
    <li className="flex flex-wrap items-start gap-3 py-3 first:pt-0 last:pb-0">
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-ink font-mono text-[12.5px]">{plugin.id}</span>
          <span className="text-ink-faint text-[11px]">{plugin.kind}</span>
          {plugin.state === "disable" ? (
            <StatusPill state="blocked">Off for this agent</StatusPill>
          ) : plugin.loads_by_default ? (
            <StatusPill state="running">Loads by default</StatusPill>
          ) : plugin.state === "enable" ? (
            <StatusPill state="running">On for this agent</StatusPill>
          ) : null}
        </div>
        <p className="text-ink-muted mt-1 text-[12px] leading-relaxed">{plugin.description}</p>
        {plugin.credentials.length ? (
          <p className="text-ink-faint mt-1 flex items-start gap-1.5 text-[11.5px]">
            <ShieldAlert className="mt-0.5 size-3 shrink-0" />
            Needs <span className="font-mono">{plugin.credentials.map((c) => c.name).join(", ")}</span> on the host.
          </p>
        ) : null}
      </div>
      <div role="group" aria-label={`${plugin.id} state`} className="flex shrink-0 gap-1">
        {STATES.map((state) => (
          <button
            key={state.id} type="button" disabled={busy}
            aria-pressed={plugin.state === state.id}
            onClick={() => onSet(state.id)}
            className={
              "rounded-lg px-2.5 py-1 text-[11.5px] font-medium disabled:opacity-40 " +
              (plugin.state === state.id
                ? "glass-solid text-ink"
                : "text-ink-faint hover:text-ink")
            }
          >
            {state.label}
          </button>
        ))}
      </div>
    </li>
  );
}
