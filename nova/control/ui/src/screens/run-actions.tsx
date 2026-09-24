import * as React from "react";
import { Play, RadioTower } from "lucide-react";
import { post } from "@/lib/api";
import type { Objective } from "./types";

/* The two control-plane actions the dashboard had routes for and no buttons: starting an
 * objective, and applying the tenant's channels to the gateway. Both change what runs next,
 * so both sit behind the same inline confirmation the task actions use, and the
 * confirmation says what will happen in words, not in route names. */

function Confirmable({
  label, busyLabel, explain, icon: Icon, run,
}: {
  label: string; busyLabel: string; explain: React.ReactNode;
  icon: typeof Play; run: () => Promise<string>;
}) {
  const [confirming, setConfirming] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [result, setResult] = React.useState<{ ok: boolean; text: string } | null>(null);

  const go = async () => {
    setBusy(true);
    setResult(null);
    try {
      setResult({ ok: true, text: await run() });
      setConfirming(false);
    } catch (error) {
      setResult({ ok: false, text: error instanceof Error ? error.message : String(error) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="border-glass-border mt-4 border-t pt-3">
      {confirming ? (
        <div className="space-y-2">
          <p className="text-ink text-[12.5px] leading-relaxed">{explain}</p>
          <div className="flex flex-wrap gap-2">
            <button type="button" disabled={busy} onClick={() => void go()}
              className="bg-accent text-accent-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium disabled:opacity-60">
              <Icon className="size-3.5" /> {busy ? busyLabel : `${label} now`}
            </button>
            <button type="button" disabled={busy} onClick={() => setConfirming(false)}
              className="glass-solid text-ink rounded-lg px-3 py-1.5 text-[12.5px] font-medium">
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <button type="button" onClick={() => { setConfirming(true); setResult(null); }}
          className="glass-solid text-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium">
          <Icon className="size-3.5" /> {label}
        </button>
      )}
      {result ? (
        <p role="status" className={`mt-2 text-[12px] leading-relaxed ${result.ok ? "text-running" : "text-blocked"}`}>
          {result.text}
        </p>
      ) : null}
    </div>
  );
}

/** Start an objective that has not been started. Re-submitting a started one is safe
 * server-side (it only fills in missing steps) but is not what an operator means by a
 * button on a running objective, so the button is offered only before the first run. */
export function StartObjective({ objective }: { objective: Objective }) {
  if (String(objective.state) !== "not_started" || objective.enabled === false) return null;
  const steps = Number(objective.total ?? (objective.steps ?? []).length) || 0;
  return (
    <Confirmable
      label="Start" busyLabel="Starting…" icon={Play}
      explain={<>
        Creates {steps ? `its ${steps} steps` : "its steps"} as tasks and hands each to the agent
        that owns it; steps that depend on others wait for them. Routing is checked first — if
        any step goes to an agent its owner may not delegate to, nothing is created.
      </>}
      run={async () => {
        const body: any = await post(`/objectives/${encodeURIComponent(objective.id)}/submit`, {});
        const created = body?.result?.items?.filter((i: any) => i.created).length ?? 0;
        return `Started — ${created} ${created === 1 ? "task" : "tasks"} created. Progress shows here as the agents work.`;
      }}
    />
  );
}

/** Write the tenant's declared channels into the gateway's configuration. */
export function ApplyChannels() {
  return (
    <Confirmable
      label="Apply channels" busyLabel="Applying…" icon={RadioTower}
      explain={<>
        Writes the channels declared for this tenant — which platforms are on and which agent
        answers each — into the gateway's configuration. The gateway uses it after its next
        restart; conversations already under way are not interrupted by applying.
      </>}
      run={async () => {
        const body: any = await post("/channels/apply", {});
        const routes = (body?.routes ?? []).length;
        const warnings: string[] = body?.warnings ?? [];
        return `Applied ${routes} ${routes === 1 ? "route" : "routes"}. Restart the gateway to put it into effect.`
          + (warnings.length ? ` Note: ${warnings.join(" · ")}` : "");
      }}
    />
  );
}
