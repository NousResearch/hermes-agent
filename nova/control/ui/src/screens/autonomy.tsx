import * as React from "react";
import { AlertTriangle, ArrowDownCircle, Check, Flag, Sparkles } from "lucide-react";
import { Chip, EmptyState, GlassPanel, SectionHeader, StatusPill } from "@/components/glass";
import { PanelBody } from "@/components/panel";
import { post } from "@/lib/api";
import type { Loaded } from "@/lib/api";

/* Earned autonomy: each approval-gated action, the record it has earned, and the acts on it.
 *
 * The false-safe count is shown first and in the warning colour, before agreement: a
 * percentage that looks good can hide the one case that matters. Promotion is proposed by
 * NOVA and confirmed here by an administrator; demotion is automatic, and also one click.
 * Anything the gate cannot observe is said in words rather than shown as a number. */

type Tally = {
  escalations: number; approved_unchanged: number; approved_edited: number | null; rejected: number;
  timed_out: number; autonomous_executed: number; autonomous_open: number; incidents: number;
  shadow_reviewed: number; shadow_agreed: number; false_safe: number;
  agreement: number | null; false_safe_rate: number | null;
};
type Row = {
  action: string; state: "supervised" | "proposed" | "graduated"; model_version: string;
  proposal: { model_version: string; ts: string } | null;
  assessment: {
    eligible: boolean; waiting_on: string[]; demote_because: string[];
    progress: { reviewed: number; needed: number; agreement: number | null; min_agreement: number;
                false_safe: number; max_false_safe: number; window: number };
  };
  ledger: { all_time: Tally; window: Tally; by_agent: Record<string, Tally> } | null;
  last_change: { state: string; changed_at: string; changed_by: string; reason: string } | null;
  last_demotion: string | null;
};
type Verdict = {
  agent_id: string; action: string; ts: string; verdict: string; mode: string; model_version: string;
  failed: Array<{ id: string; why: string }>; outcome: string; decided_by: string;
};
export type AutonomyPayload = {
  configured: boolean; provider?: string; mode?: string; data?: string;
  actions: Row[]; recent: Verdict[]; notices: Array<{ kind: string; text: string }>; caveats?: string[];
};

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${Math.round(v * 1000) / 10}%`);
const words = (id: string) => id.replace(/_/g, " ");

function StateBadge({ state }: { state: Row["state"] }) {
  const tone = state === "graduated" ? "running" : state === "proposed" ? "waiting" : "neutral";
  const label = state === "graduated" ? "Graduated" : state === "proposed" ? "Proposed" : "Supervised";
  return <StatusPill state={tone}>{label}</StatusPill>;
}

function Progress({ row }: { row: Row }) {
  const p = row.assessment.progress;
  const share = Math.min(1, p.needed ? p.reviewed / p.needed : 0);
  return (
    <div>
      <div className="text-ink-muted flex justify-between text-[12px]">
        <span>{p.reviewed} of {p.needed} reviewed safe verdicts</span>
        <span>window {p.window}</span>
      </div>
      <div className="bg-glass-border mt-1 h-1.5 w-full overflow-hidden rounded-full"
           role="progressbar" aria-valuemin={0} aria-valuemax={p.needed} aria-valuenow={p.reviewed}
           aria-label={`Progress toward promotion for ${words(row.action)}`}>
        <div className="bg-accent h-full rounded-full" style={{ width: `${share * 100}%` }} />
      </div>
    </div>
  );
}

function Act({ row, canAct, onDone }: { row: Row; canAct: boolean; onDone: () => void }) {
  const [mode, setMode] = React.useState<"" | "promote" | "demote" | "incident">("");
  const [why, setWhy] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [result, setResult] = React.useState<{ ok: boolean; text: string } | null>(null);
  if (!canAct) {
    return <p className="text-ink-faint mt-3 text-[12px]">Only an administrator can promote or demote an action.</p>;
  }
  const run = async () => {
    setBusy(true);
    setResult(null);
    try {
      const body: any = await post(`/autonomy/${encodeURIComponent(row.action)}/${mode}`, {
        reason: why.trim(), autonomous: mode === "incident",
      });
      setResult({ ok: true, text: `Done — ${words(row.action)} is now ${body?.state ?? "updated"}.` });
      setMode("");
      setWhy("");
      onDone();
    } catch (error) {
      setResult({ ok: false, text: error instanceof Error ? error.message : String(error) });
    } finally {
      setBusy(false);
    }
  };
  const explain = {
    promote: `Calls to ${words(row.action)} that pass every triage check will run without a person, on model ${row.proposal?.model_version ?? ""}. Any rejection, incident or model change sends it back to supervised. Recorded under your name.`,
    demote: "Every call goes to a person again, from the next call on. Recorded under your name.",
    incident: "Records that a call this action made without a person was wrong. The action is demoted immediately.",
    "": "",
  }[mode];
  return (
    <div className="border-glass-border mt-3 border-t pt-3">
      {mode ? (
        <div className="space-y-2">
          <p className="text-ink text-[12.5px]">{explain}</p>
          {mode !== "promote" ? (
            <textarea value={why} onChange={(e) => setWhy(e.target.value)} rows={2}
              aria-label="Reason" placeholder={mode === "incident" ? "What went wrong (required)" : "Why (optional)"}
              className="glass-solid text-ink w-full rounded-lg p-2 text-[12.5px]" />
          ) : null}
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={() => void run()} disabled={busy || (mode === "incident" && !why.trim())}
              className="bg-accent text-accent-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium disabled:opacity-60">
              {busy ? "Working…" : mode === "promote" ? "Confirm promotion" : mode === "demote" ? "Demote now" : "Record and demote"}
            </button>
            <button type="button" disabled={busy} onClick={() => setMode("")}
              className="glass-solid text-ink rounded-lg px-3 py-1.5 text-[12.5px] font-medium">Cancel</button>
          </div>
        </div>
      ) : (
        <div className="flex flex-wrap gap-2">
          {row.state === "proposed" ? (
            <button type="button" onClick={() => setMode("promote")}
              className="bg-accent text-accent-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium">
              <Check className="size-3.5" /> Approve promotion
            </button>
          ) : null}
          {row.state === "graduated" ? (
            <>
              <button type="button" onClick={() => setMode("demote")}
                className="glass-solid text-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium">
                <ArrowDownCircle className="size-3.5" /> Demote now
              </button>
              <button type="button" onClick={() => setMode("incident")}
                className="glass-solid text-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium">
                <Flag className="size-3.5" /> Report a wrong call
              </button>
            </>
          ) : null}
        </div>
      )}
      {result ? (
        <p role="status" className={`mt-2 text-[12px] ${result.ok ? "text-running" : "text-blocked"}`}>{result.text}</p>
      ) : null}
    </div>
  );
}

function ActionCard({ row, canAct, onDone }: { row: Row; canAct: boolean; onDone: () => void }) {
  const w = row.ledger?.window;
  const all = row.ledger?.all_time;
  return (
    <GlassPanel className="p-5">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-ink min-w-0 flex-1 text-[14px] font-semibold capitalize">{words(row.action)}</h3>
        <StateBadge state={row.state} />
        {row.state === "graduated" && row.model_version ? <Chip>{row.model_version}</Chip> : null}
      </div>
      <dl className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div>
          <dt className="text-ink-faint text-[11.5px]">False-safe (window)</dt>
          <dd className={`text-[18px] font-semibold ${(w?.false_safe ?? 0) > 0 ? "text-blocked" : "text-ink"}`}>
            {w?.false_safe ?? 0}
          </dd>
        </div>
        <div>
          <dt className="text-ink-faint text-[11.5px]">Shadow agreement</dt>
          <dd className="text-ink text-[18px] font-semibold">{pct(w?.agreement)}</dd>
        </div>
        <div>
          <dt className="text-ink-faint text-[11.5px]">Escalations (all time)</dt>
          <dd className="text-ink text-[18px] font-semibold">{all?.escalations ?? 0}</dd>
        </div>
        <div>
          <dt className="text-ink-faint text-[11.5px]">Ran without a person</dt>
          <dd className="text-ink text-[18px] font-semibold">{all?.autonomous_executed ?? 0}</dd>
        </div>
      </dl>
      <div className="mt-4"><Progress row={row} /></div>
      {row.state === "supervised" && row.assessment.waiting_on.length ? (
        <ul className="text-ink-muted mt-3 list-disc space-y-0.5 pl-5 text-[12px]">
          {row.assessment.waiting_on.map((w) => <li key={w}>{w}</li>)}
        </ul>
      ) : null}
      {row.last_demotion ? (
        <p className="text-waiting mt-3 flex items-start gap-1.5 text-[12px]">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" /> Last demoted: {row.last_demotion}
        </p>
      ) : null}
      {row.ledger && Object.keys(row.ledger.by_agent).length > 1 ? (
        <p className="text-ink-faint mt-2 text-[11.5px]">
          By agent: {Object.entries(row.ledger.by_agent).map(([a, t]) => `${a} ${t.shadow_agreed}/${t.shadow_reviewed}`).join(" · ")}
        </p>
      ) : null}
      <Act row={row} canAct={canAct} onDone={onDone} />
    </GlassPanel>
  );
}

export function AutonomyScreen({ autonomy, canAct, onChanged }: {
  autonomy: Loaded<AutonomyPayload>; canAct: boolean; onChanged: () => void;
}) {
  return (
    <PanelBody state={autonomy} empty={(d) => d.configured ? null : {
      title: "Autonomy is not configured",
      detail: "Add an autonomy block to policy.yaml to triage approval-gated calls. Start in shadow mode: nothing changes for your people, and the record starts building.",
    }}>
      {(data) => (
        <div className="space-y-6">
          {data.notices.map((n) => (
            <GlassPanel key={n.kind} solid className="flex items-start gap-2 p-4">
              {n.kind === "full_args" ? <AlertTriangle className="text-waiting mt-0.5 size-4 shrink-0" />
                : <Sparkles className="text-ink-muted mt-0.5 size-4 shrink-0" />}
              <span className="text-ink text-[12.5px]">{n.text}</span>
            </GlassPanel>
          ))}
          <div className="grid gap-4 lg:grid-cols-2">
            {data.actions.map((row) => <ActionCard key={row.action} row={row} canAct={canAct} onDone={onChanged} />)}
          </div>
          <GlassPanel className="p-5">
            <SectionHeader title="Recent triage verdicts"
              detail="What triage said about calls that went to a person, and what the person decided." />
            {data.recent.length ? (
              <ul className="divide-glass-border divide-y">
                {data.recent.map((v, i) => (
                  <li key={i} className="py-2.5 first:pt-0 last:pb-0">
                    <div className="flex flex-wrap items-center gap-2 text-[12.5px]">
                      <StatusPill state={v.verdict === "auto_ok" ? "running" : "waiting"}>
                        {v.verdict === "auto_ok" ? "Safe" : "Ask a person"}
                      </StatusPill>
                      <span className="text-ink capitalize">{words(v.action)}</span>
                      <span className="text-ink-faint">{v.agent_id}</span>
                      <span className="text-ink-faint ml-auto text-[11.5px]">
                        {v.outcome ? `${v.outcome.replace("_", " ")}${v.decided_by ? ` by ${v.decided_by}` : ""}` : "not answered yet"}
                      </span>
                    </div>
                    {v.failed.length ? (
                      <ul className="text-ink-muted mt-1 space-y-0.5 pl-1 text-[12px]">
                        {v.failed.map((f, j) => <li key={j}><span className="capitalize">{words(f.id || "triage")}</span>: {f.why}</li>)}
                      </ul>
                    ) : null}
                  </li>
                ))}
              </ul>
            ) : (
              <EmptyState icon={Sparkles} title="No verdicts yet"
                detail="They appear here as soon as a triaged action is escalated to a person." />
            )}
          </GlassPanel>
          {data.caveats?.length ? (
            <ul className="text-ink-faint list-disc space-y-0.5 pl-5 text-[11.5px]">
              {data.caveats.map((c) => <li key={c}>{c}</li>)}
            </ul>
          ) : null}
        </div>
      )}
    </PanelBody>
  );
}
