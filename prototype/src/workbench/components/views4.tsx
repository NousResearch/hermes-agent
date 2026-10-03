"use client";

import { useState } from "react";
import { cn } from "../utils/cn";
import { AGENTS, agentById } from "../data/catalog";
import { CHANNELS, CRON_HISTORY, CRON_JOBS, type Channel, type CronJob } from "../data/cron";
import { useApp } from "../lib/app";
import { AgentMark, IconArrowUp, IconCheck, IconClock, IconGlobe, IconHistory, IconMore, IconPause, IconPencil, IconPlay, IconPlus, IconShield, IconTrash, IconX, IconZap } from "./Icons";
import { Badge, Bar, Button, EmptyState, IconButton, Input, Menu, MenuItem, MenuSep, Modal, Segmented, Select, Sparkline, Tip, Toggle, toneBg, toneDot, toneHex, toneText, type Tone } from "./ui";
import { Grid, PageHeader } from "./Titlebar";

/* ====================================================================
   AUTOMATIONS — cron schedules + messaging channels
   The moat, made visible: scheduled agent runs, per-platform
   identities, and per-channel approval rules. No other agent
   workbench ships cron + messaging as first-class surfaces.
   ==================================================================== */
const CH_STATUS: Record<Channel["status"], Tone> = { connected: "mint", "auth-required": "amber", off: "neutral" };
const APPROVAL_TONE: Record<Channel["approvalRule"], Tone> = { "auto <$1": "iris", "ask always": "amber", readonly: "cyan" };
const CH_CODE: Record<string, string> = { telegram: "TG", whatsapp: "WA", discord: "DC", slack: "SL", signal: "SG", matrix: "MX", imessage: "iM", email: "EM", sms: "SM", teams: "TM", line: "LN", irc: "IR", ntfy: "NT", webhook: "WH" };

/* tiny natural-language → cron parser for the create modal */
function parseCron(s: string): string {
  const t = s.toLowerCase();
  if (t.includes("weekday")) return "0 9 * * 1-5";
  if (t.includes("friday")) return "0 16 * * 5";
  if (t.includes("monday")) return "0 8 * * 1";
  if (t.includes("4h")) return "0 */4 * * *";
  if (t.includes("hour")) return "0 * * * *";
  if (t.includes("night")) return "30 2 * * *";
  if (t.includes("daily") || t.includes("morning")) return "0 6 * * *";
  if (t.includes("week")) return "0 10 * * 1";
  return "0 9 * * 1-5";
}

export function AutomationsView() {
  const { toast } = useApp();
  const [tab, setTab] = useState<"schedules" | "channels">("schedules");
  const [jobs, setJobs] = useState<CronJob[]>(CRON_JOBS);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [sched, setSched] = useState("");
  const [agentId, setAgentId] = useState("aro");
  const [channelId, setChannelId] = useState("telegram");
  const [cap, setCap] = useState("5");

  const active = jobs.filter((j) => j.enabled).length;
  const paused = jobs.length - active;
  const runs = jobs.reduce((s, j) => s + j.runsThisMonth, 0);
  const spend = jobs.reduce((s, j) => s + j.monthlyCost, 0);
  const capTotal = jobs.reduce((s, j) => s + j.costCap, 0);
  const spendPct = Math.min(100, Math.round((spend / Math.max(capTotal, 1)) * 100));
  const parsed = parseCron(sched);

  const setEnabled = (id: string, v: boolean) => setJobs((all) => all.map((j) => (j.id === id ? { ...j, enabled: v } : j)));

  const create = () => {
    setJobs((all) => [{
      id: `j${Date.now()}`,
      name: name.trim(),
      schedule: sched.trim() || "weekdays · 09:00",
      cron: parsed,
      agentId,
      channel: channelId === "dashboard" ? null : channelId,
      enabled: true,
      lastRun: { ok: true, at: "—", summary: "no runs yet" },
      nextRun: "Monday 08:00",
      runsThisMonth: 0,
      monthlyCost: 0,
      costCap: Number(cap),
      tone: "iris",
    }, ...all]);
    setCreating(false);
    setName("");
    setSched("");
    toast("Automation created — first run Monday 08:00", "mint");
  };

  return (
    <div className="scroll-thin flex-1 overflow-y-auto">
      <PageHeader eyebrow="automations" title="Cron & channels" sub="The only agent workbench where automations and messaging are first-class: scheduled agent runs, per-platform identities, and approval rules."
        right={<div className="flex flex-wrap items-center gap-2">
          <Segmented value={tab} onChange={setTab} items={[{ value: "schedules", label: "Schedules", icon: IconClock }, { value: "channels", label: "Channels", icon: IconGlobe }]} />
          <Button variant="primary" icon={IconPlus} onClick={() => setCreating(true)}>New automation</Button>
        </div>} />

      <div className="space-y-4 p-5">
        <Grid className="grid-cols-2 lg:grid-cols-4">
          <div className="rounded-[11px] border border-line-soft bg-raise p-3">
            <div className="font-mono text-[9.5px] tracking-[.12em] text-ink-4 uppercase">active automations</div>
            <div className="mt-1 flex items-baseline gap-1.5">
              <span className="font-display text-[22px] leading-none font-semibold tracking-[-.02em] text-ink">{active}</span>
              <span className="font-mono text-[10px] text-ink-4">of {jobs.length}</span>
            </div>
            <div className="mt-1 text-[11px] text-ink-3">{paused} paused · 0 missed fires this month</div>
          </div>
          <div className="rounded-[11px] border border-line-soft bg-raise p-3">
            <div className="font-mono text-[9.5px] tracking-[.12em] text-ink-4 uppercase">runs this month</div>
            <div className="mt-1 flex items-baseline gap-1.5">
              <span className="font-display text-[22px] leading-none font-semibold tracking-[-.02em] text-ink">{runs}</span>
              <span className="font-mono text-[10px] text-ink-4">cron</span>
            </div>
            <div className="mt-1 text-[11px] text-ink-3">≈ {Math.max(1, Math.round(runs / 30))} / day · across {jobs.length} jobs</div>
          </div>
          <div className="rounded-[11px] border border-line-soft bg-raise p-3">
            <div className="font-mono text-[9.5px] tracking-[.12em] text-ink-4 uppercase">monthly spend</div>
            <div className="mt-1 flex items-baseline gap-1.5">
              <span className="font-display text-[22px] leading-none font-semibold tracking-[-.02em] text-ink">${spend.toFixed(2)}</span>
              <span className="font-mono text-[10px] text-ink-4">of ${capTotal} cap</span>
            </div>
            <Bar value={spendPct} tone={spendPct > 85 ? "amber" : "mint"} className="mt-2.5" />
            <div className="mt-1 text-[11px] text-ink-3">{spendPct}% of cap · resets in 3d</div>
          </div>
          <div className="rounded-[11px] border border-line-soft bg-raise p-3">
            <div className="font-mono text-[9.5px] tracking-[.12em] text-ink-4 uppercase">success rate</div>
            <div className="mt-1 flex items-baseline gap-1.5">
              <span className="font-display text-[22px] leading-none font-semibold tracking-[-.02em] text-mint">96%</span>
              <span className="font-mono text-[10px] text-ink-4">30d</span>
            </div>
            <div className="mt-1 text-[11px] text-ink-3">2 failures · both auto-retried</div>
          </div>
        </Grid>

        {tab === "schedules" ? (
          <>
            <div className="overflow-hidden rounded-[12px] border border-line-soft bg-raise">
              <div className="flex items-center gap-3 border-b border-line-soft bg-well/40 px-3.5 py-[6px] font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">
                <span className="w-[30px] shrink-0" />
                <span className="min-w-0 flex-1">automation</span>
                <span className="hidden w-[118px] shrink-0 lg:block">agent</span>
                <span className="hidden w-[104px] shrink-0 xl:block">channel</span>
                <span className="hidden w-[186px] shrink-0 md:block">last run</span>
                <span className="hidden w-[84px] shrink-0 sm:block">next</span>
                <span className="hidden w-[44px] shrink-0 text-right xl:block">runs</span>
                <span className="w-[92px] shrink-0">spend / cap</span>
                <span className="w-[26px] shrink-0" />
              </div>
              {jobs.length === 0 && (
                <EmptyState icon={IconClock} title="No automations" body="Describe a schedule in plain English — Aro parses it to cron and runs it with any agent, delivered to any channel."
                  action={<Button variant="primary" size="sm" icon={IconPlus} onClick={() => setCreating(true)}>New automation</Button>} />
              )}
              {jobs.map((j) => {
                const a = agentById(j.agentId);
                const ch = j.channel ? CHANNELS.find((c) => c.id === j.channel) : null;
                const over = j.monthlyCost / Math.max(j.costCap, 1) > 0.8;
                return (
                  <div key={j.id} className={cn("flex items-center gap-3 border-b border-line-soft/70 px-3.5 py-2.5 transition-colors last:border-0 hover:bg-hover/50", !j.enabled && "opacity-55")}>
                    <Toggle size="sm" checked={j.enabled} onChange={(v) => { setEnabled(j.id, v); toast(`${v ? "Resumed" : "Paused"} ${j.name}`, v ? "mint" : "amber"); }} />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        <i className={cn("size-[5px] shrink-0 rounded-full", j.enabled ? toneDot[j.tone] : "bg-line-strong")} />
                        <span className="truncate text-[12.5px] font-medium text-ink">{j.name}</span>
                      </div>
                      <div className="mt-0.5 flex items-center gap-1.5 pl-[11px]">
                        <span className="truncate text-[10.5px] text-ink-4">{j.schedule}</span>
                        <code className="shrink-0 rounded-[4px] bg-well px-1 py-[1px] font-mono text-[9px] text-ink-4">{j.cron}</code>
                      </div>
                    </div>
                    <div className="hidden w-[118px] shrink-0 items-center gap-1.5 lg:flex">
                      <AgentMark glyph={a.glyph} from={a.from} to={a.to} size={16} />
                      <span className="truncate font-mono text-[10px] text-ink-3">{a.name}</span>
                    </div>
                    <div className="hidden w-[104px] shrink-0 xl:block">
                      {ch ? <Tip label={ch.identity}><Badge tone="neutral" mono dot className="text-[9px]">{ch.platform}</Badge></Tip> : <span className="rounded-[4px] bg-hover px-1.5 py-[2px] font-mono text-[9px] text-ink-4">dashboard</span>}
                    </div>
                    <div className="hidden w-[186px] shrink-0 md:block">
                      <span className={cn("flex items-center gap-1.5 font-mono text-[10px]", j.lastRun.ok ? "text-ink-3" : "text-rose")}>
                        {j.lastRun.ok ? <IconCheck size={10} className="shrink-0 text-mint" /> : <IconX size={10} className="shrink-0" />}
                        <span className="truncate">{j.lastRun.at} · {j.lastRun.summary}</span>
                      </span>
                    </div>
                    <span className={cn("hidden w-[84px] shrink-0 font-mono text-[10px] sm:block", j.enabled ? "text-ink-3" : "text-ink-4")}>{j.enabled ? j.nextRun : "paused"}</span>
                    <span className="hidden w-[44px] shrink-0 text-right font-mono text-[10px] text-ink-4 xl:block">{j.runsThisMonth}</span>
                    <div className="w-[92px] shrink-0">
                      <div className="flex items-baseline justify-between gap-1 font-mono text-[9.5px]">
                        <span className={over ? "text-amber" : "text-ink-3"}>${j.monthlyCost.toFixed(2)}</span>
                        <span className="text-ink-4">${j.costCap}</span>
                      </div>
                      <Bar value={(j.monthlyCost / Math.max(j.costCap, 1)) * 100} tone={over ? "amber" : "iris"} height={3} className="mt-1" />
                    </div>
                    <Menu align="right" width={198} trigger={() => <IconButton icon={IconMore} label="Automation actions" size={24} />}>
                      {(close) => (<>
                        <MenuItem icon={<IconPlay size={12} />} label="Run now" onClick={() => { close(); toast(`Running ${j.name} now · results to ${ch ? ch.identity : "dashboard"}`, "iris"); }} />
                        <MenuItem icon={j.enabled ? <IconPause size={12} /> : <IconPlay size={12} />} label={j.enabled ? "Pause" : "Resume"} onClick={() => { close(); setEnabled(j.id, !j.enabled); toast(`${j.enabled ? "Paused" : "Resumed"} ${j.name}`, j.enabled ? "amber" : "mint"); }} />
                        <MenuItem icon={<IconPencil size={12} />} label="Edit" onClick={() => { close(); toast(`Editing ${j.name} · cron ${j.cron}`, "iris"); }} />
                        <MenuSep />
                        <MenuItem danger icon={<IconTrash size={12} />} label="Delete" onClick={() => { close(); setJobs((all) => all.filter((x) => x.id !== j.id)); toast(`${j.name} deleted`, "rose"); }} />
                      </>)}
                    </Menu>
                  </div>
                );
              })}
            </div>

            <div className="overflow-hidden rounded-[12px] border border-line-soft bg-raise">
              <div className="flex items-center justify-between border-b border-line-soft px-3.5 py-2.5">
                <h3 className="flex items-center gap-1.5 font-display text-[13px] font-semibold text-ink"><IconHistory size={12} className="text-iris-soft" />Recent runs</h3>
                <span className="font-mono text-[10px] text-ink-4">daemon 127.0.0.1:4733 · drift +0.2s</span>
              </div>
              {CRON_HISTORY.map((h) => {
                const job = CRON_JOBS.find((j) => j.id === h.jobId);
                return (
                  <div key={h.id} className="flex items-center gap-2.5 border-b border-line-soft/60 px-3.5 py-2 transition-colors last:border-0 hover:bg-hover/40">
                    <i className={cn("size-[6px] shrink-0 rounded-full", h.ok ? toneDot.mint : toneDot.rose)} />
                    <span className="w-[74px] shrink-0 font-mono text-[9.5px] text-ink-4">{h.at}</span>
                    <span className="w-[150px] shrink-0 truncate text-[11.5px] font-medium text-ink-2">{job?.name ?? "deleted job"}</span>
                    <span className="min-w-0 flex-1 truncate text-[11px] text-ink-3">{h.note}</span>
                    {!h.ok && <Badge tone="rose" mono className="shrink-0 text-[9px]">failed</Badge>}
                    <span className="w-[58px] shrink-0 text-right font-mono text-[10px] text-ink-4">{h.duration}</span>
                    <span className="w-[42px] shrink-0 text-right font-mono text-[10px] text-ink-3">${h.cost.toFixed(2)}</span>
                  </div>
                );
              })}
            </div>
          </>
        ) : (
          <>
            <div className="flex flex-col gap-4 rounded-[12px] border border-line-soft bg-raise p-4 lg:flex-row lg:items-stretch">
              <div className="flex items-start gap-3 lg:w-[42%] lg:shrink-0">
                <span className="flex size-[34px] shrink-0 items-center justify-center rounded-[10px] border border-line-soft bg-well text-iris-soft"><IconGlobe size={15} /></span>
                <div className="min-w-0">
                  <h3 className="font-display text-[13.5px] font-semibold text-ink">Aro lives where you do</h3>
                  <p className="mt-1 text-[11.5px] leading-[1.55] text-ink-3">Pick up any conversation from any platform — same memory, same approvals. Fourteen platforms, one identity, nothing lost in the handoff.</p>
                  <div className="mt-2.5 flex flex-wrap gap-1.5">
                    <Badge tone="mint" mono dot>{CHANNELS.filter((c) => c.status === "connected").length} connected</Badge>
                    <Badge tone="amber" mono>{CHANNELS.filter((c) => c.status === "auth-required").length} need auth</Badge>
                    <Badge tone="neutral" mono>{CHANNELS.length} platforms</Badge>
                  </div>
                </div>
              </div>
              <div className="min-w-0 flex-1 rounded-[10px] border border-line-soft bg-code px-3 py-2.5">
                <div className="mb-1.5 flex items-center gap-1.5 font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase"><IconArrowUp size={10} className="rotate-90 text-mint" />continuity · one thread, any surface</div>
                {[["09:41", "telegram", "@aro_agent", "“ship the release notes?” — thread #4812 opens"], ["11:03", "desktop", "workbench", "same thread · full memory · zero re-explaining"], ["14:27", "email", "aro@acme.dev", "digest lands · replies fold back into #4812"]].map(([t, p, who, msg]) => (
                  <div key={t} className="flex items-center gap-2.5 py-[3px] font-mono text-[10px]">
                    <span className="w-[32px] shrink-0 text-ink-4">{t}</span>
                    <span className={cn("w-[58px] shrink-0", p === "telegram" ? "text-mint" : p === "desktop" ? "text-iris-soft" : "text-sky")}>{p}</span>
                    <span className="w-[96px] shrink-0 truncate text-ink-3">{who}</span>
                    <span className="min-w-0 flex-1 truncate text-ink-4">{msg}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {CHANNELS.map((c) => {
                const tone = CH_STATUS[c.status];
                const linked = jobs.filter((j) => j.channel === c.id && j.enabled).length;
                return (
                  <article key={c.id} className={cn("flex flex-col overflow-hidden rounded-[12px] border bg-raise transition-all duration-200 hover:-translate-y-px hover:shadow-e3", c.status === "off" ? "border-dashed border-line opacity-80" : "border-line-soft hover:border-line-strong")}>
                    <div className="flex items-start gap-3 p-3.5">
                      <span className={cn("flex size-[32px] shrink-0 items-center justify-center rounded-[9px] border border-line-soft bg-well font-mono text-[10.5px] font-bold", toneText[tone])}>{CH_CODE[c.id] ?? c.platform.slice(0, 2).toUpperCase()}</span>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-1.5"><h3 className="truncate font-display text-[13.5px] font-semibold text-ink">{c.platform}</h3><Badge tone={tone} mono dot className="text-[9px]">{c.status === "auth-required" ? "auth required" : c.status}</Badge></div>
                        <code className="mt-0.5 block truncate font-mono text-[10.5px] text-ink-3">{c.identity}</code>
                        <div className="mt-1.5 flex items-center gap-2">
                          <span className="font-mono text-[9.5px] text-ink-4">{c.msgs24h} msgs/24h</span>
                          <Sparkline data={c.spark} w={56} h={14} color={toneHex(tone)} />
                        </div>
                      </div>
                    </div>
                    <p className="min-h-[34px] px-3.5 pb-2.5 text-[11px] leading-[1.55] text-ink-3">{c.note}</p>
                    <div className="mt-auto flex items-center gap-2 border-t border-line-soft px-3 py-2.5">
                      <span className={cn("inline-flex shrink-0 items-center gap-1 rounded-[6px] px-1.5 py-[3px] font-mono text-[9.5px]", toneBg[APPROVAL_TONE[c.approvalRule]])}><IconShield size={9} />{c.approvalRule}</span>
                      {linked >= 2 && <span className="truncate font-mono text-[9px] text-ink-4">paired with {linked} automations</span>}
                      <Button variant={c.status === "connected" ? "ghost" : "secondary"} size="xs" className="ml-auto" onClick={() => toast(c.status === "connected" ? `${c.platform} settings — identity, approvals, quiet hours` : `Connecting ${c.platform} · auth opens in a side pane`, "iris")}>{c.status === "connected" ? "Configure" : "Connect"}</Button>
                    </div>
                  </article>
                );
              })}
            </div>
          </>
        )}
      </div>

      <Modal open={creating} onClose={() => setCreating(false)} title="New automation" sub="Say when it runs in plain language — Aro parses it to cron, runs it with any agent, and delivers anywhere you live." width={560}>
        <div className="space-y-3.5 p-4">
          <div>
            <label className="mb-1 block font-mono text-[9.5px] tracking-[.12em] text-ink-4 uppercase">name</label>
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="release-notes-digest" className="font-mono text-[11.5px]" />
          </div>
          <div>
            <label className="mb-1 block font-mono text-[9.5px] tracking-[.12em] text-ink-4 uppercase">schedule</label>
            <Input value={sched} onChange={(e) => setSched(e.target.value)} placeholder="every weekday at 9am" />
            <div className="mt-1.5 flex items-center gap-1.5 font-mono text-[10px] text-ink-4">
              <IconClock size={10} className="shrink-0" />
              <span>parses to</span>
              <code className="rounded-[4px] bg-well px-1 py-[1px] text-iris-soft">{parsed}</code>
            </div>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <div>
              <label className="mb-1 block font-mono text-[9.5px] tracking-[.12em] text-ink-4 uppercase">agent</label>
              <Select value={agentId} onChange={setAgentId} options={AGENTS.filter((a) => a.status === "connected").map((a) => ({ value: a.id, label: a.name, hint: a.vendor }))} />
            </div>
            <div>
              <label className="mb-1 block font-mono text-[9.5px] tracking-[.12em] text-ink-4 uppercase">deliver to</label>
              <Select value={channelId} onChange={setChannelId} options={[{ value: "dashboard", label: "dashboard only" }, ...CHANNELS.filter((c) => c.status === "connected").map((c) => ({ value: c.id, label: c.platform, hint: c.identity }))]} />
            </div>
            <div>
              <label className="mb-1 block font-mono text-[9.5px] tracking-[.12em] text-ink-4 uppercase">cost cap</label>
              <Select value={cap} onChange={setCap} options={[{ value: "1", label: "$1 / month" }, { value: "5", label: "$5 / month" }, { value: "20", label: "$20 / month" }]} />
            </div>
          </div>
          <div className="flex items-start gap-2 rounded-[9px] border border-line-soft bg-well px-2.5 py-2 text-[11px] leading-[1.5] text-ink-3"><IconShield size={11} className="mt-0.5 shrink-0 text-iris-soft" />Runs inherit the agent's permission mode. Anything that leaves the workspace — spends money, messages a human — asks first.</div>
        </div>
        <div className="flex items-center justify-end gap-2 border-t border-line-soft bg-raise/60 px-4 py-3">
          <Button variant="ghost" onClick={() => setCreating(false)}>Cancel</Button>
          <Button variant="primary" icon={IconZap} disabled={!name.trim()} onClick={create}>Create</Button>
        </div>
      </Modal>
    </div>
  );
}
