"use client";

import React, { useState } from "react";
import { cn } from "../utils/cn";
import type { DiffFile, DiffLine, Step } from "../data/catalog";
import { agentById } from "../data/catalog";
import { IconBrain, IconCheck, IconChevronDown, IconFile, IconMcp, IconSearch, IconShield, IconSpark, IconTerminal, IconUndo, IconWarning, IconList, IconEye, IconX, IconGrid } from "./Icons";
import { Badge, Button, Kbd, Segmented } from "./ui";

/* ===================== inline markdown-ish ===================== */
function Inline({ text }: { text: string }) {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).filter(Boolean);
  return <>{parts.map((p, i) => p.startsWith("`") ? <code key={i} className="rounded-[4px] border border-line-soft bg-well px-[4px] py-[1px] font-mono text-[11.5px] text-cyan">{p.slice(1, -1)}</code>
    : p.startsWith("**") ? <strong key={i} className="font-semibold text-ink">{p.slice(2, -2)}</strong> : <span key={i}>{p}</span>)}</>;
}
export function Prose({ text, className }: { text: string; className?: string }) {
  return (
    <div className={cn("space-y-2.5 text-[13.5px] leading-[1.65] text-ink-2", className)}>
      {text.split("\n\n").map((b, i) => {
        const lines = b.split("\n");
        if (b.startsWith("## ")) return <h4 key={i} className="pt-1 text-[13.5px] font-semibold text-ink">{b.slice(3)}</h4>;
        if (lines.every((l) => /^\s*([-*]|\d+\.)\s/.test(l)))
          return <ul key={i} className="space-y-1.5 pl-0.5">{lines.map((l, j) => <li key={j} className="flex gap-2"><span className="mt-[8px] size-[4px] shrink-0 rounded-full bg-iris-soft/70" /><span><Inline text={l.replace(/^\s*([-*]|\d+\.)\s/, "")} /></span></li>)}</ul>;
        return <p key={i}><Inline text={b} /></p>;
      })}
    </div>
  );
}

/* ============================== DIFF VIEW ============================== */
function pairLines(lines: DiffLine[]) {
  const rows: { l?: DiffLine; r?: DiffLine; hunk?: string }[] = [];
  let i = 0;
  while (i < lines.length) {
    const ln = lines[i];
    if (ln.t === "hunk") { rows.push({ hunk: ln.text }); i++; continue; }
    if (ln.t === "ctx") { rows.push({ l: ln, r: ln }); i++; continue; }
    const dels: DiffLine[] = [], adds: DiffLine[] = [];
    while (i < lines.length && lines[i].t === "del") dels.push(lines[i++]);
    while (i < lines.length && lines[i].t === "add") adds.push(lines[i++]);
    for (let k = 0; k < Math.max(dels.length, adds.length); k++) rows.push({ l: dels[k], r: adds[k] });
  }
  return rows;
}
export function DiffView({ file, maxLines = 22, mode: forced, showToggle }: { file: DiffFile; maxLines?: number; mode?: "unified" | "split"; showToggle?: boolean }) {
  const [all, setAll] = useState(false);
  const [mode, setMode] = useState<"unified" | "split">(forced ?? "unified");
  const shown = all ? file.lines : file.lines.slice(0, maxLines);
  const Cell = ({ l, side }: { l?: DiffLine; side: "l" | "r" }) => (
    <div className={cn("flex min-w-0 flex-1 items-start", l?.t === "add" && "diff-add", l?.t === "del" && "diff-del", !l && "bg-well/40")}>
      <span className="w-8 shrink-0 py-[1px] pr-2 text-right text-ink-4/70 select-none">{l ? (side === "l" ? l.o ?? "" : l.n ?? "") : ""}</span>
      <span className={cn("truncate py-[1px] pr-3 whitespace-pre", l?.t === "add" ? "text-mint" : l?.t === "del" ? "text-rose" : "text-ink-2")}>{l?.text || " "}</span>
    </div>
  );
  return (
    <div className="overflow-hidden rounded-[8px] border border-line-soft bg-sunken">
      <div className="flex items-center justify-between gap-2 border-b border-line-soft bg-well px-2.5 py-[6px]">
        <div className="flex min-w-0 items-center gap-1.5"><IconFile size={11} className="shrink-0 text-ink-4" /><span className="truncate font-mono text-[11px] text-ink-2">{file.path}</span></div>
        <div className="flex shrink-0 items-center gap-2 font-mono text-[10px]">
          <span className="text-mint">+{file.adds}</span><span className="text-rose">−{file.dels}</span>
          {showToggle && <Segmented value={mode} onChange={setMode} items={[{ value: "unified", label: "Unified" }, { value: "split", label: "Split" }]} className="ml-1 scale-90" />}
        </div>
      </div>
      <div className="scroll-thin overflow-x-auto font-mono text-[11.5px] leading-[1.65]">
        {mode === "unified" ? shown.map((l, i) => (
          <div key={i} className={cn("flex items-start whitespace-pre", l.t === "add" && "diff-add", l.t === "del" && "diff-del", l.t === "hunk" && "bg-iris-tint/50")}>
            {l.t !== "hunk" ? (<>
              <span className="w-8 shrink-0 py-[1px] pr-2 text-right text-ink-4/70 select-none">{l.o ?? ""}</span>
              <span className="w-8 shrink-0 py-[1px] pr-2 text-right text-ink-4/70 select-none">{l.n ?? ""}</span>
              <span className="w-3 shrink-0 py-[1px] text-ink-4/60 select-none">{l.t === "add" ? "+" : l.t === "del" ? "−" : " "}</span>
              <span className={cn("py-[1px] pr-3", l.t === "add" ? "text-mint" : l.t === "del" ? "text-rose" : "text-ink-2")}>{l.text || " "}</span>
            </>) : <span className="py-[2px] pl-3 text-ink-3">{l.text}</span>}
          </div>
        )) : pairLines(shown).map((r, i) => r.hunk ? <div key={i} className="bg-iris-tint/50 py-[2px] pl-3 text-ink-3">{r.hunk}</div>
          : <div key={i} className="flex divide-x divide-line-soft"><Cell l={r.l} side="l" /><Cell l={r.r} side="r" /></div>)}
      </div>
      {file.lines.length > maxLines && (
        <button onClick={() => setAll((a) => !a)} className="flex w-full cursor-pointer items-center justify-center gap-1.5 border-t border-line-soft py-[6px] text-[11px] text-ink-3 transition-colors hover:bg-hover hover:text-ink">
          {all ? "Collapse" : `Show ${file.lines.length - maxLines} more lines`}<IconChevronDown size={11} className={cn("transition-transform", all && "rotate-180")} />
        </button>
      )}
    </div>
  );
}

/* ============================== TOOL CARD ============================== */
type ToolKind = "read" | "edit" | "write" | "bash" | "grep" | "mcp" | "browser" | "test" | "plan";
const toolMeta: Record<ToolKind, { icon: React.ComponentType<{ size?: number; className?: string }>; tone: string; label: string }> = {
  read: { icon: IconFile, tone: "text-sky", label: "read" }, edit: { icon: IconSpark, tone: "text-iris-soft", label: "edit" }, write: { icon: IconSpark, tone: "text-iris-soft", label: "write" },
  bash: { icon: IconTerminal, tone: "text-cyan", label: "shell" }, grep: { icon: IconSearch, tone: "text-plum", label: "search" }, mcp: { icon: IconMcp, tone: "text-mint", label: "tool" },
  browser: { icon: IconEye, tone: "text-sky", label: "browser" }, test: { icon: IconCheck, tone: "text-mint", label: "test" }, plan: { icon: IconList, tone: "text-amber", label: "plan" },
};
export function ToolCard({ step }: { step: Extract<Step, { type: "tool" }> }) {
  const meta = toolMeta[step.tool]; const Icon = meta.icon;
  const [open, setOpen] = useState(step.status === "running" || step.status === "error" || Boolean(step.diff));
  const hasBody = Boolean(step.lines?.length || step.output?.length || step.diff);
  return (
    <div className="animate-rise overflow-hidden rounded-[9px] border border-line-soft bg-well">
      <button onClick={() => hasBody && setOpen((o) => !o)} className={cn("flex w-full items-center gap-2 px-2.5 py-[8px] text-left", hasBody && "cursor-pointer hover:bg-hover/60")}>
        <span className={cn("flex size-[20px] shrink-0 items-center justify-center rounded-[5px] border border-line-soft bg-raise", meta.tone)}><Icon size={11} /></span>
        <span className="shrink-0 font-mono text-[9.5px] font-semibold tracking-[.12em] text-ink-4 uppercase">{meta.label}</span>
        {step.target && <span className="min-w-0 flex-1 truncate font-mono text-[11.5px] text-ink-2">{step.target}</span>}
        {step.status === "running" && <span className="flex shrink-0 items-center gap-1.5 font-mono text-[10.5px] text-cyan"><i className="size-[5px] animate-breathe rounded-full bg-cyan" />running</span>}
        {step.status === "done" && !!step.ms && <span className="shrink-0 font-mono text-[10px] text-ink-4">{step.ms}ms</span>}
        {step.status === "error" && <Badge tone="rose" mono>failed</Badge>}
        {hasBody && <IconChevronDown size={12} className={cn("shrink-0 text-ink-4 transition-transform duration-200", open && "rotate-180")} />}
      </button>
      {open && hasBody && (
        <div className="animate-slide-down border-t border-line-soft">
          {step.lines && <pre className="scroll-thin overflow-x-auto px-3 py-2 font-mono text-[11px] leading-[1.7] text-ink-3">{step.lines.map((l, i) => <div key={i}>{l || " "}</div>)}</pre>}
          {step.diff && <div className="p-2"><DiffView file={step.diff} /></div>}
          {step.output && (
            <div className="border-t border-line-soft/70 bg-code">
              <div className="px-3 pt-2 pb-1 font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">stdout</div>
              <pre className="scroll-thin overflow-x-auto px-3 pb-2.5 font-mono text-[11px] leading-[1.7] text-ink-3">
                {step.output.map((l, i) => <div key={i} className={cn(l.trim().startsWith("✓") && "text-mint", l.trim().startsWith("✗") && "text-rose")}>{l}</div>)}
                {step.status === "running" && <div className="text-cyan"><span className="animate-caret">▊</span></div>}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ============================== APPROVAL ============================== */
export function ApprovalCard({ step }: { step: Extract<Step, { type: "approval" }> }) {
  const [state, setState] = useState(step.state);
  const riskTone = step.risk === "high" ? "rose" : step.risk === "medium" ? "amber" : "mint";
  return (
    <div className={cn("animate-rise overflow-hidden rounded-[10px] border bg-well", state === "denied" ? "border-rose/30" : state === "allowed" ? "border-mint/25" : "border-amber/40")}>
      <div className="flex items-center gap-2 border-b border-line-soft px-3 py-2">
        <IconShield size={13} className={state === "allowed" ? "text-mint" : state === "denied" ? "text-rose" : "text-amber"} />
        <span className="text-[12.5px] font-semibold text-ink">Approval required</span>
        <Badge tone={riskTone} mono className="capitalize">{step.risk} risk</Badge>
        <span className="ml-auto font-mono text-[10px] text-ink-4">rule: ask-before-publish</span>
      </div>
      <div className="space-y-2.5 px-3 py-2.5">
        <pre className="scroll-thin overflow-x-auto rounded-[6px] border border-line-soft bg-code px-2.5 py-2 font-mono text-[11.5px] text-ink-2">{step.command}</pre>
        <p className="text-[12px] text-ink-3">{step.reason}</p>
        {state === "open" ? (
          <div className="flex flex-wrap items-center gap-2">
            <Button variant="success" icon={IconCheck} onClick={() => setState("allowed")}>Allow once</Button>
            <Button variant="outline" onClick={() => setState("allowed")}>Always in <span className="ml-1 font-mono text-[10.5px] text-ink-3">packages/**</span></Button>
            <Button variant="danger" icon={IconX} onClick={() => setState("denied")}>Deny</Button>
            <span className="ml-auto text-[10.5px] text-ink-4">auto-deny in <Kbd className="ml-1">18s</Kbd></span>
          </div>
        ) : (
          <div className={cn("flex items-center gap-2 text-[12px]", state === "denied" ? "text-rose" : "text-mint")}><IconCheck size={12} />{state === "allowed" ? "Allowed for this session" : "Denied — agent will try a different approach"}</div>
        )}
      </div>
    </div>
  );
}

function Thinking({ step }: { step: Extract<Step, { type: "thinking" }> }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-[9px] border border-dashed border-line bg-well/60">
      <button onClick={() => setOpen((o) => !o)} className="flex w-full cursor-pointer items-center gap-2 px-2.5 py-[7px] text-left">
        <IconBrain size={12} className="shrink-0 text-plum" />
        <span className="font-mono text-[9.5px] font-semibold tracking-[.12em] text-ink-4 uppercase">reasoning</span>
        {!open && <span className="min-w-0 flex-1 truncate text-[11.5px] text-ink-3 italic">{step.text}</span>}
        <span className="ml-auto shrink-0 font-mono text-[10px] text-ink-4">{(step.ms / 1000).toFixed(1)}s</span>
        <IconChevronDown size={12} className={cn("shrink-0 text-ink-4 transition-transform", open && "rotate-180")} />
      </button>
      {open && <p className="animate-slide-down border-t border-dashed border-line px-3 py-2.5 text-[12.5px] leading-[1.65] text-ink-3 italic">{step.text}</p>}
    </div>
  );
}
function Checkpoint({ step }: { step: Extract<Step, { type: "checkpoint" }> }) {
  return (
    <div className="group flex items-center gap-2 px-1 py-0.5">
      <span className="h-px flex-1 bg-line-soft" />
      <span className="flex items-center gap-1.5 font-mono text-[10px] text-ink-4"><IconUndo size={10} />checkpoint <span className="text-ink-3">{step.hash}</span> · {step.label}</span>
      <button className="cursor-pointer rounded px-1 text-[10px] text-iris-soft opacity-0 transition-opacity group-hover:opacity-100 hover:underline">restore</button>
      <span className="h-px flex-1 bg-line-soft" />
    </div>
  );
}

/* ============================ STEP ROUTER ============================ */
export function StepView({ step, agentId = "claude-code", persona }: { step: Step; agentId?: string; persona?: { name: string; glyph: string; from: string; to: string; model: string } }) {
  const a = agentById(agentId);
  const who = persona ?? { name: a.name, glyph: a.glyph, from: a.from, to: a.to, model: a.models[0].id };
  if (step.type === "user") return (
    <div className="animate-rise flex gap-3">
      <div className="flex size-[28px] shrink-0 items-center justify-center rounded-[8px] bg-hover font-mono text-[10.5px] font-bold text-ink-2 shadow-e1">DV</div>
      <div className="min-w-0 flex-1">
        <div className="mb-1.5 flex flex-wrap items-center gap-2">
          <span className="text-[12.5px] font-semibold text-ink">You</span><span className="font-mono text-[10px] text-ink-4">{step.at}</span>
          {step.attachments?.map((x) => <span key={x} className="inline-flex items-center gap-1 rounded-[5px] border border-line bg-raise px-1.5 py-[1px] font-mono text-[10px] text-ink-3"><IconFile size={9} /> {x}</span>)}
        </div>
        <div className="rounded-[10px] rounded-tl-[3px] border border-line-soft bg-raise px-3.5 py-2.5"><Prose text={step.text} /></div>
      </div>
    </div>
  );
  if (step.type === "text") return (
    <div className="animate-rise flex gap-3">
      <div className="flex size-[28px] shrink-0 items-center justify-center rounded-[8px] font-mono text-[10px] font-bold" style={{ color: who.to, background: `linear-gradient(135deg, ${who.from}26, ${who.to}1a)`, boxShadow: `inset 0 0 0 1px ${who.to}55` }}>{who.glyph}</div>
      <div className="min-w-0 flex-1 pt-0.5">
        <div className="mb-1.5 flex items-center gap-2"><span className="text-[12.5px] font-semibold text-ink">{who.name}</span><Badge tone="iris" mono className="text-[9.5px]">{who.model}</Badge></div>
        <Prose text={step.text} />
      </div>
    </div>
  );
  if (step.type === "thinking") return <Thinking step={step} />;
  if (step.type === "tool") return <div className="pl-[40px]"><ToolCard step={step} /></div>;
  if (step.type === "approval") return <div className="pl-[40px]"><ApprovalCard step={step} /></div>;
  if (step.type === "checkpoint") return <Checkpoint step={step} />;
  return <div className="flex items-center gap-2 rounded-[8px] border border-line-soft bg-raise px-3 py-2 text-[12px] text-ink-3"><IconWarning size={12} className="text-amber" />{step.text}</div>;
}
export function StreamingRow({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2.5 pl-[40px]">
      <span className="flex size-[26px] items-center justify-center rounded-[8px] border border-iris/30 bg-iris-tint text-iris-soft"><IconSpark size={12} className="animate-breathe" /></span>
      <span className="text-[12.5px] text-ink-3">{label}</span>
      <span className="flex gap-1">{[0, 1, 2].map((i) => <i key={i} className="size-[4px] rounded-full bg-iris-soft" style={{ animation: `breathe 1.2s ease-in-out ${i * 0.16}s infinite` }} />)}</span>
    </div>
  );
}
export { IconGrid };
