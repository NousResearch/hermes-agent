"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "../utils/cn";
import { AGENTS, SLASH_COMMANDS, agentById, type ModeId, type Session, type Step } from "../data/catalog";
import { ASSISTANT, INSTALLED_IDES } from "../data/extra";
import type { ModelProvider } from "../data/providers";
import { useApp } from "../lib/app";
import { ENGINES, useDictation } from "../lib/dictation";
import { AgentMark, IconArrowUp, IconAt, IconBolt, IconBrain, IconCheck, IconChevronDown, IconCloud, IconEye, IconFile, IconGit, IconHistory, IconLaptop, IconList, IconMic, IconMore, IconPaperclip, IconPlug, IconPlus, IconSearch, IconShield, IconSpark, IconStop, IconTerminal, IconUndo, IconX, IdeMark, IconBranch } from "./Icons";
import { Badge, Button, IconButton, Kbd, Menu, MenuItem, MenuLabel, MenuSep, Ring, Tip, toneText, type Tone } from "./ui";
import { StepView, StreamingRow } from "./Transcript";

/* ============================== MODE DEFINITIONS ============================== */
type ModeDef = { id: ModeId; label: string; short: string; desc: string; tone: Tone; icon: React.ComponentType<{ size?: number; className?: string }> };
export function modeDefs(product: "code" | "agent"): ModeDef[] {
  return product === "code"
    ? [
        { id: "plan", label: "Plan", short: "Plan", desc: "Researches the repo and proposes a plan. No writes.", tone: "iris", icon: IconList },
        { id: "readonly", label: "Read only", short: "Read only", desc: "Answers and explains. Nothing touches disk.", tone: "amber", icon: IconEye },
        { id: "agent", label: "Agent", short: "Agent", desc: "Edits and runs commands. Asks before risky ones.", tone: "cyan", icon: IconBolt },
        { id: "full", label: "Full access", short: "Full access", desc: "Autonomous edits, shell and network. Checkpoint per write.", tone: "rose", icon: IconShield },
      ]
    : [
        { id: "plan", label: "Plan first", short: "Plan", desc: "Proposes steps and waits for your go.", tone: "iris", icon: IconList },
        { id: "readonly", label: "Chat only", short: "Chat", desc: "Talks and researches. Takes no actions.", tone: "amber", icon: IconEye },
        { id: "agent", label: "Ask before acting", short: "Ask first", desc: "Acts inside the workspace, asks before anything leaves it.", tone: "cyan", icon: IconBolt },
        { id: "full", label: "Act autonomously", short: "Autonomous", desc: "Sends, schedules and posts without asking. Spend-capped.", tone: "rose", icon: IconShield },
      ];
}

/* ============================== HEADER ============================== */
function ThreadHeader({ session, sessions, onSelect, onNew, compact, streaming }: { session: Session; sessions: Session[]; onSelect: (id: string) => void; onNew: () => void; compact?: boolean; streaming: boolean }) {
  const { product, toast, termOpen, setTermOpen, setDockOpen } = useApp();
  const [q, setQ] = useState("");
  return (
    <div className="relative z-20 flex h-[44px] shrink-0 items-center gap-1.5 border-b border-line-soft bg-base/85 px-2.5 backdrop-blur-xl">
      <Menu width={compact ? 300 : 360} trigger={(open) => (
        <button className={cn("group flex min-w-0 cursor-pointer items-center gap-2 rounded-[7px] px-1.5 py-1 transition-colors hover:bg-hover", open && "bg-hover")}>
          <span className={cn("size-[7px] shrink-0 rounded-full", streaming ? "animate-breathe bg-cyan" : session.status === "waiting" ? "bg-amber" : "bg-mint")} />
          <span className={cn("truncate font-display font-semibold tracking-[-.015em] text-ink", compact ? "max-w-[170px] text-[12.5px]" : "max-w-[420px] text-[13.5px]")}>{session.title}</span>
          <IconChevronDown size={11} className="shrink-0 text-ink-4 transition-transform group-hover:translate-y-[1px]" />
        </button>
      )}>
        {(close) => (
          <>
            <div className="relative p-1"><IconSearch size={11} className="absolute top-1/2 left-3 -translate-y-1/2 text-ink-4" /><input autoFocus value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search history…" className="h-[28px] w-full rounded-[6px] bg-sunken pr-2 pl-7 text-[12px] text-ink placeholder:text-ink-4 focus:outline-none" /></div>
            <MenuLabel>recent</MenuLabel>
            <div className="scroll-thin max-h-[300px] overflow-y-auto">
              {sessions.filter((s) => s.title.toLowerCase().includes(q.toLowerCase())).map((s) => (
                <MenuItem key={s.id} active={s.id === session.id} onClick={() => { onSelect(s.id); close(); }}
                  icon={<span className={cn("size-[6px] rounded-full", s.status === "running" ? "bg-cyan" : s.status === "waiting" ? "bg-amber" : s.status === "failed" ? "bg-rose" : "bg-line-strong")} />}
                  label={s.title} hint={s.updated} />
              ))}
            </div>
            <MenuSep />
            <MenuItem icon={<IconPlus size={12} />} label="New session" hint="⌘N" onClick={() => { onNew(); close(); }} />
          </>
        )}
      </Menu>
      {!compact && product === "code" && <span className="hidden items-center gap-1 font-mono text-[10px] text-ink-4 lg:flex"><IconBranch size={10} />{session.branch} · {session.cost}</span>}

      <div className="ml-auto flex items-center gap-0.5">
        {product === "code" && (
          <Menu align="right" width={236} trigger={() => <Tip label="Attach this session to an editor" side="bottom"><button className="flex h-[28px] cursor-pointer items-center gap-1 rounded-[7px] px-1.5 transition-colors hover:bg-hover"><IdeMark kind="cursor" size={15} /><IconChevronDown size={10} className="text-ink-4" /></button></Tip>}>
            {(close) => (<>
              <MenuLabel>attach session to</MenuLabel>
              {INSTALLED_IDES.filter((i) => i.installed).map((i) => <MenuItem key={i.id} icon={<IdeMark kind={i.id} size={14} />} label={i.name} hint={i.attached ? "attached" : i.version} onClick={() => { close(); toast(`${i.name} attached · streaming edits both ways`, "mint"); }} />)}
            </>)}
          </Menu>
        )}
        <Menu align="right" width={236} trigger={() => <IconButton icon={IconMore} label="Session actions" size={28} />}>
          {(close) => (<>
            <MenuItem icon={<IconPlus size={12} />} label="New session" hint="⌘N" onClick={() => { onNew(); close(); }} />
            <MenuItem icon={<IconHistory size={12} />} label="Search history" onClick={() => { close(); toast("Click the session title to search history"); }} />
            <MenuSep />
            <MenuItem icon={<IconUndo size={12} />} label="Rewind to checkpoint" hint="esc esc" onClick={() => { close(); toast("Restored checkpoint a91f3c2", "amber"); }} />
            <MenuItem icon={<IconBolt size={12} />} label="Fork into parallel run" hint="⌘D" onClick={() => { close(); toast("Forked into worktree ../cnd-fork", "iris"); }} />
            <MenuItem icon={<IconSpark size={12} />} label="Compact context" hint="⌘⇧K" onClick={() => { close(); toast("Context compacted · 62% → 31%", "mint"); }} />
            <MenuSep />
            <MenuItem icon={<IconFile size={12} />} label="Export transcript (.md)" onClick={close} />
            <MenuItem icon={<IconTerminal size={12} />} label={termOpen ? "Hide terminal" : "Show terminal"} hint="⌘J" onClick={() => { close(); setTermOpen(!termOpen); }} />
            {compact && <MenuItem icon={<IconX size={12} />} label="Close agent panel" hint="⌘L" onClick={() => { close(); setDockOpen(false); }} />}
          </>)}
        </Menu>
      </div>
    </div>
  );
}

/* ============================== COMPOSER ============================== */
function Composer({ onSend, streaming, onStop, mode, setMode, agentId, setAgentId, modelId, setModelId, compact }: { onSend: (t: string) => void; streaming: boolean; onStop: () => void; mode: ModeId; setMode: (m: ModeId) => void; agentId: string; setAgentId: (a: string) => void; modelId: string; setModelId: (m: string) => void; compact?: boolean }) {
  const { product, providers, toast, setSettingsSection } = useApp();
  const [value, setValue] = useState("");
  const [slash, setSlash] = useState(false);
  const [idx, setIdx] = useState(0);
  const [effort, setEffort] = useState<"low" | "medium" | "high">("medium");
  const [env, setEnv] = useState<"local" | "worktree" | "cloud">("local");
  const ta = useRef<HTMLTextAreaElement>(null);
  const modes = modeDefs(product);
  const cur = modes.find((m) => m.id === mode)!;
  const agent = agentById(agentId);
  const model = agent.models.find((m) => m.id === modelId) ?? agent.models[0];
  const externalModel = providers.flatMap((p) => p.models.map((m) => ({ ...m, provider: p }))).find((m) => modelId === `provider:${m.provider.id}:${m.id}`);
  const selectedModelLabel = externalModel?.label ?? (modelId.startsWith("provider:") ? "Choose model" : product === "code" ? model.label : "Auto");
  const chooseProviderModel = (provider: ModelProvider, id: string) => {
    setModelId(`provider:${provider.id}:${id}`);
    toast(`Model route · ${provider.name} / ${id}`, provider.kind === "local" ? "mint" : "iris");
  };

  useEffect(() => { const el = ta.current; if (!el) return; el.style.height = "auto"; el.style.height = `${Math.min(el.scrollHeight, 220)}px`; }, [value]);
  const dictation = useDictation(
    (text) => { setValue((v) => `${v}${v && !v.endsWith(" ") ? " " : ""}${text}`); ta.current?.focus(); },
    (msg) => toast(msg, "amber"),
  );
  const engineLabel = ENGINES.find((e) => e.id === dictation.engine)?.label ?? "Browser engine";
  const engineShort = ENGINES.find((e) => e.id === dictation.engine)?.short ?? "mic";
  const submit = () => { if (!value.trim()) return; if (dictation.listening) dictation.stop(); onSend(value.trim()); setValue(""); setSlash(false); };
  useEffect(() => {
    if (compact) return;
    const h = () => { dictation.toggle(); ta.current?.focus(); };
    window.addEventListener("aro:voice:dictate", h);
    return () => window.removeEventListener("aro:voice:dictate", h);
  }, [compact, dictation]);

  const cmds = product === "code" ? SLASH_COMMANDS : [{ cmd: "/research", desc: "Browse and summarise sources", hint: "" }, { cmd: "/draft", desc: "Write a doc, email or post", hint: "" }, { cmd: "/schedule", desc: "Find time and send invites", hint: "" }, { cmd: "/automate", desc: "Make this a recurring automation", hint: "" }];
  const filtered = cmds.filter((c) => (value.startsWith("/") ? c.cmd.startsWith(value.split(" ")[0]) : true));
  const envMeta = { local: { icon: IconLaptop, label: "Local" }, worktree: { icon: IconGit, label: "Worktree" }, cloud: { icon: IconCloud, label: "Cloud" } }[env];

  const chip = "inline-flex h-[26px] cursor-pointer items-center gap-1.5 rounded-[7px] px-2 text-[11.5px] font-medium transition-colors hover:bg-hover";

  return (
    <div className={cn("relative shrink-0", compact ? "px-2.5 pb-2.5" : "px-4 pb-4")}>
      {slash && filtered.length > 0 && (
        <div className="absolute right-4 bottom-[calc(100%+2px)] left-4 z-30 animate-slide-down overflow-hidden rounded-[11px] border border-line-strong bg-overlay p-1 shadow-e4">
          {filtered.slice(0, 7).map((c, i) => (
            <button key={c.cmd} onMouseEnter={() => setIdx(i)} onClick={() => { setValue(c.cmd + " "); ta.current?.focus(); setSlash(false); }} className={cn("flex w-full cursor-pointer items-center gap-3 rounded-[7px] px-2 py-[7px] text-left", i === idx ? "bg-iris-tint" : "hover:bg-hover")}>
              <span className={cn("w-[80px] shrink-0 font-mono text-[12px] font-medium", i === idx ? "text-iris-soft" : "text-ink-2")}>{c.cmd}</span>
              <span className="min-w-0 flex-1 truncate text-[12px] text-ink-3">{c.desc}</span>{c.hint && <Kbd>{c.hint}</Kbd>}
            </button>
          ))}
        </div>
      )}

      {/* Quick context pills — scrolls horizontally on small widths */}
      {!compact && value.length === 0 && !streaming && (
        <div className="no-scrollbar mb-1.5 flex items-center gap-1.5 overflow-x-auto px-1 pb-1">
          {[
            { label: "Explain this diff", icon: IconGit, prompt: "@diff Explain what this change does and whether it looks safe to land." },
            { label: "Fix failing test", icon: IconBolt, prompt: "The reconnect test is flaky on CI. Find the root cause and fix it." },
            { label: "Summarise session", icon: IconFile, prompt: "Summarise what we've done so far in this session as release notes." },
            { label: "Research vendors", icon: IconEye, prompt: "@browser Compare the top 3 managed Postgres vendors on price and SOC2." },
          ].map((p) => (
            <button key={p.label} onClick={() => onSend(p.prompt)} className="group flex shrink-0 items-center gap-1.5 rounded-full border border-line bg-raise px-2.5 py-[5px] text-[11px] text-ink-3 transition-all hover:-translate-y-px hover:border-iris/40 hover:text-ink">
              <p.icon size={10} className="text-iris-soft" />{p.label}
            </button>
          ))}
        </div>
      )}

      <div className={cn("overflow-hidden rounded-[16px] border bg-raise shadow-e2 transition-all duration-200", streaming ? "border-cyan/40" : "border-line focus-within:border-iris/50 focus-within:shadow-glow-iris")}>
        {streaming && <div className="relative h-[2px] overflow-hidden"><div className="absolute inset-y-0 w-1/4 animate-sweep bg-gradient-to-r from-transparent via-cyan to-transparent" /></div>}
        {dictation.listening && (
          <div className="flex items-center gap-2 border-b border-rose/25 bg-rose-tint/50 px-3.5 py-1.5 text-[11px] text-ink-2">
            <span className="flex items-end gap-[2px]">{[0, 1, 2, 3, 4].map((i) => <i key={i} className="w-[2px] rounded-full bg-rose" style={{ height: 5 + ((i * 4) % 9), animation: `breathe 0.85s ease-in-out ${i * 0.1}s infinite` }} />)}</span>
            <span className="font-medium text-rose">Listening</span>
            <span className="rounded-[4px] bg-rose/15 px-1.5 font-mono text-[9px] text-rose">{engineShort}</span>
            <span className="min-w-0 flex-1 truncate italic text-ink-3">{dictation.interim || "speak — text lands in the box"}</span>
            <Menu align="right" width={250} trigger={() => <button className="cursor-pointer rounded-[5px] px-1.5 py-0.5 font-mono text-[9.5px] text-ink-3 transition-colors hover:bg-hover hover:text-ink">engine ▾</button>}>
              {(close) => (<>
                <MenuLabel>dictation engine</MenuLabel>
                {ENGINES.map((e) => (
                  <button key={e.id} onClick={() => { dictation.setEngine(e.id); localStorage.setItem("aro.voice.engine", e.id); window.dispatchEvent(new CustomEvent("aro:voice:engine", { detail: e.id })); close(); }}
                    className={cn("flex w-full cursor-pointer items-start gap-2 rounded-[7px] px-2 py-[7px] text-left transition-colors", e.id === dictation.engine ? "bg-iris-tint" : "hover:bg-hover")}>
                    <span className="min-w-0 flex-1"><span className={cn("block text-[12px] font-medium", e.id === dictation.engine ? "text-iris-soft" : "text-ink")}>{e.label}</span><span className="block text-[10.5px] leading-[1.4] text-ink-3">{e.note}</span></span>
                    {e.id === dictation.engine && <IconCheck size={12} className="mt-1 text-iris-soft" />}
                  </button>
                ))}
              </>)}
            </Menu>
            <button onClick={dictation.stop} className="cursor-pointer rounded-[5px] bg-raise px-2 py-0.5 font-mono text-[10px] text-ink-2 transition-colors hover:bg-hover hover:text-ink">done</button>
          </div>
        )}
        <textarea ref={ta} rows={compact ? 2 : 3} value={value}
          onChange={(e) => { setValue(e.target.value); setSlash(e.target.value.startsWith("/")); setIdx(0); }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (slash && filtered[idx]) { setValue(filtered[idx].cmd + " "); setSlash(false); } else submit(); }
            if (e.key === "Escape") setSlash(false);
            if (slash && (e.key === "ArrowDown" || e.key === "ArrowUp")) { e.preventDefault(); setIdx((i) => (e.key === "ArrowDown" ? (i + 1) % filtered.length : (i - 1 + filtered.length) % filtered.length)); }
          }}
          placeholder={streaming ? "Working… type to queue a follow-up" : product === "agent" ? "Ask for anything — research, write, schedule, analyse…" : mode === "plan" ? "Describe the outcome — I'll plan before touching anything" : "Ask, describe a change, or paste an error. @ for context, / for commands"}
          className={cn("scroll-thin block max-h-[260px] w-full resize-none bg-transparent leading-[1.6] text-ink placeholder:text-ink-4 focus:outline-none", compact ? "px-3.5 pt-3 pb-2 text-[13px]" : "px-4 pt-3.5 pb-2 text-[13.5px]")} />

        <div className="flex flex-wrap items-center gap-1 border-t border-line-soft/60 bg-base/40 px-1.5 py-1.5">
          {/* left cluster — small, grouped tool buttons */}
          <div className="flex items-center gap-0.5 rounded-[10px] border border-line-soft bg-sunken/60 p-0.5">
            <Menu width={240} trigger={(open) => <Tip label="Add context · @"><button className={cn("flex size-[26px] cursor-pointer items-center justify-center rounded-[7px] text-ink-3 transition-colors hover:bg-hover hover:text-ink", open && "bg-hover text-ink")} aria-label="Add context"><IconPlus size={14} /></button></Tip>}>
              {(close) => (<>
                <MenuItem icon={<IconPaperclip size={12} />} label="Upload file or image" onClick={close} />
                <MenuItem icon={<IconAt size={12} />} label="Mention file / folder" hint="@" onClick={() => { setValue((v) => v + "@"); close(); ta.current?.focus(); }} />
                {product === "code" && <MenuItem icon={<IconGit size={12} />} label="Current diff" hint="@diff" onClick={() => { setValue((v) => v + "@diff "); close(); }} />}
                {product === "code" && <MenuItem icon={<IconPlug size={12} />} label="Editor selection" hint="@editor" onClick={() => { setValue((v) => v + "@editor "); close(); }} />}
                <MenuItem icon={<IconEye size={12} />} label="Browser screenshot" hint="@browser" onClick={() => { setValue((v) => v + "@browser "); close(); }} />
                <MenuItem icon={<IconTerminal size={12} />} label="Last terminal output" hint="@terminal" onClick={() => { setValue((v) => v + "@terminal "); close(); }} />
              </>)}
            </Menu>
            <Tip label="Slash commands · /"><button onClick={() => { setValue("/"); ta.current?.focus(); setSlash(true); }} className="flex size-[26px] cursor-pointer items-center justify-center rounded-[7px] text-ink-3 transition-colors hover:bg-hover hover:text-ink" aria-label="Slash commands"><span className="font-mono text-[12px]">/</span></button></Tip>
          </div>

          {/* mode dropdown */}
          <Menu width={290} trigger={(open) => (
            <Tip label="Mode · ⇧Tab to cycle"><button className={cn(chip, toneText[cur.tone], open && "bg-hover")}><cur.icon size={12} />{cur.short}<IconChevronDown size={10} className="opacity-60" /></button></Tip>
          )}>
            {(close) => (<>
              <MenuLabel>{product === "code" ? "permission mode" : "autonomy"}</MenuLabel>
              {modes.map((m) => (
                <button key={m.id} onClick={() => { setMode(m.id); close(); }} className={cn("flex w-full cursor-pointer items-start gap-2.5 rounded-[7px] px-2 py-2 text-left transition-colors", m.id === mode ? "bg-iris-tint" : "hover:bg-hover")}>
                  <span className={cn("mt-[1px] flex size-[22px] shrink-0 items-center justify-center rounded-[6px] border border-line-soft bg-well", toneText[m.tone])}><m.icon size={12} /></span>
                  <span className="min-w-0 flex-1"><span className="block text-[12.5px] font-medium text-ink">{m.label}</span><span className="block text-[11px] leading-[1.45] text-ink-3">{m.desc}</span></span>
                  {m.id === mode && <IconCheck size={12} className="mt-1 text-iris-soft" />}
                </button>
              ))}
              <div className="mt-1 border-t border-line-soft px-2 pt-1.5 pb-0.5 font-mono text-[9.5px] text-ink-4">⇧Tab cycles · default set in Settings → Permissions</div>
            </>)}
          </Menu>

          {/* model + effort */}
          <Menu width={270} trigger={(open) => (
            <button className={cn(chip, "text-ink-2", open && "bg-hover")}>
              {product === "code" ? <AgentMark glyph={agent.glyph} from={agent.from} to={agent.to} size={14} /> : <IconBrain size={12} className="text-iris-soft" />}
              <span className="max-w-[120px] truncate">{selectedModelLabel}</span>
              {externalModel && <span className="rounded-[3px] bg-cyan-tint px-1 font-mono text-[8px] text-cyan">{externalModel.provider.kind === "local" ? "LOCAL" : "CLOUD"}</span>}
              {!compact && <span className="font-mono text-[10px] text-ink-4">{effort}</span>}
              <IconChevronDown size={10} className="opacity-60" />
            </button>
          )}>
            {(close) => (<>
              {product === "code" ? (<>
                <MenuLabel>agent</MenuLabel>
                <div className="scroll-thin max-h-[180px] overflow-y-auto">
                  {AGENTS.filter((a) => a.status === "connected").map((a) => <MenuItem key={a.id} icon={<AgentMark glyph={a.glyph} from={a.from} to={a.to} size={14} />} label={a.name} hint={a.vendor} active={a.id === agentId} onClick={() => setAgentId(a.id)} />)}
                </div>
                <MenuSep /><MenuLabel>model</MenuLabel>
                {agent.models.map((m) => <MenuItem key={m.id} label={m.label} hint={m.ctx} active={m.id === model.id} onClick={() => { setModelId(m.id); close(); }} />)}
                <MenuSep /><MenuLabel>local models</MenuLabel>
                {providers.filter((p) => p.kind === "local").map((p) => p.models.length
                  ? p.models.map((m) => <MenuItem key={`${p.id}:${m.id}`} icon={<span className="font-mono text-[8px] font-bold" style={{ color: p.accent }}>{p.glyph}</span>} label={m.label} hint={p.name} active={modelId === `provider:${p.id}:${m.id}`} onClick={() => { chooseProviderModel(p, m.id); close(); }} />)
                  : <MenuItem key={p.id} icon={<span className="font-mono text-[8px] font-bold" style={{ color: p.accent }}>{p.glyph}</span>} label={p.name} hint={p.status === "ready" ? "no models" : "offline"} onClick={() => { setSettingsSection("providers"); window.dispatchEvent(new CustomEvent("aro:settings")); close(); toast(`Connect ${p.name} in Settings → Providers & models`, "amber"); }} />)}
                <MenuLabel>cloud providers</MenuLabel>
                {providers.filter((p) => p.kind === "cloud").flatMap((p) => p.models.map((m) => <MenuItem key={`${p.id}:${m.id}`} icon={<span className="font-mono text-[8px] font-bold" style={{ color: p.accent }}>{p.glyph}</span>} label={m.label} hint={p.name} active={modelId === `provider:${p.id}:${m.id}`} onClick={() => { chooseProviderModel(p, m.id); close(); }} />))}
                <MenuSep /><MenuItem icon={<IconBrain size={12} />} label="Manage providers & models" onClick={() => { setSettingsSection("providers"); window.dispatchEvent(new CustomEvent("aro:settings")); close(); }} />
              </>) : (<>
                <MenuLabel>routing</MenuLabel>
                <MenuItem icon={<IconBrain size={12} />} label="Auto — best model per step" active onClick={close} />
                {AGENTS.slice(0, 4).map((a) => <MenuItem key={a.id} icon={<AgentMark glyph={a.glyph} from={a.from} to={a.to} size={14} />} label={`Pin ${a.models[0].label}`} hint={a.vendor} onClick={close} />)}
              </>)}
              <MenuSep /><MenuLabel>reasoning effort</MenuLabel>
              <div className="flex gap-1 px-1.5 pb-1.5">
                {(["low", "medium", "high"] as const).map((e) => <button key={e} onClick={() => setEffort(e)} className={cn("flex-1 cursor-pointer rounded-[6px] border py-1 text-[11.5px] capitalize", effort === e ? "border-iris/50 bg-iris-tint text-iris-soft" : "border-line text-ink-3 hover:text-ink")}>{e}</button>)}
              </div>
            </>)}
          </Menu>

          {/* environment */}
          {product === "code" && (
            <Menu width={250} trigger={(open) => <Tip label="Where it runs"><button className={cn(chip, "text-ink-3", open && "bg-hover")}><envMeta.icon size={12} />{!compact && envMeta.label}<IconChevronDown size={10} className="opacity-60" /></button></Tip>}>
              {(close) => (<>
                <MenuLabel>run in</MenuLabel>
                {([["local", IconLaptop, "Local", "your working tree"], ["worktree", IconGit, "Worktree", "isolated branch, merge later"], ["cloud", IconCloud, "Cloud", "background, best-of-N"]] as const).map(([id, I, l, d]) => (
                  <MenuItem key={id} icon={<I size={12} />} label={<span>{l} <span className="text-ink-4">· {d}</span></span>} active={env === id} onClick={() => { setEnv(id); close(); }} />
                ))}
              </>)}
            </Menu>
          )}

          <div className="ml-auto flex items-center gap-1.5">
            <Tip label="Context used · 124k / 200k"><button onClick={() => window.dispatchEvent(new CustomEvent("aro:pane", { detail: "context" }))} className="hidden cursor-pointer items-center gap-1 rounded-[6px] px-1 py-0.5 text-ink-4 transition-colors hover:bg-hover hover:text-ink sm:flex" aria-label="View context usage"><Ring value={62} size={18} stroke={2.5} tone="var(--color-iris)" /><span className="font-mono text-[10px] tabular">62%</span></button></Tip>

            {/* dictate — sits beside Send so you can speak-to-type */}
            <Tip label={dictation.listening ? `Stop · ${engineLabel}` : `Speak to type · ${engineLabel} · ⌘⇧V`}>
              <button onClick={dictation.toggle}
                className={cn("relative flex h-[30px] cursor-pointer items-center gap-1.5 rounded-full border px-2.5 transition-all focus-visible:ring-focus active:scale-[.97]",
                  dictation.listening ? "border-rose/50 bg-rose text-white shadow-[0_0_0_4px_color-mix(in_srgb,var(--color-rose)_18%,transparent)]" : "border-line bg-sunken text-ink-3 hover:border-iris/40 hover:text-iris-soft")}
                aria-label={dictation.listening ? "Stop dictation" : "Dictate with microphone"} aria-pressed={dictation.listening}>
                <IconMic size={13} className={dictation.listening ? "animate-pulse" : undefined} />
                {!compact && <span className="font-mono text-[10px]">{dictation.listening ? "stop" : engineShort}</span>}
                {dictation.listening && <span className="flex items-end gap-[2px]">{[0, 1, 2].map((i) => <i key={i} className="w-[2px] rounded-full bg-white" style={{ height: 5 + ((i * 4) % 6), animation: `breathe 0.85s ease-in-out ${i * 0.13}s infinite` }} />)}</span>}
              </button>
            </Tip>

            {streaming ? (
              <Button variant="danger" icon={IconStop} onClick={onStop}>Stop</Button>
            ) : (
              <button onClick={submit} disabled={!value.trim()} className={cn("inline-flex h-[30px] cursor-pointer items-center gap-1.5 rounded-full pr-3.5 pl-3.5 text-[12.5px] font-medium transition-all", value.trim() ? "bg-iris text-on-iris shadow-e1 hover:brightness-110 active:scale-[.97]" : "bg-track text-ink-4")} aria-label="Send message">
                Send<IconArrowUp size={13} />
              </button>
            )}
          </div>
        </div>
      </div>
      {!compact && <p className="mt-1.5 flex items-center justify-center gap-2 font-mono text-[9.5px] text-ink-4"><Kbd>⏎</Kbd> send <span>·</span> <Kbd>⇧⏎</Kbd> newline <span>·</span> <Kbd>⇧Tab</Kbd> mode <span>·</span> <Kbd>/</Kbd> commands <span>·</span> <Kbd>⌘⇧V</Kbd> voice</p>}
    </div>
  );
}

/* ============================== EMPTY STATE ============================== */
function EmptyThread({ agentId, onPick, compact }: { agentId: string; onPick: (t: string) => void; compact?: boolean }) {
  const { product } = useApp();
  const a = agentById(agentId);
  const who = product === "code" ? a : ASSISTANT;
  const templates = product === "code" ? [
    { icon: IconBolt, title: "Refactor across packages", prompt: "Move the snapshot logic out of server.ts into a resumable SessionChannel with a 12ms flush. Update every call site and the tests." },
    { icon: IconList, title: "Plan first", prompt: "Design the wire protocol for attaching an external editor to a live session. Produce an editable plan with open questions." },
    { icon: IconGit, title: "Fix the failing CI", prompt: "reconnect.test.ts is flaky on CI. Find the root cause, fix it, and prove it with 50 green runs." },
    { icon: IconEye, title: "Browser-driven bug", prompt: "Open localhost:5173, reproduce the dropdown clipping on the review page, and fix the CSS." },
  ] : [
    { icon: IconEye, title: "Research & compare", prompt: "Compare the top 4 managed Postgres vendors on price, p99 latency and SOC2 — table plus a recommendation." },
    { icon: IconFile, title: "Draft a document", prompt: "Draft a one-page RFC for moving our editor bridge to binary frames." },
    { icon: IconAt, title: "Schedule & coordinate", prompt: "Find 45 minutes this week with Mira, Ola and Dev for a roadmap review and send invites." },
    { icon: IconBolt, title: "Automate a routine", prompt: "Every weekday at 9am, triage new Sentry issues, open tasks for P0s and post a digest to #platform." },
  ];
  return (
    <div className={cn("animate-rise flex flex-col items-center pb-2 text-center", compact ? "pt-6" : "pt-12")}>
      <div className="flex size-[44px] items-center justify-center rounded-[13px] font-mono text-[15px] font-bold" style={{ color: who.to, background: `linear-gradient(135deg, ${who.from}26, ${who.to}1a)`, boxShadow: `inset 0 0 0 1px ${who.to}55` }}>{who.glyph}</div>
      <h3 className={cn("mt-4 font-display font-semibold tracking-[-.025em] text-ink", compact ? "text-[17px]" : "text-[23px]")}>{product === "code" ? `What should ${a.name} build?` : "What can I take off your plate?"}</h3>
      <p className="mt-1.5 max-w-[440px] text-[12.5px] leading-[1.65] text-ink-3">{product === "code" ? "Describe the outcome. Every edit stays reviewable." : "One assistant, every tool. It asks before anything leaves the workspace."}</p>
      <div className={cn("mt-5 grid w-full gap-2 text-left", compact ? "grid-cols-1" : "grid-cols-1 sm:grid-cols-2")}>
        {templates.map((t) => (
          <button key={t.title} onClick={() => onPick(t.prompt)} className="group flex cursor-pointer items-start gap-2.5 rounded-[11px] border border-line-soft bg-raise p-3 text-left transition-all hover:border-line-strong hover:shadow-e2">
            <span className="mt-[1px] flex size-[26px] shrink-0 items-center justify-center rounded-[7px] border border-line-soft bg-sunken text-iris-soft"><t.icon size={12} /></span>
            <span className="min-w-0 flex-1"><span className="block text-[12.5px] font-medium text-ink">{t.title}</span>{!compact && <span className="mt-1 block text-[11.5px] leading-[1.55] text-ink-3">{t.prompt}</span>}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

/* ============================== WORKBENCH ============================== */
export type WorkbenchProps = { session: Session; sessions: Session[]; onSelect: (id: string) => void; onNew: () => void; steps: Step[]; streaming: boolean; streamLabel: string; mode: ModeId; setMode: (m: ModeId) => void; agentId: string; setAgentId: (a: string) => void; modelId: string; setModelId: (m: string) => void; onSend: (t: string) => void; onStop: () => void; compact?: boolean };

export function Workbench(p: WorkbenchProps) {
  const { product } = useApp();
  const scroller = useRef<HTMLDivElement>(null);
  const [atBottom, setAtBottom] = useState(true);
  useEffect(() => { if (atBottom && scroller.current) scroller.current.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" }); }, [p.steps.length, p.streaming, atBottom]);
  const m = modeDefs(product).find((x) => x.id === p.mode)!;
  return (
    <div className="flex h-full min-w-0 flex-1 flex-col bg-base">
      <ThreadHeader session={p.session} sessions={p.sessions} onSelect={p.onSelect} onNew={p.onNew} compact={p.compact} streaming={p.streaming} />
      <div ref={scroller} onScroll={(e) => { const el = e.currentTarget; setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 80); }} className="scroll-thin ambient relative flex-1 overflow-y-auto">
        <div className={cn("mx-auto w-full space-y-4", p.compact ? "px-3 py-3" : "max-w-[820px] px-6 py-5")}>
          {p.steps.length === 0 && <EmptyThread agentId={p.agentId} onPick={p.onSend} compact={p.compact} />}
          {p.mode === "full" && p.steps.length > 0 && (
            <div className="flex items-center gap-2 rounded-[9px] border border-rose/25 bg-rose-tint px-3 py-2 text-[12px] text-ink-2"><IconShield size={12} className="shrink-0 text-rose" /><span><strong className="font-semibold text-rose">{m.label}.</strong> {m.desc}</span></div>
          )}
          {p.steps.map((s) => <StepView key={s.id} step={s} agentId={p.agentId} persona={product === "agent" ? ASSISTANT : undefined} />)}
          {p.streaming && <StreamingRow label={p.streamLabel} />}
        </div>
        {!atBottom && <button onClick={() => { setAtBottom(true); scroller.current?.scrollTo({ top: 1e6, behavior: "smooth" }); }} className="sticky bottom-3 left-1/2 z-10 flex -translate-x-1/2 cursor-pointer items-center gap-1.5 rounded-full border border-line-strong bg-overlay px-3 py-1.5 text-[11.5px] text-ink-2 shadow-e3"><IconChevronDown size={11} /> Latest</button>}
      </div>
      <Composer onSend={p.onSend} streaming={p.streaming} onStop={p.onStop} mode={p.mode} setMode={p.setMode} agentId={p.agentId} setAgentId={p.setAgentId} modelId={p.modelId} setModelId={p.setModelId} compact={p.compact} />
    </div>
  );
}
export { Badge };
