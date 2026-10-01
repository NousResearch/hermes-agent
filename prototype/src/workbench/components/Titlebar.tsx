"use client";

import { cn } from "../utils/cn";
import { REPOS, INSTALLED_IDES } from "../data/extra";
import { THEMES, useApp } from "../lib/app";
import { IconBranch, IconChevronDown, IconCoin, IconFolder, IconGit, IconHelp, IconMic, IconPanelLeft, IconPlug, IconPlus, IconRefresh, IconSearch, IconSettings, IconTerminal, IconUser, IdeMark } from "./Icons";
import { Badge, Button, Kbd, Menu, MenuItem, MenuLabel, MenuSep, Tip } from "./ui";
import { LayoutControl, type LayoutState } from "./LayoutControl";
import type { ViewId } from "./Sidebar";

export function Titlebar({
  onOpenCmdk, onNew, setView, layout, setLayout, panelLabel, layoutOpen, setLayoutOpen, onHelp, onToggleSidebar,
}: {
  onOpenCmdk: () => void; onNew: () => void; setView: (v: ViewId) => void;
  layout: LayoutState; setLayout: (s: LayoutState) => void; panelLabel: string;
  layoutOpen: boolean; setLayoutOpen: (b: boolean) => void; onHelp: () => void; onToggleSidebar?: () => void;
}) {
  const { product, setProduct, toast, theme, setTheme } = useApp();
  const repo = REPOS[0];
  void onNew;

  return (
    <header className="relative z-40 flex h-[46px] shrink-0 items-center gap-2 border-b border-line-soft bg-void px-2.5">
      {/* mobile: sidebar drawer toggle */}
      {onToggleSidebar && (
        <button onClick={onToggleSidebar} aria-label="Toggle sidebar" className="flex size-[30px] cursor-pointer items-center justify-center rounded-[8px] text-ink-3 transition-colors hover:bg-raise hover:text-ink">
          <IconPanelLeft size={16} />
        </button>
      )}

      {/* product mode */}
      <div className="inline-flex items-center rounded-[9px] border border-line bg-sunken p-0.5">
        {([{ id: "code", label: "Code", tip: "Coding agents on your repositories · ⇧⌘E" }, { id: "agent", label: "Agent", tip: "One assistant for any task · ⇧⌘E" }] as const).map((m) => (
          <Tip key={m.id} label={m.tip} side="bottom">
            <button onClick={() => setProduct(m.id)} className={cn("relative cursor-pointer rounded-[7px] px-3 py-[5px] text-[12.5px] font-medium transition-all duration-200", product === m.id ? "bg-raise text-ink shadow-e1 hairline" : "text-ink-3 hover:text-ink-2")}>
              {m.label}
              {product === m.id && <span className="absolute inset-x-3 -bottom-[3px] h-[2px] rounded-full bg-iris" />}
            </button>
          </Tip>
        ))}
      </div>

      <span className="hidden h-4 w-px bg-line sm:block" />

      {/* repo / workspace */}
      {product === "code" ? (
        <Menu width={304} className="hidden md:block" trigger={(open) => (
          <button className={cn("group flex min-w-0 cursor-pointer items-center gap-2 rounded-[8px] border border-line bg-raise px-2 py-[5px] transition-all hover:border-line-strong", open && "border-iris/50")}>
            <IconGit size={12} className="shrink-0 text-ink-4" />
            <span className="hidden truncate font-mono text-[11.5px] text-ink-2 md:inline">{repo.name}</span>
            <span className="hidden h-2.5 w-px bg-line md:block" />
            <IconBranch size={11} className="shrink-0 text-ink-4" />
            <span className="truncate font-mono text-[11.5px] text-iris-soft">{repo.branch}</span>
            {repo.dirty > 0 && <span className="shrink-0 rounded-full bg-amber-tint px-1.5 font-mono text-[9.5px] text-amber">{repo.dirty}</span>}
            <IconChevronDown size={11} className="shrink-0 text-ink-4 transition-transform group-hover:translate-y-[1px]" />
          </button>
        )}>
          {(close) => (<>
            <MenuLabel>repositories</MenuLabel>
            {REPOS.map((r) => <MenuItem key={r.id} icon={<IconGit size={12} />} label={r.name} hint={`↑${r.ahead} ↓${r.behind}`} active={r.default} onClick={() => { close(); setView("git"); }} />)}
            <MenuSep />
            <MenuItem icon={<IconFolder size={12} />} label="Attach local folder…" onClick={() => { close(); setView("git"); }} />
            <MenuItem icon={<IconPlus size={12} />} label="Clone from URL…" onClick={() => { close(); setView("git"); }} />
            <MenuItem icon={<IconRefresh size={12} />} label="Sync all · fetch + pull" onClick={() => { close(); toast("Synced 3 repositories", "mint"); }} />
          </>)}
        </Menu>
      ) : (
        <div className="hidden items-center gap-2 rounded-[8px] border border-line bg-raise px-2 py-[5px] sm:flex">
          <IconFolder size={12} className="text-ink-4" /><span className="font-mono text-[11.5px] text-ink-2">workspace</span>
          <span className="hidden h-2.5 w-px bg-line sm:block" /><span className="hidden font-mono text-[10.5px] text-ink-4 sm:inline">5 tools · 4 connectors</span>
        </div>
      )}

      {/* search */}
      <button onClick={onOpenCmdk} className="group mx-auto flex h-[30px] w-[min(400px,28vw)] min-w-0 max-w-full flex-1 cursor-pointer items-center gap-2 rounded-[8px] border border-line bg-sunken px-2.5 text-left transition-all hover:border-line-strong hover:bg-raise md:w-[min(400px,28vw)] md:flex-none">
        <IconSearch size={12} className="shrink-0 text-ink-4 transition-colors group-hover:text-iris-soft" />
        <span className="min-w-0 flex-1 truncate text-[12px] text-ink-4">Search sessions, tasks, files…</span>
        <Kbd className="hidden sm:inline-flex">⌘K</Kbd>
      </button>

      {/* daemon status */}
      <Tip label="aro daemon · 127.0.0.1:4733 · 99.98% uptime" side="bottom">
        <span className="hidden cursor-default items-center gap-1.5 rounded-full border border-mint/25 bg-mint-tint px-2 py-[4px] lg:flex">
          <i className="size-[5px] animate-breathe rounded-full bg-mint" /><span className="font-mono text-[10px] text-mint">daemon</span>
        </span>
      </Tip>

      {/* layout control — panels, presets */}
      <div className="hidden md:block"><LayoutControl state={layout} setState={setLayout} panelLabel={panelLabel} open={layoutOpen} setOpen={setLayoutOpen} /></div>

      {/* open in IDE */}
      <Menu align="right" width={268} className="hidden lg:block" trigger={(open) => (
        <Tip label="Open project in an editor" side="bottom">
          <button className={cn("flex h-[30px] cursor-pointer items-center gap-1.5 rounded-[8px] border border-line bg-raise pr-1.5 pl-1.5 transition-all hover:border-line-strong", open && "border-iris/50")}>
            <IdeMark kind="cursor" size={16} /><IconChevronDown size={11} className="text-ink-4" />
          </button>
        </Tip>
      )}>
        {(close) => (<>
          <MenuLabel>open {repo.name} in</MenuLabel>
          {INSTALLED_IDES.filter((i) => i.installed).map((i) => (
            <button key={i.id} onClick={() => { close(); toast(`Opening ${repo.path} in ${i.name}…`, "iris"); }} className="flex w-full cursor-pointer items-center gap-2.5 rounded-[7px] px-2 py-[6px] text-left transition-colors hover:bg-hover">
              <IdeMark kind={i.id} size={18} />
              <span className="min-w-0 flex-1"><span className="block text-[12.5px] text-ink">{i.name}</span><span className="block font-mono text-[10px] text-ink-4">{i.attached ? "session attached" : i.scheme}</span></span>
              {i.attached && <Badge tone="mint" mono dot className="text-[9px]">live</Badge>}
            </button>
          ))}
          <MenuSep />
          <MenuItem icon={<IconPlug size={12} />} label="Editor bridges" onClick={() => { close(); setView("bridges"); }} />
        </>)}
      </Menu>

      {/* help */}
      <Tip label="Shortcuts · ⌘/" side="bottom">
        <button onClick={onHelp} className="flex size-[30px] cursor-pointer items-center justify-center rounded-[8px] text-ink-3 transition-colors hover:bg-raise hover:text-ink"><IconHelp size={15} /></button>
      </Tip>

      {/* account */}
      <Menu align="right" width={228} trigger={() => (
        <button className="flex cursor-pointer items-center gap-2 rounded-[8px] border border-line bg-raise py-[3px] pr-2 pl-[3px] transition-colors hover:border-line-strong">
          <span className="flex size-[22px] items-center justify-center rounded-[6px] bg-gradient-to-br from-iris/30 to-cyan/20 font-mono text-[9.5px] font-bold text-ink">DV</span>
          <IconChevronDown size={11} className="text-ink-4" />
        </button>
      )}>
        {(close) => (<>
          <div className="px-2 pt-1.5 pb-2">
            <div className="text-[12.5px] font-semibold text-ink">Dev Vale</div>
            <div className="font-mono text-[10px] text-ink-4">dev@acme.dev · Max plan</div>
          </div>
          <div className="px-2 pt-1 pb-1.5">
            <div className="mb-1.5 font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">theme</div>
            <div className="flex gap-1.5">
              {THEMES.map((t) => (
                <button key={t.id} onClick={() => { setTheme(t.id); }} title={t.label} className={cn("flex h-[22px] flex-1 cursor-pointer overflow-hidden rounded-[5px] border transition-all hover:scale-105", theme === t.id ? "border-iris ring-1 ring-iris/40" : "border-line-strong")}>
                  {t.swatch.map((c) => <i key={c} className="h-full flex-1" style={{ background: c }} />)}
                </button>
              ))}
            </div>
          </div>
          <MenuSep />
          <MenuItem icon={<IconSettings size={12} />} label="Settings" hint="⌘," onClick={() => { close(); setView("settings"); }} />
          <MenuItem icon={<IconCoin size={12} />} label="Usage & billing" onClick={() => { close(); setView("settings"); }} />
          <MenuItem icon={<IconTerminal size={12} />} label="Daemon logs" onClick={close} />
          <MenuItem icon={<IconUser size={12} />} label="Team & seats" onClick={close} />
          <MenuSep />
          <MenuItem icon={<IconMic size={12} />} label="Hands-free assistant" onClick={() => { close(); window.dispatchEvent(new CustomEvent("aro:voice")); }} />
          <MenuItem icon={<IconHelp size={12} />} label="Keyboard shortcuts" hint="⌘/" onClick={() => { close(); onHelp(); }} />
          <MenuItem danger label="Sign out" onClick={close} />
        </>)}
      </Menu>
    </header>
  );
}

export function PageHeader({ eyebrow, title, sub, right }: { eyebrow: string; title: string; sub: string; right?: React.ReactNode }) {
  return (
    <div className="relative flex shrink-0 flex-wrap items-end justify-between gap-4 overflow-hidden border-b border-line-soft bg-base px-6 py-4">
      <div className="ambient pointer-events-none absolute inset-0 opacity-70" />
      <div className="relative min-w-0">
        <span className="font-mono text-[9.5px] font-semibold tracking-[.18em] text-iris-soft uppercase">{eyebrow}</span>
        <h1 className="mt-1 font-display text-[22px] leading-[1.15] font-semibold tracking-[-.025em] text-ink">{title}</h1>
        <p className="mt-1.5 max-w-[640px] text-[12.5px] leading-[1.6] text-ink-3">{sub}</p>
      </div>
      {right && <div className="relative flex flex-wrap items-center gap-2">{right}</div>}
    </div>
  );
}
export function Grid({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("grid gap-3", className)}>{children}</div>;
}
export { Button };
