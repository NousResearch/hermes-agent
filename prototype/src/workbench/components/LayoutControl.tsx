"use client";

import { cn } from "../utils/cn";
import { IconLayout, IconPanelBottom, IconPanelLeft, IconPanelRight, IconCheck } from "./Icons";
import { Kbd, Menu, Tip } from "./ui";

export type LayoutState = { list: boolean; terminal: boolean; panel: boolean; status: boolean };

export const PRESETS: { id: string; label: string; desc: string; state: LayoutState }[] = [
  { id: "standard", label: "Standard", desc: "List · panel · terminal", state: { list: true, terminal: true, panel: true, status: true } },
  { id: "focus", label: "Focus", desc: "Conversation only", state: { list: false, terminal: false, panel: false, status: true } },
  { id: "review", label: "Review", desc: "Wide diff, no terminal", state: { list: true, terminal: false, panel: true, status: true } },
  { id: "terminal", label: "Terminal", desc: "Shell-forward", state: { list: false, terminal: true, panel: false, status: true } },
];

function Wire({ state, onPick, interactive = true }: { state: LayoutState; onPick?: (k: keyof LayoutState) => void; interactive?: boolean }) {
  const region = (k: keyof LayoutState, cls: string, title: string, icon?: React.ReactNode) => {
    const on = state[k];
    return (
      <button
        disabled={!interactive}
        onClick={() => interactive && onPick?.(k)}
        title={title}
        className={cn(
          "group/r relative flex items-center justify-center overflow-hidden rounded-[3px] border transition-all duration-150",
          cls,
          on ? "border-iris/70 bg-iris-tint" : "border-line-strong/70 bg-well",
          interactive && "cursor-pointer hover:border-iris/60 hover:bg-hover",
        )}
      >
        {icon && <span className={cn("transition-colors", on ? "text-iris-soft" : "text-ink-4 group-hover/r:text-ink-3")}>{icon}</span>}
      </button>
    );
  };
  return (
    <div className="flex h-[84px] w-full flex-col gap-[3px] rounded-[6px] border border-line-strong/60 bg-code p-[3px]">
      <div className="flex min-h-0 flex-1 gap-[3px]">
        {region("list", "w-[20%]", "Sidebar · ⇧⌘B", <IconPanelLeft size={11} />)}
        <div className="flex min-w-0 flex-1 flex-col gap-[3px]">
          <div className="flex min-h-0 flex-1 items-center justify-center rounded-[3px] border border-dashed border-line-strong/60 bg-well/50 font-mono text-[8.5px] tracking-[.12em] text-ink-4 uppercase">main</div>
          {region("terminal", "h-[24px]", "Terminal · ⌘J", <IconPanelBottom size={11} />)}
        </div>
        {region("panel", "w-[24%]", "Side panel · ⌘L", <IconPanelRight size={11} />)}
      </div>
      {region("status", "h-[8px]", "Status bar", undefined)}
    </div>
  );
}

export function LayoutControl({
  state,
  setState,
  panelLabel,
  open,
  setOpen,
}: {
  state: LayoutState;
  setState: (s: LayoutState) => void;
  panelLabel: string;
  open: boolean;
  setOpen: (b: boolean) => void;
}) {
  const toggle = (k: keyof LayoutState) => setState({ ...state, [k]: !state[k] });
  const active = PRESETS.find((p) => JSON.stringify(p.state) === JSON.stringify(state));

  return (
    <Menu
      align="right"
      width={292}
      open={open}
      onOpenChange={setOpen}
      trigger={() => (
        <Tip label="Layout · ⌘⇧L" side="bottom">
          <button className={cn("flex h-[30px] cursor-pointer items-center gap-1.5 rounded-[8px] border bg-raise px-2 text-[12px] transition-all hover:border-line-strong", open ? "border-iris/50 text-iris-soft" : "border-line text-ink-3 hover:text-ink")}>
            <IconLayout size={14} />
            <span className="hidden lg:inline">Layout</span>
            <Kbd className="hidden xl:inline-flex">⌘⇧L</Kbd>
          </button>
        </Tip>
      )}
    >
      {() => (
        <div className="p-1">
          <div className="px-1.5 pt-1 pb-2 font-mono text-[9px] tracking-[.16em] text-ink-4 uppercase">layout</div>
          <div className="px-1.5 pb-2.5">
            <Wire state={state} onPick={toggle} />
          </div>
          <div className="space-y-0.5">
            {(
              [
                { k: "list" as const, l: "Sidebar", s: "⇧⌘B" },
                { k: "panel" as const, l: panelLabel, s: "⌘L" },
                { k: "terminal" as const, l: "Terminal", s: "⌘J" },
                { k: "status" as const, l: "Status bar", s: "" },
              ]
            ).map((r) => (
              <button
                key={r.k}
                onClick={() => toggle(r.k)}
                className={cn("flex w-full cursor-pointer items-center gap-2 rounded-[7px] px-2 py-[7px] text-left text-[12.5px] transition-colors", state[r.k] ? "text-ink hover:bg-hover" : "text-ink-3 hover:bg-hover hover:text-ink-2")}
              >
                <span className={cn("flex size-[15px] items-center justify-center rounded-[4px] border transition-colors", state[r.k] ? "border-iris bg-iris text-on-iris" : "border-line-strong")}>
                  {state[r.k] && <IconCheck size={9} />}
                </span>
                <span className="flex-1">{r.l}</span>
                {r.s && <span className="font-mono text-[10px] text-ink-4">{r.s}</span>}
              </button>
            ))}
          </div>
          <div className="mt-1 border-t border-line-soft px-1.5 pt-2 pb-1 font-mono text-[9px] tracking-[.16em] text-ink-4 uppercase">presets</div>
          <div className="grid grid-cols-2 gap-1.5 px-0.5 pb-1">
            {PRESETS.map((p) => (
              <button
                key={p.id}
                onClick={() => { setState(p.state); setOpen(false); }}
                className={cn("cursor-pointer rounded-[8px] border p-2 text-left transition-all", active?.id === p.id ? "border-iris/60 bg-iris-tint" : "border-line-soft hover:border-line-strong hover:bg-hover/60")}
              >
                <Wire state={p.state} interactive={false} />
                <div className={cn("mt-1.5 text-[11.5px] font-medium", active?.id === p.id ? "text-iris-soft" : "text-ink")}>{p.label}</div>
                <div className="font-mono text-[9px] leading-[1.3] text-ink-4">{p.desc}</div>
              </button>
            ))}
          </div>
        </div>
      )}
    </Menu>
  );
}

/** Compact icon toggles shown inline in the titlebar. */
export function LayoutChips({ state, setState, panelLabel }: { state: LayoutState; setState: (s: LayoutState) => void; panelLabel: string }) {
  const t = (k: keyof LayoutState) => setState({ ...state, [k]: !state[k] });
  const chip = (on: boolean) => cn("flex size-[28px] cursor-pointer items-center justify-center rounded-[7px] transition-all duration-150", on ? "bg-raise text-ink shadow-e1" : "text-ink-4 hover:bg-hover hover:text-ink-2");
  return (
    <div className="flex items-center gap-0.5 rounded-[9px] border border-line bg-sunken p-0.5">
      <Tip label={`Sidebar · ⇧⌘B`} side="bottom"><button onClick={() => t("list")} className={chip(state.list)}><IconPanelLeft size={14} /></button></Tip>
      <Tip label={`${panelLabel} · ⌘L`} side="bottom"><button onClick={() => t("panel")} className={chip(state.panel)}><IconPanelRight size={14} /></button></Tip>
      <Tip label={`Terminal · ⌘J`} side="bottom"><button onClick={() => t("terminal")} className={chip(state.terminal)}><IconPanelBottom size={14} /></button></Tip>
    </div>
  );
}
