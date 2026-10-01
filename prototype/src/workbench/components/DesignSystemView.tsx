"use client";

import { useState } from "react";
import { cn } from "../utils/cn";
import { THEMES, useApp } from "../lib/app";
import {
  DS_COLORS,
  DS_MOTION,
  DS_RADII,
  DS_SPACING,
  DS_TYPE,
} from "../data/catalog";
import {
  IconAt,
  IconBolt,
  IconBrain,
  IconCheck,
  IconGit,
  IconPlug,
  IconSearch,
  IconShield,
  IconSpark,
  IconTerminal,
  IconWarning,
} from "./Icons";
import {
  Badge,
  Bar,
  Button,
  Divider,
  EmptyState,
  IconButton,
  Input,
  Kbd,
  Modal,
  Ring,
  SectionLabel,
  Segmented,
  Select,
  Sparkline,
  Stat,
  Tabs,
  Toggle,
  Tip,
  toneBg,
} from "./ui";

/* ============================ section frame ============================ */
function DS({ id, eyebrow, title, body, children }: { id: string; eyebrow: string; title: string; body: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-16 border-t border-line-soft py-8 first:border-0">
      <div className="mb-5">
        <span className="font-mono text-[9.5px] font-semibold tracking-[.16em] text-iris-soft uppercase">
          {eyebrow}
        </span>
        <h2 className="mt-1.5 text-[19px] leading-tight font-semibold tracking-[-.02em] text-ink">{title}</h2>
        <p className="mt-1.5 max-w-[680px] text-[12.5px] leading-[1.65] text-ink-3">{body}</p>
      </div>
      {children}
    </section>
  );
}

function Spec({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="rounded-[10px] border border-line-soft bg-sunken p-3">
      <div className="mb-2.5 font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">{label}</div>
      {children}
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-line-soft/60 py-2.5 last:border-0">
      <span className="w-[110px] shrink-0 font-mono text-[10px] text-ink-4">{label}</span>
      {children}
    </div>
  );
}

/* ============================ color swatch ============================ */
function Swatch({ name, hex, usage }: { name: string; hex: string; usage: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => {
        navigator.clipboard?.writeText(hex);
        setCopied(true);
        setTimeout(() => setCopied(false), 1100);
      }}
      className="group flex w-full cursor-pointer flex-col overflow-hidden rounded-[10px] border border-line-soft bg-raise text-left transition-all hover:border-line-strong hover:shadow-e2"
    >
      <div className="relative h-[62px] w-full" style={{ background: hex }}>
        <div className="absolute inset-0 opacity-0 transition-opacity group-hover:opacity-100"
          style={{ background: "linear-gradient(180deg,rgba(255,255,255,.12),transparent)" }} />
        <span
          className={cn(
            "absolute top-2 right-2 rounded-[4px] px-1.5 py-[1px] font-mono text-[9px] transition-all",
            copied ? "bg-black/50 text-white opacity-100" : "bg-black/35 text-white/80 opacity-0 group-hover:opacity-100",
          )}
        >
          {copied ? "copied" : hex}
        </span>
      </div>
      <div className="px-2.5 py-2">
        <div className="font-mono text-[11px] text-ink">{name}</div>
        <div className="mt-0.5 text-[10.5px] leading-[1.4] text-ink-3">{usage}</div>
      </div>
    </button>
  );
}

/* ============================ main view ============================ */
export function DesignSystemView() {
  const { theme, setTheme } = useApp();
  const [demoTab, setDemoTab] = useState<"a" | "b" | "c">("a");
  const [seg, setSeg] = useState<"plan" | "agent">("plan");
  const [sel, setSel] = useState("opus-4.6");
  const [tog, setTog] = useState(true);
  const [tog2, setTog2] = useState(false);
  const [modal, setModal] = useState(false);

  return (
    <div className="scroll-thin flex-1 overflow-y-auto">
      {/* hero */}
      <div className="relative overflow-hidden border-b border-line-soft">
        <div className="grid-bg pointer-events-none absolute inset-0 opacity-60" />
        <div
          className="pointer-events-none absolute -top-24 -right-10 size-[420px] rounded-full opacity-[0.18] blur-[80px]"
          style={{ background: "radial-gradient(circle,#7c6bff,transparent 70%)" }}
        />
        <div className="relative px-6 py-8">
          <div className="flex items-center gap-2">
            <Badge tone="iris" mono className="text-[9.5px]">
              v1.4 · aro/primitives
            </Badge>
            <Badge tone="neutral" mono className="text-[9.5px]">
              28 components · 64 tokens
            </Badge>
          </div>
          <h1 className="mt-3 max-w-[760px] font-display text-[30px] leading-[1.08] font-semibold tracking-[-.03em] text-ink">
            A design system for{" "}
            <span className="relative whitespace-nowrap text-iris-soft">
              agent control surfaces
              <span className="absolute -bottom-1 left-0 h-[3px] w-full rounded-full bg-gradient-to-r from-iris via-cyan to-transparent opacity-70" />
            </span>
          </h1>
          <p className="mt-2.5 max-w-[640px] text-[13px] leading-[1.7] text-ink-3">
            Built for interfaces that must show <strong className="font-semibold text-ink-2">provenance</strong> (who
            did what — agent, editor or human), <strong className="font-semibold text-ink-2">intent</strong> (plan vs
            execute), and <strong className="font-semibold text-ink-2">state</strong> (streaming, waiting, blocked) at
            13px without ever feeling cramped. Dark-first, monospace-forward, hairline-separated.
          </p>
          <div className="mt-5 grid max-w-[860px] grid-cols-2 gap-3 md:grid-cols-4">
            {[
              { t: "Provenance first", d: "Every artefact carries an agent mark, a model badge and a timestamp.", i: IconShield },
              { t: "Intent is a mode", d: "Plan / Agent / Read-only / Full-access is a persistent, colour-coded state.", i: IconBolt },
              { t: "Nothing silent", d: "Tool calls, approvals and checkpoints are first-class, not log lines.", i: IconTerminal },
              { t: "Density with air", d: "11–13px type, 4px rhythm, hairlines instead of heavy chrome.", i: IconSpark },
            ].map((p) => (
              <div key={p.t} className="rounded-[11px] border border-line-soft bg-raise/70 p-3">
                <p.i size={14} className="text-iris-soft" />
                <p className="mt-2 text-[12px] font-semibold text-ink">{p.t}</p>
                <p className="mt-1 text-[11px] leading-[1.55] text-ink-3">{p.d}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="px-6 pb-16">
        {/* ---------------- THEMES ---------------- */}
        <DS id="themes" eyebrow="foundation 00" title="Themes" body="Five themes share one token contract. Signal colours are re-tuned per theme for contrast; tints are derived with color-mix so every badge, diff and approval card adapts automatically. Click to switch live.">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {THEMES.map((t) => (
              <button key={t.id} onClick={() => setTheme(t.id)} className={cn("group cursor-pointer overflow-hidden rounded-[12px] border text-left transition-all hover:shadow-e3", theme === t.id ? "border-iris/60 shadow-glow-iris" : "border-line-soft hover:border-line-strong")}>
                <div className="relative h-[92px] p-2.5" style={{ background: t.swatch[0] }}>
                  <div className="h-full rounded-[7px] p-2" style={{ background: t.swatch[1], boxShadow: `inset 0 0 0 1px ${t.swatch[2]}33` }}>
                    <div className="h-[6px] w-1/2 rounded-full" style={{ background: t.swatch[2] }} />
                    <div className="mt-1.5 h-[4px] w-3/4 rounded-full opacity-40" style={{ background: t.swatch[3] }} />
                    <div className="mt-1 h-[4px] w-2/3 rounded-full opacity-25" style={{ background: t.swatch[3] }} />
                    <div className="mt-2.5 flex gap-1"><i className="h-[12px] w-[34px] rounded-[3px]" style={{ background: t.swatch[2] }} /><i className="h-[12px] w-[24px] rounded-[3px] opacity-50" style={{ background: t.swatch[3] }} /></div>
                  </div>
                  {theme === t.id && <span className="absolute top-2 right-2 flex size-[18px] items-center justify-center rounded-full bg-iris text-on-iris"><IconCheck size={10} /></span>}
                </div>
                <div className="bg-raise px-3 py-2.5"><div className="text-[12.5px] font-semibold text-ink">{t.label}</div><div className="text-[10.5px] text-ink-3">{t.desc}</div></div>
              </button>
            ))}
          </div>
        </DS>

        {/* ---------------- COLOR ---------------- */}
        <DS
          id="color"
          eyebrow="foundation 01"
          title="Colour"
          body="Three families only: surfaces for depth, ink for hierarchy, signal for meaning. Never use signal colour decoratively — if it's cyan, work is streaming right now."
        >
          <div className="space-y-4">
            {DS_COLORS.map((g) => (
              <div key={g.group}>
                <SectionLabel className="px-0">{g.group}</SectionLabel>
                <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-6">
                  {g.swatches.map((s) => (
                    <Swatch key={s.name} {...s} />
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 grid gap-3 lg:grid-cols-3">
            <Spec label="semantic mapping">
              <div className="space-y-2">
                {[
                  { c: "iris", m: "Agent identity · primary action · selection" },
                  { c: "cyan", m: "Streaming / actively working" },
                  { c: "mint", m: "Success · additions · approvals granted" },
                  { c: "amber", m: "Caution · approval required · partial support" },
                  { c: "rose", m: "Destructive · deletions · halted runs" },
                  { c: "sky", m: "Informational · reads · file access" },
                ].map((x) => (
                  <div key={x.c} className="flex items-center gap-2">
                    <i className={cn("size-[10px] shrink-0 rounded-[3px]", toneBg[x.c as keyof typeof toneBg])} />
                    <span className="font-mono text-[10.5px] text-ink-2">{x.c}</span>
                    <span className="min-w-0 flex-1 truncate text-[10.5px] text-ink-4">{x.m}</span>
                  </div>
                ))}
              </div>
            </Spec>
            <Spec label="contrast floor">
              <div className="space-y-2.5">
                {[
                  { pair: "ink on base", r: "14.8:1", ok: true },
                  { pair: "ink-2 on base", r: "8.1:1", ok: true },
                  { pair: "ink-3 on base", r: "4.9:1", ok: true },
                  { pair: "ink-4 on base", r: "2.8:1", ok: false },
                  { pair: "white on iris", r: "5.6:1", ok: true },
                ].map((x) => (
                  <div key={x.pair} className="flex items-center gap-2">
                    <span className="min-w-0 flex-1 font-mono text-[10.5px] text-ink-2">{x.pair}</span>
                    <span className="font-mono text-[10.5px] text-ink-3">{x.r}</span>
                    <span
                      className={cn(
                        "flex size-[15px] items-center justify-center rounded-[4px]",
                        x.ok ? "bg-mint-tint text-mint" : "bg-amber-tint text-amber",
                      )}
                    >
                      {x.ok ? <IconCheck size={9} /> : <IconWarning size={9} />}
                    </span>
                  </div>
                ))}
                <p className="pt-1 text-[10.5px] leading-[1.5] text-ink-4">
                  ink-4 is decorative only — placeholders, ticks, disabled glyphs. Never body copy.
                </p>
              </div>
            </Spec>
            <Spec label="elevation ladder">
              <div className="space-y-2">
                {[
                  { n: "e1", d: "buttons, small chips", cls: "shadow-e1 bg-raise" },
                  { n: "e2", d: "cards, raised rows", cls: "shadow-e2 bg-raise" },
                  { n: "e3", d: "menus, sticky bars", cls: "shadow-e3 bg-overlay" },
                  { n: "e4", d: "modals, command palette", cls: "shadow-e4 bg-overlay" },
                ].map((x) => (
                  <div key={x.n} className={cn("rounded-[8px] px-2.5 py-2", x.cls)}>
                    <span className="font-mono text-[10.5px] text-ink">{x.n}</span>
                    <span className="ml-2 text-[10.5px] text-ink-3">{x.d}</span>
                  </div>
                ))}
              </div>
            </Spec>
          </div>
        </DS>

        {/* ---------------- TYPE ---------------- */}
        <DS
          id="type"
          eyebrow="foundation 02"
          title="Typography & spacing"
          body="Three faces, one job each. Space Grotesk carries identity — the wordmark, page titles, empty-state headlines. Inter does all reading. JetBrains Mono is reserved for machine-authored text: paths, commands, hashes, diffs, counts. A path in a proportional face is the fastest way to lose a developer."
        >
          <div className="grid gap-3 lg:grid-cols-[1.3fr_1fr]">
            <Spec label="type scale">
              <div className="space-y-3">
                {DS_TYPE.map((t) => (
                  <div key={t.token} className="border-b border-line-soft/60 pb-3 last:border-0 last:pb-0">
                    <div className="mb-1 flex items-baseline gap-2">
                      <span className="font-mono text-[9.5px] text-iris-soft">{t.token}</span>
                      <span className="font-mono text-[9.5px] text-ink-4">{t.spec}</span>
                    </div>
                    <p className={cn("truncate text-ink", t.cls)}>
                      {t.token === "mono" ? "packages/bridge/src/server.ts" : t.token === "display" ? "Agent workbench" : t.token === "overline" ? "implementation plan" : t.token === "meta" ? "148.2k tokens · $3.41 · 2m ago" : t.token === "label" ? "Claude Code · opus-4.6" : "The bridge now survives a mid-stream disconnect."}
                    </p>
                  </div>
                ))}
              </div>
            </Spec>

            <div className="space-y-3">
              <Spec label="spacing — 4px rhythm">
                <div className="space-y-1.5">
                  {DS_SPACING.map((s) => (
                    <div key={s} className="flex items-center gap-2.5">
                      <span className="w-6 shrink-0 font-mono text-[9.5px] text-ink-4">{s}</span>
                      <div className="h-[7px] rounded-[2px] bg-gradient-to-r from-iris to-iris-soft" style={{ width: s * 4 }} />
                      <span className="font-mono text-[9px] text-ink-4">
                        {s % 4 === 0 ? `var(--s-${s})` : "avoid"}
                      </span>
                    </div>
                  ))}
                </div>
              </Spec>
              <Spec label="radius">
                <div className="grid grid-cols-3 gap-2">
                  {DS_RADII.map((r) => (
                    <div key={r.token} className="text-center">
                      <div
                        className="mx-auto h-9 w-full border border-iris/35 bg-iris-tint"
                        style={{ borderRadius: r.px }}
                      />
                      <div className="mt-1 font-mono text-[9.5px] text-ink-3">{r.token}</div>
                      <div className="font-mono text-[8.5px] text-ink-4">{r.px}px</div>
                    </div>
                  ))}
                </div>
              </Spec>
            </div>
          </div>
        </DS>

        {/* ---------------- MOTION ---------------- */}
        <DS
          id="motion"
          eyebrow="foundation 03"
          title="Motion"
          body="Motion exists to explain provenance and continuity. Streams animate forever; finished work never does. Anything under 200ms is a state change, anything over 300ms is a transition."
        >
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            {DS_MOTION.map((m) => (
              <div key={m.token} className="rounded-[10px] border border-line-soft bg-raise p-3">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[11px] text-iris-soft">{m.token}</span>
                  <span className="font-mono text-[9.5px] text-ink-4">{m.dur}</span>
                </div>
                <div className="mt-2.5 flex h-[26px] items-center gap-2 overflow-hidden rounded-[6px] border border-line-soft bg-sunken px-2">
                  {m.token === "shimmer" ? (
                    <div className="skeleton h-[8px] w-full rounded-full" />
                  ) : m.token === "breathe" ? (
                    <i className="size-[8px] animate-breathe rounded-full bg-cyan" />
                  ) : m.token === "sweep" ? (
                    <div className="relative h-[4px] w-full overflow-hidden rounded-full bg-track">
                      <div className="absolute inset-y-0 left-0 w-1/4 animate-sweep rounded-full bg-iris" />
                    </div>
                  ) : m.token === "caret" ? (
                    <span className="animate-caret font-mono text-[11px] text-iris-soft">▊</span>
                  ) : (
                    <span
                      className="rounded-[4px] bg-iris px-1.5 py-[2px] font-mono text-[9px] text-white"
                      style={{ animation: `rise 2.4s ${m.curve} infinite` }}
                    >
                      {m.token}
                    </span>
                  )}
                </div>
                <p className="mt-2 text-[10.5px] text-ink-3">{m.usage}</p>
                <p className="mt-1 truncate font-mono text-[9px] text-ink-4">{m.curve}</p>
              </div>
            ))}
          </div>
        </DS>

        {/* ---------------- COMPONENTS ---------------- */}
        <DS
          id="components"
          eyebrow="primitives"
          title="Components"
          body="Every primitive is keyboard-reachable, has a hover and a focus-visible state, and degrades gracefully when an agent is mid-stream."
        >
          <div className="grid gap-3 lg:grid-cols-2">
            <Spec label="buttons">
              <Row label="primary">
                <Button variant="primary" icon={IconBolt}>Build plan</Button>
                <Button variant="primary" size="xs">Xs</Button>
                <Button variant="primary" size="lg">Lg</Button>
              </Row>
              <Row label="secondary">
                <Button variant="secondary" icon={IconPlug}>Attach editor</Button>
                <Button variant="secondary" size="xs">Xs</Button>
              </Row>
              <Row label="outline / ghost">
                <Button variant="outline">Outline</Button>
                <Button variant="ghost">Ghost</Button>
                <IconButton icon={IconSearch} label="search" />
                <IconButton icon={IconBrain} label="thinking" active />
              </Row>
              <Row label="semantic">
                <Button variant="success" icon={IconCheck}>Allow once</Button>
                <Button variant="danger" icon={IconWarning}>Deny</Button>
                <Button variant="secondary" disabled>Disabled</Button>
              </Row>
            </Spec>

            <Spec label="badges, pills & keys">
              <Row label="tones">
                {(["neutral", "iris", "cyan", "mint", "amber", "rose", "sky", "plum"] as const).map((t) => (
                  <Badge key={t} tone={t} mono dot>
                    {t}
                  </Badge>
                ))}
              </Row>
              <Row label="status">
                <Badge tone="cyan" dot>running</Badge>
                <Badge tone="amber" dot>awaiting approval</Badge>
                <Badge tone="mint" dot>landed</Badge>
                <Badge tone="rose" dot>halted</Badge>
              </Row>
              <Row label="keyboard">
                <Kbd>⌘</Kbd> <Kbd>K</Kbd> <Kbd>⇧⌘P</Kbd> <Kbd>esc esc</Kbd> <Kbd>⏎</Kbd>
              </Row>
              <Row label="icon + text">
                <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-raise py-[3px] pr-2.5 pl-2 text-[11px] text-ink-3">
                  <IconAt size={11} className="text-iris-soft" /> @file
                </span>
                <span className="inline-flex items-center gap-1.5 rounded-[5px] border border-line bg-raise px-1.5 py-[1px] font-mono text-[10px] text-ink-3">
                  <IconGit size={9} /> feat/ide-bridge
                </span>
              </Row>
            </Spec>

            <Spec label="inputs & selection">
              <div className="space-y-2.5">
                <Input placeholder="Filter sessions…" icon-right="true" />
                <div className="flex gap-2">
                  <Select
                    value={sel}
                    onChange={setSel}
                    options={[
                      { value: "opus-4.6", label: "Opus 4.6", hint: "200k" },
                      { value: "sonnet-4.6", label: "Sonnet 4.6", hint: "200k" },
                      { value: "haiku-4.6", label: "Haiku 4.6", hint: "200k" },
                    ]}
                    className="flex-1"
                  />
                  <Select
                    value={sel}
                    onChange={setSel}
                    align="right"
                    options={[{ value: sel, label: "Right aligned" }]}
                    className="flex-1"
                  />
                </div>
                <div className="flex items-center gap-4">
                  <span className="flex items-center gap-2 text-[11.5px] text-ink-2">
                    <Toggle checked={tog} onChange={setTog} /> auto-approve
                  </span>
                  <span className="flex items-center gap-2 text-[11.5px] text-ink-2">
                    <Toggle checked={tog2} onChange={setTog2} size="sm" /> network
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Segmented
                    value={seg}
                    onChange={setSeg}
                    items={[
                      { value: "plan", label: "Plan", icon: IconBrain },
                      { value: "agent", label: "Agent", icon: IconBolt },
                    ]}
                  />
                  <Tabs
                    value={demoTab}
                    onChange={setDemoTab}
                    items={[
                      { value: "a", label: "Plan" },
                      { value: "b", label: "Context", count: 4 },
                      { value: "c", label: "Bridge" },
                    ]}
                  />
                </div>
              </div>
            </Spec>

            <Spec label="data display">
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  <Stat label="tokens" value="148.2k" sub="62% of window" chart={[22, 31, 28, 44, 51, 63, 72, 96]} />
                  <Stat label="spend" value="$3.41" sub="this session" tone="iris" />
                </div>
                <div className="flex items-center gap-4 rounded-[9px] border border-line-soft bg-sunken p-3">
                  <Ring value={62} size={54} stroke={5}>
                    <span className="font-mono text-[11px] font-semibold text-ink">62%</span>
                  </Ring>
                  <Ring value={91} size={54} stroke={5} tone="#46d68f">
                    <span className="font-mono text-[11px] font-semibold text-mint">91%</span>
                  </Ring>
                  <Ring value={38} size={54} stroke={5} tone="#f2b33d">
                    <span className="font-mono text-[11px] font-semibold text-amber">38%</span>
                  </Ring>
                  <div className="ml-auto space-y-2">
                    <Bar value={62} tone="iris" />
                    <Bar value={91} tone="mint" />
                    <Bar value={38} tone="amber" />
                    <Bar indeterminate tone="cyan" />
                  </div>
                </div>
                <div className="flex items-center justify-between rounded-[9px] border border-line-soft bg-sunken px-3 py-2.5">
                  <span className="font-mono text-[10px] text-ink-4">sparkline · 12 buckets</span>
                  <div className="flex items-end gap-3">
                    <Sparkline data={[4, 8, 6, 12, 9, 14, 11, 18]} color="#46d68f" w={70} h={22} />
                    <Sparkline data={[18, 14, 16, 9, 11, 6, 8, 4]} color="#ff6b7a" w={70} h={22} />
                  </div>
                </div>
              </div>
            </Spec>

            <Spec label="feedback">
              <div className="space-y-2.5">
                <div className="flex items-center gap-2 rounded-[8px] border border-amber/25 bg-amber-tint/60 px-2.5 py-2 text-[11.5px] text-amber">
                  <IconWarning size={12} /> Approval required — <span className="font-mono">gh pr create</span>
                </div>
                <div className="flex items-center gap-2 rounded-[8px] border border-rose/25 bg-rose-tint/60 px-2.5 py-2 text-[11.5px] text-rose">
                  <IconWarning size={12} /> Spend cap reached — run halted at 44%
                </div>
                <div className="flex items-center gap-2 rounded-[8px] border border-mint/25 bg-mint-tint/60 px-2.5 py-2 text-[11.5px] text-mint">
                  <IconCheck size={12} /> 14 tests passed in 284ms
                </div>
                <div className="flex items-center gap-2 rounded-[8px] border border-dashed border-line px-2.5 py-2 text-[11.5px] text-ink-3 italic">
                  <IconBrain size={12} className="text-plum" /> reasoning · 4.2s — collapsed by default
                </div>
              </div>
            </Spec>

            <Spec label="overlay & empty">
              <div className="space-y-2.5">
                <Button variant="secondary" icon={IconBolt} onClick={() => setModal(true)}>
                  Open modal
                </Button>
                <div className="rounded-[10px] border border-dashed border-line">
                  <EmptyState
                    icon={IconSearch}
                    title="No sessions match that filter"
                    body="Try clearing the query, or start a new session with a different agent."
                    action={
                      <Button variant="secondary" size="xs">
                        Clear filter
                      </Button>
                    }
                  />
                </div>
                <div className="flex flex-wrap gap-2">
                  <Tip label="tooltip · 150ms fade">
                    <Button variant="ghost" size="xs">Hover me</Button>
                  </Tip>
                  <Tip label="⌘B">
                    <Button variant="ghost" size="xs">With shortcut</Button>
                  </Tip>
                </div>
              </div>
            </Spec>
          </div>
        </DS>

        {/* ---------------- PATTERNS ---------------- */}
        <DS
          id="patterns"
          eyebrow="patterns"
          title="Agent-specific patterns"
          body="These compositions are the reason the system exists. They are shared across every connected agent so that switching from Codex to Claude Code to Aro never changes how you read a session."
        >
          <div className="grid gap-3 lg:grid-cols-2">
            <Spec label="tool invocation card">
              <div className="space-y-2">
                <div className="flex items-center gap-2 rounded-[9px] border border-line-soft bg-well px-2.5 py-[7px]">
                  <span className="flex size-[19px] items-center justify-center rounded-[5px] border border-line-soft bg-raise text-sky">
                    <IconTerminal size={11} />
                  </span>
                  <span className="font-mono text-[9.5px] tracking-[.12em] text-ink-4 uppercase">read</span>
                  <span className="min-w-0 flex-1 truncate font-mono text-[11.5px] text-ink-2">
                    packages/bridge/src/server.ts
                  </span>
                  <span className="font-mono text-[10px] text-ink-4">180ms</span>
                </div>
                <div className="flex items-center gap-2 rounded-[9px] border border-cyan/25 bg-well px-2.5 py-[7px]">
                  <span className="flex size-[19px] items-center justify-center rounded-[5px] border border-cyan/25 bg-cyan-tint text-cyan">
                    <IconSpark size={11} />
                  </span>
                  <span className="font-mono text-[9.5px] tracking-[.12em] text-ink-4 uppercase">edit</span>
                  <span className="min-w-0 flex-1 truncate font-mono text-[11.5px] text-ink-2">reviewer.ts</span>
                  <span className="flex items-center gap-1.5 font-mono text-[10.5px] text-cyan">
                    <i className="size-[5px] animate-breathe rounded-full bg-cyan" /> running
                  </span>
                </div>
              </div>
            </Spec>

            <Spec label="provenance line">
              <div className="space-y-2">
                {[
                  { who: "Claude Code", model: "opus-4.6", act: "edited 3 files", tone: "iris" as const },
                  { who: "Dev Vale", model: "vscode 1.104", act: "edited 1 file", tone: "sky" as const },
                  { who: "Codex", model: "gpt-5.4-codex", act: "ran 12 tests", tone: "mint" as const },
                ].map((p) => (
                  <div key={p.who} className="flex items-center gap-2 rounded-[8px] border border-line-soft bg-sunken px-2.5 py-2">
                    <span className={cn("size-[6px] rounded-full", toneBg[p.tone])} />
                    <span className="text-[11.5px] font-medium text-ink">{p.who}</span>
                    <span className="font-mono text-[9.5px] text-ink-4">{p.model}</span>
                    <span className="ml-auto text-[10.5px] text-ink-3">{p.act}</span>
                  </div>
                ))}
                <p className="text-[10.5px] leading-[1.55] text-ink-4">
                  Agent-originated and editor-originated edits must be visually distinguishable at a glance, but
                  reviewed through one queue.
                </p>
              </div>
            </Spec>

            <Spec label="mode guardrail">
              <div className="space-y-2">
                {[
                  { m: "Plan", tone: "iris", rules: ["no file writes", "no shell exec", "must end with a plan"] },
                  { m: "Full access", tone: "rose", rules: ["everything allowed", "auto-checkpoint", "spend cap $25"] },
                ].map((x) => (
                  <div key={x.m} className="rounded-[9px] border border-line-soft bg-sunken p-2.5">
                    <div className="flex items-center gap-2">
                      <Badge tone={x.tone as "iris"} mono>
                        {x.m}
                      </Badge>
                      <span className="font-mono text-[9.5px] text-ink-4">effective rules</span>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {x.rules.map((r) => (
                        <span key={r} className="rounded-[4px] bg-hover px-1.5 py-[1px] font-mono text-[9.5px] text-ink-3">
                          {r}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </Spec>

            <Spec label="checkpoint strip">
              <div className="rounded-[9px] border border-line-soft bg-sunken p-3">
                <div className="space-y-2.5">
                  {[
                    { h: "a91f3c2", l: "before bridge rewrite", on: true },
                    { h: "77bd0e4", l: "channel + flush added", on: false },
                    { h: "1c3a9f8", l: "session start", on: false },
                  ].map((c, i) => (
                    <div key={c.h} className="flex items-center gap-2.5">
                      <span className="flex flex-col items-center">
                        <i
                          className={cn(
                            "size-[7px] rounded-full",
                            c.on ? "bg-iris shadow-[0_0_6px_rgba(124,107,255,.9)]" : "bg-line-strong",
                          )}
                        />
                        {i < 2 && <span className="mt-0.5 h-4 w-px bg-line" />}
                      </span>
                      <span className="font-mono text-[10.5px] text-ink-2">{c.h}</span>
                      <span className="text-[11px] text-ink-3">{c.l}</span>
                      <button className="ml-auto cursor-pointer font-mono text-[9.5px] text-iris-soft hover:underline">
                        restore
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </Spec>
          </div>
        </DS>

        {/* ---------------- TOKENS ---------------- */}
        <DS
          id="tokens"
          eyebrow="reference"
          title="Token export"
          body="The whole system compiles from these values. Drop them into any Tailwind v4 @theme block, or generate CSS custom properties for a non-Tailwind surface."
        >
          <div className="overflow-hidden rounded-[11px] border border-line-soft bg-code">
            <div className="flex items-center gap-1.5 border-b border-line-soft px-3 py-2">
              {["#ff5f57", "#febc2e", "#28c840"].map((c) => (
                <i key={c} className="size-[7px] rounded-full" style={{ background: c }} />
              ))}
              <span className="ml-1.5 font-mono text-[10px] text-ink-4">aro.theme.css</span>
              <span className="ml-auto font-mono text-[9.5px] text-ink-4">64 tokens</span>
            </div>
            <pre className="scroll-thin overflow-x-auto px-4 py-3 font-mono text-[11px] leading-[1.8] text-ink-3">
{`@theme {
  /* surfaces */
  --color-void: #06070A;      --color-base: #0C0E14;
  --color-sunken: #090B10;    --color-raise: #12151D;
  --color-hover: #1A1E29;     --color-line: #1F232F;

  /* ink */
  --color-ink: #EAECF3;       --color-ink-2: #A3ACBE;
  --color-ink-3: #6E7890;     --color-ink-4: #4B5465;

  /* signal */
  --color-iris: #7C6BFF;      --color-cyan: #35D6C4;
  --color-mint: #46D68F;      --color-amber: #F2B33D;
  --color-rose: #FF6B7A;      --color-sky:  #5BA8FF;

  /* type */
  --font-sans: "Inter", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, monospace;

  /* radius */
  --radius-xs: 3px;  --radius-sm: 5px;  --radius-md: 7px;
  --radius-lg: 10px; --radius-xl: 14px; --radius-2xl: 18px;

  /* elevation */
  --shadow-e1: 0 1px 2px rgba(0,0,0,.4);
  --shadow-e2: 0 2px 8px -2px rgba(0,0,0,.55);
  --shadow-e3: 0 12px 32px -8px rgba(0,0,0,.7);
  --shadow-e4: 0 32px 80px -24px rgba(0,0,0,.85);
}`}
            </pre>
          </div>
          <Divider className="my-6" />
          <div className="flex flex-wrap items-center gap-3">
            <Badge tone="neutral" mono>a11y: full keyboard map</Badge>
            <Badge tone="neutral" mono>tab order: rail → list → transcript → composer</Badge>
            <Badge tone="neutral" mono>reduced-motion: streams become static dots</Badge>
            <Badge tone="neutral" mono>min target 22×22px</Badge>
          </div>
        </DS>
      </div>

      <Modal
        open={modal}
        onClose={() => setModal(false)}
        title="Modal surface"
        sub="e4 elevation · 14px radius · Escape closes · backdrop blur 3px"
        width={520}
      >
        <div className="space-y-3 p-4">
          <p className="text-[12.5px] leading-[1.65] text-ink-2">
            Modals are used for decisions that change what the agent is allowed to do — never for information the
            transcript already shows.
          </p>
          <div className="rounded-[9px] border border-line-soft bg-sunken p-3">
            <SectionLabel className="px-0">example content</SectionLabel>
            <div className="space-y-2">
              <Input placeholder="Session name" />
              <Select
                value="opus-4.6"
                onChange={() => {}}
                options={[
                  { value: "opus-4.6", label: "Opus 4.6", hint: "$25 / 1M" },
                  { value: "gpt-5.4", label: "GPT-5.4-Codex", hint: "$14 / 1M" },
                ]}
              />
            </div>
          </div>
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setModal(false)}>Cancel</Button>
            <Button variant="primary" icon={IconCheck} onClick={() => setModal(false)}>
              Confirm
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
