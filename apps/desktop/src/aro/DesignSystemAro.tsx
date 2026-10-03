/**
 * ARO DESIGN SYSTEM — REFERENCE PAGE (desktop port)
 *
 * `/aro-design` — a live gallery of the Aro token layer + primitive kit,
 * adapted from the prototype's DesignSystemView. The page themes ITSELF:
 * the root wrapper carries `data-theme`, so the aro token layer resolves
 * only inside this subtree and the rest of the desktop app (its theme,
 * its scrollbars, its surfaces) is completely untouched while the page is
 * open — and reverts the moment it unmounts.
 */

import { useState } from 'react'
import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

import { AgentMark, IconAt, IconBolt, IconBrain, IconCheck, IconGit, IconPlug, IconSearch, IconShield, IconSpark, IconTerminal, IconWarning, LogoMark } from './Icons'
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
  Tip,
  Toasts,
  Toggle,
  toneBg,
} from './ui'

/* ------------------------------ theme registry ---------------------------- */
type ThemeId = 'obsidian' | 'daylight' | 'nord' | 'ember' | 'paper'

const THEMES: { id: ThemeId; label: string; desc: string; swatch: string[] }[] = [
  { id: 'obsidian', label: 'Obsidian', desc: 'Deep neutral dark · violet accent', swatch: ['#0c0e14', '#12151d', '#7c6bff', '#35d6c4'] },
  { id: 'daylight', label: 'Daylight', desc: 'Clean light · high legibility', swatch: ['#ffffff', '#f4f5f8', '#5b4ce6', '#0f9f90'] },
  { id: 'nord', label: 'Nord', desc: 'Arctic blue-grey · frost accent', swatch: ['#2a303b', '#313845', '#88c0d0', '#a3be8c'] },
  { id: 'ember', label: 'Ember', desc: 'Warm charcoal · copper accent', swatch: ['#171210', '#1f1815', '#ff8a4c', '#4fd1b5'] },
  { id: 'paper', label: 'Paper', desc: 'Warm light · ink on paper', swatch: ['#faf7f0', '#f4f0e6', '#8a5cf6', '#0f8f86'] },
]

/* ------------------------------ inline catalogue -------------------------- */
const DS_COLORS: { group: string; swatches: { name: string; cssVar: string; usage: string }[] }[] = [
  {
    group: 'Surface',
    swatches: [
      { name: 'void', cssVar: '--aro-void', usage: 'App chrome, gutters' },
      { name: 'sunken', cssVar: '--aro-sunken', usage: 'Inputs, wells' },
      { name: 'bg', cssVar: '--aro-bg', usage: 'Panels, modals' },
      { name: 'raise', cssVar: '--aro-raise', usage: 'Cards, bars, rows' },
      { name: 'hover', cssVar: '--aro-hover', usage: 'Hover target' },
      { name: 'line', cssVar: '--aro-line', usage: 'Hairline borders' },
    ],
  },
  {
    group: 'Ink',
    swatches: [
      { name: 'ink', cssVar: '--aro-ink', usage: 'Primary text' },
      { name: 'ink-2', cssVar: '--aro-ink-2', usage: 'Secondary text' },
      { name: 'ink-3', cssVar: '--aro-ink-3', usage: 'Meta, labels' },
      { name: 'ink-4', cssVar: '--aro-ink-4', usage: 'Placeholders, ticks' },
    ],
  },
  {
    group: 'Signal',
    swatches: [
      { name: 'iris', cssVar: '--aro-iris', usage: 'Brand, primary action, agent' },
      { name: 'cyan', cssVar: '--aro-cyan', usage: 'Streaming / active work' },
      { name: 'mint', cssVar: '--aro-mint', usage: 'Success, additions' },
      { name: 'amber', cssVar: '--aro-amber', usage: 'Caution, approvals' },
      { name: 'rose', cssVar: '--aro-rose', usage: 'Destructive, deletions' },
      { name: 'sky', cssVar: '--aro-sky', usage: 'Informational' },
    ],
  },
]

const DS_TYPE: { token: string; cls: string; spec: string }[] = [
  { token: 'display', cls: 'font-aro-display text-[26px] leading-[1.12] font-semibold tracking-[-.028em]', spec: 'Space Grotesk · 26/29 · 600' },
  { token: 'title', cls: 'text-[15px] leading-[1.3] font-semibold tracking-[-.01em]', spec: '15 / 20 · 600 · -1%' },
  { token: 'body', cls: 'text-[13px] leading-[1.6] font-normal', spec: '13 / 21 · 400' },
  { token: 'label', cls: 'text-[12px] leading-[1.4] font-medium', spec: '12 / 17 · 500' },
  { token: 'meta', cls: 'text-[11px] leading-[1.4] font-normal text-aro-ink-3', spec: '11 / 15 · 400' },
  { token: 'mono', cls: 'font-aro-mono text-[11.5px] leading-[1.55]', spec: 'JetBrains Mono 11.5' },
  { token: 'overline', cls: 'font-aro-mono text-[9.5px] font-semibold tracking-[.14em] uppercase text-aro-ink-4', spec: '9.5 · 600 · +14%' },
]

const DS_RADII: { token: string; px: number; usage: string }[] = [
  { token: 'xs', px: 3, usage: 'kbd, tiny chips' },
  { token: 'sm', px: 5, usage: 'badges, small buttons' },
  { token: 'md', px: 7, usage: 'buttons, inputs' },
  { token: 'lg', px: 10, usage: 'cards, panels' },
  { token: 'xl', px: 14, usage: 'modals, popovers' },
  { token: '2xl', px: 18, usage: 'hero surfaces' },
]

const DS_MOTION: { token: string; curve: string; dur: string; usage: string }[] = [
  { token: 'rise', curve: 'cubic-bezier(.22,1,.36,1)', dur: '320ms', usage: 'Rows and cards entering' },
  { token: 'pop', curve: 'cubic-bezier(.34,1.56,.64,1)', dur: '220ms', usage: 'Modals, popovers' },
  { token: 'slide-down', curve: 'cubic-bezier(.22,1,.36,1)', dur: '200ms', usage: 'Menus, disclosures' },
  { token: 'shimmer', curve: 'linear', dur: '2.2s', usage: 'Loading skeletons' },
  { token: 'breathe', curve: 'ease-in-out', dur: '2.6s', usage: 'Live agent pulse' },
  { token: 'sweep', curve: 'ease-in-out', dur: '1.6s', usage: 'Indeterminate progress' },
]

const DS_SPACING = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40]

/* ============================ section frame ============================ */
function DS({ id, eyebrow, title, body, children }: { id: string; eyebrow: string; title: string; body: string; children: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-16 border-t border-aro-line-soft py-8 first:border-0">
      <div className="mb-5">
        <span className="font-aro-mono text-[9.5px] font-semibold tracking-[.16em] text-aro-iris-soft uppercase">{eyebrow}</span>
        <h2 className="mt-1.5 text-[19px] leading-tight font-semibold tracking-[-.02em] text-aro-ink">{title}</h2>
        <p className="mt-1.5 max-w-[680px] text-[12.5px] leading-[1.65] text-aro-ink-3">{body}</p>
      </div>
      {children}
    </section>
  )
}

function Spec({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="rounded-[10px] border border-aro-line-soft bg-aro-sunken p-3">
      <div className="mb-2.5 font-aro-mono text-[9px] tracking-[.14em] text-aro-ink-4 uppercase">{label}</div>
      {children}
    </div>
  )
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-aro-line-soft/60 py-2.5 last:border-0">
      <span className="w-[110px] shrink-0 font-aro-mono text-[10px] text-aro-ink-4">{label}</span>
      {children}
    </div>
  )
}

/* ============================ color swatch ============================ */
function Swatch({ name, cssVar, usage }: { name: string; cssVar: string; usage: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      onClick={() => {
        navigator.clipboard?.writeText(`var(${cssVar})`)
        setCopied(true)
        setTimeout(() => setCopied(false), 1100)
      }}
      className="group flex w-full cursor-pointer flex-col overflow-hidden rounded-[10px] border border-aro-line-soft bg-aro-raise text-left transition-all hover:border-aro-line-strong hover:shadow-aro-e2"
    >
      <div className="relative h-[62px] w-full" style={{ background: `var(${cssVar})` }}>
        <div
          className="absolute inset-0 opacity-0 transition-opacity group-hover:opacity-100"
          style={{ background: 'linear-gradient(180deg,rgba(255,255,255,.12),transparent)' }}
        />
        <span
          className={cn(
            'absolute top-2 right-2 rounded-[4px] px-1.5 py-[1px] font-aro-mono text-[9px] transition-all',
            copied ? 'bg-black/50 text-white opacity-100' : 'bg-black/35 text-white/80 opacity-0 group-hover:opacity-100',
          )}
        >
          {copied ? 'copied' : cssVar}
        </span>
      </div>
      <div className="px-2.5 py-2">
        <div className="font-aro-mono text-[11px] text-aro-ink">{name}</div>
        <div className="mt-0.5 text-[10.5px] leading-[1.4] text-aro-ink-3">{usage}</div>
      </div>
    </button>
  )
}

/* ============================ main view ============================ */
export function AroDesignSystemPage() {
  const [theme, setTheme] = useState<ThemeId>('obsidian')
  const [demoTab, setDemoTab] = useState<'a' | 'b' | 'c'>('a')
  const [seg, setSeg] = useState<'plan' | 'agent'>('plan')
  const [sel, setSel] = useState('opus-4.6')
  const [tog, setTog] = useState(true)
  const [tog2, setTog2] = useState(false)
  const [modal, setModal] = useState(false)
  const [toasts, setToasts] = useState<{ id: number; msg: string; tone: string }[]>([])

  const toast = (msg: string, tone = 'iris') => {
    const id = Date.now()
    setToasts((t) => [...t, { id, msg, tone }])
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 2400)
  }

  return (
    // data-theme on the WRAPPER: the aro token layer resolves only inside
    // this subtree. Unmounting the page removes the attribute with it.
    <div data-theme={theme} className="aro-scroll-thin h-full overflow-y-auto bg-aro-void text-aro-ink transition-colors duration-200">
      {/* hero */}
      <div className="relative overflow-hidden border-b border-aro-line-soft">
        <div className="aro-grid-bg pointer-events-none absolute inset-0 opacity-60" />
        <div
          className="pointer-events-none absolute -top-24 -right-10 size-[420px] rounded-full opacity-[0.18] blur-[80px]"
          style={{ background: 'radial-gradient(circle,var(--aro-iris),transparent 70%)' }}
        />
        <div className="relative px-6 py-8">
          <div className="flex items-center gap-2">
            <LogoMark size={26} />
            <Badge tone="iris" mono className="text-[9.5px]">
              aro/primitives · desktop port
            </Badge>
            <Badge tone="neutral" mono className="text-[9.5px]">
              additive · namespaced · zero host risk
            </Badge>
          </div>
          <h1 className="mt-3 max-w-[760px] font-aro-display text-[30px] leading-[1.08] font-semibold tracking-[-.03em] text-aro-ink">
            A design system for{' '}
            <span className="relative whitespace-nowrap text-aro-iris-soft">
              agent control surfaces
              <span className="absolute -bottom-1 left-0 h-[3px] w-full rounded-full bg-linear-to-r from-aro-iris via-aro-cyan to-transparent opacity-70" />
            </span>
          </h1>
          <p className="mt-2.5 max-w-[640px] text-[13px] leading-[1.7] text-aro-ink-3">
            Ported additively into the desktop renderer: every token is namespaced <span className="font-aro-mono text-[12px]">--aro-*</span> and
            gated behind <span className="font-aro-mono text-[12px]">data-theme</span>, so this page themes itself without touching a single
            existing screen.
          </p>
          <div className="mt-5 grid max-w-[860px] grid-cols-2 gap-3 md:grid-cols-4">
            {[
              { t: 'Provenance first', d: 'Every artefact carries an agent mark, a model badge and a timestamp.', i: IconShield },
              { t: 'Intent is a mode', d: 'Plan / Agent / Read-only / Full-access is a persistent, colour-coded state.', i: IconBolt },
              { t: 'Nothing silent', d: 'Tool calls, approvals and checkpoints are first-class, not log lines.', i: IconTerminal },
              { t: 'Density with air', d: '11–13px type, 4px rhythm, hairlines instead of heavy chrome.', i: IconSpark },
            ].map((p) => (
              <div key={p.t} className="rounded-[11px] border border-aro-line-soft bg-aro-raise/70 p-3">
                <p.i size={14} className="text-aro-iris-soft" />
                <p className="mt-2 text-[12px] font-semibold text-aro-ink">{p.t}</p>
                <p className="mt-1 text-[11px] leading-[1.55] text-aro-ink-3">{p.d}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="px-6 pb-16">
        {/* ---------------- THEMES ---------------- */}
        <DS
          id="themes"
          eyebrow="foundation 00"
          title="Themes"
          body="Five themes share one token contract. Signal colours are re-tuned per theme for contrast; tints are derived with color-mix so every badge, diff and approval card adapts automatically. Click to switch live — the whole page retints."
        >
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {THEMES.map((t) => (
              <button
                key={t.id}
                onClick={() => setTheme(t.id)}
                className={cn(
                  'group cursor-pointer overflow-hidden rounded-[12px] border text-left transition-all hover:shadow-aro-e3',
                  theme === t.id ? 'border-aro-iris/60 shadow-aro-glow' : 'border-aro-line-soft hover:border-aro-line-strong',
                )}
              >
                <div className="relative h-[92px] p-2.5" style={{ background: t.swatch[0] }}>
                  <div className="h-full rounded-[7px] p-2" style={{ background: t.swatch[1], boxShadow: `inset 0 0 0 1px ${t.swatch[2]}33` }}>
                    <div className="h-[6px] w-1/2 rounded-full" style={{ background: t.swatch[2] }} />
                    <div className="mt-1.5 h-[4px] w-3/4 rounded-full opacity-40" style={{ background: t.swatch[3] }} />
                    <div className="mt-1 h-[4px] w-2/3 rounded-full opacity-25" style={{ background: t.swatch[3] }} />
                    <div className="mt-2.5 flex gap-1">
                      <i className="h-[12px] w-[34px] rounded-[3px]" style={{ background: t.swatch[2] }} />
                      <i className="h-[12px] w-[24px] rounded-[3px] opacity-50" style={{ background: t.swatch[3] }} />
                    </div>
                  </div>
                  {theme === t.id && (
                    <span className="absolute top-2 right-2 flex size-[18px] items-center justify-center rounded-full bg-aro-iris text-aro-on-iris">
                      <IconCheck size={10} />
                    </span>
                  )}
                </div>
                <div className="bg-aro-raise px-3 py-2.5">
                  <div className="text-[12.5px] font-semibold text-aro-ink">{t.label}</div>
                  <div className="text-[10.5px] text-aro-ink-3">{t.desc}</div>
                </div>
              </button>
            ))}
          </div>
        </DS>

        {/* ---------------- COLOR ---------------- */}
        <DS
          id="color"
          eyebrow="foundation 01"
          title="Colour"
          body="Three families only: surfaces for depth, ink for hierarchy, signal for meaning. Swatches read the live tokens — they retint with the theme above. Click a swatch to copy its CSS variable."
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
                  { c: 'iris', m: 'Agent identity · primary action · selection' },
                  { c: 'cyan', m: 'Streaming / actively working' },
                  { c: 'mint', m: 'Success · additions · approvals granted' },
                  { c: 'amber', m: 'Caution · approval required · partial support' },
                  { c: 'rose', m: 'Destructive · deletions · halted runs' },
                  { c: 'sky', m: 'Informational · reads · file access' },
                ].map((x) => (
                  <div key={x.c} className="flex items-center gap-2">
                    <i className={cn('size-[10px] shrink-0 rounded-[3px]', toneBg[x.c as keyof typeof toneBg])} />
                    <span className="font-aro-mono text-[10.5px] text-aro-ink-2">{x.c}</span>
                    <span className="min-w-0 flex-1 truncate text-[10.5px] text-aro-ink-4">{x.m}</span>
                  </div>
                ))}
              </div>
            </Spec>
            <Spec label="elevation ladder">
              <div className="space-y-2">
                {[
                  { n: 'e1', d: 'buttons, small chips', cls: 'shadow-aro-e1 bg-aro-raise' },
                  { n: 'e2', d: 'cards, raised rows', cls: 'shadow-aro-e2 bg-aro-raise' },
                  { n: 'e3', d: 'menus, sticky bars', cls: 'shadow-aro-e3 bg-aro-overlay' },
                  { n: 'e4', d: 'modals, command palette', cls: 'shadow-aro-e4 bg-aro-overlay' },
                ].map((x) => (
                  <div key={x.n} className={cn('rounded-[8px] px-2.5 py-2', x.cls)}>
                    <span className="font-aro-mono text-[10.5px] text-aro-ink">{x.n}</span>
                    <span className="ml-2 text-[10.5px] text-aro-ink-3">{x.d}</span>
                  </div>
                ))}
              </div>
            </Spec>
            <Spec label="brand accent — aro emerald">
              <div className="space-y-2.5">
                <div className="flex items-center gap-3">
                  <LogoMark size={30} />
                  <div>
                    <div className="text-[12px] font-semibold text-aro-ink">Aro mark</div>
                    <div className="font-aro-mono text-[10px] text-aro-ink-4">#10B981 · #6EE7B7 · #0D9488</div>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Badge tone="neutral" mono>
                    <span className="text-aro-emerald">●</span> text-aro-emerald
                  </Badge>
                  <Badge tone="neutral" mono>
                    <span className="text-aro-emerald-soft">●</span> text-aro-emerald-soft
                  </Badge>
                  <Badge tone="neutral" mono>
                    bg-aro-emerald-tint
                  </Badge>
                </div>
                <div className="flex items-end gap-2 rounded-[8px] border border-aro-emerald/25 bg-aro-emerald-tint px-2.5 py-2 text-[11.5px] text-aro-emerald">
                  <AgentMark glyph="A" from="#10b981" to="#0d9488" size={18} /> agent tone — the mark's emerald, usable as a live/ok accent
                </div>
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
                  <div key={t.token} className="border-b border-aro-line-soft/60 pb-3 last:border-0 last:pb-0">
                    <div className="mb-1 flex items-baseline gap-2">
                      <span className="font-aro-mono text-[9.5px] text-aro-iris-soft">{t.token}</span>
                      <span className="font-aro-mono text-[9.5px] text-aro-ink-4">{t.spec}</span>
                    </div>
                    <p className={cn('truncate text-aro-ink', t.cls)}>
                      {t.token === 'mono'
                        ? 'packages/bridge/src/server.ts'
                        : t.token === 'display'
                          ? 'Agent workbench'
                          : t.token === 'overline'
                            ? 'implementation plan'
                            : t.token === 'meta'
                              ? '148.2k tokens · $3.41 · 2m ago'
                              : t.token === 'label'
                                ? 'Aro · opus-4.6'
                                : 'The bridge now survives a mid-stream disconnect.'}
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
                      <span className="w-6 shrink-0 font-aro-mono text-[9.5px] text-aro-ink-4">{s}</span>
                      <div className="h-[7px] rounded-[2px] bg-linear-to-r from-aro-iris to-aro-iris-soft" style={{ width: s * 4 }} />
                      <span className="font-aro-mono text-[9px] text-aro-ink-4">{s % 4 === 0 ? `${s}px` : 'avoid'}</span>
                    </div>
                  ))}
                </div>
              </Spec>
              <Spec label="radius">
                <div className="grid grid-cols-3 gap-2">
                  {DS_RADII.map((r) => (
                    <div key={r.token} className="text-center">
                      <div className="mx-auto h-9 w-full border border-aro-iris/35 bg-aro-iris-tint" style={{ borderRadius: r.px }} />
                      <div className="mt-1 font-aro-mono text-[9.5px] text-aro-ink-3">aro-{r.token}</div>
                      <div className="font-aro-mono text-[8.5px] text-aro-ink-4">{r.usage}</div>
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
          body="Motion exists to explain provenance and continuity. Streams animate forever; finished work never does. Anything under 200ms is a state change, anything over 300ms is a transition. Keyframes ship aro- prefixed so they can never collide with the host's."
        >
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            {DS_MOTION.map((m) => (
              <div key={m.token} className="rounded-[10px] border border-aro-line-soft bg-aro-raise p-3">
                <div className="flex items-center justify-between">
                  <span className="font-aro-mono text-[11px] text-aro-iris-soft">aro-{m.token}</span>
                  <span className="font-aro-mono text-[9.5px] text-aro-ink-4">{m.dur}</span>
                </div>
                <div className="mt-2.5 flex h-[26px] items-center gap-2 overflow-hidden rounded-[6px] border border-aro-line-soft bg-aro-sunken px-2">
                  {m.token === 'shimmer' ? (
                    <div className="aro-skeleton h-[8px] w-full rounded-full" />
                  ) : m.token === 'breathe' ? (
                    <i className="size-[8px] animate-aro-breathe rounded-full bg-aro-cyan" />
                  ) : m.token === 'sweep' ? (
                    <div className="relative h-[4px] w-full overflow-hidden rounded-full bg-aro-track">
                      <div className="absolute inset-y-0 left-0 w-1/4 animate-aro-sweep rounded-full bg-aro-iris" />
                    </div>
                  ) : (
                    <span
                      className="rounded-[4px] bg-aro-iris px-1.5 py-[2px] font-aro-mono text-[9px] text-white"
                      style={{ animation: `aro-${m.token} 2.4s ${m.curve} infinite` }}
                    >
                      {m.token}
                    </span>
                  )}
                </div>
                <p className="mt-2 text-[10.5px] text-aro-ink-3">{m.usage}</p>
                <p className="mt-1 truncate font-aro-mono text-[9px] text-aro-ink-4">{m.curve}</p>
              </div>
            ))}
          </div>
        </DS>

        {/* ---------------- COMPONENTS ---------------- */}
        <DS
          id="components"
          eyebrow="primitives"
          title="Components"
          body="Every primitive is keyboard-reachable, has a hover and a focus-visible state, and consumes only namespaced aro tokens. These are the exact components exported from src/aro/ui.tsx."
        >
          <div className="grid gap-3 lg:grid-cols-2">
            <Spec label="buttons">
              <Row label="primary">
                <Button variant="primary" icon={IconBolt}>
                  Build plan
                </Button>
                <Button variant="primary" size="xs">
                  Xs
                </Button>
                <Button variant="primary" size="lg">
                  Lg
                </Button>
              </Row>
              <Row label="secondary">
                <Button variant="secondary" icon={IconPlug}>
                  Attach editor
                </Button>
                <Button variant="secondary" size="xs">
                  Xs
                </Button>
              </Row>
              <Row label="outline / ghost">
                <Button variant="outline">Outline</Button>
                <Button variant="ghost">Ghost</Button>
                <IconButton icon={IconSearch} label="search" />
                <IconButton icon={IconBrain} label="thinking" active />
              </Row>
              <Row label="semantic">
                <Button variant="success" icon={IconCheck}>
                  Allow once
                </Button>
                <Button variant="danger" icon={IconWarning}>
                  Deny
                </Button>
                <Button variant="secondary" disabled>
                  Disabled
                </Button>
              </Row>
            </Spec>

            <Spec label="badges, pills & keys">
              <Row label="tones">
                {(['neutral', 'iris', 'cyan', 'mint', 'amber', 'rose', 'sky', 'plum'] as const).map((t) => (
                  <Badge key={t} tone={t} mono dot>
                    {t}
                  </Badge>
                ))}
              </Row>
              <Row label="status">
                <Badge tone="cyan" dot>
                  running
                </Badge>
                <Badge tone="amber" dot>
                  awaiting approval
                </Badge>
                <Badge tone="mint" dot>
                  landed
                </Badge>
                <Badge tone="rose" dot>
                  halted
                </Badge>
              </Row>
              <Row label="keyboard">
                <Kbd>⌘</Kbd> <Kbd>K</Kbd> <Kbd>⇧⌘P</Kbd> <Kbd>esc esc</Kbd> <Kbd>⏎</Kbd>
              </Row>
              <Row label="icon + text">
                <span className="inline-flex items-center gap-1.5 rounded-full border border-aro-line bg-aro-raise py-[3px] pr-2.5 pl-2 text-[11px] text-aro-ink-3">
                  <IconAt size={11} className="text-aro-iris-soft" /> @file
                </span>
                <span className="inline-flex items-center gap-1.5 rounded-[5px] border border-aro-line bg-aro-raise px-1.5 py-[1px] font-aro-mono text-[10px] text-aro-ink-3">
                  <IconGit size={9} /> feat/ide-bridge
                </span>
              </Row>
            </Spec>

            <Spec label="inputs & selection">
              <div className="space-y-2.5">
                <Input placeholder="Filter sessions…" />
                <div className="flex gap-2">
                  <Select
                    value={sel}
                    onChange={setSel}
                    options={[
                      { value: 'opus-4.6', label: 'Opus 4.6', hint: '200k' },
                      { value: 'sonnet-4.6', label: 'Sonnet 4.6', hint: '200k' },
                      { value: 'haiku-4.6', label: 'Haiku 4.6', hint: '200k' },
                    ]}
                    className="flex-1"
                  />
                  <Select value={sel} onChange={setSel} align="right" options={[{ value: sel, label: 'Right aligned' }]} className="flex-1" />
                </div>
                <div className="flex items-center gap-4">
                  <span className="flex items-center gap-2 text-[11.5px] text-aro-ink-2">
                    <Toggle checked={tog} onChange={setTog} /> auto-approve
                  </span>
                  <span className="flex items-center gap-2 text-[11.5px] text-aro-ink-2">
                    <Toggle checked={tog2} onChange={setTog2} size="sm" /> network
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Segmented
                    value={seg}
                    onChange={setSeg}
                    items={[
                      { value: 'plan', label: 'Plan', icon: IconBrain },
                      { value: 'agent', label: 'Agent', icon: IconBolt },
                    ]}
                  />
                  <Tabs
                    value={demoTab}
                    onChange={setDemoTab}
                    items={[
                      { value: 'a', label: 'Plan' },
                      { value: 'b', label: 'Context', count: 4 },
                      { value: 'c', label: 'Bridge' },
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
                <div className="flex items-center gap-4 rounded-[9px] border border-aro-line-soft bg-aro-sunken p-3">
                  <Ring value={62} size={54} stroke={5}>
                    <span className="font-aro-mono text-[11px] font-semibold text-aro-ink">62%</span>
                  </Ring>
                  <Ring value={91} size={54} stroke={5} tone="var(--aro-mint)">
                    <span className="font-aro-mono text-[11px] font-semibold text-aro-mint">91%</span>
                  </Ring>
                  <Ring value={38} size={54} stroke={5} tone="var(--aro-amber)">
                    <span className="font-aro-mono text-[11px] font-semibold text-aro-amber">38%</span>
                  </Ring>
                  <div className="ml-auto space-y-2">
                    <Bar value={62} tone="iris" />
                    <Bar value={91} tone="mint" />
                    <Bar value={38} tone="amber" />
                    <Bar indeterminate tone="cyan" />
                  </div>
                </div>
                <div className="flex items-center justify-between rounded-[9px] border border-aro-line-soft bg-aro-sunken px-3 py-2.5">
                  <span className="font-aro-mono text-[10px] text-aro-ink-4">sparkline · 12 buckets</span>
                  <div className="flex items-end gap-3">
                    <Sparkline data={[4, 8, 6, 12, 9, 14, 11, 18]} color="var(--aro-mint)" w={70} h={22} />
                    <Sparkline data={[18, 14, 16, 9, 11, 6, 8, 4]} color="var(--aro-rose)" w={70} h={22} />
                  </div>
                </div>
              </div>
            </Spec>

            <Spec label="feedback">
              <div className="space-y-2.5">
                <div className="flex items-center gap-2 rounded-[8px] border border-aro-amber/25 bg-aro-amber-tint/60 px-2.5 py-2 text-[11.5px] text-aro-amber">
                  <IconWarning size={12} /> Approval required — <span className="font-aro-mono">gh pr create</span>
                </div>
                <div className="flex items-center gap-2 rounded-[8px] border border-aro-rose/25 bg-aro-rose-tint/60 px-2.5 py-2 text-[11.5px] text-aro-rose">
                  <IconWarning size={12} /> Spend cap reached — run halted at 44%
                </div>
                <div className="flex items-center gap-2 rounded-[8px] border border-aro-mint/25 bg-aro-mint-tint/60 px-2.5 py-2 text-[11.5px] text-aro-mint">
                  <IconCheck size={12} /> 14 tests passed in 284ms
                </div>
                <div className="flex items-center gap-2 rounded-[8px] border border-aro-emerald/25 bg-aro-emerald-tint px-2.5 py-2 text-[11.5px] text-aro-emerald">
                  <AgentMark glyph="A" from="#10b981" to="#0d9488" size={14} /> aro · brand accent — live and healthy
                </div>
              </div>
            </Spec>

            <Spec label="overlay, empty & toasts">
              <div className="space-y-2.5">
                <div className="flex flex-wrap gap-2">
                  <Button variant="secondary" icon={IconBolt} onClick={() => setModal(true)}>
                    Open modal
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={() => {
                      toast('Patch applied · 3 files', 'mint')
                      setTimeout(() => toast('Approval required', 'amber'), 700)
                    }}
                  >
                    Fire toasts
                  </Button>
                </div>
                <div className="rounded-[10px] border border-dashed border-aro-line">
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
                    <Button variant="ghost" size="xs">
                      Hover me
                    </Button>
                  </Tip>
                  <Tip label="⌘B">
                    <Button variant="ghost" size="xs">
                      With shortcut
                    </Button>
                  </Tip>
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
          body="The namespaced contract this port ships. Utilities are generated by Tailwind from the @theme bridge in src/aro/tokens.css; the raw values resolve per theme at the element that carries data-theme."
        >
          <div className="overflow-hidden rounded-[11px] border border-aro-line-soft bg-aro-code">
            <div className="flex items-center gap-1.5 border-b border-aro-line-soft px-3 py-2">
              {['#ff5f57', '#febc2e', '#28c840'].map((c) => (
                <i key={c} className="size-[7px] rounded-full" style={{ background: c }} />
              ))}
              <span className="ml-1.5 font-aro-mono text-[10px] text-aro-ink-4">src/aro/tokens.css</span>
              <span className="ml-auto font-aro-mono text-[9.5px] text-aro-ink-4">namespaced</span>
            </div>
            <pre className="aro-scroll-thin overflow-x-auto px-4 py-3 font-aro-mono text-[11px] leading-[1.8] text-aro-ink-3">
              {`@theme inline {
  /* surfaces — utilities: bg-aro-void … bg-aro-track */
  --color-aro-void: var(--aro-void);
  --color-aro-bg:   var(--aro-bg);
  --color-aro-raise: var(--aro-raise);
  --color-aro-line: var(--aro-line);

  /* ink */
  --color-aro-ink: var(--aro-ink);
  --color-aro-ink-2: var(--aro-ink-2);

  /* signal */
  --color-aro-iris: var(--aro-iris);
  --color-aro-cyan: var(--aro-cyan);
  --color-aro-mint: var(--aro-mint);
  --color-aro-amber: var(--aro-amber);
  --color-aro-rose: var(--aro-rose);
  --color-aro-sky:  var(--aro-sky);

  /* brand accent */
  --color-aro-emerald: #10b981;
  --color-aro-emerald-soft: #6ee7b7;

  /* radius + elevation + motion */
  --radius-aro-md: 7px;  --radius-aro-lg: 10px;
  --shadow-aro-e1: 0 1px 2px var(--aro-shadow-a);
  --animate-aro-rise: aro-rise .32s cubic-bezier(.22,1,.36,1) both;
}

/* five themes, gated — inert until an element opts in */
html[data-theme='obsidian'], [data-theme='obsidian'] { … }
html[data-theme='daylight'], [data-theme='daylight'] { … }
html[data-theme='nord'],      [data-theme='nord']      { … }
html[data-theme='ember'],     [data-theme='ember']     { … }
html[data-theme='paper'],     [data-theme='paper']     { … }`}
            </pre>
          </div>
          <Divider className="my-6" />
          <div className="flex flex-wrap items-center gap-3">
            <Badge tone="neutral" mono>
              a11y: full keyboard map
            </Badge>
            <Badge tone="neutral" mono>
              additive: zero existing files restyled
            </Badge>
            <Badge tone="neutral" mono>
              reduced-motion: host blanket rule applies
            </Badge>
            <Badge tone="neutral" mono>
              min target 22×22px
            </Badge>
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
          <p className="text-[12.5px] leading-[1.65] text-aro-ink-2">
            Modals are used for decisions that change what the agent is allowed to do — never for information the transcript already shows. This
            one renders inside the themed wrapper, so it keeps the active Aro theme while the rest of the window keeps the host theme.
          </p>
          <div className="rounded-[9px] border border-aro-line-soft bg-aro-sunken p-3">
            <SectionLabel className="px-0">example content</SectionLabel>
            <div className="space-y-2">
              <Input placeholder="Session name" />
              <Select
                value="opus-4.6"
                onChange={() => {}}
                options={[
                  { value: 'opus-4.6', label: 'Opus 4.6', hint: '$25 / 1M' },
                  { value: 'gpt-5.4', label: 'GPT-5.4-Codex', hint: '$14 / 1M' },
                ]}
              />
            </div>
          </div>
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setModal(false)}>
              Cancel
            </Button>
            <Button
              variant="primary"
              icon={IconCheck}
              onClick={() => {
                setModal(false)
                toast('Confirmed', 'mint')
              }}
            >
              Confirm
            </Button>
          </div>
        </div>
      </Modal>

      <Toasts items={toasts} />
    </div>
  )
}

export default AroDesignSystemPage
