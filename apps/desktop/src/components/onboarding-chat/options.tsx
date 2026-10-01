import { selectableClass } from '@/components/onboarding-chat/chip'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { IS_MAC } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'
import type { InterfaceMode } from '@/store/interface-mode'
import { readableInk } from '@/themes/color'

// Curated leaders for the first-run picker. Other enabled catalog entries
// remain searchable, so newly deployed connectors need no client list update.
export const CONNECTOR_LEAD_ORDER = [
  'gmail',
  'googlecalendar',
  'googledrive',
  'googledocs',
  'googlesheets',
  'outlook',
  'slack',
  'notion',
  'linear',
  'jira',
  'figma',
  'todoist'
]

// Connectors are the apps Hermes reads and acts on for the user. Chat channels
// (Discord, Telegram, WhatsApp) are how a user talks to Hermes; those live on
// the Messaging page, and offering them here as if they were data sources
// taught users the wrong thing about what "connect" does. The catalog
// carries them for the agent's sake; the first-run picker leaves them out.
export const CONNECTOR_PICKER_HIDDEN = new Set(['discord', 'discordbot', 'microsoft_teams'])

// A row the gateway marks `enabled: false` is a toolkit the deployment has
// turned off; the agent cannot connect it, so the picker does not offer it.
export function orderConnectorPicks<T extends { connector: string; enabled?: boolean }>(rows: T[]): T[] {
  const rank = new Map(CONNECTOR_LEAD_ORDER.map((slug, index) => [slug, index]))

  return rows
    .filter(row => row.enabled !== false && !CONNECTOR_PICKER_HIDDEN.has(row.connector))
    .sort((a, b) => {
      const ra = rank.get(a.connector) ?? Number.POSITIVE_INFINITY
      const rb = rank.get(b.connector) ?? Number.POSITIVE_INFINITY

      return ra - rb || a.connector.localeCompare(b.connector)
    })
}

// Each swatch sets the accent override, which `retintTheme` uses to repaint
// the active skin as soon as the swatch is clicked. Nous blue is the default
// and sets no override. Mono is black in light mode and white in dark mode.
export const NOUS_ACCENT = '#0053fd'

export const accentsFor = (dark: boolean): Array<{ hex: string; name: string }> => [
  { hex: dark ? '#ffffff' : '#000000', name: 'Mono' },
  { hex: '#2ea043', name: 'GitHub green' },
  { hex: '#00d5ff', name: 'Cyber cyan' },
  { hex: NOUS_ACCENT, name: 'Nous blue' },
  { hex: '#8a2be2', name: 'Ultraviolet' },
  { hex: '#e0218a', name: 'Barbie pink' },
  { hex: '#ff073a', name: 'Electric red' },
  { hex: '#ff6a00', name: 'Safety orange' }
]

export function AccentSwatch({
  active,
  hex,
  name,
  onColorChange,
  onPick
}: {
  active: boolean
  hex: string
  name: string
  onColorChange?: (hex: string) => void
  onPick?: () => void
}) {
  const className = cn(
    // The border keeps the mono swatch visible when its colour matches the background.
    'relative inline-flex size-9 items-center justify-center rounded-full border border-foreground/15 transition-transform duration-150',
    !active && 'hover:scale-105'
  )

  const style = {
    background: hex,
    boxShadow: active ? `0 0 0 2px var(--dt-background), 0 0 0 4px ${hex}` : undefined
  }

  return (
    <Tip label={name}>
      {onColorChange ? (
        <label className={cn(className, 'focus-within:outline-2 focus-within:outline-ring')} style={style}>
          <span className="flex" style={{ color: readableInk(hex) }}>
            <Codicon name="add" size="1rem" />
          </span>
          <input
            aria-label={name}
            className="absolute inset-0 size-full cursor-pointer opacity-0"
            onChange={event => onColorChange(event.target.value)}
            type="color"
            value={hex}
          />
        </label>
      ) : (
        <button
          aria-label={name}
          aria-pressed={active}
          className={className}
          onClick={onPick}
          style={style}
          type="button"
        />
      )}
    </Tip>
  )
}

// These mini trees copy the basic (BASIC_TREE) and terminal-deck
// (TERMINAL_TREE) presets in app/contrib/layout-presets.ts, drawn like the
// layout editor's thumbnails at a larger size.
export type MiniNode = 1 | { dir: 'column' | 'row'; children: MiniNode[]; weights: number[] }

export const ELITE_LAYOUT_ID = 'terminal-deck'

// Each pick is an arrangement AND an interface mode. First launch is the one
// place a single question can answer both: someone here to talk to Hermes
// should not have to find Simple mode afterwards, and a developer who asked
// for the terminal deck wants the tooling on. Basic applies Simple's own
// preset so its shelf shows the pick as active.
export const LAYOUTS: Array<{ description: string; id: string; mode: InterfaceMode; name: string; tree: MiniNode }> = [
  {
    description: 'For talking to Hermes.',
    id: 'sidebar-left',
    mode: 'simple',
    name: 'Basic',
    tree: { children: [1, 1], dir: 'row', weights: [1, 4.6] }
  },
  {
    description: 'For developers: terminal, files, diffs.',
    id: ELITE_LAYOUT_ID,
    mode: 'advanced',
    name: 'Elite',
    tree: {
      children: [{ children: [1, 1, 1], dir: 'row', weights: [1, 3.2, 1.2] }, 1],
      dir: 'column',
      weights: [3, 1]
    }
  }
]

export function MiniTree({ node }: { node: MiniNode }) {
  if (node === 1) {
    return <div className="min-h-0 min-w-0 flex-1 rounded-[3px] bg-foreground/15" />
  }

  return (
    <div className={cn('flex min-h-0 min-w-0 flex-1 gap-1', node.dir === 'row' ? 'flex-row' : 'flex-col')}>
      {node.children.map((child, i) => (
        <div className="flex min-h-0 min-w-0" key={i} style={{ flex: `${node.weights[i]} ${node.weights[i]} 0px` }}>
          <MiniTree node={child} />
        </div>
      ))}
    </div>
  )
}

/**
 * The window buttons on the preview, drawn the way this machine draws them, so the card matches the user's own window.
 * `main.ts` makes the same split: macOS puts the traffic lights on the left (`trafficLightPosition`), every other
 * platform puts monochrome native controls on the right (`titleBarOverlay`).
 */
function MiniWindowButtons() {
  if (IS_MAC) {
    return (
      <span aria-hidden className="flex gap-1">
        <span className="size-1.5 rounded-full bg-[#ff5f57]" />
        <span className="size-1.5 rounded-full bg-[#febc2e]" />
        <span className="size-1.5 rounded-full bg-[#28c840]" />
      </span>
    )
  }

  // Minimize, maximize, close. At 6 px the real glyphs are illegible, so each
  // one is a plain shape: a bar, a box, and a cross.
  return (
    <span aria-hidden className="flex items-center justify-end gap-1.5 text-foreground/40">
      <span className="h-px w-1.5 bg-current" />
      <span className="size-1.5 border border-current" />
      <span className="relative size-1.5">
        <span className="absolute top-1/2 left-0 h-px w-full rotate-45 bg-current" />
        <span className="absolute top-1/2 left-0 h-px w-full -rotate-45 bg-current" />
      </span>
    </span>
  )
}

export function LayoutPreviewCard({
  active,
  description,
  name,
  onSelect,
  tree
}: {
  active: boolean
  description?: string
  name: string
  onSelect: () => void
  tree: MiniNode
}) {
  return (
    <button aria-pressed={active} className="group flex flex-col items-center gap-2" onClick={onSelect} type="button">
      <span className={cn('flex aspect-[10/7] w-full flex-col gap-1.5 rounded-[8px] p-2', selectableClass(active))}>
        <MiniWindowButtons />
        <span className="flex min-h-0 flex-1">
          <MiniTree node={tree} />
        </span>
      </span>
      <span className="flex flex-col items-center gap-0.5">
        <span className={cn('text-xs', active ? 'text-foreground' : 'text-muted-foreground')}>{name}</span>
        {description && <span className="text-[0.68rem] text-muted-foreground/70">{description}</span>}
      </span>
    </button>
  )
}
