import { selectableClass } from '@/components/onboarding-chat/chip'
import { Tip } from '@/components/ui/tooltip'
import { IS_MAC } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'

// Which live-catalog slugs the first-run picker shows, and in what order. The
// catalog is the source of truth for WHAT can be connected; this list picks the
// few everyday apps out of it (D89). A slug the catalog no longer carries is
// simply not shown, and a slug the catalog gains is not shown until it is
// added here.
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

// Connectors are the apps Hermes reads and acts on FOR the user. Chat channels
// (Discord, Telegram, WhatsApp) are how a user talks TO Hermes — those live on
// the Messaging page, and offering them here as if they were data sources
// taught users the wrong thing about what "connect" does. The catalog
// carries them for the agent's sake; the first-run picker leaves them out.
export const CONNECTOR_PICKER_HIDDEN = new Set(['discord', 'discordbot', 'microsoft_teams'])

export function orderConnectorPicks<T extends { connector: string }>(rows: T[]): T[] {
  const rank = new Map(CONNECTOR_LEAD_ORDER.map((slug, index) => [slug, index]))

  return rows
    .filter(row => rank.has(row.connector) && !CONNECTOR_PICKER_HIDDEN.has(row.connector))
    .sort((a, b) => {
      const ra = rank.get(a.connector) ?? Number.POSITIVE_INFINITY
      const rb = rank.get(b.connector) ?? Number.POSITIVE_INFINITY

      return ra - rb || a.connector.localeCompare(b.connector)
    })
}

// Big accent swatches, Dia-style. Each seeds `retintTheme` through the accent
// override, so a click repaints the surface live. Nous blue is the default =
// no override. Mono seeds the current mode's pole — black in light, white in
// dark — for a full monochrome look.
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
  onPick
}: {
  active: boolean
  hex: string
  name: string
  onPick: () => void
}) {
  return (
    <Tip label={name}>
      <button
        aria-label={name}
        aria-pressed={active}
        className={cn(
          // The hairline keeps the mono swatch visible on its own pole.
          'size-9 rounded-full border border-foreground/15 transition-transform duration-150',
          !active && 'hover:scale-105'
        )}
        onClick={onPick}
        style={{
          background: hex,
          boxShadow: active ? `0 0 0 2px var(--dt-background), 0 0 0 4px ${hex}` : undefined
        }}
        type="button"
      />
    </Tip>
  )
}

// Mini layout trees mirror the basic (BASIC_TREE) and terminal-deck
// (TERMINAL_TREE) presets registered in app/contrib/controller.tsx, drawn in
// the layout editor's thumbnail language, upscaled.
export type MiniNode = 1 | { dir: 'column' | 'row'; children: MiniNode[]; weights: number[] }

/** The power-user layout. Picking it is the most explicit thing a user does
 *  in the whole first run to say how they work. */
export const ELITE_LAYOUT_ID = 'terminal-deck'

export const LAYOUTS: Array<{ id: string; name: string; tree: MiniNode }> = [
  { id: 'basic', name: 'Basic', tree: { children: [1, 1], dir: 'row', weights: [1, 4.6] } },
  {
    id: ELITE_LAYOUT_ID,
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
 * The window buttons on the preview, drawn the way this machine draws them.
 *
 * The card is a picture of the user's own window, so it follows the split
 * `main.ts` already makes when it builds one: macOS gets the traffic lights on
 * the left (`trafficLightPosition`), everywhere else the native controls ride
 * on the right as monochrome glyphs (`titleBarOverlay`). Three coloured dots on
 * a Windows machine is a picture of somebody else's computer — a small tell, in
 * the one moment the app is claiming to show you yours.
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

  // Minimize, maximize, close — at 6px the glyphs themselves are mush, so each
  // is the shape it would be: a bar, a box, and a cross that reads as one.
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
  name,
  onSelect,
  tree
}: {
  active: boolean
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
      <span className={cn('text-xs', active ? 'text-foreground' : 'text-muted-foreground')}>{name}</span>
    </button>
  )
}
