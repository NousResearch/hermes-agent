import { useStore } from '@nanostores/react'
import { type CSSProperties, type ReactElement, useMemo } from 'react'

import { Tip } from '@/components/ui/tooltip'
import { PROFILE_SWATCHES } from '@/lib/profile-color'
import { cn } from '@/lib/utils'

import { $pluginRecords } from './plugins-store'
import { useContributions } from './react/use-contributions'
import type { Contribution } from './types'

/**
 * Kanban card badges — small text labels a plugin puts on Kanban card tiles
 * (a tenant, a CI state, an estimate) without touching the board's markup.
 *
 *   data kind (`data`):  kanban.card.badges  (KanbanCardBadgeContribution)
 *
 * The host owns everything visible: the card, the badge row's place in it,
 * the chip's look and its tooltip. A plugin only answers `decorate(card)` with
 * text and a palette tone; the host validates, orders and caps the answers
 * and paints them. Plugins cannot reorder, hide or restyle each other.
 */
export const KANBAN_CARD_AREAS = { badges: 'kanban.card.badges' } as const

/** The badge palette: the app's twelve identity hues (`PROFILE_SWATCHES`), in
 *  hue order, so `KANBAN_CARD_BADGE_HUES[hash % 12]` spreads values evenly. */
export const KANBAN_CARD_BADGE_HUES = [
  'red',
  'orange',
  'yellow',
  'lime',
  'green',
  'mint',
  'cyan',
  'sky',
  'blue',
  'violet',
  'magenta',
  'pink'
] as const

export type KanbanCardBadgeHue = (typeof KANBAN_CARD_BADGE_HUES)[number]
export type KanbanCardBadgeTone = 'neutral' | KanbanCardBadgeHue

/** What the host accepts per card. Over-long text is shortened, never refused;
 *  badges past a plugin's cap are dropped. */
export const KANBAN_CARD_BADGE_LIMITS = { labelChars: 24, perPlugin: 2, tooltipChars: 160 } as const

/** What a decorator gets to branch on: the card's identity and routing fields,
 *  never its prose (title, body, summaries, comments, run output). Frozen.
 *  Every field is a standing compatibility promise. */
export interface KanbanCard {
  readonly assignee: null | string
  /** The task id (`t_…`): the key to look up your own per-card data by. */
  readonly id: string
  /** The column the card is in (`triage`, `ready`, `running`, …). */
  readonly status: string
  readonly tenant: null | string
}

/** Board-level facts every card on the board shares. Frozen. */
export interface KanbanCardBoard {
  /** Every tenant on the board (the list its filter menu offers), so a plugin
   *  can give each one a distinct color instead of hashing blind. */
  readonly tenants: readonly string[]
}

export interface KanbanCardBadge {
  /** Plain text, shown as-is (no markup). Trimmed; control and bidi
   *  characters removed; shortened past `labelChars`. */
  label: string
  /** A palette hue, or `'neutral'` (the default; also what anything else
   *  becomes). */
  tone?: KanbanCardBadgeTone
  /** Plain-text hover line. The host always adds the plugin's name under it. */
  tooltip?: string
}

/** Payload of a `kanban.card.badges` data contribution. Return a badge, a few
 *  badges, or `null` for cards you have nothing to say about. Keep it a pure
 *  lookup: it runs while the board renders. */
export interface KanbanCardBadgeContribution {
  decorate: (
    card: KanbanCard,
    board: KanbanCardBoard
  ) => KanbanCardBadge | null | readonly KanbanCardBadge[] | undefined
}

export interface ResolvedKanbanCardBadge {
  key: string
  label: string
  /** Provenance of the contribution (`plugin:<id>`), stamped by the host. */
  source: string
  tone: KanbanCardBadgeTone
  tooltip: null | string
}

const TONES = new Set<string>(['neutral', ...KANBAN_CARD_BADGE_HUES])

// Controls, format characters (bidi overrides, zero-width joiners) and line
// separators: invisible text a label could use to spoof or reflow the card.
const UNSAFE_CHARS = /[\p{Cc}\p{Cf}\u2028\u2029]/gu

function plainText(value: unknown, maxChars: number): null | string {
  if (typeof value !== 'string') {
    return null
  }

  const text = value.replace(UNSAFE_CHARS, '').replace(/\s+/g, ' ').trim()
  const chars = [...text]

  if (chars.length === 0) {
    return null
  }

  return chars.length > maxChars
    ? `${chars
        .slice(0, maxChars - 1)
        .join('')
        .trimEnd()}…`
    : text
}

function toBadge(raw: unknown): null | Omit<ResolvedKanbanCardBadge, 'key' | 'source'> {
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const { label, tone, tooltip } = raw as Record<string, unknown>
  const text = plainText(label, KANBAN_CARD_BADGE_LIMITS.labelChars)

  if (!text) {
    return null
  }

  return {
    label: text,
    tone: typeof tone === 'string' && TONES.has(tone) ? (tone as KanbanCardBadgeTone) : 'neutral',
    tooltip: plainText(tooltip, KANBAN_CARD_BADGE_LIMITS.tooltipChars)
  }
}

// Code-unit order, not localeCompare: the row must read the same in every locale.
const compare = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0)

// Plugin id, then contribution id. A contribution's own `order` is ignored
// here on purpose: letting plugins bid for the front of the row is exactly
// the contention this area exists to remove.
const byProvenance = (a: Contribution, b: Contribution) =>
  compare(a.source ?? 'core', b.source ?? 'core') || compare(a.id, b.id)

const warned = new Set<string>()

/** The badges a card shows, in row order. Pure over the area's contributions:
 *  each plugin gets at most `perPlugin` badges across all its contributions,
 *  and a decorator that throws or answers garbage simply contributes nothing. */
export function resolveKanbanCardBadges(
  contributions: readonly Contribution[],
  card: KanbanCard,
  board: KanbanCardBoard
): ResolvedKanbanCardBadge[] {
  const view: KanbanCard = Object.freeze({
    assignee: card.assignee ?? null,
    id: card.id,
    status: card.status,
    tenant: card.tenant ?? null
  })

  const boardView: KanbanCardBoard = Object.freeze({ tenants: Object.freeze([...board.tenants]) })

  const used = new Map<string, number>()
  const badges: ResolvedKanbanCardBadge[] = []

  for (const contribution of [...contributions].sort(byProvenance)) {
    const source = contribution.source ?? 'core'
    let count = used.get(source) ?? 0

    if (count >= KANBAN_CARD_BADGE_LIMITS.perPlugin) {
      continue
    }

    try {
      const answer = (contribution.data as Partial<KanbanCardBadgeContribution> | undefined)?.decorate?.(
        view,
        boardView
      )

      const list: readonly unknown[] = Array.isArray(answer) ? answer : answer ? [answer] : []

      for (let index = 0; index < list.length && count < KANBAN_CARD_BADGE_LIMITS.perPlugin; index += 1) {
        const badge = toBadge(list[index])

        if (badge) {
          badges.push({ key: `${contribution.id}:${index}`, source, ...badge })
          count += 1
        }
      }
    } catch (error) {
      if (!warned.has(contribution.id)) {
        warned.add(contribution.id)
        console.warn(`[${KANBAN_CARD_AREAS.badges}] ${contribution.id} decorate() threw; its badges are skipped`, error)
      }
    }

    used.set(source, count)
  }

  return badges
}

/** The fields of a board task the badge row reads, and the board's tenants. */
export interface KanbanCardBadgesProps {
  card: { assignee?: null | string; id: string; status: string; tenant?: null | string }
  tenants: readonly string[]
}

const HUE_STYLE =
  'bg-[color-mix(in_srgb,var(--kanban-badge-hue)_16%,transparent)] text-[color-mix(in_oklab,var(--kanban-badge-hue)_40%,var(--ui-text-primary,currentColor))] shadow-[inset_0_0_0_1px_color-mix(in_srgb,var(--kanban-badge-hue)_45%,transparent)]'

// Only the host's own palette reaches `style`; no plugin string ever does.
const hueVar = (tone: KanbanCardBadgeTone): CSSProperties | undefined => {
  const index = KANBAN_CARD_BADGE_HUES.indexOf(tone as KanbanCardBadgeHue)

  return index < 0 ? undefined : ({ '--kanban-badge-hue': PROFILE_SWATCHES[index] } as CSSProperties)
}

const pluginName = (source: string, records: ReturnType<typeof $pluginRecords.get>) => {
  const id = source.startsWith('plugin:') ? source.slice('plugin:'.length) : source

  return records[id]?.name || id
}

/** The badge row a Kanban card renders above its title. Renders nothing when
 *  no plugin has a badge for this card, so an unused area costs no space. */
export function KanbanCardBadges({ card, tenants }: KanbanCardBadgesProps): null | ReactElement {
  const contributions = useContributions(KANBAN_CARD_AREAS.badges)
  const records = useStore($pluginRecords)
  const { assignee = null, id, status, tenant = null } = card

  const badges = useMemo(
    () => resolveKanbanCardBadges(contributions, { assignee, id, status, tenant }, { tenants }),
    [assignee, contributions, id, status, tenant, tenants]
  )

  if (badges.length === 0) {
    return null
  }

  return (
    <div className="flex min-w-0 flex-wrap items-center gap-1" data-slot="kanban-card-badges">
      {badges.map(badge => (
        <Tip
          key={badge.key}
          label={
            <span className="flex flex-col gap-0.5">
              {badge.tooltip && <span>{badge.tooltip}</span>}
              <span className="text-[0.625rem] opacity-70">{pluginName(badge.source, records)}</span>
            </span>
          }
        >
          <span
            className={cn(
              'inline-flex max-w-full items-center truncate rounded-[3px] px-1.5 py-0.5 text-[0.625rem] font-medium leading-none',
              badge.tone === 'neutral' ? 'bg-muted text-muted-foreground' : HUE_STYLE
            )}
            data-slot="kanban-card-badge"
            data-tone={badge.tone}
            style={hueVar(badge.tone)}
          >
            {badge.label}
          </span>
        </Tip>
      ))}
    </div>
  )
}
