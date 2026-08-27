/**
 * How the roster groups and labels what it lists: the gateway filter, the
 * per-connection bucketing, and the foldable headings those buckets render.
 *
 * Presentation below the pane. The headings take everything they draw as
 * props and the bucketing is pure, so the roster pane can compose them
 * without either half knowing about a bot row.
 */

import * as sdk from '@hermes/plugin-sdk'
import { cn, Codicon, DisclosureCaret, RowButton, Tip } from '@hermes/plugin-sdk'

import { botHandle, botRosterKey, botSourceStatus, filterBots } from './data'
import { displayName } from './labels'
import { botRosterMeta } from './routing'
import type { BotMeta, GatewaySource, RosterRow } from './types'

export function filterBotsByGateway(roster: RosterRow[], connectionId: string) {
  if (!connectionId || connectionId === 'all') {
    return roster
  }
  return (roster || []).filter(bot => String(bot?.connectionId || '') === connectionId)
}
export function botNeedsHandleLabel(bot: RosterRow, roster: RosterRow[], metaByName: Record<string, BotMeta>) {
  const identity = displayName(bot, botRosterMeta(bot, metaByName)).trim().toLowerCase()
  const connectionId = String(bot?.connectionId || '')
  return (roster || []).some(
    candidate =>
      botRosterKey(candidate) !== botRosterKey(bot) &&
      String(candidate?.connectionId || '') === connectionId &&
      displayName(candidate, botRosterMeta(candidate, metaByName)).trim().toLowerCase() === identity &&
      botHandle(candidate.name, candidate) !== botHandle(bot.name, bot)
  )
}
export function groupMatchesRosterFilters(
  name: string,
  members: RosterRow[],
  metaByName: Record<string, BotMeta>,
  query: string,
  connectionId: string
) {
  const inGateway = filterBotsByGateway(members, connectionId)
  if (connectionId && connectionId !== 'all' && inGateway.length === 0) {
    return false
  }
  const needle = String(query || '')
    .trim()
    .toLowerCase()
    .replace(/^@/, '')
  return (
    !needle ||
    String(name || '')
      .toLowerCase()
      .includes(needle) ||
    filterBots(inGateway, metaByName, needle).length > 0
  )
}

/** A gateway source as the roster's picker carries it: a running row count,
 *  and `reachable` widened to the roster row's null ("status unknown"), which
 *  GatewaySource does not model. */
interface RosterGatewayOption extends Omit<GatewaySource, 'reachable'> {
  count?: number
  reachable?: boolean | null
}

/** A sidebar roster row (`{ kind, bot, … }`), or a bare roster row for
 *  callers that hand the bots over directly. */
interface RosterGatewayRow extends Partial<RosterRow> {
  bot?: RosterRow
}

interface RosterGatewaySection<TRow extends RosterGatewayRow = RosterGatewayRow> {
  id: string
  option: null | RosterGatewayOption
  rows: TRow[]
}

/** Sections built from rows that always carry their resolved roster row. */
export type ResolvedRosterGatewaySection = RosterGatewaySection<RosterGatewayRow & { bot: RosterRow }>
export function rosterGatewayOptions(sources: GatewaySource[], roster: RosterRow[]) {
  const byId = new Map<string, RosterGatewayOption & { count: number }>()
  for (const source of Array.isArray(sources) ? sources : []) {
    const id = String(source?.connectionId || '').trim()
    if (id) {
      byId.set(id, {
        ...source,
        connectionId: id,
        count: 0
      })
    }
  }
  for (const bot of roster || []) {
    const id = String(bot?.connectionId || '').trim()
    if (!id) {
      continue
    }
    const source = byId.get(id) || {
      connectionId: id,
      kind: bot.connectionKind,
      label: bot.connectionLabel || id,
      reachable: bot.sourceReachable,
      error: bot.sourceError,
      count: 0
    }
    source.count += 1
    byId.set(id, source)
  }
  return [...byId.values()].sort((a, b) =>
    String(a.label || a.connectionId).localeCompare(String(b.label || b.connectionId), undefined, {
      sensitivity: 'base'
    })
  )
}
export function rosterGatewaySections<TRow extends RosterGatewayRow>(
  botRows: TRow[],
  gatewayOptions: RosterGatewayOption[],
  gatewayFilter = 'all'
): { sectioned: boolean; sections: RosterGatewaySection<TRow>[] } {
  const rows = Array.isArray(botRows) ? botRows : []
  const options = Array.isArray(gatewayOptions) ? gatewayOptions : []
  if (gatewayFilter !== 'all' || options.length <= 1) {
    return {
      sectioned: false,
      sections: [
        {
          id: 'all',
          option: null,
          rows
        }
      ]
    }
  }
  const byId = new Map<string, TRow[]>()
  for (const row of rows) {
    const bot = row?.bot || row
    const id = String(bot?.connectionId || 'legacy').trim() || 'legacy'
    const bucket = byId.get(id) || []
    bucket.push(row)
    byId.set(id, bucket)
  }
  const known = new Set<string>()
  const sections: RosterGatewaySection<TRow>[] = []
  for (const option of options) {
    const id = String(option?.connectionId || '').trim()
    const sectionRows = byId.get(id)
    if (!id || !sectionRows?.length) {
      continue
    }
    known.add(id)
    sections.push({
      id,
      option,
      rows: sectionRows
    })
  }
  for (const [id, sectionRows] of byId) {
    if (known.has(id)) {
      continue
    }
    const bot = sectionRows[0]?.bot || sectionRows[0]
    sections.push({
      id,
      option: {
        connectionId: id,
        kind: bot?.connectionKind || 'remote',
        label: bot?.connectionLabel || (id === 'legacy' ? 'Current gateway' : id),
        reachable: bot?.sourceReachable,
        error: bot?.sourceError
      },
      rows: sectionRows
    })
  }
  return {
    sectioned: true,
    sections
  }
}
function gatewayKindIcon(kind?: string) {
  const icons: Partial<typeof sdk.icons> = (typeof sdk === 'undefined' ? null : sdk.icons) || {}
  if (kind === 'local') return icons.Monitor
  if (kind === 'cloud') return icons.Cloud
  if (kind === 'ssh') return icons.Terminal
  return icons.Network
}
function gatewayKindCodicon(kind?: string) {
  if (kind === 'local') return 'device-desktop'
  if (kind === 'cloud') return 'cloud'
  if (kind === 'ssh') return 'terminal'
  return 'remote-explorer'
}

interface GatewayKindGlyphProps {
  className?: string
  kind?: string
}

/** Match the gateway switcher's Tabler glyphs while keeping older SDK shells
 * usable until they expose the shared icon namespace. */
export function GatewayKindGlyph({ className, kind }: GatewayKindGlyphProps) {
  const Icon = gatewayKindIcon(kind)
  return (
    <span
      aria-hidden
      className={cn('grid size-3.5 shrink-0 place-items-center', className)}
      data-connection-kind={kind || 'remote'}
      data-slot="connection-glyph"
    >
      {Icon ? <Icon className="size-3" /> : <Codicon name={gatewayKindCodicon(kind)} className="text-[0.75rem]" />}
    </span>
  )
}

/** Foldable roster heading. It organizes rows visually but never supplies or
 * reconstructs ownership; every action still receives the full bot row. */
interface RosterSectionHeaderProps {
  collapsed: boolean
  count: number
  gatewayKind?: string
  icon?: string
  label: string
  onToggle: () => void
  status?: { available: boolean; label: string }
  tip?: string
}
export function RosterSectionHeader({
  collapsed,
  count,
  gatewayKind,
  icon,
  label,
  onToggle,
  status,
  tip
}: RosterSectionHeaderProps) {
  const button = (
    <RowButton
      aria-expanded={!collapsed}
      className="mt-1 flex w-full min-w-0 items-center gap-1.5 rounded-md px-2 py-1.5 text-left text-[0.6875rem] font-semibold uppercase tracking-wider text-(--ui-text-quaternary) transition-colors hover:bg-(--chrome-action-hover) hover:text-(--ui-text-secondary)"
      onClick={onToggle}
    >
      <DisclosureCaret open={!collapsed} />
      {gatewayKind ? (
        <GatewayKindGlyph kind={gatewayKind} />
      ) : (
        // TODO(bot-mode-types): neither `gatewayKind` nor `icon` is required, so a header
        // given neither renders `codicon-undefined`. Both current callers pass exactly one.
        // @ts-expect-error `icon` is optional here; Codicon's `name` is required.
        <Codicon name={icon} className="shrink-0" />
      )}
      <span className="flex min-w-0 items-center gap-1">
        <span className="min-w-0 truncate">{label}</span>
        {status && !status.available ? <span className="sr-only">{status.label}</span> : null}
      </span>
      <span className="min-w-0 flex-1" aria-hidden />
      <span className="shrink-0 font-normal tabular-nums text-(--ui-text-quaternary)">{count}</span>
      {status && !status.available ? (
        <Codicon name="debug-disconnect" className="shrink-0 text-amber-600 dark:text-amber-300" aria-hidden />
      ) : null}
    </RowButton>
  )
  return tip ? <Tip label={tip}>{button}</Tip> : button
}
interface GatewaySectionHeadingProps {
  collapsed: boolean
  count: number
  onToggle: () => void
  option?: RosterGatewayOption | null
}
export function GatewaySectionHeading({ collapsed, count, onToggle, option }: GatewaySectionHeadingProps) {
  const status = botSourceStatus({
    sourceError: option?.error,
    sourceReachable: option?.reachable
  })
  const label = option?.label || option?.connectionId || 'Current gateway'
  const kind = option?.kind || 'remote'
  return (
    <RosterSectionHeader
      collapsed={collapsed}
      count={count}
      gatewayKind={kind}
      label={label}
      onToggle={onToggle}
      status={status}
      tip={`${label} · ${kind} · ${status.label}`}
    />
  )
}
