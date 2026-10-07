import { connectorTitle } from '@/lib/connector-tools'

import type {
  BundledEntryInput,
  ConnectorCardModel,
  ConnectorFact,
  ConnectorState,
  ConnectorStateWord,
  ConnectorWayLocal,
  ConnectorsFilter,
  LocalServerInput,
  LocalServerStatus
} from './types'

export const EMPTY_CONNECTORS_FILTER: ConnectorsFilter = { query: '', segment: 'all' }

interface Phase {
  reason: 'serverError' | 'serverNeedsAuth' | undefined
  state: ConnectorState
  verb: ConnectorWayLocal['verb']
}

const LOCAL_PHASES = {
  error: { reason: 'serverError', state: 'broken', verb: 'openLogs' },
  'needs-auth': { reason: 'serverNeedsAuth', state: 'broken', verb: 'authenticate' },
  off: { reason: undefined, state: 'off', verb: undefined },
  ok: { reason: undefined, state: 'connected', verb: undefined },
  probing: { reason: undefined, state: 'connecting', verb: undefined },
  unknown: { reason: undefined, state: 'connecting', verb: undefined }
} satisfies Record<LocalServerStatus, Phase>

const LOCAL_WORDS = {
  available: 'available',
  broken: 'serverError',
  connected: 'serverOn',
  connecting: 'serverConnecting',
  expired: 'serverError',
  off: 'serverOff',
  unknown: 'serverConnecting'
} satisfies Record<ConnectorState, ConnectorStateWord>

function localFact(server: LocalServerInput, state: ConnectorState): ConnectorFact | undefined {
  if (state !== 'connected' || server.unused === true || server.toolsTotal === undefined) {
    return undefined
  }

  if (server.toolsOn === undefined) {
    return { count: server.toolsTotal, key: 'tools' }
  }

  return server.toolsOn < server.toolsTotal
    ? { count: server.toolsTotal, key: 'toolsSomeOn', on: server.toolsOn }
    : { count: server.toolsOn, key: 'toolsOn' }
}

export function localWay(server: LocalServerInput): ConnectorWayLocal {
  const status: LocalServerStatus = server.enabled ? server.status : 'off'
  const phase = LOCAL_PHASES[status]

  return {
    fact: localFact(server, phase.state),
    inCatalog: server.inCatalog,
    installed: true,
    plugin: server.plugin,
    reason: phase.reason ? { key: phase.reason } : undefined,
    serverEnabled: server.enabled,
    serverName: server.name,
    state: phase.state,
    target: server.target,
    unused: server.unused,
    verb: phase.verb === 'authenticate' && server.canAuthenticate === false ? 'openLogs' : phase.verb
  }
}

export function bundledWay(entry: BundledEntryInput): ConnectorWayLocal {
  return {
    authType: entry.authType,
    entryName: entry.name,
    inCatalog: true,
    installed: false,
    needsEnv: entry.needsEnv,
    state: 'available',
    verb: 'install'
  }
}

export function localWord(way: ConnectorWayLocal): ConnectorStateWord {
  if (way.reason?.key === 'serverNeedsAuth') {
    return 'serverNeedsAuth'
  }

  return way.state === 'connected' && way.unused === true ? 'serverOnUnused' : LOCAL_WORDS[way.state]
}

export function mergeCard({
  description,
  inCatalog,
  name,
  slug,
  way
}: {
  description?: string
  inCatalog: boolean
  name: string
  slug: string
  way: ConnectorWayLocal
}): ConnectorCardModel {
  return {
    description,
    fact: way.fact,
    inCatalog: inCatalog || way.inCatalog === true,
    name,
    plugin: way.plugin,
    reason: way.reason,
    slug,
    state: way.state,
    stateWord: localWord(way),
    verb: way.verb,
    ways: way
  }
}

export function localServerName(card: ConnectorCardModel): string {
  return card.ways.serverName ?? card.slug
}

export const localCardKey = (name: string) => `local:${name}`

export const cardKey = (card: ConnectorCardModel): string => localCardKey(card.slug)

interface CardParts {
  bundled?: BundledEntryInput
  local?: LocalServerInput
}

function slotAt(parts: Map<string, CardParts>, key: string): CardParts {
  const found = parts.get(key)

  if (found) {
    return found
  }

  const fresh: CardParts = {}
  parts.set(key, fresh)

  return fresh
}

const mergeKey = (connectorSlug: string | undefined, name: string) => connectorSlug ?? localCardKey(name)

// The slug stays the server's config key, so a deep link can still address the card.
function slugOf({ bundled, local }: CardParts): string {
  return local?.name ?? bundled?.name ?? ''
}

function descriptionOf({ bundled, local }: CardParts): string | undefined {
  return local?.description ?? bundled?.description
}

// An install must not rename the app: the bundled entry and the server it becomes read the same way.
function nameOf({ bundled, local }: CardParts, slug: string): string {
  return local?.title ?? connectorTitle(local?.name ?? bundled?.name ?? slug)
}

function cardOf(slot: CardParts): ConnectorCardModel | null {
  const way = slot.local ? localWay(slot.local) : slot.bundled ? bundledWay(slot.bundled) : null

  if (!way) {
    return null
  }

  const slug = slugOf(slot)

  return mergeCard({
    description: descriptionOf(slot),
    inCatalog: slot.local?.inCatalog ?? slot.bundled !== undefined,
    name: nameOf(slot, slug),
    slug,
    way
  })
}

export interface DeriveCardsInput {
  bundled?: readonly BundledEntryInput[]
  local: readonly LocalServerInput[]
}

export function deriveCards({ bundled = [], local }: DeriveCardsInput): ConnectorCardModel[] {
  const parts = new Map<string, CardParts>()

  for (const server of local) {
    slotAt(parts, mergeKey(server.connectorSlug, server.name)).local = server
  }

  for (const entry of bundled) {
    const slot = slotAt(parts, mergeKey(entry.connectorSlug, entry.name))

    // An installed server always beats the bundled entry of the same app.
    if (!slot.local) {
      slot.bundled = entry
    }
  }

  const cards: ConnectorCardModel[] = []

  for (const slot of parts.values()) {
    const card = cardOf(slot)

    if (card) {
      cards.push(card)
    }
  }

  return cards
}
