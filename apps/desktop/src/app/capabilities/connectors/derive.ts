import { connectorTitle } from '@/lib/connector-tools'

import type {
  BundledEntryInput,
  ConnectorCardModel,
  ConnectorFact,
  ConnectorReason,
  ConnectorsFilter,
  ConnectorState,
  ConnectorStateWord,
  ConnectorVerb,
  ConnectorWayHosted,
  ConnectorWayLocal,
  ConnectorWays,
  HostedConnectorInput,
  HostedPhase,
  LocalServerInput,
  LocalServerStatus
} from './types'

export const EMPTY_CONNECTORS_FILTER: ConnectorsFilter = { query: '', segment: 'all' }

interface Phase {
  reason: ConnectorReason['key'] | undefined
  state: ConnectorState
  verb: ConnectorVerb | undefined
}

const HOSTED_PHASES = {
  active: { reason: undefined, state: 'connected', verb: undefined },
  expired: { reason: 'reconnect', state: 'expired', verb: 'reconnect' },
  failed: { reason: 'reconnect', state: 'broken', verb: 'tryAgain' },
  inactive: { reason: 'reconnect', state: 'expired', verb: 'reconnect' },
  pending: { reason: 'finishSignIn', state: 'connecting', verb: 'stopWaiting' },
  revoked: { reason: 'reconnect', state: 'expired', verb: 'reconnect' }
} satisfies Record<string, Phase>

const UNKNOWN_HOSTED_PHASE: Phase = { reason: undefined, state: 'unknown', verb: undefined }

function hostedPhaseFor(status: string): Phase {
  if (!Object.hasOwn(HOSTED_PHASES, status)) {
    return UNKNOWN_HOSTED_PHASE
  }

  // SAFETY: guarded by `Object.hasOwn` on the line above.
  return HOSTED_PHASES[status as keyof typeof HOSTED_PHASES]
}

const LOCAL_PHASES = {
  error: { reason: 'serverError', state: 'broken', verb: 'openLogs' },
  'needs-auth': { reason: 'serverNeedsAuth', state: 'broken', verb: 'authenticate' },
  off: { reason: undefined, state: 'off', verb: undefined },
  ok: { reason: undefined, state: 'connected', verb: undefined },
  probing: { reason: undefined, state: 'connecting', verb: undefined },
  unknown: { reason: undefined, state: 'connecting', verb: undefined }
} satisfies Record<LocalServerStatus, Phase>

const HOSTED_WORDS = {
  available: 'available',
  broken: 'couldNotConnect',
  connected: 'connected',
  connecting: 'connecting',
  expired: 'accessExpired',
  off: 'offForYou',
  unknown: 'connectionUnknown'
} satisfies Record<ConnectorState, ConnectorStateWord>

const LOCAL_WORDS = {
  available: 'available',
  broken: 'serverError',
  connected: 'serverOn',
  connecting: 'serverConnecting',
  expired: 'serverError',
  off: 'serverOff',
  unknown: 'serverConnecting'
} satisfies Record<ConnectorState, ConnectorStateWord>

export function hostedWay(row: HostedConnectorInput): ConnectorWayHosted {
  const base = {
    accountLabel: row.accountLabel,
    connected: row.connected,
    connectedAt: row.connectedAt,
    disabledTools: row.disabledTools
  }

  if (row.orgLocked) {
    return { ...base, offBy: 'org', state: 'off' }
  }

  if (!row.enabled) {
    return { ...base, offBy: 'me', state: 'off', verb: 'turnBackOn' }
  }

  const status = row.connectionStatus ?? 'active'
  const phase = row.connected ? hostedPhaseFor(status) : undefined

  if (!phase) {
    return { ...base, state: 'available', verb: 'connect' }
  }

  if (phase.state === 'connecting') {
    return {
      ...base,
      accountLabel: undefined,
      connectedAt: undefined,
      reason: { key: 'finishSignIn' },
      state: phase.state,
      verb: phase.verb
    }
  }

  return {
    ...base,
    fact:
      phase.state === 'connected' && row.toolsOff && row.toolsOff > 0
        ? { count: row.toolsOff, key: 'toolsOff' }
        : undefined,
    reason: phase.reason
      ? { key: phase.reason, text: phase.reason === 'reconnect' ? row.statusReason : undefined }
      : undefined,
    state: phase.state,
    verb: phase.verb
  }
}

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

export function bothWaysOn({ hosted, local }: ConnectorWays): boolean {
  return (
    hosted?.state === 'connected' &&
    local?.installed === true &&
    local.serverEnabled === true &&
    local.state === 'connected'
  )
}

type Speaker = { kind: 'hosted'; way: ConnectorWayHosted } | { kind: 'local'; way: ConnectorWayLocal }

function speakerOf(ways: ConnectorWays): Speaker {
  if (ways.hosted && ways.hosted.state !== 'available') {
    return { kind: 'hosted', way: ways.hosted }
  }

  if (ways.local?.installed === true) {
    return { kind: 'local', way: ways.local }
  }

  return ways.hosted === null ? { kind: 'local', way: ways.local } : { kind: 'hosted', way: ways.hosted }
}

export function hostedStateWord(way: ConnectorWayHosted): ConnectorStateWord {
  return way.offBy === 'org' ? 'offByYourOrganisation' : HOSTED_WORDS[way.state]
}

function localWord(way: ConnectorWayLocal): ConnectorStateWord {
  if (way.reason?.key === 'serverNeedsAuth') {
    return 'serverNeedsAuth'
  }

  return way.state === 'connected' && way.unused === true ? 'serverOnUnused' : LOCAL_WORDS[way.state]
}

export interface MergeCardInput {
  description?: string
  inCatalog: boolean
  name: string
  slug: string
  ways: ConnectorWays
}

export function mergeCard({ description, inCatalog, name, slug, ways }: MergeCardInput): ConnectorCardModel {
  const speaker = speakerOf(ways)

  const base = {
    description,
    fact: speaker.way.fact,
    inCatalog: inCatalog || ways.local?.inCatalog === true,
    name,
    reason: speaker.way.reason,
    slug,
    ways
  }

  if (speaker.kind === 'hosted') {
    return {
      ...base,
      offBy: speaker.way.offBy,
      residency: 'hosted',
      state: speaker.way.state,
      stateWord: hostedStateWord(speaker.way),
      verb: speaker.way.verb
    }
  }

  return {
    ...base,
    offBy: speaker.way.state === 'off' ? 'me' : undefined,
    plugin: speaker.way.plugin,
    residency: 'local',
    state: speaker.way.state,
    stateWord: localWord(speaker.way),
    verb: speaker.way.verb
  }
}

export function forgetAbandoned(
  rows: readonly HostedConnectorInput[],
  abandoned: ReadonlySet<string>
): HostedConnectorInput[] {
  return rows.map(row =>
    abandoned.has(row.slug) && row.connectionStatus === 'pending'
      ? { ...row, accountLabel: undefined, connected: false, connectedAt: undefined, connectionStatus: undefined }
      : row
  )
}

export function localServerName(card: ConnectorCardModel): string {
  return card.ways.local?.serverName ?? card.slug
}

export type ConnectorTwinPill = 'alsoLocal' | 'hostedTwin' | null

export function twinPillOf({ residency, ways }: ConnectorCardModel): ConnectorTwinPill {
  if (residency === 'local') {
    return ways.hosted && ways.hosted.state !== 'off' ? 'hostedTwin' : null
  }

  return ways.local?.installed === true ? 'alsoLocal' : null
}

export interface DeriveCardsInput {
  bundled?: readonly BundledEntryInput[]
  hosted: readonly HostedConnectorInput[]
  local: readonly LocalServerInput[]
  titles?: Readonly<Record<string, string>>
}

interface CardParts {
  bundled?: BundledEntryInput
  hosted?: HostedConnectorInput
  local?: LocalServerInput
}

export const hostedCardKey = (slug: string) => `hosted:${slug}`

export const localCardKey = (name: string) => `local:${name}`

export const cardKey = (card: ConnectorCardModel): string =>
  card.ways.hosted === null ? localCardKey(card.slug) : hostedCardKey(card.slug)

const mergeKey = (connectorSlug: string | undefined, name: string) => connectorSlug ?? localCardKey(name)

function waysOf(parts: CardParts): ConnectorWays | null {
  const local = parts.local ? localWay(parts.local) : parts.bundled ? bundledWay(parts.bundled) : null

  if (parts.hosted) {
    return { hosted: hostedWay(parts.hosted), local }
  }

  return local ? { hosted: null, local } : null
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

// The slug stays the hosted slug, or the server's config key, so a deep link can still address the card.
function slugOf({ bundled, hosted, local }: CardParts): string {
  return hosted?.slug ?? local?.name ?? bundled?.name ?? ''
}

function descriptionOf({ bundled, hosted, local }: CardParts): string | undefined {
  return hosted?.description ?? local?.description ?? bundled?.description
}

// An install must not rename the app: the bundled entry and the server it becomes read the same way.
function nameOf({ bundled, hosted, local }: CardParts, titles: Readonly<Record<string, string>>, slug: string): string {
  const key = hosted?.slug ?? local?.connectorSlug ?? bundled?.connectorSlug ?? slug

  return titles[key] ?? local?.title ?? connectorTitle(local?.name ?? bundled?.name ?? slug)
}

function cardOf(slot: CardParts, titles: Readonly<Record<string, string>>): ConnectorCardModel | null {
  const ways = waysOf(slot)

  if (!ways) {
    return null
  }

  const slug = slugOf(slot)

  return mergeCard({
    description: descriptionOf(slot),
    inCatalog: slot.hosted?.inCatalog ?? false,
    name: nameOf(slot, titles, slug),
    slug,
    ways
  })
}

export function deriveCards({ bundled = [], hosted, local, titles = {} }: DeriveCardsInput): ConnectorCardModel[] {
  const parts = new Map<string, CardParts>()

  for (const row of hosted) {
    slotAt(parts, row.slug).hosted = row
  }

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
    const card = cardOf(slot, titles)

    if (card) {
      cards.push(card)
    }
  }

  return cards
}

export interface HostedPhaseInput {
  available?: boolean
  errored: boolean
  pending: boolean
  reason: null | string
}

export function hostedPhase({ available, errored, pending, reason }: HostedPhaseInput): HostedPhase {
  if (reason === 'NEEDS_NOUS_AUTH') {
    return 'signedOut'
  }

  if (reason === 'CONNECTORS_UNAVAILABLE') {
    return 'unavailable'
  }

  if (errored) {
    return 'failed'
  }

  if (pending) {
    return 'loading'
  }

  return available === false ? 'unavailable' : 'ready'
}
