// Every derivation the Connectors page needs, in one place, so the components
// take derived props and never the raw world. No React, no i18n, no dates — a
// model carries a key and a count, and the component says the sentence.
//
// `derive-tools.ts` owns everything inside an opened connector; it is re-exported
// here so a call site has one import.

import type {
  ConnectorCardModel,
  ConnectorCategoryOption,
  ConnectorFact,
  ConnectorGroupId,
  ConnectorGroupModel,
  ConnectorPillId,
  ConnectorPillModel,
  ConnectorReason,
  ConnectorResidency,
  ConnectorsFilter,
  ConnectorState,
  ConnectorStateWord,
  ConnectorVerb,
  HostedConnectorInput,
  LocalServerInput,
  LocalServerStatus
} from './types'

export * from './derive-tools'

interface Phase {
  state: ConnectorState
  verb: ConnectorVerb | undefined
  word: ConnectorStateWord
}

/** One table instead of a ladder: every gateway account status decides its own
 *  state, word and verb. `inactive` and `revoked` both mean the sign-in stopped
 *  working, which is the same repair as an expiry. */
const HOSTED_PHASES = {
  active: { state: 'connected', verb: undefined, word: 'connected' },
  expired: { state: 'expired', verb: 'reconnect', word: 'accessExpired' },
  failed: { state: 'broken', verb: 'tryAgain', word: 'couldNotConnect' },
  inactive: { state: 'expired', verb: 'reconnect', word: 'accessExpired' },
  pending: { state: 'connecting', verb: 'stopWaiting', word: 'connecting' },
  revoked: { state: 'expired', verb: 'reconnect', word: 'accessExpired' }
} satisfies Record<string, Phase>

/** The same table for a server on this machine. Its own switch lives on the card,
 *  so `ok` and `off` carry no verb — the dialog is never needed to turn a server
 *  off. `unknown` is a configured server nothing has probed yet. */
const LOCAL_PHASES = {
  error: { state: 'broken', verb: 'openLogs', word: 'serverError' },
  'needs-auth': { state: 'broken', verb: 'authenticate', word: 'serverNeedsAuth' },
  off: { state: 'off', verb: undefined, word: 'serverOff' },
  ok: { state: 'connected', verb: undefined, word: 'serverOn' },
  probing: { state: 'connecting', verb: undefined, word: 'serverConnecting' },
  unknown: { state: 'connecting', verb: undefined, word: 'serverConnecting' }
} satisfies Record<LocalServerStatus, Phase>

const LOCAL_REASONS = {
  error: 'serverError',
  'needs-auth': 'serverNeedsAuth'
} satisfies Partial<Record<LocalServerStatus, ConnectorReason['key']>>

/** Attention sorts before everything else inside Connected: a connection that
 *  broke is the only thing on this page that stopped working on its own. */
const STATE_RANK = {
  available: 4,
  broken: 0,
  connected: 3,
  connecting: 2,
  expired: 1,
  off: 5
} satisfies Record<ConnectorState, number>

export const ATTENTION_STATES: readonly ConnectorState[] = ['broken', 'expired']

export function isAttention(card: ConnectorCardModel): boolean {
  return ATTENTION_STATES.includes(card.state)
}

function hostedReason(state: ConnectorState, statusReason?: string): ConnectorReason | undefined {
  if (state === 'connecting') {
    return { key: 'finishSignIn' }
  }

  // The provider's own sentence beats ours when it sent one: it is the only
  // source that knows whether the token expired or the account was revoked.
  return state === 'broken' || state === 'expired' ? { key: 'reconnect', text: statusReason } : undefined
}

/** One hosted app. `name` is the caller's: `connectorTitle` lives in
 *  `lib/connector-tools.ts` and the wiring slice passes its result in. */
export function hostedCard(row: HostedConnectorInput, name: string): ConnectorCardModel {
  const base = {
    category: row.category,
    description: row.description,
    hostedTwinAvailable: false,
    inCatalog: row.inCatalog ?? false,
    name,
    residency: 'hosted' as const,
    slug: row.slug
  }

  // Policy comes before status: an app the org took away has no useful state of
  // its own, and offering a verb on it would promise something that cannot work.
  if (row.orgLocked) {
    return { ...base, offBy: 'org', state: 'off', stateWord: 'offByYourOrganisation' }
  }

  if (!row.enabled) {
    return { ...base, offBy: 'me', state: 'off', stateWord: 'offForYou', verb: 'turnBackOn' }
  }

  const phase = row.connected ? HOSTED_PHASES[row.connectionStatus ?? 'active'] : undefined

  if (!phase) {
    return { ...base, state: 'available', stateWord: 'available', verb: 'connect' }
  }

  return {
    ...base,
    fact:
      phase.state === 'connected' && row.toolsOff && row.toolsOff > 0
        ? { count: row.toolsOff, key: 'toolsOff' }
        : undefined,
    reason: hostedReason(phase.state, row.statusReason),
    state: phase.state,
    stateWord: phase.word,
    verb: phase.verb
  }
}

/** A working server says how much of itself is live, because that is the only
 *  number a person can act on here. "31 tools, 29 on" when some are off, "8 tools
 *  on" when none are, and the bare total when the probe never reported a split.
 *
 *  An idle server says nothing: the lane holds one fact, and "On, unused" is the
 *  one worth reading. Returning a count here would print it over that word and
 *  leave the state unreachable. */
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

/** One server on this machine. `name` is the card's title; the server's config
 *  key stays the slug so the wiring slice can address it. */
export function localCard(server: LocalServerInput, name: string, hostedTwinAvailable = false): ConnectorCardModel {
  const status: LocalServerStatus = server.enabled ? server.status : 'off'
  const phase = LOCAL_PHASES[status]
  const reasonKey = LOCAL_REASONS[status as keyof typeof LOCAL_REASONS]
  const idle = phase.state === 'connected' && server.unused === true

  return {
    category: server.category,
    description: server.description,
    fact: localFact(server, phase.state),
    hostedTwinAvailable,
    inCatalog: false,
    name,
    offBy: phase.state === 'off' ? 'me' : undefined,
    reason: reasonKey ? { key: reasonKey } : undefined,
    residency: 'local',
    serverEnabled: server.enabled,
    slug: server.name,
    state: phase.state,
    stateWord: idle ? 'serverOnUnused' : phase.word,
    target: server.target,
    verb: phase.verb
  }
}

export interface DeriveCardsInput {
  hosted: readonly HostedConnectorInput[]
  local: readonly LocalServerInput[]
  /** Slug → display title. The wiring slice passes `connectorTitle` results. */
  titles?: Readonly<Record<string, string>>
}

/** The twin is worth a pill only if the person could actually fall back on it.
 *  An app the org took away, or one they turned off themselves, is not another
 *  backing — advertising it would promise a door that does not open. */
function twinIsAvailable(twin: HostedConnectorInput | undefined): boolean {
  return twin !== undefined && twin.orgLocked !== true && twin.enabled
}

/** One card per app.
 *
 * A local server that backs an app which also exists hosted collapses the two
 * rows into one card — the server's, because the server is the thing configured
 * on this machine — and the hosted twin survives as a quiet pill rather than a
 * second card the person must reconcile.
 *
 * The card survives the switch going off. A card that vanished when its own
 * switch was pressed would take the switch, the state and the way back with it,
 * and the person would be left looking for the server they just turned off. So
 * a disabled server keeps its card, in `On this Mac` where it lives, reading
 * `Off` — which is exactly what the `Turned off` pill counts.
 */
export function deriveCards({ hosted, local, titles = {} }: DeriveCardsInput): ConnectorCardModel[] {
  const nameOf = (slug: string) => titles[slug] ?? slug
  const twinned = new Set<string>()
  const cards: ConnectorCardModel[] = []

  for (const server of local) {
    const twin = server.hostedSlug ? hosted.find(row => row.slug === server.hostedSlug) : undefined

    if (twin) {
      twinned.add(twin.slug)
    }

    cards.push(localCard(server, nameOf(server.hostedSlug ?? server.name), twinIsAvailable(twin)))
  }

  for (const row of hosted) {
    if (!twinned.has(row.slug)) {
      cards.push(hostedCard(row, nameOf(row.slug)))
    }
  }

  return cards
}

const GROUP_OF = {
  available: 'available',
  broken: 'connected',
  connected: 'connected',
  connecting: 'connected',
  expired: 'connected',
  off: 'off'
} satisfies Record<ConnectorState, ConnectorGroupId>

const GROUP_ORDER: readonly ConnectorGroupId[] = ['connected', 'local', 'available', 'off']

function byStateThenName(a: ConnectorCardModel, b: ConnectorCardModel): number {
  return STATE_RANK[a.state] - STATE_RANK[b.state] || a.name.localeCompare(b.name)
}

/** Groups in reading order, empty ones dropped. Every local card stays in
 *  `On this Mac` whatever its state: that group is a place, not a status, and
 *  filing a broken server under `Connected` hides where it actually lives. */
export function groupCards(cards: readonly ConnectorCardModel[]): ConnectorGroupModel[] {
  const buckets = new Map<ConnectorGroupId, ConnectorCardModel[]>()

  for (const card of cards) {
    const id = card.residency === 'local' ? 'local' : GROUP_OF[card.state]
    const bucket = buckets.get(id)

    if (bucket) {
      bucket.push(card)
    } else {
      buckets.set(id, [card])
    }
  }

  return GROUP_ORDER.filter(id => (buckets.get(id)?.length ?? 0) > 0).map(id => ({
    cards: [...buckets.get(id)!].sort(byStateThenName),
    id
  }))
}

const PILL_ORDER: readonly ConnectorPillId[] = ['all', 'attention', 'connected', 'local', 'available', 'off']

const PILL_MATCHES = {
  all: () => true,
  attention: isAttention,
  available: card => card.state === 'available',
  connected: card => card.state === 'connected' || card.state === 'connecting',
  local: card => card.residency === 'local',
  off: card => card.state === 'off'
} satisfies Record<ConnectorPillId, (card: ConnectorCardModel) => boolean>

export function cardMatchesPill(card: ConnectorCardModel, pill: ConnectorPillId): boolean {
  return PILL_MATCHES[pill](card)
}

/** Counts for the state pills. A pill with a zero count is omitted: there is
 *  exactly one way to filter, and a pill that can only empty the page is a
 *  control that does nothing. */
export function pillCounts(cards: readonly ConnectorCardModel[]): ConnectorPillModel[] {
  return PILL_ORDER.map(id => ({ count: cards.filter(PILL_MATCHES[id]).length, id })).filter(pill => pill.count > 0)
}

export function cardMatchesQuery(card: ConnectorCardModel, query: string): boolean {
  const needle = query.trim().toLowerCase()

  return needle.length === 0 || card.slug.toLowerCase().includes(needle) || card.name.toLowerCase().includes(needle)
}

export function filterCards(
  cards: readonly ConnectorCardModel[],
  { category, pill, query, residency }: ConnectorsFilter
): ConnectorCardModel[] {
  return cards.filter(
    card =>
      cardMatchesPill(card, pill) &&
      cardMatchesQuery(card, query) &&
      (residency === null || card.residency === residency) &&
      (category === null || card.category === category)
  )
}

/** Category options for the select, counted under everything except the category
 *  filter itself — a select whose every option reads zero is a trap. */
export function cardCategoryOptions(
  cards: readonly ConnectorCardModel[],
  filter: ConnectorsFilter
): ConnectorCategoryOption[] {
  const counts = new Map<string, number>()

  for (const card of filterCards(cards, { ...filter, category: null })) {
    if (card.category) {
      counts.set(card.category, (counts.get(card.category) ?? 0) + 1)
    }
  }

  return [...counts.entries()]
    .map(([name, count]) => ({ count, name }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
}

/** How the inventory line counts itself: "64 hosted · 6 on this Mac". */
export function inventoryCounts(cards: readonly ConnectorCardModel[]): Record<ConnectorResidency, number> {
  return {
    hosted: cards.filter(card => card.residency === 'hosted').length,
    local: cards.filter(card => card.residency === 'local').length
  }
}
