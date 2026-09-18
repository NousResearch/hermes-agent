// The Connectors page speaks in two vocabularies, and this file holds both.
//
// The `*Input` types mirror the facts the backend already has (a hosted
// connector row, a local MCP server, a tool listing). The models below them are
// what the components render. Nothing here is a wire type: the wiring slice maps
// RPC payloads onto the inputs, `derive.ts` turns inputs into models, and the
// components only ever see models. That seam is why the page can be built and
// tested before the RPCs exist.
//
// Copy is NOT derived. A pure function cannot reach `useI18n()`, so derivations
// return *descriptors* — a key plus its count — and the component resolves them
// against `t.connectorsPage`. Never put an English sentence in a model.

import type { ConnectionStatus } from '@/lib/connector-tools'

/** Where the connector actually runs. One card shows one backing. */
export type ConnectorResidency = 'hosted' | 'local'

/** `mcp-tab.tsx` already speaks this vocabulary for a local server; keep it identical. */
export type LocalServerStatus = 'error' | 'needs-auth' | 'off' | 'ok' | 'probing' | 'unknown'

/** How a local server is reached. A detail of an opened server, never a row label. */
export type LocalServerTransport = 'program' | 'url'

// ---------------------------------------------------------------- inputs

/** One hosted connector, as the gateway's connector list describes it. */
export interface HostedConnectorInput {
  /** The signed-in identity Hermes acts as. Shown in the dialog, never on a card. */
  accountLabel?: string
  /** Grouping for the category select. Absent apps fall in no bucket. */
  category?: string
  connected: boolean
  /** ISO timestamp of the connection. The dialog prints a date, so keep it raw here. */
  connectedAt?: string
  connectionStatus?: ConnectionStatus
  description?: string
  /** The person's own on/off switch. False means "Off for you". */
  enabled: boolean
  /** The catalog check mark: this app ships in the Hermes catalog. */
  inCatalog?: boolean
  /** The org took the whole app away. Nothing the person does here can widen it. */
  orgLocked?: boolean
  slug: string
  /** The provider's own words for a broken connection. */
  statusReason?: string
  toolsOff?: number
  toolsTotal?: number
}

/** One server configured on this machine. Mirrors what `mcp-tab.tsx` reads today. */
export interface LocalServerInput {
  /** Set when this server is the local face of an app that also exists hosted. */
  category?: string
  description?: string
  enabled: boolean
  /** The app slug this server backs, when it is one of the known apps. */
  hostedSlug?: string
  name: string
  status: LocalServerStatus
  target: string
  toolsOn?: number
  toolsTotal?: number
  transport: LocalServerTransport
  /** No call in the usage window. A quiet pill, not a problem. */
  unused?: boolean
}

/** One tool as the 24 h tool-list cache holds it. `facet` and `hints` stay open
 *  strings: the provider may add a value before we ship a word for it. */
export interface ToolInput {
  categories: string[]
  deprecated: boolean
  description: string
  facet: string
  hints: string[]
  name: string
  slug: string
}

// ---------------------------------------------------------------- card models

/** The machine state of one app. The group it lands in is derived from this. */
export type ConnectorState = 'available' | 'broken' | 'connected' | 'connecting' | 'expired' | 'off'

/** Who turned it off. The two read very differently: one is reversible here. */
export type ConnectorOffBy = 'me' | 'org'

/** The one word in the card's state lane. The key, not the word. Hosted apps and
 *  servers on this Mac keep separate words: "Connected" and "On" are different
 *  claims, and a server is never "connected" to anything but this machine. */
export type ConnectorStateWord =
  | 'accessExpired'
  | 'available'
  | 'connected'
  | 'connecting'
  | 'couldNotConnect'
  | 'offByYourOrganisation'
  | 'offForYou'
  | 'serverConnecting'
  | 'serverError'
  | 'serverNeedsAuth'
  | 'serverOff'
  | 'serverOn'
  | 'serverOnUnused'

/** The single count a card may carry. Never two facts in one lane. */
export type ConnectorFactKey = 'tools' | 'toolsOff' | 'toolsOn' | 'toolsSomeOn'

export interface ConnectorFact {
  count: number
  key: ConnectorFactKey
  /** The second number of `toolsSomeOn` ("31 tools, 29 on"). */
  on?: number
}

/** Why a broken connection broke. `text` carries the provider's own sentence when
 *  it sent one; the component prefers it over the generic key. */
export interface ConnectorReason {
  key: 'finishSignIn' | 'reconnect' | 'serverError' | 'serverNeedsAuth'
  text?: string
}

/** The one verb at the card's right edge. */
export type ConnectorVerb =
  'authenticate' | 'connect' | 'openLogs' | 'reconnect' | 'stopWaiting' | 'tryAgain' | 'turnBackOn'

export interface ConnectorCardModel {
  category?: string
  description?: string
  fact?: ConnectorFact
  /** The quiet `Hosted version available` pill on a local-backed card. */
  hostedTwinAvailable: boolean
  inCatalog: boolean
  name: string
  offBy?: ConnectorOffBy
  reason?: ConnectorReason
  residency: ConnectorResidency
  /** Local cards only: the server's own switch, rendered in the dialog. */
  serverEnabled?: boolean
  slug: string
  state: ConnectorState
  stateWord: ConnectorStateWord
  /** Local cards only: the endpoint or program, printed in mono under the name
   *  where a hosted card prints its description. */
  target?: string
  verb?: ConnectorVerb
}

export type ConnectorGroupId = 'available' | 'connected' | 'local' | 'off'

export interface ConnectorGroupModel {
  cards: ConnectorCardModel[]
  id: ConnectorGroupId
}

/** The state filter. `all` is always offered; the rest are omitted at zero. */
export type ConnectorPillId = 'all' | 'attention' | 'available' | 'connected' | 'local' | 'off'

export interface ConnectorPillModel {
  count: number
  id: ConnectorPillId
}

export interface ConnectorsFilter {
  category: null | string
  pill: ConnectorPillId
  query: string
  residency: ConnectorResidency | null
}

export interface ConnectorCategoryOption {
  count: number
  name: string
}

// ---------------------------------------------------------------- tool models

export interface ToolRowModel {
  categories: string[]
  deprecated: boolean
  description: string
  facet: string
  hints: string[]
  /** Struck and unswitchable. Only the org can give it back. */
  lockedBy: 'org' | null
  name: string
  on: boolean
  slug: string
}

export interface ToolsFilter {
  category: null | string
  facet: null | string
  hint: null | string
  query: string
  showDeprecated: boolean
}

/** Every state the right column of the dialog can be in. `ready` is the only one
 *  that renders a list. */
export type ToolsEditorPhase = 'conflict' | 'gone' | 'loading' | 'ready' | 'saving' | 'signedOut' | 'unavailable'

/** The dirty footer reads exactly these two numbers. */
export interface ToolsEditorCounts {
  backOn: number
  off: number
}

export interface ToolsEditorState {
  counts: ToolsEditorCounts
  dirty: boolean
  phase: ToolsEditorPhase
}

/** Where this tool list came from and how old it is. Drives the freshness cue. */
export interface ToolsFreshness {
  fetchedAt: number
  source: 'cache' | 'network' | 'revalidated'
  stale: boolean
}

export type QuickActionId = 'everything-on' | 'no-destructive' | 'read-only'

export interface QuickAction {
  /** The facets this action turns OFF. Empty means it turns everything back on. */
  facets: readonly string[]
  id: QuickActionId
}

/** What the other editor's saved version does that this one does not. Counts
 *  only — the sentence lives in i18n. */
export interface ConflictDifference {
  /** Tools they left on that this editor turned off. */
  theyOn: number
  /** Tools they turned off that this editor has on. */
  theyOff: number
}
