export type ConnectorResidency = 'hosted' | 'local'

export type LocalServerStatus = 'error' | 'needs-auth' | 'off' | 'ok' | 'probing' | 'unknown'

export interface HostedConnectorInput {
  accountLabel?: string
  connected: boolean
  connectedAt?: string
  connectionStatus?: string
  description?: string
  disabledTools?: readonly string[]
  enabled: boolean
  inCatalog?: boolean
  orgLocked?: boolean
  slug: string
  statusReason?: string
  toolsOff?: number
}

export interface LocalServerInput {
  canAuthenticate?: boolean
  connectorSlug?: string
  description?: string
  enabled: boolean
  inCatalog?: boolean
  name: string
  plugin?: string
  status: LocalServerStatus
  target: string
  title?: string
  toolsOn?: number
  toolsTotal?: number
  unused?: boolean
}

export type ConnectorAuthType = 'apiKey' | 'none' | 'oauth'

export interface BundledEntryInput {
  authType: ConnectorAuthType
  connectorSlug?: string
  description?: string
  name: string
  needsEnv: boolean
}

export interface ToolInput {
  categories: string[]
  deprecated: boolean
  description: string
  facet: string
  hints: string[]
  name: string
  slug: string
}

export type ConnectorState = 'available' | 'broken' | 'connected' | 'connecting' | 'expired' | 'off' | 'unknown'

export type ConnectorOffBy = 'me' | 'org'

export type ConnectorStateWord =
  | 'accessExpired'
  | 'available'
  | 'connected'
  | 'connecting'
  | 'connectionUnknown'
  | 'couldNotConnect'
  | 'offByYourOrganisation'
  | 'offForYou'
  | 'serverConnecting'
  | 'serverError'
  | 'serverNeedsAuth'
  | 'serverOff'
  | 'serverOn'
  | 'serverOnUnused'

export type ConnectorFactKey = 'tools' | 'toolsOff' | 'toolsOn' | 'toolsSomeOn'

export interface ConnectorFact {
  count: number
  key: ConnectorFactKey
  on?: number
}

export interface ConnectorReason {
  key: 'finishSignIn' | 'reconnect' | 'serverError' | 'serverNeedsAuth'
  text?: string
}

export type ConnectorVerb =
  'authenticate' | 'connect' | 'install' | 'openLogs' | 'reconnect' | 'stopWaiting' | 'tryAgain' | 'turnBackOn'

export interface ConnectorWayHosted {
  accountLabel?: string
  connected: boolean
  connectedAt?: string
  disabledTools?: readonly string[]
  fact?: ConnectorFact
  offBy?: ConnectorOffBy
  reason?: ConnectorReason
  state: ConnectorState
  verb?: ConnectorVerb
}

export interface ConnectorWayLocal {
  authType?: ConnectorAuthType
  entryName?: string
  fact?: ConnectorFact
  inCatalog?: boolean
  installed?: boolean
  needsEnv?: boolean
  plugin?: string
  reason?: ConnectorReason
  serverEnabled?: boolean
  serverName?: string
  state: ConnectorState
  target?: string
  unused?: boolean
  verb?: ConnectorVerb
}

export type ConnectorWays =
  { hosted: ConnectorWayHosted; local: ConnectorWayLocal | null } | { hosted: null; local: ConnectorWayLocal }

export interface ConnectorCardModel {
  description?: string
  fact?: ConnectorFact
  inCatalog: boolean
  name: string
  offBy?: ConnectorOffBy
  plugin?: string
  reason?: ConnectorReason
  residency: ConnectorResidency
  slug: string
  state: ConnectorState
  stateWord: ConnectorStateWord
  verb?: ConnectorVerb
  ways: ConnectorWays
}

export type ConnectorGroupId = 'available' | 'connected' | 'local' | 'off'

export interface ConnectorGroupModel {
  cards: ConnectorCardModel[]
  id: ConnectorGroupId
}

export type ConnectorSegmentId = 'all' | ConnectorGroupId

export interface ConnectorSegmentModel {
  count: number
  id: ConnectorSegmentId
}

export interface ConnectorsFilter {
  query: string
  segment: ConnectorSegmentId
}

export interface ConnectorPageModel {
  groups: ConnectorGroupModel[]
  hiddenMatches: number
  segment: ConnectorSegmentId
  segments: ConnectorSegmentModel[]
}

export type HostedPhase = 'failed' | 'loading' | 'ready' | 'signedOut' | 'unavailable'

export interface ToolRowModel {
  categories: string[]
  deprecated: boolean
  description: string
  facet: string
  hints: string[]
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

export type ToolsEditorPhase =
  'conflict' | 'gone' | 'loading' | 'needsAuth' | 'off' | 'ready' | 'saving' | 'signedOut' | 'unavailable'

export type ToolsEditorStatus = Extract<
  ToolsEditorPhase,
  'gone' | 'loading' | 'needsAuth' | 'off' | 'signedOut' | 'unavailable'
>

export interface FacetSummaryRow {
  facet: string
  locked: boolean
  on: number
  switchState: 'mixed' | 'off' | 'on'
  total: number
}

export interface ToolsEditorCounts {
  backOn: number
  off: number
}

export type QuickActionId = 'everything-on' | 'no-destructive' | 'read-only'

export interface QuickAction {
  facets: readonly string[]
  id: QuickActionId
}

export interface ConflictDifference {
  theyOn: number
  theyOff: number
}
