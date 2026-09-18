// Wire rows become the `*Input` types `derive.ts` takes. Pure: no React, no
// query client, no i18n — every fact arrives as an argument so each join is
// provable on its own.
//
// Three reads make one hosted card. The tool gateway's list knows which apps are
// connected, the catalog knows their words, and the policy knows who is allowed
// to use them. None of the three is a superset of the others, so the page joins
// them here rather than asking one of them to grow.

import type {
  ConnectorAccountRow,
  ConnectorCatalogRow,
  ConnectorRow as ConnectorListRow,
  ConnectorPolicyLayer,
  ConnectorToolsResult
} from '@hermes/shared'

import type { McpCatalogEntry } from '@/hermes'
import { connectorTitle } from '@/lib/connector-tools'
import type { McpServers } from '@/lib/mcp-servers'

import { toolRows } from '../derive-tools'
import type {
  HostedConnectorInput,
  LocalServerInput,
  LocalServerStatus,
  ToolInput,
  ToolRowModel,
  ToolsFreshness
} from '../types'

// ── policy ─────────────────────────────────────────────────────────────────

/** Behaviour hints a layer rules on wholesale. `disable` takes a tool away;
 *  `enable` is an allow-list, so a tool carrying none of its tags is locked. */
export interface ConnectorPolicyTagRules {
  disable: readonly string[]
  enable: readonly string[]
}

/** One policy layer, flattened so every mode reads the same way. Four wire modes
 *  become two facts: what this layer permits and what it forbids. */
export interface ConnectorPolicyRules {
  /** Connectors this layer permits; `null` means "every connector". */
  allowed: ReadonlySet<string> | null
  /** Connectors this layer forbids by name. */
  denied: ReadonlySet<string>
  /** Compare-and-set token for this layer. */
  revision: string
  tags: ConnectorPolicyTagRules
  /** Named tool rules, keyed by connector slug. */
  tools: Readonly<Record<string, readonly string[]>>
}

export interface ConnectorPolicyView {
  /** The one layer this page writes. Absent until the member has a rule. */
  member: ConnectorPolicyRules | null
  /** Every other layer (org, role). Read-only here, and any of them can lock. */
  others: readonly ConnectorPolicyRules[]
}

export const EMPTY_POLICY: ConnectorPolicyView = { member: null, others: [] }

const NO_TAGS: ConnectorPolicyTagRules = { disable: [], enable: [] }

function rulesOf(layer: ConnectorPolicyLayer): ConnectorPolicyRules {
  const body = layer.body
  const base = { revision: layer.revision, tags: NO_TAGS, tools: {} }

  if (body.mode === 'unrestricted') {
    return { ...base, allowed: null, denied: new Set() }
  }

  // Deny-all permits nothing, which is an allow-list with no entries in it —
  // the same shape as `allow` with `connectors: []`, so it needs no third case.
  if (body.mode === 'deny-all') {
    return { ...base, allowed: new Set(), denied: new Set() }
  }

  const tags: ConnectorPolicyTagRules = {
    disable: body.tags?.disable ?? [],
    enable: body.tags?.enable ?? []
  }

  return body.mode === 'allow'
    ? { allowed: new Set(body.connectors), denied: new Set(), revision: layer.revision, tags, tools: body.tools }
    : {
        allowed: null,
        denied: new Set(body.disabled_connectors),
        revision: layer.revision,
        tags,
        tools: body.tools
      }
}

export function readPolicy(layers: readonly ConnectorPolicyLayer[]): ConnectorPolicyView {
  const member = layers.find(layer => layer.kind === 'member')

  return {
    member: member ? rulesOf(member) : null,
    others: layers.filter(layer => layer.kind !== 'member').map(rulesOf)
  }
}

const layerAllows = (rules: ConnectorPolicyRules, slug: string): boolean =>
  (rules.allowed === null || rules.allowed.has(slug)) && !rules.denied.has(slug)

/** The person's own switch. No member layer at all means nothing is switched off. */
export const memberEnables = (policy: ConnectorPolicyView, slug: string): boolean =>
  policy.member === null || layerAllows(policy.member, slug)

/** A layer the person cannot write took the whole app away. */
export const orgLocks = (policy: ConnectorPolicyView, slug: string): boolean =>
  policy.others.some(rules => !layerAllows(rules, slug))

/** The member's saved rule for one connector — the editor's baseline. */
export const memberDisabledTools = (policy: ConnectorPolicyView, slug: string): readonly string[] =>
  policy.member?.tools[slug] ?? []

/** The CAS token the editor must send back with a tools write. */
export const memberRevision = (policy: ConnectorPolicyView): string | undefined => policy.member?.revision

/** Tools a non-member layer locks: named rules plus the two tag rules. A tag
 *  rule is checked against the tool's own hints, so a connector whose tools were
 *  never fetched cannot be resolved — which is why the card counts named rules
 *  only and the locked rows appear when the dialog opens. */
export function orgLockedTools(
  policy: ConnectorPolicyView,
  slug: string,
  tools: readonly ToolInput[]
): Set<string> {
  const locked = new Set<string>()

  for (const rules of policy.others) {
    for (const tool of rules.tools[slug] ?? []) {
      locked.add(tool)
    }

    for (const tool of tools) {
      const hints = new Set(tool.hints)
      const disabledByTag = rules.tags.disable.some(tag => hints.has(tag))
      const missingEveryEnabledTag = rules.tags.enable.length > 0 && !rules.tags.enable.some(tag => hints.has(tag))

      if (disabledByTag || missingEveryEnabledTag) {
        locked.add(tool.slug)
      }
    }
  }

  return locked
}

/** The rows the tool list renders: the wire tools, the member's rule, and
 *  whatever a layer above the member has locked. */
export function connectorToolRows(
  policy: ConnectorPolicyView,
  slug: string,
  tools: readonly ToolInput[]
): ToolRowModel[] {
  return toolRows(tools, new Set(memberDisabledTools(policy, slug)), orgLockedTools(policy, slug, tools))
}

// ── hosted ─────────────────────────────────────────────────────────────────

export interface HostedJoinInput {
  accounts: readonly ConnectorAccountRow[]
  catalog: readonly ConnectorCatalogRow[]
  list: readonly ConnectorListRow[]
  policy: ConnectorPolicyView
}

const text = (value: unknown): string | undefined =>
  typeof value === 'string' && value.trim() !== '' ? value : undefined

/** The account a card speaks for. A connector can carry several; the live one
 *  wins, and among equals the newest, because that is the one the person just
 *  finished connecting. */
export function pickAccount(
  accounts: readonly ConnectorAccountRow[],
  slug: string
): ConnectorAccountRow | undefined {
  const mine = accounts.filter(account => account.connector === slug)
  const rank = (account: ConnectorAccountRow) => (account.active ? 0 : 1)

  return [...mine].sort((a, b) => rank(a) - rank(b) || b.created_at.localeCompare(a.created_at))[0]
}

/** Slug → the title a card prints. The catalog's own name beats our table, and
 *  our table beats a bare slug. */
export function connectorTitles(input: Pick<HostedJoinInput, 'catalog' | 'list'>): Record<string, string> {
  const titles: Record<string, string> = {}

  for (const slug of hostedSlugs(input)) {
    const fromCatalog = text(input.catalog.find(row => row.slug === slug)?.name)
    const fromList = text(input.list.find(row => row.connector === slug)?.name)

    titles[slug] = fromCatalog ?? fromList ?? connectorTitle(slug)
  }

  return titles
}

/** Slug → shelf category, so a local server that backs a known app filters into
 *  the same bucket as the app it backs. */
export function connectorCategories(catalog: readonly ConnectorCatalogRow[]): Record<string, string> {
  return Object.fromEntries(catalog.filter(row => row.category).map(row => [row.slug, row.category]))
}

/** Every app the page knows about: the catalog is the shelf, the list is what
 *  the account has actually touched, and neither contains the other. */
function hostedSlugs({ catalog, list }: Pick<HostedJoinInput, 'catalog' | 'list'>): string[] {
  const slugs = new Set(catalog.map(row => row.slug))

  for (const row of list) {
    if (row.connector) {
      slugs.add(row.connector)
    }
  }

  return [...slugs]
}

export function joinHostedConnectors({ accounts, catalog, list, policy }: HostedJoinInput): HostedConnectorInput[] {
  return hostedSlugs({ catalog, list }).map(slug => {
    const entry = catalog.find(row => row.slug === slug)
    const row = list.find(candidate => candidate.connector === slug)
    const account = pickAccount(accounts, slug)
    const named = memberDisabledTools(policy, slug).length

    return {
      accountLabel: account?.label,
      category: entry?.category,
      // An account row exists for a connection that is still being authorized and
      // for one that expired, and both are states the card has a word for. Only
      // an app with no account row at all is "Available", so existence — not
      // `active` — is what `connected` means here; `connectionStatus` says which.
      connected: account !== undefined || row?.connected === true,
      connectedAt: account?.created_at,
      connectionStatus: account?.status ?? undefined,
      description: entry?.description ?? text(row?.description),
      // Two sources can say "off" and both are authoritative: the member policy
      // layer and the tool gateway's own flag. On means neither said so.
      enabled: memberEnables(policy, slug) && row?.enabled !== false,
      inCatalog: entry !== undefined,
      orgLocked: orgLocks(policy, slug),
      slug,
      statusReason: text(account?.status_reason) ?? text(row?.statusReason),
      // Named rules only: a tag rule needs the tool list, which a card never
      // fetches. The dialog shows the full picture once it opens.
      toolsOff: named > 0 ? named : undefined
    }
  })
}

// ── local ──────────────────────────────────────────────────────────────────

export interface LocalJoinInput {
  /** The bundled MCP catalog. Its `connector` field is the only source for the
   *  hosted slug a local server backs. */
  catalog: readonly McpCatalogEntry[]
  /** Hosted slug → its shelf category, so a server that backs a known app lands
   *  in the same bucket as the app. `connectorCategories` builds it. */
  categories?: Readonly<Record<string, string>>
  /** The `mcp_servers` map out of the scoped profile's config. */
  servers: McpServers
  /** Server name → the status the MCP tab's probe table computed. */
  status: Readonly<Record<string, LocalServerStatus>>
  /** Server name → its tool split, when a probe reported one. */
  toolCounts?: Readonly<Record<string, { on: number; total: number }>>
  /** Server name → calls in the usage window. Zero is the quiet `unused` pill. */
  usage?: Readonly<Record<string, number>>
}

const targetOf = (entry: Record<string, unknown>): string => {
  const url = text(entry.url)

  if (url !== undefined) {
    return url
  }

  const args = Array.isArray(entry.args) ? entry.args.filter((arg): arg is string => typeof arg === 'string') : []

  return [text(entry.command) ?? '', ...args].join(' ').trim()
}

export function joinLocalServers({
  catalog,
  categories,
  servers,
  status,
  toolCounts,
  usage
}: LocalJoinInput): LocalServerInput[] {
  return Object.entries(servers).map(([name, entry]) => {
    const bundled = catalog.find(candidate => candidate.name === name)
    const hostedSlug = text(bundled?.connector)
    const counts = toolCounts?.[name]
    const calls = usage?.[name]

    return {
      category: hostedSlug === undefined ? undefined : categories?.[hostedSlug],
      description: text(bundled?.description),
      enabled: entry.enabled !== false,
      // The manifest's join key is what collapses a local `notion` and a hosted
      // `notion` into one card. A hand-written server has none, and stays its
      // own card — which is right: nothing says the two are the same app.
      hostedSlug,
      name,
      status: status[name] ?? 'unknown',
      target: targetOf(entry),
      toolsOn: counts?.on,
      toolsTotal: counts?.total,
      transport: text(entry.url) !== undefined ? ('url' as const) : ('program' as const),
      unused: calls === undefined ? undefined : calls === 0
    }
  })
}

// ── tools ──────────────────────────────────────────────────────────────────

/** The freshness cue behind the Refresh button. The wire timestamp is Unix
 *  SECONDS (`time.time()`); `freshnessLabel` subtracts it from `Date.now()`, so
 *  it has to arrive in milliseconds. */
export const toolsFreshness = (result: ConnectorToolsResult): ToolsFreshness => ({
  fetchedAt: result.fetched_at * 1000,
  source: result.source,
  stale: result.stale
})
