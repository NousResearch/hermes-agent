import {
  addMcpServer,
  authMcpServer,
  cancelMcpOAuthFlow,
  getMcpCatalog,
  getMcpOAuthFlow,
  listMcpServers,
  removeMcpServer
} from '@/hermes'
import { translateNow } from '@/i18n'
import { completeMcpDesktopOAuth, McpOAuthCancelled } from '@/lib/mcp-dashboard-oauth'
import { MCP_DIRECTORY } from '@/lib/mcp-directory'
import { prettyName } from '@/lib/text'
import { type ComposerSuggestion, registerDraftProvider } from '@/store/composer-suggestions'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'

/**
 * The MCP draft provider — the suggestion bus's founding member (PR #85036).
 *
 * Matches the draft against the Nous-approved MCP catalog's `suggest`
 * metadata (`GET /api/mcp/catalog` — the same reviewed manifests behind
 * `hermes mcp catalog`), by whole-word keyword and pasted-link host suffix,
 * excluding servers already configured. The catalog is the single source of
 * truth for suggestible servers; the renderer-local `lib/mcp-directory.ts`
 * remains only as a compatibility rung for older backends whose catalog
 * entries carry no `suggest` field. A suggestion's invoke runs the whole
 * connect: validated config write → browser OAuth → live tool reload, with
 * rollback on cancel/failure so a decline never strands a half-configured
 * server.
 */

const CONFIGURED_TTL_MS = 5 * 60_000
const CATALOG_TTL_MS = 5 * 60_000

// Names already present in mcp_servers config (enabled or not) — those need a
// toggle/auth at most, not an "add this server" pill. Cached briefly; a miss
// (older backend, transient error) suggests nothing rather than nagging.
let configuredNames: Set<string> | null = null
let configuredAt = 0

interface SuggestibleServer {
  server: string
  keywords: string[]
  hosts?: string[]
  /** Streamable-HTTP/SSE endpoint written to config on invoke. */
  url: string
}

// Suggestible servers from the catalog (entries with `suggest` + an http
// url), or the static directory on backends that predate `suggest`.
let suggestible: SuggestibleServer[] | null = null
let suggestibleAt = 0

/** Drop the caches (profile switch / after an install). */
export function invalidateMcpSuggestionIndex(): void {
  configuredNames = null
  configuredAt = 0
  suggestible = null
  suggestibleAt = 0
}

async function loadConfiguredNames(): Promise<Set<string>> {
  if (configuredNames && Date.now() - configuredAt < CONFIGURED_TTL_MS) {
    return configuredNames
  }

  const { servers } = await listMcpServers()

  configuredNames = new Set(servers.map(server => server.name))
  configuredAt = Date.now()

  return configuredNames
}

async function loadSuggestible(): Promise<SuggestibleServer[]> {
  if (suggestible && Date.now() - suggestibleAt < CATALOG_TTL_MS) {
    return suggestible
  }

  const { entries } = await getMcpCatalog()

  const fromCatalog: SuggestibleServer[] = entries
    .filter(entry => entry.suggest && entry.url && (entry.suggest.keywords.length > 0 || entry.suggest.hosts.length > 0))
    .map(entry => ({
      hosts: entry.suggest!.hosts,
      keywords: entry.suggest!.keywords,
      server: entry.name,
      url: entry.url!
    }))

  // Compatibility rung: an older backend serves the catalog without any
  // `suggest` metadata. Fall back to the static directory rather than
  // silently losing the feature (remove once the backend contract bumps).
  suggestible =
    fromCatalog.length > 0
      ? fromCatalog
      : MCP_DIRECTORY.map(entry => ({
          hosts: entry.hosts,
          keywords: entry.keywords,
          server: entry.name,
          url: entry.url
        }))
  suggestibleAt = Date.now()

  return suggestible
}

interface KeywordEntry {
  server: string
  keywords: string[]
  /** Hostname suffixes ("atlassian.net") matched against URLs in the draft. */
  hosts?: string[]
}

// Hostnames of http(s) URLs in the draft. Loose on purpose — a draft is not
// a document, so a trailing-punctuation host ("linear.app,") still counts.
const URL_HOST_RE = /https?:\/\/([^\s/,)\]}"'<>]+)/gi

const draftHosts = (text: string): string[] =>
  [...text.matchAll(URL_HOST_RE)].map(match => {
    const raw = match[1]!.toLowerCase()
    // Strip credentials and port: user@host:443 → host.
    const withoutCredentials = raw.slice(raw.lastIndexOf('@') + 1)

    return withoutCredentials.replace(/:\d+$/, '')
  })

// Strict suffix-on-dot-boundary: "myorg.atlassian.net" matches "atlassian.net";
// "notlinear.app" and "linear.app.example.com" do not match "linear.app".
const hostMatches = (host: string, suffix: string): boolean => host === suffix || host.endsWith(`.${suffix}`)

export interface McpMatch {
  server: string
  /** The keyword or host that matched, for the pill's tooltip. */
  keyword: string
}

const MAX_MATCHES = 2

// Whole-word (unicode-aware) keyword hit that the user has FINISHED typing:
// at least one character must follow the match (the lookahead already
// guarantees it's a boundary). A hit still under the caret — "figma" as the
// last thing typed, debounce elapsed mid-thought — is not intent yet, it's
// eavesdropping on a word in progress; the pill waits for the space/period.
const keywordHit = (haystack: string, candidate: string): boolean => {
  const pattern = new RegExp(
    `(?<![\\p{L}\\p{N}])${candidate.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![\\p{L}\\p{N}])`,
    'gu'
  )

  for (const match of haystack.matchAll(pattern)) {
    if (match.index + match[0].length < haystack.length) {
      return true
    }
  }

  return false
}

/** Pure matcher, exported for tests: pasted-link host hits (the strongest
 *  intent signal) and completed whole-word keyword hits against the draft,
 *  capped at MAX_MATCHES. Host hits skip the completed-word guard — a paste
 *  is a deliberate act, and the URL routinely ends the draft. */
export function matchSuggestions(text: string, index: KeywordEntry[]): McpMatch[] {
  const haystack = text.toLowerCase()
  const hosts = draftHosts(text)
  const matches: McpMatch[] = []

  for (const entry of index) {
    // A pasted vendor link beats any keyword: report the host as the trigger.
    const host = entry.hosts?.find(suffix => hosts.some(candidate => hostMatches(candidate, suffix)))

    // Whole-word match so "linearly" doesn't suggest Linear. Suggest
    // keywords are lowercase; multi-word keywords match as phrases.
    const keyword = host ?? entry.keywords.find(candidate => keywordHit(haystack, candidate))

    if (keyword) {
      matches.push({ keyword, server: entry.server })

      if (matches.length >= MAX_MATCHES) {
        break
      }
    }
  }

  return matches
}

async function connect(known: SuggestibleServer, sessionId: string | null, cancelled: () => boolean): Promise<void> {
  try {
    await addMcpServer({ name: known.server, url: known.url })

    try {
      await completeMcpDesktopOAuth({
        serverName: known.server,
        start: authMcpServer,
        status: getMcpOAuthFlow,
        cancelled,
        cancel: cancelMcpOAuthFlow,
        openExternal: url => window.hermesDesktop.openExternal(url)
      })
    } catch (error) {
      // Decline/failure means "no server" — roll back the config write
      // rather than stranding an unauthorized entry (authoritative-write
      // rule). Best-effort; the primary error wins.
      await removeMcpServer(known.server).catch(() => {})
      throw error
    }

    // Tools reach the live session before the pill claims success — the
    // same write-through the Capabilities tab and the setup card use.
    await $gateway
      .get()
      ?.request('reload.mcp', { confirm: true, session_id: sessionId ?? undefined })
      .catch(() => {})

    invalidateMcpSuggestionIndex()
  } catch (error) {
    if (!(error instanceof McpOAuthCancelled)) {
      notifyError(error, translateNow('composer.mcpSuggestions.connectFailed', prettyName(known.server)))
    }

    throw error
  }
}

function toSuggestion(match: McpMatch, known: SuggestibleServer, sessionId: string | null): ComposerSuggestion {
  const name = prettyName(match.server)
  const copy = (key: string, ...args: unknown[]) => translateNow(`composer.mcpSuggestions.${key}`, ...args)

  return {
    brand: match.server,
    doneLabel: copy('added', name),
    doneTip: copy('addedTip'),
    id: match.server,
    // The pill's session wins over the one captured at sample time: the reload
    // has to reach the session the user is actually looking at.
    invoke: context => connect(known, context.sessionId ?? sessionId, context.cancelled),
    label: copy('label', name),
    provider: 'mcp',
    tip: copy('tip', match.keyword),
    workingLabel: copy('connecting', name),
    workingTip: copy('cancelTip')
  }
}

registerDraftProvider('mcp', async ({ sessionId, text }) => {
  // Catalog unreachable — suggest nothing rather than mis-suggest.
  const index = await loadSuggestible()
  const candidates = matchSuggestions(text, index)

  // Fast path: no keyword hit at all → skip the servers fetch.
  if (candidates.length === 0) {
    return []
  }

  const configured = await loadConfiguredNames()
  const byName = new Map(index.map(entry => [entry.server, entry]))

  return candidates
    .filter(candidate => !configured.has(candidate.server))
    .map(match => toSuggestion(match, byName.get(match.server)!, sessionId))
})
