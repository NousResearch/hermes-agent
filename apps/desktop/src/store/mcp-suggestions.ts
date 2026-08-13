import { atom } from 'nanostores'

import { listMcpServers } from '@/hermes'
import { MCP_DIRECTORY } from '@/lib/mcp-directory'

/**
 * Keyword-triggered MCP suggestions for the composer pill strip.
 *
 * Source: the desktop's own suggestion directory (`lib/mcp-directory.ts`) —
 * NOT the Nous install catalog, whose contents are a reviewed trust boundary
 * we don't grow from the renderer. While the user types, the composer samples
 * the draft (debounced — never per keystroke) and this store matches it
 * against directory keywords and pasted-link hosts, excluding servers already
 * configured in `mcp_servers`. Matches surface as pills above the composer; a
 * pill click drafts a setup request, and the agent's `setup_mcp` tool takes
 * it from there with the inline consent card.
 *
 * The pills are self-limiting rather than dismissible: they only exist while
 * a trigger is in the draft, vanish once the server is configured, and cap at
 * MAX_SUGGESTIONS — so there is deliberately no per-server opt-out state.
 */
export interface McpSuggestion {
  server: string
  /** The keyword or host that matched, for the pill's tooltip. */
  keyword: string
}

const SAMPLE_DEBOUNCE_MS = 600
const CONFIGURED_TTL_MS = 5 * 60_000
const MAX_SUGGESTIONS = 2

/**
 * Suggestions keyed by RUNTIME session id, exactly like
 * `$composerActionsBySession`: drafts are per-session state, so the pills
 * computed from a draft are too. A single global slot made whichever session
 * sampled last leak its pills into every other tab (#draft-restore re-samples
 * on switch), which read as "Add GitHub" following you around the app.
 */
export const $mcpSuggestionsBySession = atom<Record<string, McpSuggestion[]>>({})

const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

const sameSuggestions = (a: readonly McpSuggestion[], b: readonly McpSuggestion[]) =>
  a.length === b.length && a.every((x, i) => x.server === b[i]!.server && x.keyword === b[i]!.keyword)

function setSuggestions(sessionId: string | null, suggestions: McpSuggestion[]): void {
  const key = keyFor(sessionId)
  const current = $mcpSuggestionsBySession.get()
  const existing = current[key] ?? []

  // Unchanged sets keep their reference so the strip doesn't re-render.
  if (sameSuggestions(existing, suggestions)) {
    return
  }

  const next = { ...current }

  if (suggestions.length > 0) {
    next[key] = suggestions
  } else {
    delete next[key]
  }

  $mcpSuggestionsBySession.set(next)
}

interface KeywordEntry {
  server: string
  keywords: string[]
  /** Hostname suffixes ("atlassian.net") matched against URLs in the draft. */
  hosts?: string[]
}

// Names already present in mcp_servers config (enabled or not) — those need a
// toggle/auth at most, not a "add this server" pill. Cached briefly; a miss
// (older backend, transient error) suggests nothing rather than nagging.
let configuredNames: Set<string> | null = null
let configuredAt = 0

/** Drop the configured-servers cache (profile switch / after an install). */
export function invalidateMcpSuggestionIndex(): void {
  configuredNames = null
  configuredAt = 0
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

// Hostnames of http(s) URLs in the draft. Loose on purpose — a draft is not
// a document, so a trailing-punctuation host ("linear.app,") still counts.
const URL_HOST_RE = /https?:\/\/([^\s/,)\]}"'<>]+)/gi

function draftHosts(text: string): string[] {
  const hosts: string[] = []

  for (const match of text.matchAll(URL_HOST_RE)) {
    const host = match[1]?.split('@').pop()?.split(':')[0]?.toLowerCase()

    if (host) {
      hosts.push(host)
    }
  }

  return hosts
}

const hostMatches = (host: string, suffix: string) => host === suffix || host.endsWith(`.${suffix}`)

/** Pure matcher, exported for tests: pasted-link host hits (the strongest
 *  intent signal) and whole-word (unicode-aware) keyword hits against the
 *  draft, capped at MAX_SUGGESTIONS. */
export function matchSuggestions(text: string, index: KeywordEntry[]): McpSuggestion[] {
  const haystack = ` ${text.toLowerCase()} `
  const hosts = draftHosts(text)
  const matches: McpSuggestion[] = []

  for (const entry of index) {
    // A pasted vendor link beats any keyword: report the host as the trigger.
    const host = entry.hosts?.find(suffix => hosts.some(candidate => hostMatches(candidate, suffix)))

    // Whole-word match so "linearly" doesn't suggest Linear. Directory
    // keywords are lowercase; multi-word keywords match as phrases.
    const keyword =
      host ??
      entry.keywords.find(candidate =>
        new RegExp(
          `(?<![\\p{L}\\p{N}])${candidate.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![\\p{L}\\p{N}])`,
          'u'
        ).test(haystack)
      )

    if (keyword) {
      matches.push({ keyword, server: entry.server })

      if (matches.length >= MAX_SUGGESTIONS) {
        break
      }
    }
  }

  return matches
}

// Per-session debounce/generation so a tile composer's sampling never stomps
// the primary's (each session settles independently).
const sampleTimers = new Map<string, number>()
const sampleGenerations = new Map<string, number>()

/**
 * Feed a session's draft snapshot into the matcher. Called from that
 * composer's runtime subscription on every change, but internally debounced
 * and change-gated: the store only writes when the session's matched set
 * actually differs, so typing within a line costs nothing downstream.
 */
export function sampleComposerDraftForMcpSuggestions(sessionId: string | null | undefined, text: string): void {
  const key = keyFor(sessionId)

  window.clearTimeout(sampleTimers.get(key))

  const generation = (sampleGenerations.get(key) ?? 0) + 1
  sampleGenerations.set(key, generation)

  // Too short to mean anything — clear instead of hitting the matcher.
  if (text.trim().length < 3) {
    setSuggestions(sessionId ?? null, [])

    return
  }

  sampleTimers.set(
    key,
    window.setTimeout(() => {
      // Fast path: no keyword hit at all → clear without touching the network.
      const index = MCP_DIRECTORY.map(entry => ({ hosts: entry.hosts, keywords: entry.keywords, server: entry.name }))
      const candidates = matchSuggestions(text, index)

      if (candidates.length === 0) {
        setSuggestions(sessionId ?? null, [])

        return
      }

      void loadConfiguredNames()
        .then(configured => {
          // A newer sample for THIS session superseded this one mid-load.
          if (generation !== sampleGenerations.get(key)) {
            return
          }

          setSuggestions(
            sessionId ?? null,
            candidates.filter(candidate => !configured.has(candidate.server))
          )
        })
        .catch(() => {
          // Server list unreachable — suggest nothing rather than mis-suggest.
        })
    }, SAMPLE_DEBOUNCE_MS)
  )
}

/** Drop a session's pills outright (composer unmount / session close). */
export function clearMcpSuggestions(sessionId: string | null | undefined): void {
  window.clearTimeout(sampleTimers.get(keyFor(sessionId)))
  setSuggestions(sessionId ?? null, [])
}
