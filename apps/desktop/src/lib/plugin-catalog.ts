/**
 * The curated Hermes plugin catalog as the Desktop sees it.
 *
 * `PLUGIN_CATALOG_URL` (`/docs/api/plugins.json`, generated from the repo's
 * `plugin-catalog/` directory at site build) is the catalog the Capabilities →
 * Plugins card catalog browses. A `hermes://plugin/install?catalog=<name>`
 * deep link resolves against the SAME document so a link and an in-app pick
 * always agree on the repo, subdir and reviewed pin for a name.
 *
 * The lookup asks the backend (`plugins.catalog.lookup`), which reads the live
 * catalog with its cache, kill list and in-tree fallback: the same catalog the
 * install resolves its pin from. A renderer fetch of the JSON also depends on
 * the docs host's cross-origin redirect carrying a CORS header, which it did
 * not until 2026-10-13 (hermes-agent 136483). Only a backend that predates the
 * method falls back to the direct fetch.
 *
 * Everything here only classifies — nothing installs. Callers open the Install
 * Plugin dialog, which still requires the user's explicit confirmation.
 */

import type { ConnectionState } from '@hermes/shared'
import type { ReadableAtom } from 'nanostores'

import { isMissingRpcMethod } from '@/lib/gateway-rpc'
import { BACKEND_BOOT_WAIT_TIMEOUT_MS, withTimeout } from '@/lib/with-timeout'
import { activeGatewayProfileKey, requestGatewayForProfile } from '@/store/gateway'
import { $gatewayState } from '@/store/session'

export const CATALOG_ORIGIN = 'https://hermes-agent.nousresearch.com'
export const PLUGIN_CATALOG_URL = `${CATALOG_ORIGIN}/docs/api/plugins.json`
export const PLUGIN_CATALOG_DOCS_URL = `${CATALOG_ORIGIN}/docs/user-guide/features/plugin-catalog`
export const SKILLS_DOCS_URL = `${CATALOG_ORIGIN}/docs/skills/`
export const SUBMIT_MOD_URL = 'https://github.com/NousResearch/hermes-agent/compare'

/** Catalog names are directory names under `plugin-catalog/`; anything else is not a lookup key. */
export const PLUGIN_CATALOG_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/

const FETCH_TIMEOUT_MS = 15_000
const LOOKUP_TIMEOUT_MS = 20_000

/** The fields an install needs — the same ones the embedded picker posts on a pick. */
export interface PluginCatalogEntry {
  name: string
  repo: string
  /** Reviewed pin (40-hex) the backend installs at; may be missing on a malformed feed. */
  sha?: string
  /** Sub-directory of a monorepo the plugin lives in; empty when the repo root is the plugin. */
  subdir?: string
  /** The description's first sentence: what the plugin does, in one line. */
  summary?: string
  /** official / verified / community; badges the install dialog. */
  tier?: string
}

export type PluginCatalogLookupError = 'invalid_name' | 'unavailable' | 'unknown'

export type PluginCatalogLookup =
  { ok: false; error: PluginCatalogLookupError } | { ok: true; entry: PluginCatalogEntry }

/** Mirrors `plugin_catalog_presence._first_sentence`, so the dialog and the setup card agree. */
function firstSentence(text: string): string | undefined {
  const flat = text.split(/\s+/).filter(Boolean).join(' ')
  const end = flat.indexOf('. ')

  return (end >= 0 ? flat.slice(0, end + 1) : flat) || undefined
}

function asEntry(raw: unknown): null | PluginCatalogEntry {
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const row = raw as Record<string, unknown>

  // A field that is not a string reads as absent.
  const text = (field: 'description' | 'name' | 'repo' | 'sha' | 'subdir' | 'tier'): string | undefined => {
    const value = row[field]

    return typeof value === 'string' ? value : undefined
  }

  const name = text('name')
  const repo = text('repo')

  if (name === undefined || !repo) {
    return null
  }

  return {
    name,
    repo,
    sha: text('sha') || undefined,
    subdir: text('subdir') || undefined,
    summary: firstSentence(text('description') ?? ''),
    tier: text('tier') || undefined
  }
}

/** What `plugins.catalog.lookup` answers; `entry` is parsed by `asEntry`, so a malformed one degrades. */
export interface CatalogLookupReply {
  entry?: unknown
  status?: string
}

/** The lookup RPC for one name, as `requestGatewayForProfile` sends it for the active profile. */
export type CatalogRequest = (
  method: 'plugins.catalog.lookup',
  params: { name: string }
) => Promise<CatalogLookupReply | null | undefined>

/**
 * Run `send` once the active gateway is open. A link that launches Desktop is
 * delivered as soon as the renderer mounts, before the backend socket exists,
 * so sending at once fails before anything reaches the backend. The wait shares
 * the renderer's cold-boot budget; past it the caller gets a rejection, never a
 * promise that stays pending.
 */
export function sendWhenGatewayOpen<T>(
  send: () => Promise<T>,
  {
    state = $gatewayState,
    timeoutMs = BACKEND_BOOT_WAIT_TIMEOUT_MS
  }: {
    state?: ReadableAtom<ConnectionState>
    timeoutMs?: number
  } = {}
): Promise<T> {
  if (state.get() === 'open') {
    return send()
  }

  let stop = () => {}

  const opened = new Promise<void>(resolve => {
    stop = state.listen(next => {
      if (next === 'open') {
        stop()
        resolve()
      }
    })
  })

  return withTimeout(opened, timeoutMs, 'Hermes backend did not connect', () => stop()).then(send)
}

// The route lookup inside requestGatewayForProfile has no deadline of its own, so the
// whole call is bounded here, not just the RPC.
const activeProfileRequest: CatalogRequest = (method, params) =>
  sendWhenGatewayOpen(() =>
    withTimeout(
      requestGatewayForProfile<CatalogLookupReply>(activeGatewayProfileKey(), method, params, LOOKUP_TIMEOUT_MS),
      LOOKUP_TIMEOUT_MS,
      `request timed out: ${method}`
    )
  )

async function lookupFromFeed(name: string, fetchImpl: typeof fetch): Promise<PluginCatalogLookup> {
  let rows: unknown

  try {
    const response = await fetchImpl(PLUGIN_CATALOG_URL, {
      cache: 'no-cache',
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS)
    })

    if (!response.ok) {
      return { ok: false, error: 'unavailable' }
    }

    rows = (await response.json()) as unknown
  } catch {
    return { ok: false, error: 'unavailable' }
  }

  if (!Array.isArray(rows)) {
    return { ok: false, error: 'unavailable' }
  }

  // Catalog names are directory names and the CLI treats them case-sensitively.
  const match = rows.map(asEntry).find(entry => entry?.name === name)

  return match ? { ok: true, entry: match } : { ok: false, error: 'unknown' }
}

/**
 * Look one name up in the live catalog. Unknown names are a hard `unknown`,
 * never a guess: the deep link must not turn an arbitrary string into a git
 * identifier the dialog would then clone. The backend answers; only a backend
 * older than the method falls back to the direct fetch.
 */
export async function lookupPluginCatalogEntry(
  name: string,
  {
    fetchImpl = (...args) => fetch(...args),
    request = activeProfileRequest
  }: { fetchImpl?: typeof fetch; request?: CatalogRequest } = {}
): Promise<PluginCatalogLookup> {
  const trimmed = name.trim()

  if (!PLUGIN_CATALOG_NAME_RE.test(trimmed)) {
    return { ok: false, error: 'invalid_name' }
  }

  let reply: CatalogLookupReply | null | undefined

  try {
    reply = await request('plugins.catalog.lookup', { name: trimmed })
  } catch (error) {
    // Only a backend that predates the method falls back; an unreachable one can't run the install either.
    return isMissingRpcMethod(error) ? lookupFromFeed(trimmed, fetchImpl) : { ok: false, error: 'unavailable' }
  }

  // A malformed success reply (null, no status) degrades like every other failure.
  const status = reply?.status

  if (status === 'found') {
    const entry = asEntry(reply?.entry)

    return entry ? { ok: true, entry } : { ok: false, error: 'unavailable' }
  }

  return {
    ok: false,
    error: status === 'unknown' || status === 'invalid_name' ? status : 'unavailable'
  }
}
