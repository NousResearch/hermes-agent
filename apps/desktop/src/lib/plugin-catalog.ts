/**
 * The curated Hermes plugin catalog as the Desktop sees it.
 *
 * The Capabilities → Plugins tab embeds the docs-site catalog page
 * (`CATALOG_PICKER_URL`) as a one-click picker; that page renders
 * `PLUGIN_CATALOG_URL` (`/docs/api/plugins.json`, generated from the repo's
 * `plugin-catalog/` directory at site build). A `hermes://plugin/install?catalog=<name>`
 * deep link resolves against the SAME document so a link and an in-app pick
 * always agree on the repo, subdir and reviewed pin for a name.
 *
 * Everything here only classifies — nothing installs. Callers open the Install
 * Plugin dialog, which still requires the user's explicit confirmation.
 */

export const CATALOG_ORIGIN = 'https://hermes-agent.nousresearch.com'
export const CATALOG_PICKER_URL = `${CATALOG_ORIGIN}/docs/plugins?embed=picker`
export const PLUGIN_CATALOG_URL = `${CATALOG_ORIGIN}/docs/api/plugins.json`

/** Catalog names are directory names under `plugin-catalog/`; anything else is not a lookup key. */
export const PLUGIN_CATALOG_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/

const FETCH_TIMEOUT_MS = 15_000

/** The fields an install needs — the same ones the embedded picker posts on a pick. */
export interface PluginCatalogEntry {
  name: string
  repo: string
  /** Reviewed pin (40-hex) the backend installs at; may be missing on a malformed feed. */
  sha?: string
  /** Sub-directory of a monorepo the plugin lives in; empty when the repo root is the plugin. */
  subdir?: string
}

export type PluginCatalogLookupError = 'invalid_name' | 'unavailable' | 'unknown'

export type PluginCatalogLookup =
  { ok: false; error: PluginCatalogLookupError } | { ok: true; entry: PluginCatalogEntry }

function asEntry(raw: unknown): null | PluginCatalogEntry {
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const row = raw as Record<string, unknown>

  if (typeof row.name !== 'string' || typeof row.repo !== 'string' || !row.repo) {
    return null
  }

  return {
    name: row.name,
    repo: row.repo,
    sha: typeof row.sha === 'string' && row.sha ? row.sha : undefined,
    subdir: typeof row.subdir === 'string' && row.subdir ? row.subdir : undefined
  }
}

/**
 * Look one name up in the live catalog. Unknown names are a hard `unknown`,
 * never a guess: the deep link must not turn an arbitrary string into a git
 * identifier the dialog would then clone.
 */
export async function lookupPluginCatalogEntry(
  name: string,
  fetchImpl: typeof fetch = (...args) => fetch(...args)
): Promise<PluginCatalogLookup> {
  const trimmed = name.trim()

  if (!PLUGIN_CATALOG_NAME_RE.test(trimmed)) {
    return { ok: false, error: 'invalid_name' }
  }

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
  const match = rows.map(asEntry).find(entry => entry?.name === trimmed)

  return match ? { ok: true, entry: match } : { ok: false, error: 'unknown' }
}
