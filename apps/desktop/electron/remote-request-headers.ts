import { modeIsRemoteLike, remoteRequestMatchesBaseUrl } from './connection-config'

type ConnectionRegistry = {
  connections?: Array<{ headers?: unknown; id?: unknown; kind?: unknown; url?: unknown }>
  primary?: unknown
}

/**
 * Select the stored header envelopes that authorize a REST request to the
 * active remote gateway. Registry-primary remotes are intentionally a
 * fallback: a v1 remote setting remains the authoritative route whenever it
 * is configured, even if a different registry entry has the same origin.
 */
export function headersForRemoteRequest(
  requestUrl: string,
  config: { mode?: unknown; remote?: { headers?: unknown; url?: unknown } },
  registry: ConnectionRegistry
): unknown {
  if (modeIsRemoteLike(config.mode) && config.remote?.url) {
    return remoteRequestMatchesBaseUrl(requestUrl, config.remote.url) ? config.remote.headers : {}
  }

  const primaryId = String(registry?.primary || '').trim()
  const primary = registry?.connections?.find(entry => entry.id === primaryId)

  if (
    primary &&
    (primary.kind === 'remote' || primary.kind === 'cloud') &&
    primary.url &&
    remoteRequestMatchesBaseUrl(requestUrl, primary.url)
  ) {
    return primary.headers
  }

  return {}
}
