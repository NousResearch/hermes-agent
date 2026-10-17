import { normalizeRemoteBaseUrl } from './connection-config'
import type { RemotePluginRegistry } from './remote-plugin-registry'

/**
 * Effective base url for a saved remote: a remote-plugin-backed url dials first
 * to its loopback access point; a plain https url normalizes unchanged.
 */
export async function resolveRemoteBaseUrl(
  registry: RemotePluginRegistry,
  rawUrl: string,
  log: (msg: string) => void
): Promise<string> {
  const dialed = await registry.ensure(rawUrl, log)

  return normalizeRemoteBaseUrl(dialed ? dialed.baseUrl : rawUrl)
}
