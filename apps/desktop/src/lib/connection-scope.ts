import type { HermesConnection } from '@/global'

function normalizedRemoteTarget(connection: HermesConnection): string {
  if (connection.remoteKind === 'ssh') {
    return (connection.remoteIdentity || connection.remoteHost || connection.baseUrl || 'remote').trim()
  }

  const raw = (connection.baseUrl || connection.remoteIdentity || connection.remoteHost || 'remote').trim()

  try {
    const url = new URL(raw)
    url.username = ''
    url.password = ''
    url.search = ''
    url.hash = ''
    url.pathname = url.pathname.replace(/\/+$/, '')

    return url.toString().replace(/\/$/, '')
  } catch {
    return raw.replace(/\/+$/, '')
  }
}

/** Stable identity for renderer state that belongs to one backend connection. */
export function desktopConnectionScope(connection: HermesConnection | null): string | null {
  if (!connection) {
    return null
  }

  const mode = connection.mode || 'local'
  const profile = connection.profile?.trim() || 'default'

  if (mode !== 'remote') {
    return `local:${profile}`
  }

  return `remote:${connection.remoteKind || 'url'}:${profile}:${normalizedRemoteTarget(connection)}`
}
