import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $connection } from '@/store/session'
import { knownOwnerForSession } from '@/store/session-states'

/** Ambient preview IPC/FS bridges can only read the active connection/profile.
 * A visible tile may belong to a different backend; never externalize its
 * file:// path (or resolve its localhost) through the wrong machine. */
export function previewOwnerIsAmbient(sessionId: null | string | undefined): boolean {
  if (!sessionId) {
    return true // unsent draft in the active connection
  }

  const owner = knownOwnerForSession(sessionId)

  if (!owner) {
    return sessionId === $activeSessionId.get()
  }

  const activeProfile = $activeGatewayProfile.get()

  if (typeof owner === 'string') {
    return owner === activeProfile
  }

  const connection = $connection.get()
  const connectionId = connection?.connectionId || (connection?.mode === 'local' ? 'local' : '')

  return Boolean(connectionId && owner.connectionId === connectionId && owner.profile === activeProfile)
}
