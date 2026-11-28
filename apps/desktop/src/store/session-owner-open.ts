import { translateNow } from '@/i18n'
import { notify } from '@/store/notifications'

import type { SessionProfileRoute } from './session-request-router'
import { sessionOwnerOpenable } from './session-states'

/** The user's open of a chat: `sessionOwnerOpenable`, telling them why when it refuses. */
export function canOpenSessionOwner(storedSessionId: string, owner?: SessionProfileRoute): boolean {
  if (sessionOwnerOpenable(storedSessionId, owner)) {
    return true
  }

  notify({
    kind: 'error',
    title: translateNow('desktop.sessionOwnerConflictTitle'),
    message: translateNow('desktop.sessionOwnerConflictBody')
  })

  return false
}
