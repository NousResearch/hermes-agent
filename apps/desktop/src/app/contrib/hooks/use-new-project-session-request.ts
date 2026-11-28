import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { $newProjectSessionRequest } from '@/store/projects'

import type { useSessionActions } from '../../session/hooks/use-session-actions'

type OpenNewSessionTile = ReturnType<typeof useSessionActions>['openNewSessionTile']

/**
 * "New project" DRAG completion: the dialog created a project that was dropped onto a chat zone
 * (tab-strip slot / pane edge / pane center). Open its fresh session draft exactly there — the same
 * `openNewSessionTile` create path the new-session drags use — so the project starts, and stays,
 * where it was dropped. Consume-once: the request is dropped once handled.
 */
export function useNewProjectSessionRequest(openNewSessionTile: OpenNewSessionTile): void {
  const request = useStore($newProjectSessionRequest)

  useEffect(() => {
    if (!request) {
      return
    }

    const { path, placement } = request

    $newProjectSessionRequest.set(null)
    void openNewSessionTile(placement.dir, {
      anchor: placement.anchor,
      before: placement.before,
      cwd: path,
      // Same draft-tab contract as onNewSessionSplit: a center/strip drop is an unlisted draft tab
      // until its first turn; an edge split lists.
      listed: placement.dir === 'center' ? false : undefined
    })
  }, [openNewSessionTile, request])
}
