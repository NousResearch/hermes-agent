// Opening a folder is the IDE's one workspace gesture: the app's native
// directory dialog picks it, and the choice becomes the IDE workspace — the
// explorer root and the cwd every new IDE session anchors at. The last choice
// persists per window (see state.ts for the seed/persist precedence).

import { notifyError } from '@/store/notifications'

import { setIdeWorkspaceRoot } from './state'

export async function openIdeFolder(failureMessage: string): Promise<null | string> {
  const bridge = window.hermesDesktop

  if (!bridge?.selectPaths) {
    return null
  }

  try {
    const paths = await bridge.selectPaths({ directories: true })
    const chosen = paths?.[0]?.trim()

    if (!chosen) {
      return null
    }

    setIdeWorkspaceRoot(chosen)

    return chosen
  } catch (error) {
    notifyError(error, failureMessage)

    return null
  }
}
