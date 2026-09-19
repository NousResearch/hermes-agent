import { type Codec, persistentAtom } from '@/lib/persisted'
import { ideSeedCwd } from '@/store/windows'

export const IDE_WORKSPACE_STORAGE_KEY = 'hermes.desktop.ideWorkspace.v1'

const codec: Codec<null | string> = {
  decode: raw => (raw && raw !== 'null' ? raw : null),
  encode: value => (value === null ? null : value)
}

const seeded = ideSeedCwd()

/**
 * The workspace root the IDE is showing: the explorer rail, the status bar, and
 * every new IDE session's cwd. The opener's seed wins on boot — the window URL
 * carries the cwd the IDE was opened from — while a folder the user picked
 * inside the IDE persists for boots that arrive without a seed. Window-scoped:
 * this atom lives only in the IDE renderer.
 */
export const $ideWorkspaceRoot = persistentAtom<null | string>(IDE_WORKSPACE_STORAGE_KEY, seeded, codec)

if (seeded && $ideWorkspaceRoot.get() !== seeded) {
  $ideWorkspaceRoot.set(seeded)
}

export function setIdeWorkspaceRoot(root: null | string | undefined) {
  $ideWorkspaceRoot.set(root?.trim() || null)
}

/** The trailing path segment of a workspace root ("D:\\apps\\COAI" → "COAI"). */
export function workspaceBasename(root: null | string): null | string {
  if (!root) {
    return null
  }

  return root.split(/[\\/]/).filter(Boolean).pop() ?? root
}
