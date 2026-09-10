import type { HermesConnection } from '@/global'
import { desktopDefaultCwd } from '@/lib/desktop-fs'
import {
  $activeSessionId,
  $currentCwd,
  $newChatWorkspaceTarget,
  ensureDefaultWorkspaceCwd,
  setCurrentBranch,
  setCurrentCwd
} from '@/store/session'

// Seed the working dir on a fresh view (nothing open yet): the remembered
// per-(connection, profile) workspace first (ensureDefaultWorkspaceCwd), then
// the backend's own default (/api/fs/default-cwd on remote). Shared by boot,
// the soft connection switch, and the post-profile-swap reseed in wiring.
// `shouldPublish` lets a superseded caller bail before touching live state.
export async function seedDefaultCwd(shouldPublish: () => boolean = () => true): Promise<void> {
  await ensureDefaultWorkspaceCwd(shouldPublish)

  if (!shouldPublish()) {
    return
  }

  const remoteDefault = await desktopDefaultCwd().catch(() => null)

  if (shouldPublish() && remoteDefault?.cwd && !$activeSessionId.get() && !$currentCwd.get()) {
    setCurrentCwd(remoteDefault.cwd)
    setCurrentBranch(remoteDefault.branch || '')
  }
}

// A profile switch drops to a fresh draft BEFORE the swapped connection is
// published (selectProfile fires requestFreshSession synchronously; the
// descriptor lands once the swap resolves). A REMOTE draft therefore resolved
// its cwd under the OUTGOING profile's remembered workspace — the key is
// per-(host, profile) — which is blank whenever that profile never had one, and
// nothing reseeded it afterwards; boot and the soft connection switch both do.
// Local drafts are unaffected: a bare local chat is detached by design and
// every local profile shares one remembered key, so the lookup can't drift.
// True when the draft on `connection` should be reseeded the way boot seeds it.
export function draftNeedsReseed(connection: HermesConnection | null | undefined): boolean {
  return (
    connection?.mode === 'remote' &&
    !$activeSessionId.get() &&
    !$currentCwd.get().trim() &&
    $newChatWorkspaceTarget.get() === undefined
  )
}
