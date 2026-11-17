import type { PreviewSigninsResult } from '@/global'

/** The (connection, profile) a config record was read from: the same scope its toggle writes. */
export interface PreviewSigninsScope {
  connectionId?: null | string
  profile?: null | string
}

// Last `enabled` synced per scope, shared by the boot hook and the explicit Settings actions, so a toggle
// the panel already synced is not imported a second time when the cache update reaches the hook.
const lastSynced = new Map<string, boolean>()

export const previewSigninsScopeKey = (scope: PreviewSigninsScope | undefined) =>
  `${(scope?.connectionId ?? '').trim()}::${(scope?.profile ?? '').trim() || 'default'}`

/** Ask Electron main to sign the in-app browser in (or out) for `scope`. Main re-checks that the backend is
 *  local and the backend re-checks consent; the reply carries counts only. Null off-Desktop. */
export async function syncPreviewSignins(
  enabled: boolean,
  scope: PreviewSigninsScope | undefined
): Promise<null | PreviewSigninsResult> {
  const bridge = window.hermesDesktop?.syncPreviewSignins

  if (!bridge) {
    return null
  }

  const result = await bridge({ connectionId: scope?.connectionId ?? null, enabled, profile: scope?.profile ?? null })

  // P2-8 (renderer): only mark the scope synced AFTER a successful reply. Stamping before the bridge
  // resolves (or on a `status: 'error'` reply) marked a failed import/clear as done, so the boot hook
  // skipped the real retry. A failure leaves `lastSynced` untouched so the next `syncPreviewSigninsIfChanged`
  // tries again.
  if (result && result.status !== 'error') {
    lastSynced.set(previewSigninsScopeKey(scope), enabled)
  }

  return result
}

/** A Settings toggle / consent prompt is about to write `enabled` optimistically into the config cache: claim
 *  it so the boot hook does not race the save (the backend would still read the old value), then the caller
 *  syncs explicitly once the save has landed. */
export function claimPreviewSigninsSync(enabled: boolean, scope: PreviewSigninsScope | undefined) {
  lastSynced.set(previewSigninsScopeKey(scope), enabled)
}

/** Sync only when `enabled` differs from what this window last synced for `scope` (boot, external edits). */
export function syncPreviewSigninsIfChanged(enabled: boolean, scope: PreviewSigninsScope | undefined) {
  if (lastSynced.get(previewSigninsScopeKey(scope)) === enabled) {
    return null
  }

  return syncPreviewSignins(enabled, scope)
}
