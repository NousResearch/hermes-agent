import { useStore } from '@nanostores/react'
import { computed } from 'nanostores'
import { useMemo, useRef } from 'react'

import { useHermesConfigRecord } from '@/app/hooks/use-config-record'
import { getNested } from '@/app/settings/helpers'
import { useI18n } from '@/i18n'
import { selectDesktopPaths } from '@/lib/desktop-fs'
import {
  $codingWorkspaceDrafts, codingWorkspaceDraftKey, codingWorkspaceKey,
  type CodingWorkspaceOwner, enableCodingWorkspaceControls, registerCodingWorkspaceFolder
} from '@/store/coding-workspaces'
import { $gateway, activeGatewayConnectionId } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, $newChatConnectionId, $newChatProfile, $newChatRoute, resolveNewChatBackendOwner } from '@/store/profile'
import { $connection } from '@/store/session'

import { selectCodingWorkspaceIntent } from './coding-workspace-selection'

const $codingOwnerRoute = computed([$gateway, $connection, $activeGatewayProfile, $newChatConnectionId, $newChatProfile, $newChatRoute], () => resolveNewChatBackendOwner())

/** Presentation only: preparation belongs exclusively to the first-Send gate. */
export function useCodingWorkspace(draftKey: string | null, fresh: boolean) {
  const { t } = useI18n()
  const route = useStore($codingOwnerRoute)
  const owner = useMemo<CodingWorkspaceOwner | null>(() => route && fresh ? { ...route, draftKey: codingWorkspaceDraftKey(draftKey) } : null, [route, draftKey, fresh])
  const key = owner ? codingWorkspaceKey(owner) : null
  const currentKey = useRef(key)
  currentKey.current = key
  const generation = useRef(0)
  const { data: config } = useHermesConfigRecord(route ?? undefined)
  const draft = useStore(useMemo(() => computed($codingWorkspaceDrafts, drafts => key ? drafts[key] : undefined), [key]))
  const visible = Boolean(owner && (draft?.controlsEnabled || draft?.intent || getNested(config ?? {}, 'desktop.coding.show_controls') === true))

  const enable = () => { if (owner) {enableCodingWorkspaceControls(owner)} }

  const selectFolder = async (path?: string) => {
    if (!owner) {return}
    const capturedKey = key
    const capturedDraft = $codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)]
    const request = ++generation.current

    const isCurrent = () => currentKey.current === capturedKey && generation.current === request &&
      $codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)]?.requestId === capturedDraft?.requestId

    const ownsPicker = () => (owner.connectionId === null ? activeGatewayConnectionId() === null : $connection.get()?.connectionId === owner.connectionId) && ($activeGatewayProfile.get() || 'default') === owner.profile

    try {
      if (!path && !ownsPicker()) {throw new Error(t.codingWorkspace.unavailable)}
      const selected = path ?? (await selectDesktopPaths({ directories: true, multiple: false, title: t.codingWorkspace.workInProject }))[0]

      if (!selected || !isCurrent() || (!path && !ownsPicker())) {return}
      const previous = $codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)]
      const project = await registerCodingWorkspaceFolder(owner, selected)

      if (!isCurrent() || previous !== $codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)]) {return}
      const mode = getNested(config ?? {}, 'desktop.coding.default_checkout') === 'current' ? 'current' : 'worktree'
      await selectCodingWorkspaceIntent(owner, { projectId: project.id, path: selected, mode })
    } catch (error) {
      if (isCurrent()) {notifyError(error, t.codingWorkspace.folderFailed)}
    }
  }

  return { owner, draft, visible, enable, selectFolder }
}
