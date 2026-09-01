import { useStore } from '@nanostores/react'
import { useQueryClient } from '@tanstack/react-query'
import { useCallback } from 'react'

import { useModelControls } from '@/app/session/hooks/use-model-controls'
import type { ModelSelection } from '@/app/shell/model-menu-panel'
import { ModelPickerDialog } from '@/components/model-picker'
import type { HermesGateway } from '@/hermes'
import { resolveModelPickerOwner } from '@/lib/model-picker-owner'
import { useStoreSelector } from '@/lib/use-session-slice'
import {
  $activeSessionId,
  $currentModel,
  $currentProvider,
  $gatewayState,
  $modelPickerOpen,
  $selectedStoredSessionId,
  setModelPickerOpen
} from '@/store/session'
import { requestForSessionProfile } from '@/store/session-request-router'
import {
  $focusedRuntimeId,
  $focusedSessionState,
  $focusedStoredSessionId,
  $sessionTiles
} from '@/store/session-states'

interface ModelPickerOverlayProps {
  gateway?: HermesGateway
  onSelect: (selection: ModelSelection) => void
  ownerConnectionId?: string
  profile: string
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

export function ModelPickerOverlay({
  gateway,
  onSelect,
  ownerConnectionId,
  profile,
  requestGateway
}: ModelPickerOverlayProps) {
  const queryClient = useQueryClient()
  const primarySessionId = useStore($activeSessionId)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const primaryModel = useStore($currentModel)
  const primaryProvider = useStore($currentProvider)
  const focusedRuntimeId = useStore($focusedRuntimeId)
  const focusedStoredSessionId = useStore($focusedStoredSessionId)
  const sessionTiles = useStore($sessionTiles)
  // `$focusedSessionState` is a projection of `$sessionStates`, republished on
  // EVERY message delta — and this overlay is mounted app-wide. Only two
  // fields are read off it, so subscribing to the whole object re-rendered
  // this component (and the un-memoized closed dialog below) per token while
  // the focused session streamed. Select each scalar so an unchanged
  // model/provider bails out instead — same fix as the statusbar (#72163).
  const focusedModel = useStoreSelector($focusedSessionState, state => state?.model ?? null)
  const focusedProvider = useStoreSelector($focusedSessionState, state => state?.provider ?? null)
  const gatewayOpen = useStore($gatewayState) === 'open'
  const open = useStore($modelPickerOpen)

  const pickerOwner = resolveModelPickerOwner({
    ambientConnectionId: ownerConnectionId,
    ambientProfile: profile,
    focusedStoredSessionId,
    selectedStoredSessionId,
    sessionTiles
  })

  const requestPickerGateway = useCallback(
    <T,>(method: string, params?: Record<string, unknown>): Promise<T> =>
      requestForSessionProfile<T>(pickerOwner.route, requestGateway, method, params),
    [pickerOwner.route, requestGateway]
  )

  const { selectModel: selectFocusedModel } = useModelControls({
    cacheOwnerConnectionId: pickerOwner.connectionId,
    cacheProfile: pickerOwner.profile,
    queryClient,
    requestGateway: requestPickerGateway
  })

  // Prefer the focused tile's runtime when the overlay opens from a tile that
  // lacked a live menu (gateway closed → fallback path).
  const sessionId = focusedRuntimeId ?? primarySessionId
  const currentModel = focusedRuntimeId && focusedModel !== null ? focusedModel : primaryModel
  const currentProvider = focusedRuntimeId && focusedProvider !== null ? focusedProvider : primaryProvider

  if (!gatewayOpen) {
    return null
  }

  return (
    <ModelPickerDialog
      currentModel={currentModel}
      currentProvider={currentProvider}
      gw={gateway}
      onOpenChange={setModelPickerOpen}
      onSelect={selection =>
        (pickerOwner.route ? selectFocusedModel : onSelect)({ ...selection, sessionId })
      }
      open={open}
      ownerConnectionId={pickerOwner.connectionId}
      profile={pickerOwner.profile}
      request={pickerOwner.route ? requestPickerGateway : undefined}
      sessionId={sessionId}
    />
  )
}
