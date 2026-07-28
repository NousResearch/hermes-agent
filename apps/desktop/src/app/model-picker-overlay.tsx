import { useStore } from '@nanostores/react'

import type { ModelSelection } from '@/app/shell/model-menu-panel'
import { ModelPickerDialog } from '@/components/model-picker'
import type { HermesGateway } from '@/hermes'
import { useStoreSelector } from '@/lib/use-session-slice'
import {
  $activeSessionId,
  $currentModel,
  $currentProvider,
  $gatewayState,
  $modelPickerOpen,
  setModelPickerOpen
} from '@/store/session'
import { $focusedRuntimeId, $focusedSessionState } from '@/store/session-states'

interface ModelPickerOverlayProps {
  gateway?: HermesGateway
  onSelect: (selection: ModelSelection) => void
  profile: string
}

export function ModelPickerOverlay({ gateway, onSelect, profile }: ModelPickerOverlayProps) {
  const primarySessionId = useStore($activeSessionId)
  const primaryModel = useStore($currentModel)
  const primaryProvider = useStore($currentProvider)
  const focusedRuntimeId = useStore($focusedRuntimeId)
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
      onSelect={selection => onSelect({ ...selection, sessionId })}
      open={open}
      profile={profile}
      sessionId={sessionId}
    />
  )
}
