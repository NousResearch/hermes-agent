import { useStore } from '@nanostores/react'

import type { ModelSelection } from '@/app/shell/model-menu-panel'
import { ModelPickerDialog } from '@/components/model-picker'
import type { HermesGateway } from '@/hermes'
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
}

export function ModelPickerOverlay({ gateway, onSelect }: ModelPickerOverlayProps) {
  const primarySessionId = useStore($activeSessionId)
  const primaryModel = useStore($currentModel)
  const primaryProvider = useStore($currentProvider)
  const focusedRuntimeId = useStore($focusedRuntimeId)
  const focusedState = useStore($focusedSessionState)
  const gatewayOpen = useStore($gatewayState) === 'open'
  const open = useStore($modelPickerOpen)

  // Prefer the focused tile's runtime when the overlay opens from a tile that
  // lacked a live menu (gateway closed → fallback path).
  const sessionId = focusedRuntimeId ?? primarySessionId
  const currentModel = focusedRuntimeId && focusedState ? focusedState.model : primaryModel
  const currentProvider = focusedRuntimeId && focusedState ? focusedState.provider : primaryProvider

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
      sessionId={sessionId}
    />
  )
}
