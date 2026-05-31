import { useStore } from '@nanostores/react'
import type * as React from 'react'

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

interface ModelPickerOverlayProps {
  gateway?: HermesGateway
  onSelect: React.ComponentProps<typeof ModelPickerDialog>['onSelect']
}

export function ModelPickerOverlay({ gateway, onSelect }: ModelPickerOverlayProps) {
  const activeSessionId = useStore($activeSessionId)
  const currentModel = useStore($currentModel)
  const currentProvider = useStore($currentProvider)
  const gatewayOpen = useStore($gatewayState) === 'open'
  const open = useStore($modelPickerOpen)

  if (!gatewayOpen) {
    return null
  }

  return (
    <ModelPickerDialog
      currentModel={currentModel}
      currentProvider={currentProvider}
      gw={gateway}
      onOpenChange={setModelPickerOpen}
      onSelect={onSelect}
      open={open}
      sessionId={activeSessionId}
    />
  )
}
