import { useStore } from '@nanostores/react'

import { useHudComposerDrag } from '@/app/hud/composer-drag'
import { $hudMode } from '@/store/hud'

/** Native windowing facts, kept outside the composer coordinator. */
export function useComposerWindowing() {
  const hudMode = useStore($hudMode)
  const windowing = window.hermesDesktop?.hud?.windowing
  const hudNativeDrag = hudMode && windowing?.nativeDrag === true

  const { grabbing: hudGrabbing, onPointerDown: onHudDragPointerDown } = useHudComposerDrag(hudMode && !hudNativeDrag, {
    controlDrag: windowing?.controlDrag === true,
    workspaceTransfer: windowing?.workspaceTransfer === true
  })

  return { hudMode, hudNativeDrag, hudGrabbing, onHudDragPointerDown }
}
