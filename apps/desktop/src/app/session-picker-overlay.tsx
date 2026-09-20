import { useStore } from '@nanostores/react'

import { SessionPickerDialog } from '@/components/session-picker'
import {
  $gatewayState,
  $sessionPickerCallerTileId,
  $sessionPickerOpen,
  setSessionPickerOpen
} from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-states'

interface SessionPickerOverlayProps {
  onResume: (storedSessionId: string, targetTileId?: string | null) => void
}

/**
 * Mounts the session picker that `/resume` (and `/sessions`, `/switch`) opens —
 * the desktop equivalent of the TUI's sessions overlay. Resuming runs through
 * the active tab / session tile when opened from a tile, else main.
 */
export function SessionPickerOverlay({ onResume }: SessionPickerOverlayProps) {
  const open = useStore($sessionPickerOpen)
  const gatewayOpen = useStore($gatewayState) === 'open'
  const activeStoredSessionId = useStore($focusedStoredSessionId)
  const callerTileId = useStore($sessionPickerCallerTileId)

  if (!gatewayOpen) {
    return null
  }

  return (
    <SessionPickerDialog
      activeStoredSessionId={activeStoredSessionId}
      onOpenChange={open => {
        setSessionPickerOpen(open)

        if (!open) {
          $sessionPickerCallerTileId.set(null)
        }
      }}
      onResume={storedSessionId => onResume(storedSessionId, callerTileId)}
      open={open}
    />
  )
}
