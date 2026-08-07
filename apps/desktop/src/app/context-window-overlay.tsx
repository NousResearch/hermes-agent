import { useStore } from '@nanostores/react'

import { ContextWindowDialog } from '@/components/context-window-dialog'
import { $contextWindowOpen, setContextWindowOpen } from '@/store/context-window'
import { $gatewayState } from '@/store/session'

interface ContextWindowOverlayProps {
  profile: string
}

export function ContextWindowOverlay({ profile }: ContextWindowOverlayProps) {
  const gatewayOpen = useStore($gatewayState) === 'open'
  const open = useStore($contextWindowOpen)

  if (!gatewayOpen) {
    return null
  }

  return <ContextWindowDialog onOpenChange={setContextWindowOpen} open={open} profile={profile} />
}
