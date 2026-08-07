// Avatar3DMascot — 3D Commander mascot for Hermes Desktop
// Replaces the 2D FloatingPet with the baked GLB avatar

import { useStore } from '@nanostores/react'
import { useEffect, useMemo } from 'react'

import { AvatarCanvas, type AvatarState } from '@/components/avatar'
import { $petState, $petActivity } from '@/store/pet'
import { $busy } from '@/store/session'
import { $gatewayState } from '@/store/session'

/**
 * Map PetState → AvatarRuntimeEvent for the 3D avatar.
 * PetState: idle | wave | run | failed | review | jump | waiting
 * AvatarState (AvatarRuntimeEvent): idle | scan | working | step | speaking | notify | error | success
 */
function petStateToAvatarState(petState: string): AvatarState {
  switch (petState) {
    case 'failed':
      return 'error'
    case 'jump':
      return 'success'
    case 'wave':
      return 'success'
    case 'waiting':
      return 'notify'
    case 'run':
      return 'working'
    case 'review':
      return 'thinking'
    case 'idle':
    default:
      return 'idle'
  }
}

export function Avatar3DMascot() {
  const petState = useStore($petState)
  const petActivity = useStore($petActivity)
  const busy = useStore($busy)
  const gatewayState = useStore($gatewayState)

  // Only show when gateway is connected and pet would be active
  const isActive = gatewayState === 'open'

  // Map pet state to avatar state
  const avatarState = useMemo(() => petStateToAvatarState(petState), [petState])

  // Debug logging
  useEffect(() => {
    console.log('[Avatar3D] petState:', petState, '→ avatarState:', avatarState, 'busy:', busy, 'activity:', petActivity)
  }, [petState, avatarState, busy, petActivity])

  if (!isActive) {
    return null
  }

  return (
    <AvatarCanvas
      state={avatarState}
      width={200}
      height={300}
      background="transparent"
    />
  )
}