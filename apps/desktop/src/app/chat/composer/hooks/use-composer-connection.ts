import { useStore } from '@nanostores/react'

import {
  $backendTransitions,
  type BackendTransitionKind,
  changesBackend,
  transitionProfileKey
} from '@/store/backend-transition'
import { $activeGatewayProfile } from '@/store/profile'
import { $gatewayState } from '@/store/session'

import { shouldDisableComposerInput } from '../composer-utils'

/**
 * The composer's view of its connection. `gatewayDisabled` is the chat's own
 * verdict (no open gateway). `backendTransition` is the planned transition of
 * the composer's profile, if any (store/backend-transition.ts: a plugin
 * starting, stopping or restarting its backend). It decides what a message
 * does meanwhile, so that none starts on a backend about to go away:
 *
 * - starting / stopping: the profile moves to another backend. The message
 *   stays in the composer until the move is done (use-composer-submit) and the
 *   controls are off; nothing is sent or queued across the change.
 * - restarting: the same backend and its sessions come back. The composer
 *   stays usable with the connection down and the message waits in the chat's
 *   local queue (use-composer-submit, use-composer-queue).
 */
export function useComposerConnection(gatewayDisabled: boolean, profile: null | string | undefined) {
  const gatewayState = useStore($gatewayState)
  const activeProfile = useStore($activeGatewayProfile)
  const backendTransition = useStore($backendTransitions)[transitionProfileKey(profile || activeProfile)]?.kind ?? null

  return {
    backendTransition,
    disabled: gatewayDisabled && backendTransition !== 'restarting',
    inputDisabled: shouldDisableComposerInput(gatewayDisabled, gatewayState),
    reconnecting: gatewayState !== 'open'
  }
}

/**
 * Send and the controls beside it are off while the gateway is away, and while
 * the profile moves to another backend. The input stays usable either way.
 */
export const controlsDisabled = (disabled: boolean, backendTransition: BackendTransitionKind | null) =>
  disabled || changesBackend(backendTransition)
