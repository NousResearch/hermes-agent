import type { ConnectionState } from '@hermes/shared'

import { useComposerInputHeld } from '@/store/composer-holds'

import { shouldDisableComposerInput } from '../composer-utils'
import type { ComposerTarget } from '../focus'

/** The composer's text input is read-only while the app disables it (a live
 *  gateway with `disabled` set) or while a feature holds it (see
 *  `store/composer-holds`). */
export function useComposerInputDisabled(disabled: boolean, gatewayState: ConnectionState, target: ComposerTarget) {
  const held = useComposerInputHeld(target)

  return held || shouldDisableComposerInput(disabled, gatewayState)
}
