import { useStore } from '@nanostores/react'

import { $bindings } from '@/store/keybinds'

import { KEYBIND_READONLY } from './actions'
import { formatCombo } from './combo'

// The formatted first combo for `actionId`, or null when unbound. Rebindable
// actions read live from the store; readonly shortcuts (e.g. `composer.steer`)
// fall back to their fixed combo. Returns null for unknown action ids so the
// tooltip shows just the text label with no trailing hint.
export function useKeybindHint(actionId: string): string | null {
  const bindings = useStore($bindings)

  const rebindable = bindings[actionId]?.[0]

  if (rebindable) {
    return formatCombo(rebindable)
  }

  const readonly = KEYBIND_READONLY.find(entry => entry.id === actionId)

  if (readonly) {
    return formatCombo(readonly.keys[0])
  }

  return null
}
