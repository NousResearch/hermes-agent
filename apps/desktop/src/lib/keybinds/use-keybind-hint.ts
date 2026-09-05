import { useStore } from '@nanostores/react'

import { $registryVersion } from '@/contrib/registry'
import { $bindings, bindingsFor } from '@/store/keybinds'

import { KEYBIND_READONLY } from './actions'
import { formatCombo } from './combo'

// The formatted first combo for `actionId`, or null when unbound. Rebindable
// actions read live from the store; readonly shortcuts (e.g. `composer.steer`)
// fall back to their fixed combo. Returns null for unknown action ids so the
// tooltip shows just the text label with no trailing hint.
export function useKeybindHint(actionId: string): string | null {
  const bindings = useStore($bindings)

  // `bindingsFor`, not a raw `bindings[id]`: $bindings is seeded at module init
  // from the actions known THEN, so a plugin action contributed later isn't in
  // it and a raw lookup renders no hint at all. The resolver falls through to
  // the stored override and the action's own defaults. Subscribing to the
  // registry version repaints the hint when that late registration lands.
  useStore($registryVersion)

  const rebindable = bindingsFor(actionId, bindings)[0]

  if (rebindable) {
    return formatCombo(rebindable)
  }

  const readonly = KEYBIND_READONLY.find(entry => entry.id === actionId)

  if (readonly) {
    return formatCombo(readonly.keys[0])
  }

  return null
}
