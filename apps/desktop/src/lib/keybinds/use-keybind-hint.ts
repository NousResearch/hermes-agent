import { useStore } from '@nanostores/react'

import { $registryVersion } from '@/contrib/registry'
import { $composerSendPrefs } from '@/store/composer-send'
import { $bindings, bindingsFor } from '@/store/keybinds'

import { readonlyShortcuts } from './actions'
import { formatCombo } from './combo'

// The formatted first combo for `actionId`, or null when unbound. Rebindable
// actions read live from the store; readonly shortcuts (e.g. `composer.steer`)
// fall back to their fixed combo. Returns null for unknown action ids so the
// tooltip shows just the text label with no trailing hint.
export function useKeybindHint(actionId: string): string | null {
  const bindings = useStore($bindings)
  // Composer rows are mode-dependent, so the hint has to follow the setting —
  // otherwise the Send button keeps advertising Enter after the user moved
  // sending to ⌘Enter.
  const sendPrefs = useStore($composerSendPrefs)

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

  const readonly = readonlyShortcuts(sendPrefs).find(entry => entry.id === actionId)

  if (readonly) {
    const [first, second] = readonly.keys

    // A double-tap row repeats one combo, and a lone "Enter" hint would be a
    // lie — spell both presses out.
    return second !== undefined && second === first
      ? `${formatCombo(first)} ${formatCombo(first)}`
      : formatCombo(first)
  }

  return null
}
