// Built-in app actions a plugin may run by id (`ctx.runAction`), through the
// SAME handler the keyboard shortcut and the command palette dispatch. A
// plugin that wants "toggle the browser panel" asks for the action, not for
// the chord: rebinding the shortcut in Settings ▸ Keyboard Shortcuts never
// changes what the plugin runs, and an unbound action still runs.
//
// Only view / navigation actions are exposed. Nothing that deletes, archives,
// signs out, answers an approval, switches profile / model / provider, spawns
// a shell, opens native dialogs, or touches app lifecycle (update, quit,
// reload) — those stay user-initiated.

import { runtimeTranslations } from '@/i18n/runtime'

import {
  KEYBIND_ACTION_IDS,
  keybindAction,
  type KeybindCategory,
  PLUGIN_APP_ACTIONS,
  type PluginAppActionId
} from './actions'

export { PLUGIN_APP_ACTIONS, type PluginAppActionId } from './actions'

/** One row of `ctx.listActions()`. */
export interface PluginAppActionInfo {
  id: PluginAppActionId
  /** The localized label the Keyboard Shortcuts panel shows. */
  label: string
  category: KeybindCategory
}

/** What `ctx.runAction` resolves to. It never throws. */
export type PluginRunActionResult =
  | { ok: true }
  | {
      ok: false
      /** `unknown`: no such action. `denied`: a real action that is not on
       *  the plugin allowlist. `unavailable`: the app shell that owns the
       *  handlers is not mounted (e.g. during early startup). */
      reason: 'denied' | 'unavailable' | 'unknown'
      error: string
    }

const ALLOWED: ReadonlySet<string> = new Set(PLUGIN_APP_ACTIONS)

export function isPluginAppAction(id: unknown): id is PluginAppActionId {
  return typeof id === 'string' && ALLOWED.has(id)
}

/** Runs a built-in action's handler by id; false when it has none. */
type BuiltinActionRunner = (id: string) => boolean

let runner: BuiltinActionRunner | null = null

/** The keybinding dispatcher (`useKeybinds`) installs its handler table here
 *  while it is mounted, so plugin runs and keypresses share one handler. */
export function registerBuiltinActionRunner(next: BuiltinActionRunner): () => void {
  runner = next

  return () => {
    if (runner === next) {
      runner = null
    }
  }
}

export function listPluginAppActions(): PluginAppActionInfo[] {
  // Action ids contain dots, so read the block rather than a dot-path key.
  const labels: Record<string, string | undefined> = runtimeTranslations().keybinds.actions

  return PLUGIN_APP_ACTIONS.map(id => ({
    id,
    label: labels[id] ?? id,
    category: keybindAction(id)?.category ?? 'view'
  }))
}

function refuse(pluginId: string, reason: 'denied' | 'unavailable' | 'unknown', error: string): PluginRunActionResult {
  console.warn(`[plugin:${pluginId}] runAction: ${error}`)

  return { ok: false, reason, error }
}

export function runPluginAppAction(pluginId: string, id: unknown): PluginRunActionResult {
  const label = typeof id === 'string' ? `'${id}'` : String(id)

  if (!isPluginAppAction(id)) {
    const known = typeof id === 'string' && KEYBIND_ACTION_IDS.includes(id)

    return known
      ? refuse(pluginId, 'denied', `${label} is not available to plugins. See ctx.listActions() for the allowed ids.`)
      : refuse(pluginId, 'unknown', `${label} is not a Hermes app action. See ctx.listActions() for the allowed ids.`)
  }

  if (!runner || !runner(id)) {
    return refuse(pluginId, 'unavailable', `${label} cannot run yet: the app shell is not ready.`)
  }

  return { ok: true }
}
