/**
 * Typed capabilities bridge for desktop plugins (`host.skills`, `host.toolsets`,
 * `host.profiles`, `host.pluginDecisions`).
 *
 * Plugins that configure capabilities used to call `window.hermesDesktop.api`
 * raw and read the persisted plugin-decisions map straight out of
 * localStorage. These wrappers give them the SAME doors the Capabilities page
 * uses — same endpoints, same `ProfileScope` handling — so a plugin acting on
 * skills or toolsets behaves exactly like the page. Nothing new is arbitrated
 * here: every RPC below is already reachable via `host.request`; the value is
 * typing plus profile scoping.
 *
 * `pluginDecisions` is READ-ONLY on purpose. A `set()` would let one plugin
 * flip another plugin's enable state, which is exactly "plugins messing with
 * each other" and the singleton `host` cannot tell which plugin is calling to
 * restrict it to its own id. Enabling/disabling plugins stays in the app's
 * Capabilities → Plugins tab (`host.navigate('/capabilities?tab=plugins')`).
 */
import type { ReadableAtom } from 'nanostores'

import type { ProfileScope } from '@/api/client'
import { getProfiles } from '@/api/profiles'
import { getSkills, setSkillEnabled } from '@/api/skills'
import { getToolsets, setToolsetEnabled } from '@/api/toolsets'
import { $pluginDecisions } from '@/contrib/plugins-store'

/** Skills for a profile scope — omit `profile` for the app-wide active one. */
export const skills = {
  /** Every skill the backend reports for the scope. */
  list: (profile?: ProfileScope) => getSkills(profile),
  /** Enable/disable a skill (the Capabilities toggle). */
  setEnabled: (name: string, enabled: boolean, profile?: ProfileScope) => setSkillEnabled(name, enabled, profile)
}

export const toolsets = {
  /** Every toolset with its enabled state for the scope. */
  list: (profile?: ProfileScope) => getToolsets(profile),
  /** Enable/disable a toolset (the Capabilities toggle). */
  setEnabled: (name: string, enabled: boolean, profile?: ProfileScope) => setToolsetEnabled(name, enabled, profile)
}

export const profiles = {
  /** The profile list as the app's own surfaces read it (same endpoint and
   *  startup timeout as the profile rail). */
  list: (scope?: ProfileScope) => getProfiles(scope)
}

/** This window's plugin enable/disable decisions (id → enabled; absence means
 *  the user never chose, so the plugin's own `defaultEnabled` applies).
 *  A hand-built view rather than a `computed`/type cast: both of those still
 *  carry `.set` at runtime, and the point is that a plugin cannot cast its way
 *  into another plugin's toggle. */
export const pluginDecisions: ReadableAtom<Record<string, boolean>> = {
  get: () => $pluginDecisions.get(),
  get init() {
    return $pluginDecisions.init
  },
  get lc() {
    return $pluginDecisions.lc
  },
  listen: listener => $pluginDecisions.listen(listener),
  notify: oldValue => $pluginDecisions.notify(oldValue),
  off: () => $pluginDecisions.off(),
  subscribe: listener => $pluginDecisions.subscribe(listener),
  get value() {
    return $pluginDecisions.value
  }
}
