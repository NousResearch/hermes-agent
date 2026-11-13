import { useStore } from '@nanostores/react'
import { computed } from 'nanostores'

import { persistentAtom } from '@/lib/persisted'

import { DESKTOP_BUTTON_ACTIONS, recordAction, recordFeatureToggle } from './desktop-metrics'
import { providePortalPresentation } from './shell-presentation'
import { isBrowserWindow, isHudWindow } from './windows'

/** The persisted explicit choice: null until the user picks, `false` = classic rollback. */
export const $shellChoice = persistentAtom<boolean | null>('hermes.desktop.agentShell.v1', null, {
  decode: raw => (raw === 'true' ? true : raw === 'false' ? false : null),
  encode: value => (value === null ? null : String(value))
})

/** The user's redesign preference (on unless rolled back). Window-independent: read it for
 *  behaviour every window must agree on — `shell-policy.ts` names each such behaviour. */
export const $portalPreferred = computed($shellChoice, choice => choice !== false)

/** Whether THIS window renders the portal shell for a stored choice. The chromeless HUD and
 *  popped-out Browser keep the classic look in every mode; their layouts were never redesigned. */
export const rendersPortalShell = (choice: boolean | null): boolean =>
  choice !== false && !isHudWindow() && !isBrowserWindow()

/** Whether THIS window renders the redesigned (portal) shell — the one presentation read, mirrored
 *  to `<html data-shell="portal">` for CSS. */
export const $portalShell = computed($shellChoice, rendersPortalShell)

// The icon layer reads this through the dependency-free `shell-presentation` leaf.
providePortalPresentation($portalShell)

export function usePortalShell(): boolean {
  return useStore($portalShell)
}

export function setPortalPreferred(enabled: boolean): void {
  const previous = $portalPreferred.get()

  if (previous === enabled) {
    return
  }

  $shellChoice.set(enabled)
  const actions = DESKTOP_BUTTON_ACTIONS

  recordAction(enabled ? actions.sidebarEnableSettings : actions.sidebarDisable, 'click')
  recordFeatureToggle('simplified_sidebar', previous, enabled)
}

/** Mirrors `$portalShell` onto `<html data-shell>`; called once at boot, before the first render, so
 *  body-portalled surfaces never paint a mixed shell. Classic leaves no attribute. */
export function installShellAttribute(root: HTMLElement = document.documentElement): () => void {
  return $portalShell.subscribe(portal => {
    if (portal) {
      root.dataset.shell = 'portal'
    } else {
      delete root.dataset.shell
    }
  })
}
