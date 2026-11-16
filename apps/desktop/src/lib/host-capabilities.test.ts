// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest'

import { hostHas } from './host-capabilities'
import { isBrowserHostedDesktop } from './platform'

interface HostGlobals {
  __HERMES_AUTH_REQUIRED__?: boolean
  __HERMES_UI_SURFACE__?: string
  hermesDesktop?: unknown
}

const win = window as unknown as HostGlobals

function reset() {
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  delete win.__HERMES_AUTH_REQUIRED__
  delete win.__HERMES_UI_SURFACE__
  delete win.hermesDesktop
}

// Every way the renderer meets its host: before and after the bridge writes the marker.
const HOST_STATES: { browser: boolean; name: string; setup: () => void }[] = [
  {
    browser: true,
    name: 'browser marker',
    setup: () => (document.documentElement.dataset.hermesDesktopHost = 'browser')
  },
  { browser: true, name: 'gated host before the marker', setup: () => (win.__HERMES_AUTH_REQUIRED__ = true) },
  { browser: true, name: 'Webapp surface before the marker', setup: () => (win.__HERMES_UI_SURFACE__ = 'webapp') },
  {
    browser: false,
    name: 'electron marker',
    setup: () => (document.documentElement.dataset.hermesDesktopHost = 'electron')
  },
  {
    browser: false,
    name: 'Electron preload with a stale Webapp global',
    setup: () => {
      win.hermesDesktop = {}
      win.__HERMES_UI_SURFACE__ = 'webapp'
    }
  },
  { browser: false, name: 'no host signal', setup: () => undefined }
]

// Every migrated call site, keyed by its line on the base commit: `old` is the condition it read
// from `isBrowserHostedDesktop()` (as a function of the host), `now` the expression it reads
// instead. Both select the same branch, so a capability given the wrong host fails here.
const CALL_SITES: [site: string, old: (browser: boolean) => boolean, now: () => boolean][] = [
  ['api/plugins.ts:195', b => b, () => !hostHas('urlTokenAuth')],
  ['app/chat/composer/controls.tsx:33', b => b, () => hostHas('keepFocusOnPress')],
  ['app/chat/composer/controls.tsx:139', b => b, () => hostHas('touchSizedControls')],
  ['app/chat/composer/hooks/use-composer-metrics.ts:70', b => !b, () => !hostHas('touchSizedControls')],
  ['app/chat/composer/hooks/use-mic-recorder.ts:54', b => b, () => hostHas('webappCopy')],
  ['app/chat/right-rail/preview-pane.tsx:128', b => !b, () => hostHas('webviewPreview')],
  ['app/chat/right-rail/preview-pane.tsx:1115', b => b, () => !hostHas('webviewPreview')],
  ['app/chat/right-rail/preview-pane.tsx:1448', b => b, () => !hostHas('appMouseHistoryButtons')],
  ['app/chat/sidebar/profile-dropdown-switcher.tsx:93', b => b, () => !hostHas('profileRailSquares')],
  ['app/chat/sidebar/profile-switcher.tsx:125', b => b, () => !hostHas('profileRailSquares')],
  ['app/chat/sidebar/unserved-profiles.tsx:23', b => b, () => hostHas('servedProfileRoster')],
  ['app/context-menu/app-context-menu.tsx:71', b => b, () => hostHas('browserContextMenu')],
  ['app/context-menu/app-context-menu.tsx:102', b => b, () => !hostHas('mainProcessEditVerbs')],
  ['app/context-menu/app-context-menu.tsx:703', b => b, () => hostHas('browserContextMenu')],
  ['app/gateway/hooks/use-gateway-boot.ts:747', b => b, () => hostHas('servingOriginConnection')],
  ['app/gateway/hooks/use-gateway-boot.ts:775', b => b, () => hostHas('servingOriginConnection')],
  ['app/right-sidebar/terminal/terminal-session-attempt.ts:255', b => b, () => hostHas('retryTerminalSpawn')],
  ['app/session/hooks/use-prompt-actions/browser-manage-refusal.ts:7', b => b, () => hostHas('webappCopy')],
  ['app/settings/gateway-settings.tsx:180', b => b, () => !hostHas('gatewayManagement')],
  ['app/settings/other-subpages.ts:7', b => !b, () => hostHas('gatewayManagement')],
  ['app/settings/sessions-settings.tsx:355', b => b, () => hostHas('originScopedPrefs')],
  ['app/settings/sessions-settings.tsx:386', b => b, () => !hostHas('nativeFolderPicker')],
  ['app/settings/web-access-settings.tsx:96', b => !b, () => hostHas('webAccessHost')],
  ['app/shell/titlebar-controls.tsx:163', b => b, () => hostHas('narrowOverlayChrome')],
  ['app/shell/use-touch-titlebar.ts:10', b => b, () => hostHas('touchSizedControls')],
  ['components/chat/expandable-block.tsx:33', b => b, () => hostHas('sidewaysCodeFences')],
  ['components/chat/shiki-highlighter.tsx:207', b => b, () => hostHas('sidewaysCodeFences')],
  ['components/pane-shell/tree/renderer/narrow-overlays.tsx:144', b => b, () => hostHas('narrowOverlayChrome')],
  ['components/pane-shell/tree/renderer/narrow-overlays.tsx:212', b => b, () => hostHas('tapOutsideDismiss')],
  ['components/pane-shell/tree/renderer/narrow-overlays.tsx:284', b => b, () => hostHas('narrowOverlayChrome')],
  ['lib/browser-profile-prefs.ts:12', b => b, () => hostHas('originScopedPrefs')],
  ['lib/clipboard.ts:61', b => !b, () => hostHas('clipboardIpc')],
  ['lib/desktop-fs.ts:48', b => b, () => !hostHas('nativeHostFiles')],
  ['lib/external-link.tsx:272', b => b, () => !hostHas('inAppLinks')],
  ['lib/keybinds/actions.ts:31', b => !b, () => !hostHas('browserShortcutDefaults')],
  ['lib/media.ts:125', b => !b, () => hostHas('nativeHostFiles')],
  ['lib/media.ts:286', b => !b, () => hostHas('urlTokenAuth')],
  ['store/browser-tab-title.ts:86', b => !b, () => !hostHas('tabTitleStatus')],
  ['store/profile.ts:1031', b => b, () => hostHas('tabStartupProfile')],
  ['store/session.ts:1877', b => b, () => hostHas('originScopedPrefs')],
  ['store/translucency.ts:73', b => !b, () => hostHas('windowMaterial')],
  ['store/translucency.ts:84', b => !b, () => hostHas('windowMaterial')]
]

afterEach(reset)

describe('hostHas', () => {
  for (const state of HOST_STATES) {
    it(`keeps every call site's branch: ${state.name}`, () => {
      state.setup()
      expect(isBrowserHostedDesktop()).toBe(state.browser)

      for (const [site, old, now] of CALL_SITES) {
        expect([site, now()]).toEqual([site, old(state.browser)])
      }
    })
  }

  it('follows the host marker written after import', () => {
    expect(hostHas('clipboardIpc')).toBe(true)
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    expect(hostHas('clipboardIpc')).toBe(false)
    expect(hostHas('webappCopy')).toBe(true)
  })
})
