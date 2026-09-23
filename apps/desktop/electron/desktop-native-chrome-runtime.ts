import path from 'node:path'

import type { BrowserWindow } from 'electron'
import { app, Menu, type MenuItemConstructorOptions, session } from 'electron'

import { applyZoomLevel, DEFAULT_ZOOM_LEVEL, ZOOM_STEP, ZOOM_STORAGE_KEY } from './zoom'

// Native menu, shortcuts, zoom, context-menu, and session capability wiring.
// Window and flag getters retain the original main-process live state at the
// moment Electron delivers an event.
export function createDesktopNativeChromeRuntime(deps: {
  APP_NAME: string
  IS_MAC: boolean
  closeHudWindow: () => void
  extensionForMimeType: (mimeType: string) => string
  getCreateInstanceWindow: () => () => BrowserWindow
  getF12Blocked: () => boolean
  getHudWindow: () => BrowserWindow | null
  getMainWindow: () => BrowserWindow | null
  readZoomState: () => number | null
  rememberLog: (line: string) => void
  sendClosePreviewRequested: () => void
  sendOpenFolderRequested: () => void
  sendOpenUpdatesRequested: () => void
  sendPreviewNavCommand: (command: 'back' | 'forward' | 'reload') => void
  showAboutPanelFresh: () => void
  writeZoomState: (level: number) => void
}) {
  const {
    APP_NAME,
    IS_MAC,
    closeHudWindow,
    extensionForMimeType,
    getCreateInstanceWindow,
    getF12Blocked,
    getHudWindow,
    getMainWindow,
    readZoomState,
    rememberLog,
    sendClosePreviewRequested,
    sendOpenFolderRequested,
    sendOpenUpdatesRequested,
    sendPreviewNavCommand,
    showAboutPanelFresh,
    writeZoomState
  } = deps

  function buildApplicationMenu() {
    const template: MenuItemConstructorOptions[] = []

    const checkForUpdatesItem = {
      label: 'Check for Updates…',
      click: () => sendOpenUpdatesRequested()
    }

    if (IS_MAC) {
      template.push({
        label: APP_NAME,
        submenu: [
          { label: `About ${APP_NAME}`, click: () => showAboutPanelFresh() },
          checkForUpdatesItem,
          { type: 'separator' },
          { role: 'services' },
          { type: 'separator' },
          { role: 'hide' },
          { role: 'hideOthers' },
          { role: 'unhide' },
          { type: 'separator' },
          { role: 'quit' }
        ]
      })
    }

    template.push({
      label: 'File',
      submenu: [
        // No accelerator: ⌘⇧N is a rebindable renderer keybind (session.newWindow);
        // a menu accelerator would fight the rebind panel and (on macOS) be
        // swallowed before the renderer sees it. Here purely for discoverability.
        { click: () => getCreateInstanceWindow()(), label: 'New Window' },
        // Same no-accelerator rationale: ⌘O is the rebindable renderer keybind
        // (workspace.openFolder). Clicking runs the same open-folder-as-project
        // flow through the renderer.
        { click: () => sendOpenFolderRequested(), label: 'Open Folder…' },
        { type: 'separator' },
        IS_MAC
          ? {
              // NO accelerator: on macOS a registered ⌘W is consumed by the OS
              // menu before the web contents ever sees it (and registerAccelerator
              // false is a no-op on mac — electron#18295). Leaving it off lets the
              // `before-input-event` handler below intercept ⌘W and route it to the
              // renderer's close-active-tab. Clicking the item still closes the tab
              // (or window) via the same request.
              click: () => sendClosePreviewRequested(),
              label: 'Close'
            }
          : { role: 'quit' }
      ]
    })
    template.push({
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        // ⌘⇧V is only wired up by this item existing: an accelerator with no menu
        // entry is never translated into an editor command, so the chord was a
        // no-op in every input in the app. The composer inserts plain text on
        // every paste anyway, so this is the same result as ⌘V there — it's the
        // terminal, preview, and other editable surfaces that need the strip.
        { role: 'pasteAndMatchStyle' },
        { role: 'delete' },
        { role: 'selectAll' },
        ...(IS_MAC
          ? ([
              { type: 'separator' },
              {
                label: 'Substitutions',
                submenu: [{ role: 'showSubstitutions' }, { type: 'separator' }, { role: 'toggleTextReplacement' }]
              }
            ] satisfies MenuItemConstructorOptions[])
          : [])
      ]
    })
    template.push({
      label: 'View',
      submenu: [
        // Not `role: 'reload'`: that hard-reloads the RENDERER (every pane, the
        // whole shell) and a focused in-app browser needs ⌘R to mean "reload
        // this page", the way it does in every other browser. ⇧⌘R
        // (`forceReload`) below stays the unconditional escape hatch.
        //
        // No accelerator: ⌘R is claimed in `installPreviewShortcut`, which works
        // on every platform (this menu exists only on macOS). Declaring it here
        // too would fire the item and the input hook for one keypress.
        { click: () => sendPreviewNavCommand('reload'), label: 'Reload' },
        { role: 'forceReload' },
        {
          label: 'Toggle Developer Tools',
          accelerator: process.platform === 'darwin' ? 'Alt+Cmd+I' : 'Ctrl+Shift+I',
          click: (_menuItem, browserWindow) => toggleDevTools(browserWindow || getMainWindow())
        },
        { type: 'separator' },
        {
          label: 'Actual Size',
          accelerator: 'CommandOrControl+0',
          click: () => {
            setAndPersistZoomLevel(getMainWindow(), DEFAULT_ZOOM_LEVEL)
          }
        },
        {
          label: 'Zoom In',
          accelerator: 'CommandOrControl+Plus',
          click: () => {
            if (getMainWindow() && !getMainWindow().isDestroyed()) {
              setAndPersistZoomLevel(getMainWindow(), getMainWindow().webContents.getZoomLevel() + ZOOM_STEP)
            }
          }
        },
        {
          label: 'Zoom Out',
          accelerator: 'CommandOrControl+-',
          click: () => {
            if (getMainWindow() && !getMainWindow().isDestroyed()) {
              setAndPersistZoomLevel(getMainWindow(), getMainWindow().webContents.getZoomLevel() - ZOOM_STEP)
            }
          }
        },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    })
    template.push({
      label: 'Window',
      submenu: IS_MAC
        ? [{ role: 'minimize' }, { role: 'zoom' }, { role: 'front' }]
        : [{ role: 'minimize' }, { role: 'close' }]
    })
    template.push({
      label: 'Help',
      role: 'help',
      submenu: [checkForUpdatesItem]
    })

    return Menu.buildFromTemplate(template)
  }

  function toggleDevTools(window) {
    // DevTools is enabled in packaged builds so users can diagnose renderer
    // issues without needing a dev build. Trade-off: tiny attack surface
    // increase versus a much better support story when WS connection or
    // CSP issues surface in the field.
    const { webContents } = window

    if (webContents.isDevToolsOpened()) {
      webContents.closeDevTools()
    } else {
      webContents.openDevTools({ mode: 'detach' })
    }
  }

  function installDevToolsShortcut(window) {
    // Only Ctrl+Shift+I (or Cmd+Opt+I on Mac) opens DevTools.
    // F12 is explicitly blocked so Chromium's built-in handler doesn't open it.
    window.webContents.on('before-input-event', (event, input) => {
      const key = input.key.toLowerCase()

      // F12 opens DevTools by default; block only when the user disabled it.
      if (input.key === 'F12') {
        if (getF12Blocked()) {
          event.preventDefault()

          return
        }
        // Not blocked — fall through to open DevTools.
      }

      const isInspectShortcut =
        input.key === 'F12' ||
        (IS_MAC && input.meta && input.alt && key === 'i') ||
        (!IS_MAC && input.control && input.shift && key === 'i')

      if (!isInspectShortcut) {
        return
      }

      event.preventDefault()
      toggleDevTools(window)
    })
  }

  function installPreviewShortcut(window) {
    window.webContents.on('before-input-event', (event, input) => {
      const key = String(input.key || '').toLowerCase()
      const accel = (IS_MAC ? input.meta : input.control) && !input.alt
      const isCloseTabShortcut = key === 'w' && accel && !input.shift

      // Always claim ⌘W here (the File>Close item deliberately has no
      // accelerator, so nothing else does). The renderer decides tab-vs-window
      // — no `previewShortcutActive` gate, so it works for every closeable tab.
      if (isCloseTabShortcut) {
        event.preventDefault()

        // ⌘W in the HUD is "leave HUD mode", not "close a tab in the app
        // window". Routing it to the main renderer closed the app's tab out
        // from under the user while the HUD stayed put; routing it through the
        // HUD's own close path hands the session back like the exit button.
        const hudWindow = getHudWindow()

        if (hudWindow && !hudWindow.isDestroyed() && window === hudWindow) {
          closeHudWindow()

          return
        }

        sendClosePreviewRequested()

        return
      }

      // ⌘R rides here rather than on the View menu item for the same reason:
      // the application menu only exists on macOS (it is set to null elsewhere,
      // see #77845), so a menu accelerator would leave Windows and Linux with no
      // way to reload a page at all. ⇧⌘R is left alone — that is `forceReload`,
      // the unconditional whole-window escape hatch.
      if (key === 'r' && accel && !input.shift) {
        event.preventDefault()
        sendPreviewNavCommand('reload')
      }
    })
  }

  // Zoom level is persisted in the renderer's own localStorage (per-origin,
  // survives reloads/restarts) rather than a main-process JSON file. The main
  // process owns setZoomLevel, so we mirror each change into localStorage and
  // read it back on did-finish-load to re-apply after reloads or crash recovery.

  function setAndPersistZoomLevel(window, zoomLevel) {
    if (!window || window.isDestroyed()) {
      return
    }

    // Apply + notify in one funnel so the settings UI stays in sync, including
    // changes made via the keyboard shortcuts or the View menu.
    const next = applyZoomLevel(window.webContents, zoomLevel)

    // Primary store: main-process JSON (survives crash recovery — #56726).
    writeZoomState(next)
    // Secondary mirror: renderer localStorage (legacy store; kept in sync so a
    // downgrade or JSON read failure still finds a sane value).
    window.webContents
      .executeJavaScript(
        `try { localStorage.setItem(${JSON.stringify(ZOOM_STORAGE_KEY)}, ${JSON.stringify(String(next))}) } catch {
      void 0
    }`
      )
      .catch(error => rememberLog(`[zoom] persist failed: ${error?.message || error}`))
  }

  function restorePersistedZoomLevel(window) {
    if (!window || window.isDestroyed()) {
      return
    }

    // Prefer the JSON file — it survives crash recovery wiping Electron's
    // cache/storage folders (#56726). applyZoomLevel notifies the renderer so
    // the Appearance UI Scale control stays in sync.
    const saved = readZoomState()

    if (saved != null) {
      // Drift-guard: skip when this window already shows the persisted level.
      // Blindly re-applying on every resize/move would race the compositor's
      // surface reconfigure during a Wayland resize storm (Cosmic tiled mode
      // fires one whenever a new session window opens — #84818) and keep the
      // renderer notification stream churning for no gain. The settle-verify
      // chain in installZoomReassertOnWindowEvents re-applies only when the
      // window actually drifted from the persisted level.
      const current = window.webContents?.getZoomLevel?.()

      if (current != null && Math.abs(current - saved) < 1e-9) {
        return
      }

      applyZoomLevel(window.webContents, saved)

      return
    }

    // No JSON yet: paint the shipped default immediately so a fresh install
    // doesn't flash Chromium 100%, then try localStorage for pre-JSON installs
    // and overwrite if a legacy value is there.
    applyZoomLevel(window.webContents, DEFAULT_ZOOM_LEVEL)

    window.webContents
      .executeJavaScript(
        `(() => { try { return localStorage.getItem(${JSON.stringify(ZOOM_STORAGE_KEY)}) } catch { return null } })()`
      )
      .then(stored => {
        if (!window || window.isDestroyed()) {
          return
        }

        const level = stored == null ? DEFAULT_ZOOM_LEVEL : Number(stored)
        const applied = applyZoomLevel(window.webContents, level)
        writeZoomState(applied)
      })
      .catch(error => rememberLog(`[zoom] restore failed: ${error?.message || error}`))
  }

  function installZoomShortcuts(window) {
    // Override Ctrl/Cmd + +/-/0 with half Chromium's default zoom step (ZOOM_STEP
    // is 0.1 vs Chromium's 0.2). The menu items handle this on macOS (where the
    // menu is always present), but on Linux/Windows the menu is null and
    // Chromium's default handler would use the full 0.2 step, so we intercept
    // here for consistency. Ctrl/Cmd+0 resets to DEFAULT_ZOOM_LEVEL, not Chromium 0.
    window.webContents.on('before-input-event', (event, input) => {
      const mod = IS_MAC ? input.meta : input.control

      if (!mod || input.alt) {
        return
      }

      const key = input.key

      if (key === '0') {
        if (input.shift) {
          return // Ctrl/Cmd+Shift+0 is not a zoom chord — leave it alone
        }

        event.preventDefault()
        setAndPersistZoomLevel(window, DEFAULT_ZOOM_LEVEL)
      } else if (key === '=' || key === '+') {
        // Zoom-in must accept the shift modifier: on US layouts Plus is
        // physically Shift+=, so Cmd+Plus arrives as Cmd+Shift+'+' (or '='
        // depending on platform). The old blanket shift guard silently
        // dropped keyboard zoom-in on macOS (#43517).
        event.preventDefault()
        setAndPersistZoomLevel(window, window.webContents.getZoomLevel() + ZOOM_STEP)
      } else if (key === '-') {
        if (input.shift) {
          return // Shift+'-' is '_' territory on most layouts, not zoom-out
        }

        event.preventDefault()
        setAndPersistZoomLevel(window, window.webContents.getZoomLevel() - ZOOM_STEP)
      }
    })

    // Ctrl/Cmd + mouse wheel — the standard desktop/browser zoom gesture
    // (#40295). Chromium surfaces it as the main-process 'zoom-changed' event
    // (wheel events are DOM-side, so before-input-event never sees them).
    // Route through the same persist+notify funnel as the keyboard shortcuts
    // so wheel zoom survives restarts and the settings Scale control stays in
    // sync, and use the same half step for consistency.
    window.webContents.on('zoom-changed', (event, zoomDirection) => {
      event.preventDefault()
      const delta = zoomDirection === 'in' ? ZOOM_STEP : -ZOOM_STEP
      setAndPersistZoomLevel(window, window.webContents.getZoomLevel() + delta)
    })
  }

  /**
   * The custom (renderer) context menu's main-process half.
   *
   * The app popups no native menus: the renderer owns the menu UI so labels
   * are translated with the rest of the app. Main keeps only what Chromium
   * reports here and the renderer cannot see:
   *  - spell-check facts (misspelled word + suggestions) — forwarded so the
   *    renderer appends them to its already-open menu,
   *  - the gesture coordinates — kept for copyImageAt, which needs them.
   */
  const lastContextMenuPoint = new Map<number, { x: number; y: number }>()

  function installContextMenuBridge(window: BrowserWindow) {
    window.webContents.on('context-menu', (_event, params) => {
      lastContextMenuPoint.set(window.webContents.id, { x: params.x, y: params.y })

      const suggestions = Array.isArray(params.dictionarySuggestions) ? params.dictionarySuggestions : []

      if (params.isEditable && params.misspelledWord) {
        window.webContents.send('hermes:context-menu-spellcheck', {
          misspelledWord: params.misspelledWord,
          suggestions
        })
      }
    })
  }

  // Microphone and camera capture. The voice composer drives mic access and
  // renderer features (e.g. desktop plugins) can drive camera access, both
  // through getUserMedia, which Chromium gates behind these two session hooks.
  //
  // The naive `details.mediaTypes.includes('audio')` check works on macOS but
  // breaks on Windows: Chromium frequently fires the request with an empty or
  // undefined `mediaTypes`, so a strict check denies it and getUserMedia throws
  // NotAllowedError. We therefore allow the capture permissions and treat absent
  // metadata as allowed.
  //
  // Granting here is not the last gate: the OS still applies its own capture
  // permission (macOS TCC prompts on first use, per the NSMicrophone/NSCamera
  // usage strings), so the user keeps a real allow/deny and can revoke it in
  // System Settings afterwards.
  function isMediaCapturePermission(permission, details) {
    // HTML5 video/audio fullscreen asks the request handler for 'fullscreen'
    // and the check handler for 'automatic-fullscreen'. Both must be allowed
    // or the native fullscreen button on <video controls> does nothing.
    if (permission === 'fullscreen' || permission === 'automatic-fullscreen') {
      return true
    }

    if (permission === 'audioCapture' || permission === 'videoCapture') {
      return true
    }

    if (permission !== 'media') {
      return false
    }

    const mediaTypes = details?.mediaTypes

    // Windows: mediaTypes is often empty for a capture request. Don't deny on
    // missing metadata.
    if (!Array.isArray(mediaTypes) || mediaTypes.length === 0) {
      return true
    }

    return mediaTypes.includes('audio') || mediaTypes.includes('video')
  }

  // Chromium-initiated downloads (renderer anchor/blob downloads, drag-outs)
  // land here. Without a handler the OS save dialog opens with the process cwd
  // as the default directory (win-unpacked in packaged installs) and whatever
  // extensionless name the anchor carried. Route every download to the user's
  // Downloads directory and guarantee a MIME-derived extension.
  function installDownloadHandling() {
    session.defaultSession.on('will-download', (_event, item) => {
      const suggested = item.getFilename() || 'download'
      const hasExtension = Boolean(path.extname(suggested))
      const extension = hasExtension ? '' : extensionForMimeType(item.getMimeType())
      const filename = `${suggested}${extension}`

      try {
        item.setSaveDialogOptions({
          title: 'Save File',
          defaultPath: path.join(app.getPath('downloads'), filename),
          filters:
            extension || /^image\//i.test(item.getMimeType() || '')
              ? [
                  { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'svg'] },
                  { name: 'All Files', extensions: ['*'] }
                ]
              : undefined
        })
      } catch {
        // No Downloads directory to offer — keep Chromium's default prompt.
      }
    })
  }

  function installMediaPermissions() {
    // Async request handler: the prompt-style path (most platforms).
    session.defaultSession.setPermissionRequestHandler((_webContents, permission, callback, details) => {
      callback(isMediaCapturePermission(permission, details))
    })

    // Synchronous check handler: Chromium consults this for getUserMedia on
    // Windows in addition to (or instead of) the request handler. Without it,
    // the check defaults to false and capture is denied before the request
    // handler ever runs.
    session.defaultSession.setPermissionCheckHandler((_webContents, permission) => {
      return (
        permission === 'media' ||
        (permission as string) === 'automatic-fullscreen' ||
        permission === ('audioCapture' as any) /* todo: is this needed? */ ||
        permission === ('videoCapture' as any)
      )
    })
  }

  return {
    buildApplicationMenu,
    installDevToolsShortcut,
    installPreviewShortcut,
    setAndPersistZoomLevel,
    restorePersistedZoomLevel,
    installZoomShortcuts,
    lastContextMenuPoint,
    installContextMenuBridge,
    installDownloadHandling,
    installMediaPermissions
  }
}
