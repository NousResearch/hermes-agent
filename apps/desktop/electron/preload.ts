import { contextBridge, ipcRenderer, webFrame, webUtils } from 'electron'

import type { DesktopProfileRoute } from './desktop-profile'
import type { HudModifierApi, HudModifierStatus } from './hud-modifier-types'
import { customWindowControlsEnabled } from './window-controls'

// Which translucency the OS can back. Asked synchronously because the renderer
// needs it before its first paint, and answered by main because deciding it
// needs `os.release()` — a sandboxed preload may only require electron, events,
// timers and url, so importing node:os here throws before contextBridge runs
// and takes the ENTIRE bridge down with it (window.rabbitDesktop undefined =>
// "Desktop IPC bridge is unavailable"). No reply means no glass, which degrades
// to an ordinary opaque window rather than a page thinned over nothing.
const translucencySupport = ipcRenderer.sendSync('rabbit:translucency:support')
const hudWindowing = ipcRenderer.sendSync('rabbit:hud:windowing')
const hudNativeDrag = hudWindowing?.nativeDrag === true

const launchFlags: { localModels?: boolean; guestOnboarding?: boolean } | undefined =
  ipcRenderer.sendSync('rabbit:feature-flags')

// Local, sanitized skin payload for the first renderer theme paint. This does
// not wait on `gateway.ready`, so an unreachable remote primary cannot force
// the built-in palette over the skin configured on this machine.
const localSkin = ipcRenderer.sendSync('rabbit:skin:local')

import { unwrapExpectedNotFound } from './api-expected-404'

contextBridge.exposeInMainWorld('rabbitDesktop', {
  glassSupported: translucencySupport?.glass === true,
  translucencySupported: translucencySupport?.translucency === true,
  // Launch-flag fact: the app was started with --local, so the renderer may
  // show the local-models surfaces. Static for the window's lifetime.
  localModelsEnabled: launchFlags?.localModels === true,
  // Launch-flag fact: the Nous free tier is on for this launch
  // (RABBIT_GUEST_ONBOARDING=1 or --guest-onboarding). Read-only; the same
  // decision is stamped onto every backend the app spawns.
  guestOnboardingEnabled: launchFlags?.guestOnboarding === true,
  localSkin: localSkin && typeof localSkin === 'object' ? localSkin : null,
  getConnection: (profile, opts) => ipcRenderer.invoke('rabbit:connection', profile, opts),
  // Loopback origin that hosts YouTube's player for the file:// renderer.
  getEmbedHostOrigin: () => ipcRenderer.invoke('rabbit:embed-host:origin'),
  // Registry-scoped backend resolution: { connectionId, profile } → descriptor.
  getConnectionFor: payload => ipcRenderer.invoke('rabbit:connection:for', payload),
  getProfileRoutes: profiles => ipcRenderer.invoke('rabbit:plugin-profile-routes', profiles),
  revalidateConnection: () => ipcRenderer.invoke('rabbit:connection:revalidate'),
  touchBackend: (profile, options) => ipcRenderer.invoke('rabbit:backend:touch', profile, options),
  getPoolLimits: () => ipcRenderer.invoke('rabbit:pool-limits:get'),
  setPoolLimits: limits => ipcRenderer.invoke('rabbit:pool-limits:set', limits),
  getGatewayWsUrl: profile => ipcRenderer.invoke('rabbit:gateway:ws-url', profile),
  // Registry-scoped fresh WS URL: { connectionId, profile } → result shape of
  // getGatewayWsUrl, minted against that connection's backend.
  getGatewayWsUrlFor: payload => ipcRenderer.invoke('rabbit:gateway:ws-url-for', payload),
  // Union agent roster across every registered connection.
  getAgentRoster: () => ipcRenderer.invoke('rabbit:agents:roster'),
  openSessionWindow: (sessionId, opts) => ipcRenderer.invoke('rabbit:window:openSession', sessionId, opts),
  openSessionInTerminal: (sessionId, opts) => ipcRenderer.invoke('rabbit:window:openInTerminal', sessionId, opts),
  openWindow: (options?: DesktopProfileRoute) => ipcRenderer.invoke('rabbit:window:openInstance', options),
  openBrowserWindow: tabId => ipcRenderer.invoke('rabbit:window:openBrowser', tabId),
  windowRelay: {
    send: payload => ipcRenderer.send('rabbit:window:relay', payload),
    onMessage: callback => {
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on('rabbit:window:relay', listener)

      return () => ipcRenderer.removeListener('rabbit:window:relay', listener)
    }
  },
  onBrowserPopoutClosed: callback => {
    const listener = (_event, tabId) => callback(tabId)
    ipcRenderer.on('rabbit:browser-popout:closed', listener)

    return () => ipcRenderer.removeListener('rabbit:browser-popout:closed', listener)
  },
  claimAmbientCue: key => ipcRenderer.invoke('rabbit:ambient:claim', key),
  windowControls: {
    custom: customWindowControlsEnabled(),
    minimize: () => ipcRenderer.send('rabbit:window-control', 'minimize'),
    toggleMaximize: () => ipcRenderer.send('rabbit:window-control', 'toggle-maximize'),
    close: () => ipcRenderer.send('rabbit:window-control', 'close')
  },
  wakeIndicator: {
    getState: () => ipcRenderer.invoke('rabbit:wake-indicator:get'),
    setState: state => ipcRenderer.send('rabbit:wake-indicator:set', state),
    onState: callback => {
      const listener = (_event, state) => callback(state)
      ipcRenderer.on('rabbit:wake-indicator:state', listener)

      return () => ipcRenderer.removeListener('rabbit:wake-indicator:state', listener)
    }
  },
  chatOnboarding: {
    grow: request => ipcRenderer.send('rabbit:chat-onboarding:grow', request),
    soloBoot: () => ipcRenderer.send('rabbit:chat-onboarding:solo-boot')
  },
  petOverlay: {
    // Main renderer → main process: window lifecycle + drag. `request` is
    // `{ bounds, screen }`; resolves with the screen bounds it actually used.
    open: request => ipcRenderer.invoke('rabbit:pet-overlay:open', request),
    close: () => ipcRenderer.invoke('rabbit:pet-overlay:close'),
    setBounds: bounds => ipcRenderer.send('rabbit:pet-overlay:set-bounds', bounds),
    setIgnoreMouse: ignore => ipcRenderer.send('rabbit:pet-overlay:ignore-mouse', ignore),
    // Flip the overlay focusable (and focus it) while the composer needs keys.
    setFocusable: focusable => ipcRenderer.send('rabbit:pet-overlay:set-focusable', focusable),
    // Main renderer → overlay (forwarded by main): push the latest pet state.
    pushState: payload => ipcRenderer.send('rabbit:pet-overlay:state', payload),
    // Overlay → main renderer (forwarded by main): pop back in / composer submit.
    control: payload => ipcRenderer.send('rabbit:pet-overlay:control', payload),
    // Overlay subscribes to state pushes.
    onState: callback => {
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on('rabbit:pet-overlay:state', listener)

      return () => ipcRenderer.removeListener('rabbit:pet-overlay:state', listener)
    },
    // Main renderer subscribes to overlay control messages.
    onControl: callback => {
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on('rabbit:pet-overlay:control', listener)

      return () => ipcRenderer.removeListener('rabbit:pet-overlay:control', listener)
    }
  },
  // HUD mode: the chrome-free floating chat. A full app renderer (own gateway)
  // sized as a floating bar, so it mounts the real composer. Main owns the
  // window; `onChanged` keeps every window's toggle truthful.
  hud: {
    nativeDrag: hudNativeDrag,
    windowing: {
      clientPlacement: hudWindowing?.clientPlacement !== false,
      controlDrag: hudWindowing?.controlDrag === true,
      nativeDrag: hudNativeDrag,
      solid: hudWindowing?.solid === true,
      workspaceTransfer: hudWindowing?.workspaceTransfer === true
    },
    open: request => ipcRenderer.invoke('rabbit:hud:open', request),
    close: () => ipcRenderer.invoke('rabbit:hud:close'),
    setIgnoreMouse: ignore => ipcRenderer.send('rabbit:hud:ignore-mouse', ignore),
    beginMove: () => ipcRenderer.send('rabbit:hud:begin-move'),
    endMove: () => ipcRenderer.send('rabbit:hud:end-move'),
    moveBy: delta => ipcRenderer.send('rabbit:hud:move-by', delta),
    setWorkspaceTransfer: transferring => ipcRenderer.send('rabbit:hud:workspace-transfer', transferring),
    setBounds: bounds => ipcRenderer.send('rabbit:hud:set-bounds', bounds),
    resetLayout: () => ipcRenderer.invoke('rabbit:hud:reset-layout'),
    // Whether the band covers the window below the bar. Main pairs it with the
    // user's translucency setting to decide the native frost (macOS vibrancy /
    // Windows 11 DWM backdrop) — see hudFrostFor.
    setFrost: showing => ipcRenderer.invoke('rabbit:hud:frost', showing),
    // The HUD tells main which session it is on; main hands that back to the
    // app window when the HUD closes, so the app can re-home onto it.
    setSession: sessionId => ipcRenderer.send('rabbit:hud:session', sessionId),
    onGoto: callback => {
      const listener = (_event, sessionId) => callback(sessionId)
      ipcRenderer.on('rabbit:hud:goto', listener)

      return () => ipcRenderer.removeListener('rabbit:hud:goto', listener)
    },
    onChanged: callback => {
      const listener = (_event, state) => callback(state)
      ipcRenderer.on('rabbit:hud:changed', listener)

      return () => ipcRenderer.removeListener('rabbit:hud:changed', listener)
    },
    // Linux only, and silent elsewhere: where the cursor is, in page
    // coordinates, or null when it has left the window. Stands in for the
    // mousemove that `setIgnoreMouseEvents(true, { forward: true })` delivers on
    // macOS and Windows but not here.
    onCursor: callback => {
      const listener = (_event, point) => callback(point)
      ipcRenderer.on('rabbit:hud:cursor', listener)

      return () => ipcRenderer.removeListener('rabbit:hud:cursor', listener)
    },
    // Main's game-overlay watch: whether a fullscreen app (a game) is under
    // the HUD, so the renderer can step back to the low-opacity overlay
    // treatment while one owns the screen.
    onGameOverlay: callback => {
      const listener = (_event, state) => callback(state)
      ipcRenderer.on('rabbit:hud:game-overlay', listener)

      return () => ipcRenderer.removeListener('rabbit:hud:game-overlay', listener)
    }
  },
  hudModifier: {
    getSettings: () => ipcRenderer.invoke('rabbit:hud-modifier:settings:get'),
    setEnabled: enabled => ipcRenderer.invoke('rabbit:hud-modifier:settings:set', enabled),
    openPermissionSettings: () => ipcRenderer.invoke('rabbit:hud-modifier:permission'),
    onStatus: callback => {
      const listener = (_event: Electron.IpcRendererEvent, status: HudModifierStatus) => callback(status)
      ipcRenderer.on('rabbit:hud-modifier:status', listener)

      return () => ipcRenderer.removeListener('rabbit:hud-modifier:status', listener)
    }
  } satisfies HudModifierApi,
  // macOS native screenshot gesture; captures require a main-issued request.
  screenshot:
    process.platform === 'darwin'
      ? {
          getSettings: () => ipcRenderer.invoke('rabbit:screenshot:settings:get'),
          setEnabled: enabled => ipcRenderer.invoke('rabbit:screenshot:settings:set', enabled),
          openPermissionSettings: kind => ipcRenderer.invoke('rabbit:screenshot:permission', kind),
          capture: requestId => ipcRenderer.invoke('rabbit:screenshot:capture', requestId),
          onStatus: callback => {
            const listener = (_event, status) => callback(status)
            ipcRenderer.on('rabbit:screenshot:status', listener)

            return () => ipcRenderer.removeListener('rabbit:screenshot:status', listener)
          },
          onRequest: callback => {
            const channel = 'rabbit:screenshot:request'
            const listener = (_event, requestId) => callback(requestId)

            if (ipcRenderer.listenerCount(channel) === 0) {
              ipcRenderer.send('rabbit:screenshot:subscribe', true)
            }

            ipcRenderer.on(channel, listener)

            return () => {
              ipcRenderer.removeListener(channel, listener)

              if (ipcRenderer.listenerCount(channel) === 0) {
                ipcRenderer.send('rabbit:screenshot:subscribe', false)
              }
            }
          }
        }
      : undefined,
  // Quick Entry: the global-hotkey mini composer window. Main owns the OS
  // shortcut + the persisted preference; the quick window only captures text
  // and hands it back, and the primary renderer submits it through the normal
  // prompt path.
  quickEntry: {
    getSettings: () => ipcRenderer.invoke('rabbit:quick-entry:settings:get'),
    setSettings: patch => ipcRenderer.invoke('rabbit:quick-entry:settings:set', patch),
    // Invoke returns the delivery result so the draft is not lost (#85590).
    submit: payload => ipcRenderer.invoke('rabbit:quick-entry:submit', payload),
    // Main cannot invoke the primary renderer, so it receives this ack (#85590).
    ackSubmit: (correlationId, result) => ipcRenderer.send('rabbit:quick-entry:ack', { correlationId, result }),
    dismiss: () => ipcRenderer.send('rabbit:quick-entry:dismiss'),
    // Primary renderer → main → quick window: gateway connection state + the
    // recent-session options the target picker offers. Main caches the latest
    // payload so a freshly spawned quick window starts from truth.
    pushState: payload => ipcRenderer.send('rabbit:quick-entry:state', payload),
    // Quick window subscribes to those pushes.
    onState: callback => {
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on('rabbit:quick-entry:state', listener)

      return () => ipcRenderer.removeListener('rabbit:quick-entry:state', listener)
    },
    // Main → primary renderer: a submit captured by the quick window.
    onSubmit: callback => {
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on('rabbit:quick-entry:submit', listener)

      return () => ipcRenderer.removeListener('rabbit:quick-entry:submit', listener)
    },
    // Main → quick window: you were just summoned (reset draft + refocus).
    onShown: callback => {
      const listener = () => callback()
      ipcRenderer.on('rabbit:quick-entry:shown', listener)

      return () => ipcRenderer.removeListener('rabbit:quick-entry:shown', listener)
    },
    // Main → quick window: the outcome of a submit whose relay already timed
    // out. Delivery is now KNOWN — reconcile the unknown state instead of
    // leaving the user to resend a prompt that may already be delivered.
    onLateResult: callback => {
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on('rabbit:quick-entry:late-result', listener)

      return () => ipcRenderer.removeListener('rabbit:quick-entry:late-result', listener)
    }
  },
  getBootProgress: () => ipcRenderer.invoke('rabbit:boot-progress:get'),
  getConnectionConfig: profile => ipcRenderer.invoke('rabbit:connection-config:get', profile),
  saveConnectionConfig: payload => ipcRenderer.invoke('rabbit:connection-config:save', payload),
  applyConnectionConfig: payload => ipcRenderer.invoke('rabbit:connection-config:apply', payload),
  testConnectionConfig: payload => ipcRenderer.invoke('rabbit:connection-config:test', payload),
  // Opt-in OS-keychain encryption for stored gateway secrets (default off —
  // see secret-storage-policy.ts). get never touches the OS keychain.
  getSecretStorageEncryption: () => ipcRenderer.invoke('rabbit:secret-storage:get'),
  setSecretStorageEncryption: (on: boolean) => ipcRenderer.invoke('rabbit:secret-storage:set', on),
  // v2 multi-connection registry: named agent sources (local / remote / ssh).
  connections: {
    list: () => ipcRenderer.invoke('rabbit:connections:list'),
    save: payload => ipcRenderer.invoke('rabbit:connections:save', payload),
    remove: id => ipcRenderer.invoke('rabbit:connections:remove', id),
    setPrimary: id => ipcRenderer.invoke('rabbit:connections:set-primary', id),
    setLaunchMode: mode => ipcRenderer.invoke('rabbit:connections:set-launch-mode', mode),
    setLastUsed: id => ipcRenderer.invoke('rabbit:connections:set-last-used', id),
    test: id => ipcRenderer.invoke('rabbit:connections:test', id),
    updateManaged: id => ipcRenderer.invoke('rabbit:connections:update-managed', id),
    // Fan out `rabbit update` to every eligible registered connection.
    // Optional excludeIds skips rows the caller updates through another path.
    updateAll: options => ipcRenderer.invoke('rabbit:connections:update-all', options),
    // Registry lifecycle push (main → renderer): a connection was removed or
    // materially edited, so secondaries scoped to it must be disposed (and,
    // for edits, re-dialed at the new target).
    onChanged: callback => {
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on('rabbit:connections:changed', listener)

      return () => ipcRenderer.removeListener('rabbit:connections:changed', listener)
    }
  },
  sshConfigHosts: () => ipcRenderer.invoke('rabbit:ssh-config:hosts'),
  sshResolveHost: host => ipcRenderer.invoke('rabbit:ssh-config:resolve', host),
  probeConnectionConfig: remoteUrl => ipcRenderer.invoke('rabbit:connection-config:probe', remoteUrl),
  // `options` lets a registry-editor draft sign in BEFORE it is saved: the
  // main process settles the draft's connection id up front so the login
  // window writes into the per-connection cookie jar the saved entry will
  // read (not the legacy shared jar an unsaved URL would fall back to).
  oauthLoginConnectionConfig: (remoteUrl, options) =>
    ipcRenderer.invoke('rabbit:connection-config:oauth-login', remoteUrl, options),
  oauthLogoutConnectionConfig: remoteUrl => ipcRenderer.invoke('rabbit:connection-config:oauth-logout', remoteUrl),
  profile: {
    getDefault: () => ipcRenderer.invoke('rabbit:profile:default:get'),
    setDefault: (route: DesktopProfileRoute) => ipcRenderer.invoke('rabbit:profile:default:set', route),
    onDefaultChanged: (callback: (route: DesktopProfileRoute | null) => void) => {
      const listener = (_event: Electron.IpcRendererEvent, route: DesktopProfileRoute | null) => callback(route)
      ipcRenderer.on('rabbit:profile:default:changed', listener)

      return () => ipcRenderer.removeListener('rabbit:profile:default:changed', listener)
    },
    get: () => ipcRenderer.invoke('rabbit:profile:get'),
    remember: name => ipcRenderer.invoke('rabbit:profile:remember', name),
    set: name => ipcRenderer.invoke('rabbit:profile:set', name)
  },
  // The handler resolves an expected 404 with a sentinel instead of rejecting
  // (Electron logs a stack for every rejected invoke). Turn it back into the
  // rejection the renderer expects — see electron/api-expected-404.ts.
  api: request => ipcRenderer.invoke('rabbit:api', request).then(unwrapExpectedNotFound),
  notify: payload => ipcRenderer.invoke('rabbit:notify', payload),
  claimStartupLatency: () => ipcRenderer.invoke('rabbit:startup-latency:claim'),
  requestMicrophoneAccess: () => ipcRenderer.invoke('rabbit:requestMicrophoneAccess'),
  readWindowBelow: () => ipcRenderer.invoke('rabbit:window:readBelow'),
  readFileDataUrl: filePath => ipcRenderer.invoke('rabbit:readFileDataUrl', filePath),
  readFileDataUrlForAttach: filePath => ipcRenderer.invoke('rabbit:readFileDataUrlForAttach', filePath),
  dataUrlReadMax: {
    get: () => ipcRenderer.invoke('rabbit:data-url-read-max:get'),
    set: maxMb => ipcRenderer.invoke('rabbit:data-url-read-max:set', maxMb)
  },
  readFileText: filePath => ipcRenderer.invoke('rabbit:readFileText', filePath),
  readPluginSource: (filePath: string) => ipcRenderer.invoke('rabbit:readPluginSource', filePath),
  selectPaths: options => ipcRenderer.invoke('rabbit:selectPaths', options),
  selectSavePath: options => ipcRenderer.invoke('rabbit:selectSavePath', options),
  writeClipboard: text => ipcRenderer.invoke('rabbit:writeClipboard', text),
  readClipboard: () => ipcRenderer.invoke('rabbit:readClipboard'),
  saveGatewayFile: payload => ipcRenderer.invoke('rabbit:saveGatewayFile', payload),
  saveImageFromUrl: url => ipcRenderer.invoke('rabbit:saveImageFromUrl', url),
  contextMenuEdit: command => ipcRenderer.invoke('rabbit:context-menu:edit', command),
  contextMenuCopyImage: () => ipcRenderer.invoke('rabbit:context-menu:copy-image'),
  contextMenuSpellcheck: action => ipcRenderer.invoke('rabbit:context-menu:spellcheck', action),
  contextMenuGuestAddWord: payload => ipcRenderer.invoke('rabbit:context-menu:guest-add-word', payload),
  onContextMenuSpellcheck: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('rabbit:context-menu-spellcheck', listener)

    return () => ipcRenderer.removeListener('rabbit:context-menu-spellcheck', listener)
  },
  saveImageBuffer: (data, ext, name) => ipcRenderer.invoke('rabbit:saveImageBuffer', { data, ext, name }),
  capturePreview: payload => ipcRenderer.invoke('rabbit:capturePreview', payload),
  savePastedText: text => ipcRenderer.invoke('rabbit:savePastedText', { text }),
  saveClipboardImage: () => ipcRenderer.invoke('rabbit:saveClipboardImage'),
  getPathForFile: file => {
    try {
      return webUtils.getPathForFile(file) || ''
    } catch {
      return ''
    }
  },
  normalizePreviewTarget: (target, baseDir) => ipcRenderer.invoke('rabbit:normalizePreviewTarget', target, baseDir),
  watchPreviewFile: url => ipcRenderer.invoke('rabbit:watchPreviewFile', url),
  watchDirectory: dir => ipcRenderer.invoke('rabbit:watchDirectory', dir),
  stopPreviewFileWatch: id => ipcRenderer.invoke('rabbit:stopPreviewFileWatch', id),
  setActiveWork: payload => ipcRenderer.send('rabbit:active-work', payload),
  setTitleBarTheme: payload => ipcRenderer.send('rabbit:titlebar-theme', payload),
  setNativeTheme: mode => ipcRenderer.send('rabbit:native-theme', mode),
  setTranslucency: payload => ipcRenderer.send('rabbit:translucency', payload),
  setKeepAwake: mode => ipcRenderer.send('rabbit:keep-awake', mode),
  minimizeToTray: {
    get: () => ipcRenderer.invoke('rabbit:minimize-to-tray:get'),
    set: on => ipcRenderer.invoke('rabbit:minimize-to-tray:set', on),
    onChanged: callback => {
      const listener = (_event, status) => callback(status)
      ipcRenderer.on('rabbit:minimize-to-tray:changed', listener)

      return () => ipcRenderer.removeListener('rabbit:minimize-to-tray:changed', listener)
    }
  },
  setDisableF12: blocked => ipcRenderer.send('rabbit:devtools:disable-f12', blocked),
  setF12ShortcutActive: active => ipcRenderer.send('rabbit:f12ShortcutActive', Boolean(active)),
  onF12Shortcut: callback => {
    const listener = (_event, input) => callback(input)
    ipcRenderer.on('rabbit:f12-shortcut', listener)

    return () => ipcRenderer.removeListener('rabbit:f12-shortcut', listener)
  },
  setPreviewShortcutActive: active => ipcRenderer.send('rabbit:previewShortcutActive', Boolean(active)),
  setPreviewGuestHidden: (webContentsId, hidden) =>
    ipcRenderer.send('rabbit:preview-guest-hidden', { webContentsId, hidden: Boolean(hidden) }),
  openExternal: url => ipcRenderer.invoke('rabbit:openExternal', url),
  mcpOauth: {
    // One-shot loopback listener for MCP OAuth against remote backends: bind
    // on this machine, hand redirectUri to mcp.servers.oauth.start, then wait
    // for the provider redirect and relay code/state via oauth.callback.
    listen: () => ipcRenderer.invoke('rabbit:mcp-oauth:listen'),
    wait: (id, timeoutMs) => ipcRenderer.invoke('rabbit:mcp-oauth:wait', id, timeoutMs),
    cancel: id => ipcRenderer.invoke('rabbit:mcp-oauth:cancel', id)
  },
  openPreviewInBrowser: url => ipcRenderer.invoke('rabbit:openPreviewInBrowser', url),
  reachPreviewUrl: url => ipcRenderer.invoke('rabbit:preview:reach', url),
  setActiveConnectionRoute: route => ipcRenderer.send('rabbit:connection:active-route', route),
  fetchLinkTitle: url => ipcRenderer.invoke('rabbit:fetchLinkTitle', url),
  resolveFavicon: url => ipcRenderer.invoke('rabbit:resolveFavicon', url),
  sanitizeWorkspaceCwd: cwd => ipcRenderer.invoke('rabbit:workspace:sanitize', cwd),
  settings: {
    getDefaultProjectDir: () => ipcRenderer.invoke('rabbit:setting:defaultProjectDir:get'),
    setDefaultProjectDir: dir => ipcRenderer.invoke('rabbit:setting:defaultProjectDir:set', dir),
    pickDefaultProjectDir: () => ipcRenderer.invoke('rabbit:setting:defaultProjectDir:pick')
  },
  zoom: {
    // Current zoom of this window, as { level, percent }.
    get: () => ipcRenderer.invoke('rabbit:zoom:get'),
    // Synchronous zoom factor (1 = 100%). Coordinate math needs it in the
    // same tick as the event it converts, so no IPC round-trip here.
    factor: () => webFrame.getZoomFactor(),
    setPercent: percent => ipcRenderer.send('rabbit:zoom:set-percent', percent),
    // Fires on every zoom change, including the Ctrl/Cmd +/-/0 shortcuts,
    // so the settings UI can stay in sync with the keyboard.
    onChanged: callback => {
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on('rabbit:zoom:changed', listener)

      return () => ipcRenderer.removeListener('rabbit:zoom:changed', listener)
    }
  },
  revealLogs: () => ipcRenderer.invoke('rabbit:logs:reveal'),
  getRecentLogs: () => ipcRenderer.invoke('rabbit:logs:recent'),
  // Fire-and-forget: persists a renderer error-boundary catch (with component
  // stack) to desktop.log so crashes survive the window (#79428).
  reportRendererError: report => ipcRenderer.send('rabbit:logs:renderer-error', report),
  logLine: (line: string): void => ipcRenderer.send('rabbit:logs:renderer-line', line),
  readDir: dirPath => ipcRenderer.invoke('rabbit:fs:readDir', dirPath),
  gitRoot: startPath => ipcRenderer.invoke('rabbit:fs:gitRoot', startPath),
  revealPath: targetPath => ipcRenderer.invoke('rabbit:fs:reveal', targetPath),
  openDir: dirPath => ipcRenderer.invoke('rabbit:fs:openDir', dirPath),
  desktopPluginsRoot: () => ipcRenderer.invoke('rabbit:fs:desktopPluginsRoot'),
  reconcileDesktopPlugins: () => ipcRenderer.invoke('rabbit:fs:reconcileDesktopPlugins'),
  logsRoot: (profile?: string) => ipcRenderer.invoke('rabbit:fs:logsRoot', profile),
  renamePath: (targetPath, newName) => ipcRenderer.invoke('rabbit:fs:rename', targetPath, newName),
  writeTextFile: (filePath, content) => ipcRenderer.invoke('rabbit:fs:writeText', filePath, content),
  trashPath: targetPath => ipcRenderer.invoke('rabbit:fs:trash', targetPath),
  git: {
    worktreeList: repoPath => ipcRenderer.invoke('rabbit:git:worktreeList', repoPath),
    worktreeAdd: (repoPath, options) => ipcRenderer.invoke('rabbit:git:worktreeAdd', repoPath, options),
    worktreeRemove: (repoPath, worktreePath, options) =>
      ipcRenderer.invoke('rabbit:git:worktreeRemove', repoPath, worktreePath, options),
    branchSwitch: (repoPath, branch) => ipcRenderer.invoke('rabbit:git:branchSwitch', repoPath, branch),
    branchList: repoPath => ipcRenderer.invoke('rabbit:git:branchList', repoPath),
    baseBranchList: repoPath => ipcRenderer.invoke('rabbit:git:baseBranchList', repoPath),
    repoStatus: repoPath => ipcRenderer.invoke('rabbit:git:repoStatus', repoPath),
    fileDiff: (repoPath, filePath) => ipcRenderer.invoke('rabbit:git:fileDiff', repoPath, filePath),
    scanRepos: (roots, options) => ipcRenderer.invoke('rabbit:git:scanRepos', roots, options),
    review: {
      list: (repoPath, scope, baseRef) => ipcRenderer.invoke('rabbit:git:review:list', repoPath, scope, baseRef),
      diff: (repoPath, filePath, scope, baseRef, staged) =>
        ipcRenderer.invoke('rabbit:git:review:diff', repoPath, filePath, scope, baseRef, staged),
      stage: (repoPath, filePath) => ipcRenderer.invoke('rabbit:git:review:stage', repoPath, filePath),
      unstage: (repoPath, filePath) => ipcRenderer.invoke('rabbit:git:review:unstage', repoPath, filePath),
      revert: (repoPath, filePath) => ipcRenderer.invoke('rabbit:git:review:revert', repoPath, filePath),
      revParse: (repoPath, ref) => ipcRenderer.invoke('rabbit:git:review:revParse', repoPath, ref),
      commit: (repoPath, message, push) => ipcRenderer.invoke('rabbit:git:review:commit', repoPath, message, push),
      commitContext: repoPath => ipcRenderer.invoke('rabbit:git:review:commitContext', repoPath),
      push: repoPath => ipcRenderer.invoke('rabbit:git:review:push', repoPath),
      shipInfo: repoPath => ipcRenderer.invoke('rabbit:git:review:shipInfo', repoPath),
      prList: (repoPath, branches, numbers) =>
        ipcRenderer.invoke('rabbit:git:review:prList', repoPath, branches, numbers),
      createPr: repoPath => ipcRenderer.invoke('rabbit:git:review:createPr', repoPath)
    }
  },
  terminal: {
    attach: id => ipcRenderer.invoke('rabbit:terminal:attach', id),
    cwd: id => ipcRenderer.invoke('rabbit:terminal:cwd', id),
    dispose: id => ipcRenderer.invoke('rabbit:terminal:dispose', id),
    resize: (id, size) => ipcRenderer.invoke('rabbit:terminal:resize', id, size),
    start: options => ipcRenderer.invoke('rabbit:terminal:start', options),
    write: (id, data) => ipcRenderer.invoke('rabbit:terminal:write', id, data),
    onData: (id, callback) => {
      const channel = `rabbit:terminal:${id}:data`
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on(channel, listener)

      return () => ipcRenderer.removeListener(channel, listener)
    },
    onExit: (id, callback) => {
      const channel = `rabbit:terminal:${id}:exit`
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on(channel, listener)

      return () => ipcRenderer.removeListener(channel, listener)
    }
  },
  onClosePreviewRequested: callback => {
    const listener = () => callback()
    ipcRenderer.on('rabbit:close-preview-requested', listener)

    return () => ipcRenderer.removeListener('rabbit:close-preview-requested', listener)
  },
  onPreviewNav: callback => {
    const listener = (_event, command) => callback(command)
    ipcRenderer.on('rabbit:preview-nav', listener)

    return () => ipcRenderer.removeListener('rabbit:preview-nav', listener)
  },
  onOpenFolderRequested: callback => {
    const listener = () => callback()
    ipcRenderer.on('rabbit:open-folder-requested', listener)

    return () => ipcRenderer.removeListener('rabbit:open-folder-requested', listener)
  },
  onOpenUpdatesRequested: callback => {
    const listener = () => callback()
    ipcRenderer.on('rabbit:open-updates', listener)

    return () => ipcRenderer.removeListener('rabbit:open-updates', listener)
  },
  onDeepLink: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('rabbit:deep-link', listener)

    return () => ipcRenderer.removeListener('rabbit:deep-link', listener)
  },
  signalDeepLinkReady: () => ipcRenderer.invoke('rabbit:deep-link-ready'),
  probePluginRepo: payload => ipcRenderer.invoke('rabbit:plugin:probe', payload),
  installDesktopPlugin: payload => ipcRenderer.invoke('rabbit:plugin:installDesktop', payload),
  removeDesktopPlugin: payload => ipcRenderer.invoke('rabbit:plugin:removeDesktop', payload),
  onWindowStateChanged: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('rabbit:window-state-changed', listener)

    return () => ipcRenderer.removeListener('rabbit:window-state-changed', listener)
  },
  onFocusSession: callback => {
    const listener = (_event, sessionId) => callback(sessionId)
    ipcRenderer.on('rabbit:focus-session', listener)

    return () => ipcRenderer.removeListener('rabbit:focus-session', listener)
  },
  onNotificationAction: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('rabbit:notification-action', listener)

    return () => ipcRenderer.removeListener('rabbit:notification-action', listener)
  },
  onNotificationActivate: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('rabbit:notification-activate', listener)

    return () => ipcRenderer.removeListener('rabbit:notification-activate', listener)
  },
  onExternalOpenFailed: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('rabbit:external-open-failed', listener)

    return () => ipcRenderer.removeListener('rabbit:external-open-failed', listener)
  },
  onPreviewFileChanged: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('rabbit:preview-file-changed', listener)

    return () => ipcRenderer.removeListener('rabbit:preview-file-changed', listener)
  },
  onBackendExit: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('rabbit:backend-exit', listener)

    return () => ipcRenderer.removeListener('rabbit:backend-exit', listener)
  },
  // Cooperative pool retirement (main → renderer): the pooled backend under
  // `poolKey` is being stopped for a foreground open. Park that scope; do not
  // redial into the slot it vacated.
  onPoolBackendRetiring: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('rabbit:pool:retiring', listener)

    return () => ipcRenderer.removeListener('rabbit:pool:retiring', listener)
  },
  // Soft gateway-mode apply finished tearing down the primary backend. Renderer
  // should wipe session lists + re-dial without a window reload.
  onConnectionApplied: callback => {
    const listener = () => callback()
    ipcRenderer.on('rabbit:connection:applied', listener)

    return () => ipcRenderer.removeListener('rabbit:connection:applied', listener)
  },
  onPowerResume: callback => {
    const listener = () => callback()
    ipcRenderer.on('rabbit:power-resume', listener)

    return () => ipcRenderer.removeListener('rabbit:power-resume', listener)
  },
  // AC ↔ battery transitions; renderers slow their backstop polls on battery.
  getOnBattery: () => ipcRenderer.invoke('rabbit:power-battery:get'),
  onBatteryChanged: callback => {
    const listener = (_event, onBattery) => callback(Boolean(onBattery))
    ipcRenderer.on('rabbit:power-battery', listener)

    return () => ipcRenderer.removeListener('rabbit:power-battery', listener)
  },
  onBootProgress: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('rabbit:boot-progress', listener)

    return () => ipcRenderer.removeListener('rabbit:boot-progress', listener)
  },
  // First-launch bootstrap progress -- emitted by the install.ps1 stage
  // runner in main.ts (apps/desktop/electron/bootstrap-runner.ts).
  // Renderer's install overlay subscribes to live events and queries the
  // current snapshot via getBootstrapState() to recover after a devtools
  // reload mid-bootstrap.
  getBootstrapState: () => ipcRenderer.invoke('rabbit:bootstrap:get'),
  probeLocalBackend: () => ipcRenderer.invoke('rabbit:local-backend:probe'),
  continueBootstrapLocal: () => ipcRenderer.invoke('rabbit:bootstrap:continue-local'),
  recycleBackend: profile => ipcRenderer.invoke('rabbit:backend:recycle', profile),
  resetBootstrap: () => ipcRenderer.invoke('rabbit:bootstrap:reset'),
  repairBootstrap: () => ipcRenderer.invoke('rabbit:bootstrap:repair'),
  cancelBootstrap: () => ipcRenderer.invoke('rabbit:bootstrap:cancel'),
  onBootstrapEvent: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('rabbit:bootstrap:event', listener)

    return () => ipcRenderer.removeListener('rabbit:bootstrap:event', listener)
  },
  getVersion: () => ipcRenderer.invoke('rabbit:version'),
  relaunchApp: () => ipcRenderer.invoke('rabbit:app:relaunch'),
  getMachineProfile: () => ipcRenderer.invoke('rabbit:machine:profile'),
  getRemoteDisplayReason: () => ipcRenderer.invoke('rabbit:get-remote-display-reason'),
  uninstall: {
    summary: () => ipcRenderer.invoke('rabbit:uninstall:summary'),
    run: mode => ipcRenderer.invoke('rabbit:uninstall:run', { mode })
  },
  updates: {
    check: opts => ipcRenderer.invoke('rabbit:updates:check', opts),
    apply: opts => ipcRenderer.invoke('rabbit:updates:apply', opts),
    getBranch: () => ipcRenderer.invoke('rabbit:updates:branch:get'),
    setBranch: name => ipcRenderer.invoke('rabbit:updates:branch:set', name),
    onProgress: callback => {
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on('rabbit:updates:progress', listener)

      return () => ipcRenderer.removeListener('rabbit:updates:progress', listener)
    },
    takePendingRun: () => ipcRenderer.invoke('rabbit:updates:metric:take'),
    ackPendingRun: sent => ipcRenderer.invoke('rabbit:updates:metric:ack', sent),
    onPendingRun: callback => {
      const listener = () => callback()
      ipcRenderer.on('rabbit:updates:metric:pending', listener)

      return () => ipcRenderer.removeListener('rabbit:updates:metric:pending', listener)
    }
  },
  desktopMetrics: {
    setEnabled: (on, profile) => ipcRenderer.invoke('rabbit:desktop-metrics:set-enabled', on, profile),
    takeRendererCrashes: () => ipcRenderer.invoke('rabbit:desktop-metrics:crash:take'),
    ackRendererCrashes: sent => ipcRenderer.invoke('rabbit:desktop-metrics:crash:ack', sent)
  },
  themes: {
    fetchMarketplace: id => ipcRenderer.invoke('rabbit:vscode-theme:fetch', id),
    searchMarketplace: query => ipcRenderer.invoke('rabbit:vscode-theme:search', query)
  },
  // Find-in-page (Ctrl/Cmd+F): delegates to Electron's
  // webContents.findInPage on the IPC sender's window so a Cmd+F pressed
  // in a secondary session window searches THAT window, not the primary.
  // `onFoundInPage` returns the unsubscribe fn; the renderer wires it via
  // `initFindInPageListener` in store/find-in-page.ts and tears it down
  // when the FindBar unmounts.
  findInPage: (query, options) => ipcRenderer.invoke('rabbit:find-in-page', query, options),
  stopFindInPage: () => ipcRenderer.invoke('rabbit:stop-find-in-page'),
  onFoundInPage: callback => {
    const listener = (_event, result) => callback(result)
    ipcRenderer.on('rabbit:found-in-page', listener)

    return () => ipcRenderer.removeListener('rabbit:found-in-page', listener)
  },
  // Main-process `before-input-event` forwards Ctrl/Cmd+F here so renderer
  // can open the FindBar even when the GTK compositor has already grabbed
  // the chord at the windowing layer (#81727).
  onOpenFindBarRequested: callback => {
    const listener = () => callback()
    ipcRenderer.on('rabbit:open-find-bar', listener)

    return () => ipcRenderer.removeListener('rabbit:open-find-bar', listener)
  }
})
