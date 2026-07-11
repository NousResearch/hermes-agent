const { contextBridge, ipcRenderer, webUtils } = require('electron')

contextBridge.exposeInMainWorld('hermesDesktop', {
  getConnection: profile => ipcRenderer.invoke('hermes:connection', profile),
  revalidateConnection: () => ipcRenderer.invoke('hermes:connection:revalidate'),
  touchBackend: profile => ipcRenderer.invoke('hermes:backend:touch', profile),
  getGatewayWsUrl: profile => ipcRenderer.invoke('hermes:gateway:ws-url', profile),
  openSessionWindow: (sessionId, opts) => ipcRenderer.invoke('hermes:window:openSession', sessionId, opts),
  openNewSessionWindow: () => ipcRenderer.invoke('hermes:window:openNewSession'),
  getBootProgress: () => ipcRenderer.invoke('hermes:boot-progress:get'),
  getConnectionConfig: profile => ipcRenderer.invoke('hermes:connection-config:get', profile),
  saveConnectionConfig: payload => ipcRenderer.invoke('hermes:connection-config:save', payload),
  applyConnectionConfig: payload => ipcRenderer.invoke('hermes:connection-config:apply', payload),
  testConnectionConfig: payload => ipcRenderer.invoke('hermes:connection-config:test', payload),
  probeConnectionConfig: remoteUrl => ipcRenderer.invoke('hermes:connection-config:probe', remoteUrl),
  oauthLoginConnectionConfig: remoteUrl => ipcRenderer.invoke('hermes:connection-config:oauth-login', remoteUrl),
  oauthLogoutConnectionConfig: remoteUrl => ipcRenderer.invoke('hermes:connection-config:oauth-logout', remoteUrl),
  profile: {
    get: () => ipcRenderer.invoke('hermes:profile:get'),
    set: name => ipcRenderer.invoke('hermes:profile:set', name)
  },
  // ApexNodes managed-LLM (zero-key) default path. See electron/apex-managed.cjs.
  managed: {
    status: () => ipcRenderer.invoke('hermes:managed:status'),
    // hc-512: live relay model-catalog state for the model menu ('ok' |
    // 'unauthorized' | 'unreachable' | 'unknown'); { refresh: true } re-probes.
    relayCatalog: opts => ipcRenderer.invoke('hermes:managed:relayCatalog', opts),
    signIn: payload => ipcRenderer.invoke('hermes:managed:signIn', payload),
    browserSignIn: payload => ipcRenderer.invoke('hermes:managed:browserSignIn', payload),
    signOut: () => ipcRenderer.invoke('hermes:managed:signOut'),
    // On-demand relay-key self-heal after a chat turn hit a relay auth error
    // (HTTP 401/403): re-provision + report whether it healed or the user must
    // sign in again. See electron/main.cjs hermes:managed:selfHeal.
    selfHeal: () => ipcRenderer.invoke('hermes:managed:selfHeal')
  },
  // hc-444: desktop ↔ cloud Feishu bridge — mirror the signed-in user's own
  // Feishu app credential down to light up the Feishu adapter + lark tools. See
  // electron/apex-feishu.cjs. No secret crosses to the renderer: status returns
  // only display fields; sync/disconnect return status objects.
  feishu: {
    status: () => ipcRenderer.invoke('hermes:feishu:status'),
    sync: () => ipcRenderer.invoke('hermes:feishu:sync'),
    disconnect: () => ipcRenderer.invoke('hermes:feishu:disconnect'),
    openBind: () => ipcRenderer.invoke('hermes:feishu:openBind')
  },
  // Platform client-config sync — informational read of the cached versioned
  // config (no network). Application happens in the MAIN process pre-gateway
  // (main.cjs applyClientConfigToRuntime); the renderer no longer applies.
  clientConfig: {
    get: () => ipcRenderer.invoke('hermes:clientConfig:get')
  },
  // Continuous auth gate: main broadcasts when a backend call returns 401
  // (login lost) or 403 account_disabled (account abnormal). The renderer
  // clears auth and returns to the login screen. See main.cjs broadcastAuthGate.
  onAuthGate: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('hermes:auth-gate', listener)
    return () => ipcRenderer.removeListener('hermes:auth-gate', listener)
  },
  // Runtime 3-end consistency — desktop opt-in engine update (R5). checkUpdate
  // compares the installed runtime against the admin-set default; applyUpdate
  // re-points the pin and re-runs bootstrap (renderer reloads when
  // reloadRequired is true). Both are safe no-ops offline.
  runtime: {
    // R6: installed engine version, read locally from the bootstrap marker.
    // No network / no state change — the About panel calls this on open.
    getVersion: () => ipcRenderer.invoke('hermes:runtime:version'),
    checkUpdate: () => ipcRenderer.invoke('hermes:runtime:check-update'),
    applyUpdate: () => ipcRenderer.invoke('hermes:runtime:apply-update')
  },
  // 壳(Electron 应用本体)自更新 — electron-updater,和上面的引擎(runtime)
  // 更新是两条通道。机制全在主进程(electron/shell-updater.cjs):静默检查+
  // 下载,状态经 onEvent 推给侧栏胶囊;install = quitAndInstall(应用退出重装)。
  shellUpdate: {
    getState: () => ipcRenderer.invoke('hermes:shell-update:get'),
    install: () => ipcRenderer.invoke('hermes:shell-update:install'),
    onEvent: callback => {
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on('hermes:shell-update:event', listener)
      return () => ipcRenderer.removeListener('hermes:shell-update:event', listener)
    }
  },
  api: request => ipcRenderer.invoke('hermes:api', request),
  notify: payload => ipcRenderer.invoke('hermes:notify', payload),
  requestMicrophoneAccess: () => ipcRenderer.invoke('hermes:requestMicrophoneAccess'),
  readFileDataUrl: filePath => ipcRenderer.invoke('hermes:readFileDataUrl', filePath),
  readFileText: filePath => ipcRenderer.invoke('hermes:readFileText', filePath),
  selectPaths: options => ipcRenderer.invoke('hermes:selectPaths', options),
  writeClipboard: text => ipcRenderer.invoke('hermes:writeClipboard', text),
  saveImageFromUrl: url => ipcRenderer.invoke('hermes:saveImageFromUrl', url),
  saveImageBuffer: (data, ext) => ipcRenderer.invoke('hermes:saveImageBuffer', { data, ext }),
  saveClipboardImage: () => ipcRenderer.invoke('hermes:saveClipboardImage'),
  getPathForFile: file => {
    try {
      return webUtils.getPathForFile(file) || ''
    } catch {
      return ''
    }
  },
  normalizePreviewTarget: (target, baseDir) => ipcRenderer.invoke('hermes:normalizePreviewTarget', target, baseDir),
  watchPreviewFile: url => ipcRenderer.invoke('hermes:watchPreviewFile', url),
  stopPreviewFileWatch: id => ipcRenderer.invoke('hermes:stopPreviewFileWatch', id),
  setTitleBarTheme: payload => ipcRenderer.send('hermes:titlebar-theme', payload),
  setNativeTheme: mode => ipcRenderer.send('hermes:native-theme', mode),
  setTranslucency: payload => ipcRenderer.send('hermes:translucency', payload),
  setPreviewShortcutActive: active => ipcRenderer.send('hermes:previewShortcutActive', Boolean(active)),
  openExternal: url => ipcRenderer.invoke('hermes:openExternal', url),
  fetchLinkTitle: url => ipcRenderer.invoke('hermes:fetchLinkTitle', url),
  sanitizeWorkspaceCwd: cwd => ipcRenderer.invoke('hermes:workspace:sanitize', cwd),
  createProjectDir: (parentDir, name) => ipcRenderer.invoke('hermes:workspace:createDir', parentDir, name),
  settings: {
    getDefaultProjectDir: () => ipcRenderer.invoke('hermes:setting:defaultProjectDir:get'),
    setDefaultProjectDir: dir => ipcRenderer.invoke('hermes:setting:defaultProjectDir:set', dir),
    pickDefaultProjectDir: () => ipcRenderer.invoke('hermes:setting:defaultProjectDir:pick')
  },
  revealLogs: () => ipcRenderer.invoke('hermes:logs:reveal'),
  getRecentLogs: () => ipcRenderer.invoke('hermes:logs:recent'),
  readDir: dirPath => ipcRenderer.invoke('hermes:fs:readDir', dirPath),
  gitRoot: startPath => ipcRenderer.invoke('hermes:fs:gitRoot', startPath),
  worktrees: cwds => ipcRenderer.invoke('hermes:fs:worktrees', cwds),
  terminal: {
    dispose: id => ipcRenderer.invoke('hermes:terminal:dispose', id),
    resize: (id, size) => ipcRenderer.invoke('hermes:terminal:resize', id, size),
    start: options => ipcRenderer.invoke('hermes:terminal:start', options),
    write: (id, data) => ipcRenderer.invoke('hermes:terminal:write', id, data),
    onData: (id, callback) => {
      const channel = `hermes:terminal:${id}:data`
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on(channel, listener)
      return () => ipcRenderer.removeListener(channel, listener)
    },
    onExit: (id, callback) => {
      const channel = `hermes:terminal:${id}:exit`
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on(channel, listener)
      return () => ipcRenderer.removeListener(channel, listener)
    }
  },
  onClosePreviewRequested: callback => {
    const listener = () => callback()
    ipcRenderer.on('hermes:close-preview-requested', listener)
    return () => ipcRenderer.removeListener('hermes:close-preview-requested', listener)
  },
  onOpenUpdatesRequested: callback => {
    const listener = () => callback()
    ipcRenderer.on('hermes:open-updates', listener)
    return () => ipcRenderer.removeListener('hermes:open-updates', listener)
  },
  onDeepLink: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('hermes:deep-link', listener)
    return () => ipcRenderer.removeListener('hermes:deep-link', listener)
  },
  signalDeepLinkReady: () => ipcRenderer.invoke('hermes:deep-link-ready'),
  onWindowStateChanged: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('hermes:window-state-changed', listener)
    return () => ipcRenderer.removeListener('hermes:window-state-changed', listener)
  },
  onFocusSession: callback => {
    const listener = (_event, sessionId) => callback(sessionId)
    ipcRenderer.on('hermes:focus-session', listener)
    return () => ipcRenderer.removeListener('hermes:focus-session', listener)
  },
  onNotificationAction: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('hermes:notification-action', listener)
    return () => ipcRenderer.removeListener('hermes:notification-action', listener)
  },
  onPreviewFileChanged: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('hermes:preview-file-changed', listener)
    return () => ipcRenderer.removeListener('hermes:preview-file-changed', listener)
  },
  onBackendExit: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('hermes:backend-exit', listener)
    return () => ipcRenderer.removeListener('hermes:backend-exit', listener)
  },
  onPowerResume: callback => {
    const listener = () => callback()
    ipcRenderer.on('hermes:power-resume', listener)
    return () => ipcRenderer.removeListener('hermes:power-resume', listener)
  },
  onBootProgress: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('hermes:boot-progress', listener)
    return () => ipcRenderer.removeListener('hermes:boot-progress', listener)
  },
  // First-launch bootstrap progress -- emitted by the install.ps1 stage
  // runner in main.cjs (apps/desktop/electron/bootstrap-runner.cjs).
  // Renderer's install overlay subscribes to live events and queries the
  // current snapshot via getBootstrapState() to recover after a devtools
  // reload mid-bootstrap.
  getBootstrapState: () => ipcRenderer.invoke('hermes:bootstrap:get'),
  resetBootstrap: () => ipcRenderer.invoke('hermes:bootstrap:reset'),
  repairBootstrap: () => ipcRenderer.invoke('hermes:bootstrap:repair'),
  cancelBootstrap: () => ipcRenderer.invoke('hermes:bootstrap:cancel'),
  onBootstrapEvent: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('hermes:bootstrap:event', listener)
    return () => ipcRenderer.removeListener('hermes:bootstrap:event', listener)
  },
  getVersion: () => ipcRenderer.invoke('hermes:version'),
  uninstall: {
    summary: () => ipcRenderer.invoke('hermes:uninstall:summary'),
    run: mode => ipcRenderer.invoke('hermes:uninstall:run', { mode })
  },
  updates: {
    check: () => ipcRenderer.invoke('hermes:updates:check'),
    apply: opts => ipcRenderer.invoke('hermes:updates:apply', opts),
    getBranch: () => ipcRenderer.invoke('hermes:updates:branch:get'),
    setBranch: name => ipcRenderer.invoke('hermes:updates:branch:set', name),
    onProgress: callback => {
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on('hermes:updates:progress', listener)
      return () => ipcRenderer.removeListener('hermes:updates:progress', listener)
    }
  },
  themes: {
    fetchMarketplace: id => ipcRenderer.invoke('hermes:vscode-theme:fetch', id),
    searchMarketplace: query => ipcRenderer.invoke('hermes:vscode-theme:search', query)
  }
})
