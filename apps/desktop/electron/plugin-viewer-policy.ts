import type { App, session } from 'electron'

/** Install before creating app windows: attachment precedes the guest's first navigation. */
export function installViewerGuestPolicy(app: Pick<App, 'on'>, sessions: Pick<typeof session, 'fromPartition'>) {
  app.on('web-contents-created', (_event, contents) => {
    contents.on('will-attach-webview', (event, preferences, params) => {
      const partition = params.partition
      const isolatedViewer = /^hermes-viewer-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(partition ?? '')
      const browserPartition = partition === 'persist:hermes-preview' || partition === 'persist:hermes-embed'

      // The renderer may choose a Browser jar or a fresh viewer, not OAuth/app
      // storage. Electron's session preference takes precedence over partition.
      if (
        (!isolatedViewer && !browserPartition) ||
        preferences.preload ||
        ('preloadURL' in preferences && preferences.preloadURL) ||
        preferences.session ||
        (preferences.partition !== undefined && preferences.partition !== partition)
      ) {
        event.preventDefault()

        return
      }

      Object.assign(preferences, {
        partition,
        nodeIntegration: false,
        nodeIntegrationInWorker: false,
        nodeIntegrationInSubFrames: false,
        contextIsolation: true,
        sandbox: true,
        webviewTag: false,
        webSecurity: true,
        allowRunningInsecureContent: false
      })

      // Leave ordinary Browser permission handlers and cookies untouched.
      if (isolatedViewer) {
        const isolated = sessions.fromPartition(partition)
        isolated.setPermissionCheckHandler(() => false)
        isolated.setPermissionRequestHandler((_contents, _permission, callback) => callback(false))
      }
    })
  })
}

/** Shared renderer/main policy: never promote a viewer URL to native privileges. */
export function safeViewerUrl(value: unknown): string | null {
  if (typeof value !== 'string' || value.length > 16384 || /[\s\\]/.test(value)) {
    return null
  }

  try {
    const url = new URL(value)

    return ['http:', 'https:'].includes(url.protocol) && url.hostname && !url.username && !url.password ? value : null
  } catch {
    return null
  }
}
