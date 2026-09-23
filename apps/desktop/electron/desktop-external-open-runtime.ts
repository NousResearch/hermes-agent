// Main retains the native APIs and path guard. The two opening paths share
// URL validation while preserving the file, WSL, and browser-specific routes.
interface DesktopExternalOpenDeps {
  IS_WSL: boolean
  shell: any
  spawn: (...args: any[]) => any
  pathToFileURL: (...args: any[]) => any
  resolveRequestedPathForIpc: (...args: any[]) => any
  rememberLog: (message: string) => void
}

export function createDesktopExternalOpenRuntime(deps: DesktopExternalOpenDeps) {
  const { IS_WSL, shell, spawn, pathToFileURL, resolveRequestedPathForIpc, rememberLog } = deps

  function openExternalUrl(rawUrl) {
    const raw = String(rawUrl || '').trim()

    if (!raw) {
      return false
    }

    let parsed

    try {
      parsed = new URL(raw)
    } catch {
      return false
    }

    // `file://` URLs come from the artifacts panel (the renderer can't open
    // them itself because Chromium blocks file:// navigation from the app
    // origin). Hand them to `shell.openPath`, which dispatches to the OS
    // file association. If the OS can't open it (`error` is a non-empty
    // string), fall back to revealing the file in the system file manager.
    if (parsed.protocol === 'file:') {
      let localPath

      try {
        localPath = resolveRequestedPathForIpc(parsed.toString(), { purpose: 'Open external file' })
      } catch {
        return false
      }

      void shell
        .openPath(localPath)
        .then(error => {
          if (!error) {
            return
          }

          rememberLog(`[file] openPath failed: ${error}; revealing in folder instead`)

          try {
            shell.showItemInFolder(localPath)
          } catch (revealError) {
            rememberLog(`[file] showItemInFolder failed: ${revealError.message}`)
          }
        })
        .catch(error => rememberLog(`[file] openPath rejected: ${error.message}`))

      return true
    }

    if (!['http:', 'https:', 'mailto:'].includes(parsed.protocol)) {
      return false
    }

    const url = parsed.toString()

    if (IS_WSL) {
      rememberLog(`[link] opening via WSL→Windows: ${url}`)

      const proc = spawn('cmd.exe', ['/c', 'start', '""', url], {
        detached: true,
        stdio: 'ignore',
        windowsHide: true
      })

      proc.on('error', error => {
        rememberLog(`[link] cmd.exe start failed: ${error.message}; falling back to xdg-open`)
        shell.openExternal(url).catch(fallback => rememberLog(`[link] xdg-open failed: ${fallback.message}`))
      })
      proc.unref()

      return true
    }

    shell.openExternal(url).catch(error => rememberLog(`[link] openExternal failed: ${error.message}`))

    return true
  }

  async function openPreviewInBrowser(rawUrl) {
    const raw = String(rawUrl || '').trim()

    if (!raw) {
      return false
    }

    let parsed

    try {
      parsed = new URL(raw)
    } catch {
      return false
    }

    if (parsed.protocol === 'file:') {
      let localPath

      try {
        localPath = resolveRequestedPathForIpc(parsed.toString(), { purpose: 'Open preview in browser' })
      } catch {
        return false
      }

      await shell.openExternal(pathToFileURL(localPath).toString())

      return true
    }

    return openExternalUrl(raw)
  }

  return { openExternalUrl, openPreviewInBrowser }
}
