import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

// Live-reload watching for previewed files.
//
// The watcher is registered on the file's PARENT DIRECTORY (a plain file
// watch does not survive atomic save-by-rename, which is how most editors
// write). A deleted/unmounted/permission-revoked watch directory makes the
// FSWatcher emit 'error'; with no listener Node raises it as an uncaught
// exception and takes down the whole main process. Watcher failure is
// therefore treated as terminal for that watch only: the registry entry is
// dropped and the preview simply stops live-reloading instead of crashing
// the app.
export function createPreviewWatchRegistry({
  fileExists,
  sendChanged,
  debounceMs,
  watchImpl = fs.watch,
  log = console.warn
}) {
  const watchers = new Map()

  function watch(filePath) {
    const watchDir = path.dirname(filePath)
    const targetName = path.basename(filePath)
    const id = crypto.randomBytes(12).toString('base64url')
    let timer = null
    // Flipped by the error path and by close(); guards against a change
    // event racing watcher teardown (some platforms deliver a final event
    // during close()).
    let active = true

    const clearPending = () => {
      if (timer) {
        clearTimeout(timer)
        timer = null
      }
    }

    let watcher

    try {
      watcher = watchImpl(watchDir, (_eventType: any, filename: any) => {
        if (!active) {
          return
        }

        // filename is a utf8 string with the default encoding, but decode
        // defensively: String(buffer) would yield "<Buffer …>" garbage.
        const raw = typeof filename === 'string' ? filename : filename ? filename.toString() : ''
        const changedName = raw ? path.basename(raw) : ''

        if (changedName && changedName !== targetName) {
          return
        }

        clearPending()

        timer = setTimeout(() => {
          timer = null

          if (!fileExists(filePath)) {
            return
          }

          sendChanged({ id, path: filePath })
        }, debounceMs)
      })
    } catch (err) {
      // fs.watch can throw SYNCHRONOUSLY (e.g. ENOENT/EPERM when the watch
      // directory was deleted before the call). Log for diagnostics, then
      // rethrow: inside ipcMain.handle this becomes a rejected invoke() for
      // the renderer — terminal for this watch, never a main-process crash.
      log(`[preview-watch] failed to watch ${watchDir}:`, err)
      throw err
    }

    watcher.on('error', err => {
      // Some platforms deliver a final 'error' during close(); the first one
      // already tore this watch down.
      if (!active) {
        return
      }

      log(`[preview-watch] watcher failed for ${watchDir}, live reload disabled:`, err)
      active = false
      clearPending()
      watchers.delete(id)
      watcher.close()
    })

    watchers.set(id, {
      close: () => {
        active = false
        clearPending()
        watcher.close()
      }
    })

    return { id, path: filePath }
  }

  /**
   * Watch a DIRECTORY for entry churn (folders/files appearing or vanishing).
   * Used by the desktop plugins door to detect new/removed plugins without
   * polling. Unlike watch(), this does not filter by filename — any change
   * in the directory fires the debounced callback. The same error-containment
   * and lifecycle management applies.
   */
  function watchDirectory(
    dirPath,
    { dirExists }: { dirExists: (p: string) => boolean } = { dirExists: p => fs.existsSync(p) }
  ) {
    const id = crypto.randomBytes(12).toString('base64url')
    let timer = null
    let active = true

    const clearPending = () => {
      if (timer) {
        clearTimeout(timer)
        timer = null
      }
    }

    let watcher

    try {
      watcher = watchImpl(dirPath, () => {
        if (!active) {
          return
        }

        clearPending()

        timer = setTimeout(() => {
          timer = null

          if (!dirExists(dirPath)) {
            return
          }

          sendChanged({ id, path: dirPath })
        }, debounceMs)
      })
    } catch (err) {
      log(`[preview-watch] failed to watch directory ${dirPath}:`, err)
      throw err
    }

    watcher.on('error', err => {
      if (!active) {
        return
      }

      log(`[preview-watch] directory watcher failed for ${dirPath}, live reload disabled:`, err)
      active = false
      clearPending()
      watchers.delete(id)
      watcher.close()
    })

    watchers.set(id, {
      close: () => {
        active = false
        clearPending()
        watcher.close()
      }
    })

    return { id, path: dirPath }
  }

  function stop(id) {
    const entry = watchers.get(id)

    if (!entry) {
      return false
    }

    entry.close()
    watchers.delete(id)

    return true
  }

  function closeAll() {
    for (const id of [...watchers.keys()]) {
      stop(id)
    }
  }

  return { watch, watchDirectory, stop, closeAll, size: () => watchers.size }
}
