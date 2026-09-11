import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

/**
 * Live-reload watching for previewed files and the disk-plugin door, extracted
 * from main.ts so the lifecycle is unit-testable (Vitest project `electron`).
 *
 * A watch exists because a renderer asked the main process to report when
 * something changes on disk: the preview pane reloads its file, and the
 * desktop plugin loader reconciles its folder set. Both consumers only care
 * that "it changed", so events are debounced per watch — editors write in
 * bursts (in-place save, rename dance) and one tick per burst is enough.
 *
 * File watches are registered on the file's PARENT directory: a watch on the
 * file itself does not survive atomic save-by-rename, which is how most
 * editors write. A change to a sibling in that directory is filtered out by
 * name.
 *
 * main.ts keeps what does not belong here: resolving a preview URL to a
 * readable path (the IPC path-security check), the directory-exists check,
 * and shaping the renderer payload for `hermes:preview-file-changed`.
 */

/** What a watch reports when its target changed; main.ts shapes the IPC. */
export interface PreviewWatchPayload {
  id: string
  path: string
}

/**
 * Directory + listener -> a close()-able watcher. Injected so tests can use a
 * deterministic EventEmitter instead of real fs events; defaults to fs.watch.
 */
export type PreviewWatchImpl = (
  dirPath: string,
  listener: (eventType: string, filename: Buffer | null | string) => void
) => { close: () => void }

export interface PreviewWatchDeps {
  /** Whether the watched file still exists (the debounce may fire post-delete). */
  fileExists: (filePath: string) => boolean
  /** Renderer-facing sink, called at most once per debounce window. */
  sendChanged: (payload: PreviewWatchPayload) => void
  /** Coalescing window per watch, in milliseconds. */
  debounceMs: number
  /** Injectable for tests. */
  watchImpl?: PreviewWatchImpl
}

export interface PreviewWatchHandle {
  id: string
  path: string
}

export interface PreviewWatchRegistry {
  closeAll: () => void
  /** Registered watches right now; bookkeeping counterpart to stop(). */
  size: () => number
  /** Drop one watch. False when the id is unknown (already stopped). */
  stop: (id: string) => boolean
  watch: (filePath: string) => PreviewWatchHandle
  watchDirectory: (dirPath: string) => PreviewWatchHandle
}

export function createPreviewWatchRegistry({
  fileExists,
  sendChanged,
  debounceMs,
  watchImpl = fs.watch as PreviewWatchImpl
}: PreviewWatchDeps): PreviewWatchRegistry {
  const watchers = new Map<string, { close: () => void }>()

  function watch(filePath: string): PreviewWatchHandle {
    const watchDir = path.dirname(filePath)
    const targetName = path.basename(filePath)
    const id = crypto.randomBytes(12).toString('base64url')
    let timer: null | ReturnType<typeof setTimeout> = null

    const watcher = watchImpl(watchDir, (_eventType, filename) => {
      const changedName = filename ? path.basename(String(filename)) : ''

      if (changedName && changedName !== targetName) {
        return
      }

      if (timer) {
        clearTimeout(timer)
      }

      timer = setTimeout(() => {
        timer = null

        if (!fileExists(filePath)) {
          return
        }

        sendChanged({ id, path: filePath })
      }, debounceMs)
    })

    watchers.set(id, {
      close: () => {
        if (timer) {
          clearTimeout(timer)
        }

        watcher.close()
      }
    })

    return { id, path: filePath }
  }

  function watchDirectory(dirPath: string): PreviewWatchHandle {
    const id = crypto.randomBytes(12).toString('base64url')
    let timer: null | ReturnType<typeof setTimeout> = null

    const watcher = watchImpl(dirPath, () => {
      if (timer) {
        clearTimeout(timer)
      }

      timer = setTimeout(() => {
        timer = null
        sendChanged({ id, path: dirPath })
      }, debounceMs)
    })

    watchers.set(id, {
      close: () => {
        if (timer) {
          clearTimeout(timer)
        }

        watcher.close()
      }
    })

    return { id, path: dirPath }
  }

  function stop(id: string): boolean {
    const watcher = watchers.get(id)

    if (!watcher) {
      return false
    }

    watcher.close()
    watchers.delete(id)

    return true
  }

  function closeAll(): void {
    for (const id of [...watchers.keys()]) {
      stop(id)
    }
  }

  return { watch, watchDirectory, stop, closeAll, size: () => watchers.size }
}
