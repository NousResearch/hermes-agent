import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

/**
 * Live-reload watching for previewed files and the disk-plugin door, extracted
 * from main.ts so the lifecycle is unit-testable (Vitest project `electron`).
 *
 * A watch exists because a RENDERER asked the main process to report when
 * something changes on disk: the preview pane reloads its file, and the
 * desktop plugin loader reconciles its folder set. Both consumers receive
 * change events on `hermes:preview-file-changed` and each filters them by its
 * own watch id — so a watch's events must go back to the WebContents that
 * created it, not to whichever window happens to be the main one. The owner
 * is captured at watch creation and delivery is routed to it; a window that
 * died while its watch lived takes the watch down instead of delivering
 * nowhere.
 *
 * Events are debounced per watch — editors write in bursts (in-place save,
 * rename dance) and one tick per burst is enough. File watches are registered
 * on the file's PARENT directory: a watch on the file itself does not survive
 * atomic save-by-rename, which is how most editors write. A change to a
 * sibling in that directory is filtered out by name.
 *
 * main.ts keeps what does not belong here: resolving a preview URL to a
 * readable path (the IPC path-security check) and the directory-exists check.
 */

export const PREVIEW_FILE_CHANGED_CHANNEL = 'hermes:preview-file-changed'

/** The renderer that created a watch; an Electron WebContents satisfies this. */
export interface PreviewWatchOwner {
  isDestroyed: () => boolean
  send: (channel: string, payload: PreviewFileChangedPayload) => void
}

/** What a watch reports when its target changed. */
export interface PreviewFileChangedPayload {
  id: string
  path: string
  url: string
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
  watch: (filePath: string, owner: PreviewWatchOwner) => PreviewWatchHandle
  watchDirectory: (dirPath: string, owner: PreviewWatchOwner) => PreviewWatchHandle
}

export function createPreviewWatchRegistry({
  fileExists,
  debounceMs,
  watchImpl = fs.watch as PreviewWatchImpl
}: PreviewWatchDeps): PreviewWatchRegistry {
  const watchers = new Map<string, { close: () => void }>()

  /** Deliver to the owner that created the watch, and only to it. A window
   *  that died while its watch lived takes the watch down with the event. */
  const deliver = (id: string, owner: PreviewWatchOwner, targetPath: string) => {
    if (owner.isDestroyed()) {
      stop(id)

      return
    }

    owner.send(PREVIEW_FILE_CHANGED_CHANNEL, {
      id,
      path: targetPath,
      url: pathToFileURL(targetPath).toString()
    })
  }

  function watch(filePath: string, owner: PreviewWatchOwner): PreviewWatchHandle {
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

        deliver(id, owner, filePath)
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

  function watchDirectory(dirPath: string, owner: PreviewWatchOwner): PreviewWatchHandle {
    const id = crypto.randomBytes(12).toString('base64url')
    let timer: null | ReturnType<typeof setTimeout> = null

    const watcher = watchImpl(dirPath, () => {
      if (timer) {
        clearTimeout(timer)
      }

      timer = setTimeout(() => {
        timer = null
        deliver(id, owner, dirPath)
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
