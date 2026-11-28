// Unsaved Files-pane edits must not vanish without a warning.
//
// Inspired by Claude Cowork (desktop changelog v7.37095.0, 2026-11-24): "Fixed
// unsaved work, such as an edit in the Files pane, being lost without a warning
// when quitting while Claude works, relaunching to update, or reloading the
// window." Hermes had the same three holes: the preview editor's draft lives in
// the renderer only, and nothing between the renderer and the quit/update path
// knew it existed.
//
// The renderer reports dirty preview files on the active-work channel and
// cancels `beforeunload` while any are dirty. Three exits use that report:
//   - closing a window or reloading it: Electron fires `will-prevent-unload`;
//     `installUnsavedEditsUnloadGuard` asks here (Electron otherwise cancels
//     the close or reload silently).
//   - quitting: before-quit's quitPromptFor lists the files (quit-guard.ts),
//     and once the quit is going ahead `allowUnload` lets every window go.
//   - relaunching to update: `confirmUpdateDiscardsUnsaved` asks BEFORE the
//     updater is spawned. A modal after that would strand the detached script.

import type { BrowserWindow, MessageBoxOptions, MessageBoxSyncOptions } from 'electron'

import { type ActiveWork, mergeActiveWork, type QuitPrompt, unsavedEditsPrompt } from './quit-guard'

export const UNSAVED_EDITS_ERROR = 'unsaved-edits'

interface PromptHost {
  showMessageBox: (parent: BrowserWindow, options: MessageBoxOptions) => Promise<{ response: number }>
  showMessageBoxSync: (parent: BrowserWindow, options: MessageBoxSyncOptions) => number
}

/** A QuitPrompt as message-box options: button 0 is always the safe choice. */
export function promptDialogOptions(prompt: QuitPrompt, type: 'question' | 'warning' = 'warning'): MessageBoxSyncOptions {
  return {
    buttons: [...prompt.buttons],
    cancelId: 0,
    defaultId: 0,
    detail: prompt.detail,
    message: prompt.message,
    type
  }
}

export interface UnloadGuardOptions {
  /** True while the app itself is going away (confirmed quit, update hand-off, tests). */
  allowUnload: () => boolean
  host: PromptHost
  /** This window's last reported unsaved files. */
  unsavedFiles: () => readonly string[]
}

export function installUnsavedEditsUnloadGuard(win: BrowserWindow, options: UnloadGuardOptions): void {
  win.webContents.on('will-prevent-unload', event => {
    // preventDefault() here means "ignore the page's beforeunload and unload".
    if (options.allowUnload() || win.isDestroyed()) {
      event.preventDefault()

      return
    }

    const prompt = unsavedEditsPrompt(options.unsavedFiles(), 'leave')

    if (options.host.showMessageBoxSync(win, promptDialogOptions(prompt)) === 1) {
      event.preventDefault()
    }
  })
}

/**
 * Ask before an update relaunch throws away unsaved edits. Resolves to a wire
 * result to return instead of updating, or null to go ahead.
 */
export async function confirmUpdateDiscardsUnsaved(
  reports: Iterable<ActiveWork>,
  parent: BrowserWindow | null | undefined,
  host: PromptHost
): Promise<null | { error: string; message: string; ok: false }> {
  const files = mergeActiveWork(reports).unsavedFiles ?? []

  if (files.length < 1 || !parent || parent.isDestroyed()) {
    return null
  }

  const { response } = await host.showMessageBox(parent, promptDialogOptions(unsavedEditsPrompt(files, 'update')))

  // The renderer shows this on the update's usual error card, with Retry.
  return response === 1
    ? null
    : {
        error: UNSAVED_EDITS_ERROR,
        message: 'Update cancelled so your unsaved changes are kept. Save them, then update again.',
        ok: false
      }
}
