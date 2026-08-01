import { execFile } from 'node:child_process'
import path from 'node:path'

export interface OpenLocalFileDeps {
  /** `shell.openPath` — resolves `''` on success or a non-empty error string. */
  openPath: (target: string) => Promise<string>
  /** `shell.showItemInFolder`. */
  showItemInFolder: (target: string) => void
  /** `process.platform` at the call site. Defaults to the running platform. */
  platform?: NodeJS.Platform
  /** Structured logger. Defaults to a no-op. */
  log?: (message: string) => void
  /**
   * Opens `target` with macOS Preview, resolving `null` on success or the
   * failure message. Injectable for tests; defaults to `open -a Preview`.
   */
  openWithMacPreview?: (target: string) => Promise<string | null>
}

const openWithPreview = (target: string): Promise<string | null> =>
  new Promise(resolve => {
    // `execFile` takes an argv array (not a shell string), so filenames with
    // spaces or punctuation stay safe.
    execFile('open', ['-a', 'Preview', target], error => resolve(error ? error.message : null))
  })

/**
 * Open a local file for the user, degrading gracefully when the OS can't.
 *
 * `shell.openPath` dispatches to the OS default handler for the file type, so we
 * try it first for every file — this honors the user's chosen PDF application
 * (Acrobat, PDF Expert, etc.) rather than overriding it. Only when `openPath`
 * reports a non-empty error for a macOS PDF do we reach for Preview: a failed
 * open on macOS commonly means a stale/broken LaunchServices association —
 * typically a `com.adobe.pdf` entry left behind after Adobe Acrobat is removed
 * or relocated — where `open -a Preview <file>` still works. Preview is bundled
 * with macOS and bypasses that broken default. If Preview also fails (and for
 * any other failed open), we reveal the file in the system file manager.
 */
export async function openLocalFile(localPath: string, deps: OpenLocalFileDeps): Promise<void> {
  const platform = deps.platform ?? process.platform
  const log = deps.log ?? (() => undefined)
  const tryPreview = deps.openWithMacPreview ?? openWithPreview
  const isMacPdf = platform === 'darwin' && path.extname(localPath).toLowerCase() === '.pdf'

  let openError: string

  try {
    openError = await deps.openPath(localPath)
  } catch (error) {
    log(`[file] openPath rejected: ${(error as Error).message}`)

    return
  }

  if (!openError) {
    return
  }

  if (isMacPdf) {
    log(`[file] openPath failed: ${openError}; trying Preview`)

    const previewError = await tryPreview(localPath)

    if (!previewError) {
      return
    }

    log(`[file] Preview open failed: ${previewError}; revealing in folder instead`)
  } else {
    log(`[file] openPath failed: ${openError}; revealing in folder instead`)
  }

  try {
    deps.showItemInFolder(localPath)
  } catch (revealError) {
    log(`[file] showItemInFolder failed: ${(revealError as Error).message}`)
  }
}
