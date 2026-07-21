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
 * On macOS, `shell.openPath` dispatches to the LaunchServices default handler
 * for the file type. When that association is a stale/broken binding — commonly
 * a `com.adobe.pdf` entry left behind after Adobe Acrobat is removed or
 * relocated — opening a PDF fails even though `open -a Preview <file>` works. So
 * for macOS PDFs we try Preview (bundled with macOS, bypasses a broken default
 * handler) first, then fall back to the default handler, then reveal the file in
 * the system file manager. All other files use the default handler directly, as
 * before.
 */
export async function openLocalFile(localPath: string, deps: OpenLocalFileDeps): Promise<void> {
  const platform = deps.platform ?? process.platform
  const log = deps.log ?? (() => undefined)
  const tryPreview = deps.openWithMacPreview ?? openWithPreview

  if (platform === 'darwin' && path.extname(localPath).toLowerCase() === '.pdf') {
    const previewError = await tryPreview(localPath)

    if (!previewError) {
      return
    }

    log(`[file] Preview open failed: ${previewError}; trying default handler`)
  }

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

  log(`[file] openPath failed: ${openError}; revealing in folder instead`)

  try {
    deps.showItemInFolder(localPath)
  } catch (revealError) {
    log(`[file] showItemInFolder failed: ${(revealError as Error).message}`)
  }
}
