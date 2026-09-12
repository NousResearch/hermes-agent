/**
 * Board export / import flows — the same shape as the profile ones
 * (`src/store/profile-share.ts`): pick a path with the native dialog, hand
 * the PATH to the backend, toast the outcome.
 *
 * Bytes never cross the renderer. The picker and the backend are on the same
 * machine, so the backend does the reading and writing; that also keeps a
 * multi-hundred-megabyte board of attachments out of the renderer heap.
 */

import { host, type PluginOs } from '@hermes/plugin-sdk'

import { exportBoard, importBoard } from './api'
import type { KanbanText } from './i18n'
import { errText } from './ui'

const ARCHIVE_FILTERS = [{ extensions: ['tar.gz', 'tgz'], name: 'Hermes board' }]

/** Pick a destination and export `slug`. Returns the archive path, or null
 *  when the user cancelled or the export failed. */
export async function runExportBoardFlow(os: PluginOs, k: KanbanText, slug: string): Promise<null | string> {
  const output = await os.pickSavePath({
    title: k.exportBoardTitle,
    defaultPath: `${slug}.tar.gz`,
    filters: ARCHIVE_FILTERS
  })

  if (!output) {
    return null
  }

  try {
    const result = await exportBoard(slug, output)
    host.notify({ kind: 'success', message: k.boardExported(result.archive) })

    return result.archive
  } catch (error) {
    host.notify({ kind: 'error', message: errText(error) })

    return null
  }
}

/** Pick an archive and import it as a new board. Returns the new board's slug,
 *  or null when cancelled or failed. */
export async function runImportBoardFlow(os: PluginOs, k: KanbanText): Promise<null | string> {
  const archive = await os.pickOpenPath({ title: k.importBoardTitle, filters: ARCHIVE_FILTERS })

  if (!archive) {
    return null
  }

  try {
    const result = await importBoard(archive)
    host.notify({ kind: 'success', message: k.boardImported(result.name) })

    // The slug auto-suffixes on collision, and warnings cover tasks parked
    // for an unresolvable workspace — both change what the user sees on the
    // board they just opened, so neither is allowed to pass silently.
    if (result.renamed) {
      host.notify({ kind: 'info', message: k.boardImportedAs(result.board) })
    }

    for (const warning of result.warnings) {
      host.notify({ kind: 'warning', message: warning })
    }

    return result.board
  } catch (error) {
    host.notify({ kind: 'error', message: errText(error) })

    return null
  }
}
