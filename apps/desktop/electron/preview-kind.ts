/**
 * Which preview affordance a file on disk gets.
 *
 * Extracted from `previewFileTarget` in `main.ts` so the RULE is provable
 * without booting Electron. The main process is a single large module with no
 * exported seams; the precedent for a small dependency-free sibling that owns
 * one decision is `preview-capture.ts` / `preview-reach.ts`.
 *
 * The bug this closes: the classifier had html/image/pdf branches and then
 * fell through to `metadata.binary ? 'binary' : 'text'`. An audio file is
 * sniffed binary (it is not UTF-8), so a `.wav` was refused with "This looks
 * like a binary file" — despite Desktop owning a Range-capable
 * `hermes-media://` stream protocol and the transcript already rendering
 * `<audio controls>` for the very same paths.
 */

export type PreviewKind = 'audio' | 'binary' | 'html' | 'image' | 'pdf' | 'text'

export interface PreviewKindInput {
  /** Result of the byte sniff in `previewFileMetadata`. */
  binary: boolean
  /** Lowercased extension INCLUDING the dot, or '' when the name has none. */
  extension: string
  /** MIME from the main process's `MEDIA_MIME_TYPES` table. */
  mimeType: string
}

const HTML_EXTENSIONS = new Set(['.htm', '.html'])
const PDF_EXTENSIONS = new Set(['.pdf'])

/**
 * Audio is decided by MIME, not by a second extension list.
 *
 * `mimeTypeForPath` derives the MIME from `MEDIA_MIME_TYPES` — the process's
 * own table of what each extension IS — so an extension set here would be a
 * redundant copy that can only disagree with it, and could never be the
 * deciding factor for a real file. MIME also means a format added to that table
 * becomes previewable with no edit here.
 */
function isAudio(mimeType: string): boolean {
  return mimeType.startsWith('audio/')
}

export function previewKindForFile(input: PreviewKindInput): PreviewKind {
  if (HTML_EXTENSIONS.has(input.extension)) {
    return 'html'
  }

  if (input.mimeType.startsWith('image/')) {
    return 'image'
  }

  if (PDF_EXTENSIONS.has(input.extension) || input.mimeType === 'application/pdf') {
    return 'pdf'
  }

  // Checked BEFORE the binary fallthrough: a successful sniff of "not text"
  // says nothing about whether the rail can play the file.
  if (isAudio(input.mimeType)) {
    return 'audio'
  }

  return input.binary ? 'binary' : 'text'
}
