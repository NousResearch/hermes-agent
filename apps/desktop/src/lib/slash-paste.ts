import { isPastedContentPath } from '@/app/chat/composer/large-paste'
import { isReadFileErrorResult } from '@/lib/desktop-fs'
import type { ComposerAttachment } from '@/store/composer'

export type ExpandPastedSlashArgsResult =
  { ok: true; text: string } | { ok: false; reason: 'non-paste-attachment' | 'read-failed' }

export function isPastedAttachment(
  chip: ComposerAttachment | null | undefined
): chip is ComposerAttachment & { kind: 'file'; path: string } {
  return Boolean(chip && chip.kind === 'file' && typeof chip.path === 'string' && isPastedContentPath(chip.path))
}

export function hasNonPasteAttachment(chips: readonly (ComposerAttachment | null | undefined)[]): boolean {
  return chips.some(chip => !isPastedAttachment(chip))
}

/**
 * Trim a slash command for dispatch. A multiline command carries a paste body, which is
 * sent verbatim: only its leading whitespace goes. A one-line command is trimmed both ends.
 */
export function trimSlashCommandText(text: string): string {
  return text.includes('\n') ? text.trimStart() : text.trim()
}

/**
 * Inline large-paste chips into a slash command's argument text.
 *
 * Reads the local composer-paste file directly via `window.hermesDesktop.readFileText`
 * (never `readDesktopFileText`, which reads the remote agent host in remote mode).
 * Refuses if any chip is not a paste chip or if any paste read fails, is binary,
 * or was truncated at the 512 KiB bridge cap.
 */
export async function expandPastedSlashArgs(
  text: string,
  chips: readonly (ComposerAttachment | null | undefined)[]
): Promise<ExpandPastedSlashArgsResult> {
  if (chips.length === 0) {
    return { ok: true, text }
  }

  if (hasNonPasteAttachment(chips)) {
    return { ok: false, reason: 'non-paste-attachment' }
  }

  const readFileText = window.hermesDesktop?.readFileText

  if (!readFileText) {
    return { ok: false, reason: 'read-failed' }
  }

  let expanded = text

  for (const chip of chips) {
    if (!isPastedAttachment(chip)) {
      return { ok: false, reason: 'non-paste-attachment' }
    }

    let result: Awaited<ReturnType<typeof readFileText>>

    try {
      result = await readFileText(chip.path)
    } catch {
      return { ok: false, reason: 'read-failed' }
    }

    if (
      !result ||
      typeof result !== 'object' ||
      isReadFileErrorResult(result) ||
      result.binary === true ||
      result.truncated === true ||
      typeof result.text !== 'string'
    ) {
      return { ok: false, reason: 'read-failed' }
    }

    if (expanded.length > 0 && !/\s$/.test(expanded)) {
      expanded += '\n'
    }

    expanded += result.text
  }

  return { ok: true, text: expanded }
}
