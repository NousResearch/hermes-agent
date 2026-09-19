// preview-home-fallback.ts — home/attachments retry for desktop file previews.
//
// Attachment refs stored in chat history are frequently home-relative
// (e.g. `AppData/Local/hermes/attachments/foo.xlsx`, backslashes possible).
// The default preview base (the agent working directory) does not contain
// them, so `previewFileTarget` in main.ts retries the relative non-`file:`
// target against the candidates below before giving up. Pure functions so
// the OS-specific part (home dir, existence checks) stays injectable.

import fs from 'node:fs'
import path from 'node:path'

export interface PreviewHomeFallbackOptions {
  homeDir: string
  exists?: (candidate: string) => boolean
}

function defaultExists(candidate: string): boolean {
  try {
    return fs.statSync(candidate).isFile()
  } catch {
    return false
  }
}

export function previewHomeFallbackCandidates(rawTarget: string, homeDir: string): string[] {
  const requested = String(rawTarget || '').trim().replace(/^file:\/\//i, '')
  const normalized = requested.replace(/\\/g, '/')
  const home = String(homeDir || '').trim()

  if (!normalized || !home || path.isAbsolute(normalized) || /^file:/i.test(normalized)) {
    return []
  }

  return [
    path.join(home, normalized),
    path.join(home, 'AppData', 'Local', 'hermes', 'attachments', path.basename(normalized))
  ]
}

export function resolvePreviewHomeFallback(
  rawTarget: string,
  options: PreviewHomeFallbackOptions
): string | null {
  const exists = options.exists ?? defaultExists

  for (const candidate of previewHomeFallbackCandidates(rawTarget, options.homeDir)) {
    if (exists(candidate)) {
      return candidate
    }
  }

  return null
}
