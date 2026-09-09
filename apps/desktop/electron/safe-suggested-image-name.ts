import path from 'node:path'

const WINDOWS_RESERVED_RE = /^(CON|PRN|AUX|NUL|COM\d|LPT\d)$/i
const MAX_STEM = 120

function stripTrailingDotsAndSpaces(value: string): string {
  return value.replace(/[. ]+$/g, '')
}

function sanitizeBase(value: unknown): string {
  return Array.from(path.basename(String(value || '')).replace(/[<>:"/\\|?*]/g, '_'), char =>
    char.charCodeAt(0) < 32 ? '_' : char
  )
    .join('')
    .trim()
}

export function safeSuggestedImageName(value: unknown, extension: string): string {
  const fallbackExt = extension || '.png'
  const base = stripTrailingDotsAndSpaces(sanitizeBase(value))

  if (!base || !base.replace(/[._\s]/g, '')) {
    return `image${fallbackExt}`
  }

  const currentExt = path.extname(base)
  let stem = stripTrailingDotsAndSpaces(currentExt ? base.slice(0, -currentExt.length) : base)

  if (!stem || WINDOWS_RESERVED_RE.test(stem)) {
    stem = stem ? `_${stem}` : 'image'
  }

  if (stem.length > MAX_STEM) {
    stem = stem.slice(0, MAX_STEM)
  }

  return `${stem}${currentExt || fallbackExt}`
}
