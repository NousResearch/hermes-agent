import path from 'node:path'

/** The file name a URL's path ends in, or ``fallback`` when it names none. */
export function filenameFromUrl(rawUrl: string, fallback: string = 'image'): string {
  try {
    const parsed = new URL(rawUrl)
    const base = path.basename(decodeURIComponent(parsed.pathname || ''))

    return base && base.includes('.') ? base : fallback
  } catch {
    return fallback
  }
}
