import type { IBufferLine, ILink, ILinkProvider, Terminal } from '@xterm/xterm'

import { isDesktopFsRemoteMode } from '@/lib/desktop-fs'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { openPreview } from '@/store/preview'

import { isTerminalLinkActivation } from './links'

// File paths printed by compilers, test runners, `git status`, `ls`, tracebacks
// (`src/app.ts:12:3`, `./build/out.log`, `~/notes.md`, `/etc/hosts`) become
// ⌘/Ctrl-click links into the preview pane, the same gesture as a URL. A token
// is underlined only once the main process confirms it exists on this machine
// (resolved against the shell's cwd), so version strings and prose never light up.

const MAX_LINE_LENGTH = 2000
const MAX_CANDIDATES_PER_LINE = 10
const MAX_PATH_LENGTH = 1024
const CACHE_LIMIT = 500
const CACHE_TTL_MS = 10_000

// Runs of non-space path characters. Quotes, brackets and trailing punctuation
// are trimmed per token below rather than excluded here, so `"x.py",` and
// `(src/a.ts)` still yield their path.
const TOKEN_RE = /[^\s"'`<>()[\]{}|,;]+/g
// `:12`, `:12:3`, `:12-20` and GitHub-style `#L12` / `#L12-L20` location suffixes.
const LOCATION_SUFFIX_RE = /(?::\d+(?::\d+)?(?:-\d+(?::\d+)?)?|#L\d+(?:C\d+)?(?:-L?\d+(?:C\d+)?)?):?$/
const TRAILING_PUNCT_RE = /[.:!?]+$/

export interface PathCandidate {
  /** 0-based column where the clickable range starts (suffix included). */
  start: number
  /** Exclusive end column of the clickable range. */
  end: number
  /** The path as printed, without its location suffix. */
  path: string
}

/** Path-shaped tokens in one terminal line. Pure: existence is checked later. */
export function pathCandidates(line: string): PathCandidate[] {
  if (!line || line.length > MAX_LINE_LENGTH) {
    return []
  }

  const out: PathCandidate[] = []

  for (const match of line.matchAll(TOKEN_RE)) {
    let token = match[0]
    const start = match.index ?? 0

    // A URL belongs to the web-links provider.
    if (/^[a-z][a-z0-9+.-]*:\/\//i.test(token)) {
      continue
    }

    token = token.replace(TRAILING_PUNCT_RE, '')
    const end = start + token.length
    const path = token.replace(LOCATION_SUFFIX_RE, '')

    if (!looksLikePath(path)) {
      continue
    }

    out.push({ end, path, start })

    if (out.length >= MAX_CANDIDATES_PER_LINE) {
      break
    }
  }

  return out
}

function looksLikePath(path: string): boolean {
  if (!path || path.length > MAX_PATH_LENGTH || /^[-+]/.test(path)) {
    return false
  }

  // A separator (`src/a`, `./a`, `~/a`, `/a`, `C:\a`) or a dotted file name
  // (`README.md`, `.env`); bare words and numbers are never paths here.
  if (/[\\/]/.test(path)) {
    return /[\w.~]/.test(path)
  }

  // Stem and extension each carry a letter, so `0.5s` and `v1.2.3` stay text.
  const dot = path.lastIndexOf('.')
  const stem = path.slice(0, dot)
  const ext = path.slice(dot + 1)

  return (
    /^\.?[\w-][\w.-]*\.[A-Za-z0-9]{1,10}$/.test(path) &&
    /[A-Za-z]/.test(ext) &&
    (stem === '' || /[A-Za-z_]/.test(stem))
  )
}

interface Resolution {
  at: number
  directory: boolean
  path: string | null
}

const cache = new Map<string, Resolution>()

function cacheGet(key: string): Resolution | undefined {
  const hit = cache.get(key)

  if (hit && Date.now() - hit.at < CACHE_TTL_MS) {
    return hit
  }

  cache.delete(key)

  return undefined
}

function cacheSet(key: string, value: Resolution) {
  cache.delete(key)
  cache.set(key, value)

  if (cache.size > CACHE_LIMIT) {
    cache.delete(cache.keys().next().value as string)
  }
}

async function resolveOnHost(path: string, cwd: string): Promise<Resolution> {
  const key = `${cwd}\0${path}`
  const hit = cacheGet(key)

  if (hit) {
    return hit
  }

  let resolved: Resolution = { at: Date.now(), directory: false, path: null }

  try {
    const target = await window.hermesDesktop?.normalizePreviewTarget?.(path, cwd || undefined)

    if (target?.kind === 'file' && target.path && target.previewKind !== 'missing') {
      resolved = { at: Date.now(), directory: target.previewKind === 'directory', path: target.path }
    }
  } catch {
    // Unreadable / refused by the IPC path guard: not a link.
  }

  cacheSet(key, resolved)

  return resolved
}

const MAX_WRAPPED_ROWS = 12

interface LogicalLine {
  /** 0-based buffer row the logical line starts on. */
  firstRow: number
  text: string
}

/** The soft-wrapped line containing buffer row `row` (0-based), joined back
 *  into one string so a path the terminal wrapped mid-name still matches. */
function logicalLine(term: Terminal, row: number): LogicalLine {
  const buffer = term.buffer.active
  let first = row

  while (first > 0 && row - first < MAX_WRAPPED_ROWS && buffer.getLine(first)?.isWrapped) {
    first--
  }

  let last = row

  while (last - first < MAX_WRAPPED_ROWS && buffer.getLine(last + 1)?.isWrapped) {
    last++
  }

  let text = ''

  for (let index = first; index <= last; index++) {
    const line: IBufferLine | undefined = buffer.getLine(index)
    // Interior rows keep their full width so offsets map back to columns.
    text += line?.translateToString(index === last) ?? ''
  }

  return { firstRow: first, text }
}

/** 0-based offset into a logical line → 1-based xterm cell. */
function cellAt(offset: number, firstRow: number, cols: number) {
  return { x: (offset % cols) + 1, y: firstRow + Math.floor(offset / cols) + 1 }
}

async function openResolved(resolution: Resolution) {
  if (!resolution.path) {
    return
  }

  if (resolution.directory) {
    await window.hermesDesktop?.openDir?.(resolution.path)

    return
  }

  const target = await normalizeOrLocalPreviewTarget(resolution.path)

  if (target) {
    openPreview(target)
  }
}

export interface TerminalFileLinkOptions {
  /** Directory relative paths resolve against: the shell's last known cwd. */
  resolveCwd: () => string
  /** False when the printed paths live on another machine (SSH shell, remote backend). */
  isLocal: () => boolean
}

/** Registers the file-path link provider on `term`; returns its disposer. */
export function registerTerminalFileLinks(term: Terminal, options: TerminalFileLinkOptions): () => void {
  const provider: ILinkProvider = {
    provideLinks(y, callback) {
      if (!window.hermesDesktop?.normalizePreviewTarget || isDesktopFsRemoteMode() || !options.isLocal()) {
        callback(undefined)

        return
      }

      const { firstRow, text } = logicalLine(term, y - 1)
      const cols = term.cols

      // xterm asks once per row; answer only for links that touch this row.
      const candidates = pathCandidates(text).filter(candidate => {
        const start = cellAt(candidate.start, firstRow, cols).y
        const end = cellAt(candidate.end - 1, firstRow, cols).y

        return start <= y && y <= end
      })

      if (!candidates.length) {
        callback(undefined)

        return
      }

      const cwd = options.resolveCwd()

      void Promise.all(candidates.map(candidate => resolveOnHost(candidate.path, cwd))).then(resolutions => {
        const links: ILink[] = []

        resolutions.forEach((resolution, index) => {
          const candidate = candidates[index]

          if (!resolution.path) {
            return
          }

          links.push({
            activate: event => {
              if (isTerminalLinkActivation(event)) {
                void openResolved(resolution)
              }
            },
            range: { end: cellAt(candidate.end - 1, firstRow, cols), start: cellAt(candidate.start, firstRow, cols) },
            text: text.slice(candidate.start, candidate.end)
          })
        })

        callback(links.length ? links : undefined)
      })
    }
  }

  const disposable = term.registerLinkProvider(provider)

  return () => disposable.dispose()
}

/** Test seam: forget cached existence answers. */
export function clearTerminalFileLinkCache() {
  cache.clear()
}
