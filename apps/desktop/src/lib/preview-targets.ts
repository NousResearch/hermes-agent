const PREVIEW_MARKDOWN_RE = /\[Preview:[^\]]+\]\((?<href>#preview[:/][^)]+)\)/gi

export function stripPreviewTargets(text: string): string {
  return text
    .replace(PREVIEW_MARKDOWN_RE, '')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

export function extractPreviewTargets(text: string): string[] {
  const targets: string[] = []
  const seen = new Set<string>()

  for (const match of text.matchAll(PREVIEW_MARKDOWN_RE)) {
    const target = previewTargetFromMarkdownHref(match.groups?.href)

    if (target && !seen.has(target)) {
      seen.add(target)
      targets.push(target)
    }
  }

  return targets
}

export function previewMarkdownHref(target: string): string {
  return `#preview/${encodeURIComponent(target)}`
}

export function previewTargetFromMarkdownHref(href?: string): string | null {
  if (!href?.startsWith('#preview:') && !href?.startsWith('#preview/')) {
    return null
  }

  try {
    return decodeURIComponent(href.slice('#preview'.length + 1))
  } catch {
    return null
  }
}

export function previewName(target: string): string {
  if (/^[A-Za-z]:\\/.test(target)) {
    return target.split('\\').filter(Boolean).pop() || target
  }

  try {
    const url = new URL(target)

    if (url.protocol === 'file:') {
      return decodeURIComponent(url.pathname).split(/[\\/]/).filter(Boolean).pop() || target
    }

    const file = url.pathname.split('/').filter(Boolean).pop()

    return file || url.host
  } catch {
    return target.split(/[\\/]/).filter(Boolean).pop() || target
  }
}

export function previewDisplayLabel(target: string): string {
  const escaped = previewName(target).replace(/[[\]\\]/g, '\\$&')

  return `Preview: ${escaped}`
}

const PREVIEWABLE_EXTENSIONS = [
  'html?',
  'md',
  'markdown',
  'svg',
  'png',
  'jpe?g',
  'gif',
  'webp',
  'mp4',
  'mov',
  'webm',
  'pdf',
  'csv',
  'txt',
  'json',
  'ya?ml',
  'toml'
] as const

const PREVIEWABLE_EXTENSION_PATTERN = `(?:${PREVIEWABLE_EXTENSIONS.join('|')})`
const PATH_LEAD_PATTERN = '(^|[\\s([{"\'`])'
const PATH_BODY_PATTERN = '[^\\n\\r\\]}"\'`,;:!?]*?'
const PATH_END_PATTERN = '(?!\\.[A-Za-z0-9])(?=$|[\\s.)\\]}"\'`,;:!?])'

function pathPattern(prefix: string): RegExp {
  return new RegExp(
    `${PATH_LEAD_PATTERN}(?<path>${prefix}${PATH_BODY_PATTERN}\\.${PREVIEWABLE_EXTENSION_PATTERN})${PATH_END_PATTERN}`,
    'gim'
  )
}

const FILE_URL_RE = pathPattern('file:\\/\\/(?:localhost)?\\/')
const POSIX_PATH_RE = pathPattern('\\/')
const HOME_PATH_RE = pathPattern('~\\/')
const WINDOWS_PATH_RE = pathPattern('[A-Za-z]:(?:\\\\|\\/)')
const RELATIVE_PATH_RE = pathPattern('\\.\\.?(?:\\\\|\\/)')
const PREVIEWABLE_FILE_EXTENSION_RE = new RegExp(`\\.${PREVIEWABLE_EXTENSION_PATTERN}$`, 'i')
const PRIVATE_FILE_EXTENSION_RE = /\.(?:key|p12|pem|pfx)$/i

const SECRET_LOCAL_PATH_RE =
  /(?:^|[\\/])(?:\.env(?:\..*)?|\.ssh|\.aws|\.gnupg|\.kube|id_(?:rsa|dsa|ecdsa|ed25519)(?:\.pub)?|[^\\/]*(?:api[-_]?key|private[-_]?key|secret|token|credential|password|passwd)[^\\/]*)/i

const FENCED_CODE_RE =
  /(^|\n)[ \t]*(`{3,}|~{3,})[^\n]*\n[\s\S]*?(?:\n[ \t]*\2[ \t]*(?=\n|$)|$)/g

const INLINE_CODE_RE = /(`+)[^\n]*?\1/g

const MARKDOWN_IMAGE_RE = /!\[[^\]\n]*\]\([^)\n]+\)/g
const MARKDOWN_IMAGE_REFERENCE_RE = /!\[([^\]\n]*)\]\[([^\]\n]*)\]/g
const MARKDOWN_IMAGE_SHORTCUT_RE = /!\[([^\]\n]+)\](?!\s*\()/g
const MARKDOWN_REFERENCE_DEFINITION_RE = /^\s*\[([^\]\n]+)\]:/

const INDENTED_CODE_LINE_RE = /^(?: {4}|\t).*$/gm

const DIFF_START_LINE_RE = /^\s*(?:diff --git\b|---\s|\+\+\+\s|@@\s)/

const DIFF_BODY_LINE_RE =
  /^(?:[- +]|\\ No newline at end of file|index\s|(?:old|new|deleted) file mode\s|similarity index\s|(?:rename|copy) (?:from|to)\s|Binary files\s|GIT binary patch\s|(?:literal|delta) \d+\s*$)/

const IGNORED_OUTPUT_LINE_RE =
  /^\s*(?:Traceback\b|File "[^"]+", line\b|at\s+\S+|Caused by:\s|[A-Za-z_$][\w.$]*(?:Error|Exception):|(?:\d{4}-\d{2}-\d{2}(?:T|\s)|\[?\d{2}:\d{2}:\d{2}(?:\.\d+)?\]?|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})|(?:TRACE|DEBUG|INFO|WARN|ERROR|FATAL)\b)/i

const LOCAL_MARKDOWN_LINK_RE = /(?<!!)\[[^\]\n]*\]\(([^)\n]+)\)/g

export interface AutoPreviewTargetOptions {
  includeRelative?: boolean
  maxTargets?: number
}

function maskMatch(match: string): string {
  return match.replace(/[^\n]/g, ' ')
}

function withoutIgnoredContent(text: string): string {
  let inDiff = false

  let source = text
    .replace(FENCED_CODE_RE, maskMatch)
    .replace(INLINE_CODE_RE, maskMatch)
    .replace(INDENTED_CODE_LINE_RE, maskMatch)

  const imageReferenceLabels = new Set<string>()

  MARKDOWN_IMAGE_REFERENCE_RE.lastIndex = 0

  for (const match of source.matchAll(MARKDOWN_IMAGE_REFERENCE_RE)) {
    imageReferenceLabels.add((match[2] || match[1] || '').trim().toLowerCase())
  }

  MARKDOWN_IMAGE_SHORTCUT_RE.lastIndex = 0

  for (const match of source.matchAll(MARKDOWN_IMAGE_SHORTCUT_RE)) {
    imageReferenceLabels.add((match[1] || '').trim().toLowerCase())
  }

  source = source
    .replace(MARKDOWN_IMAGE_RE, maskMatch)
    .replace(MARKDOWN_IMAGE_REFERENCE_RE, maskMatch)
    .replace(MARKDOWN_IMAGE_SHORTCUT_RE, maskMatch)

  return source
    .split('\n')
    .map(line => {
      const referenceDefinition = line.match(MARKDOWN_REFERENCE_DEFINITION_RE)

      if (referenceDefinition && imageReferenceLabels.has(referenceDefinition[1].trim().toLowerCase())) {
        return maskMatch(line)
      }

      if (DIFF_START_LINE_RE.test(line)) {
        inDiff = true

        return maskMatch(line)
      }

      if (inDiff && DIFF_BODY_LINE_RE.test(line)) {
        return maskMatch(line)
      }

      inDiff = false

      return IGNORED_OUTPUT_LINE_RE.test(line) ? maskMatch(line) : line
    })
    .join('\n')
}

function validatedAutoPreviewTarget(candidate: string, includeRelative: boolean): string | null {
  const target = candidate.replace(/\\ /g, ' ').trim()
  let pathForChecks = target

  if (/^file:\/\//i.test(target)) {
    try {
      const url = new URL(target)

      if (url.protocol !== 'file:' || (url.hostname && url.hostname.toLowerCase() !== 'localhost')) {
        return null
      }

      pathForChecks = decodeURIComponent(url.pathname)
    } catch {
      return null
    }
  } else {
    const isAbsolute =
      target.startsWith('/') || target.startsWith('~/') || /^[A-Za-z]:(?:\\|\/)/.test(target)

    const isRelative = /^\.\.?[\\/]/.test(target)

    if (!isAbsolute && (!includeRelative || !isRelative)) {
      return null
    }
  }

  if (
    !PREVIEWABLE_FILE_EXTENSION_RE.test(pathForChecks) ||
    PRIVATE_FILE_EXTENSION_RE.test(pathForChecks) ||
    SECRET_LOCAL_PATH_RE.test(pathForChecks)
  ) {
    return null
  }

  return target
}

export function extractLocalMarkdownPreviewTargets(markdown: string, includeRelative: boolean): string[] {
  const source = withoutIgnoredContent(markdown)
  const targets: string[] = []
  const seen = new Set<string>()

  LOCAL_MARKDOWN_LINK_RE.lastIndex = 0

  for (const match of source.matchAll(LOCAL_MARKDOWN_LINK_RE)) {
    const target = validatedAutoPreviewTarget(match[1]?.trim() ?? '', includeRelative)

    if (target && !seen.has(target)) {
      seen.add(target)
      targets.push(target)
    }
  }

  return targets
}

export function extractAutoPreviewTargets(
  text: string,
  options: AutoPreviewTargetOptions = {}
): string[] {
  const includeRelative = options.includeRelative === true
  const requestedMax = options.maxTargets ?? 3
  const maxTargets = Number.isFinite(requestedMax) ? Math.max(0, Math.floor(requestedMax)) : 3

  if (!text || maxTargets === 0) {
    return []
  }

  const source = withoutIgnoredContent(text)
  const matches: Array<{ index: number; target: string }> = []
  const patterns = [FILE_URL_RE, POSIX_PATH_RE, HOME_PATH_RE, WINDOWS_PATH_RE]

  if (includeRelative) {
    patterns.push(RELATIVE_PATH_RE)
  }

  for (const pattern of patterns) {
    pattern.lastIndex = 0

    for (const match of source.matchAll(pattern)) {
      const candidate = match.groups?.path
      const target = candidate ? validatedAutoPreviewTarget(candidate, includeRelative) : null

      if (target) {
        matches.push({ index: (match.index ?? 0) + match[1].length, target })
      }
    }
  }

  matches.sort((a, b) => a.index - b.index)

  const targets: string[] = []
  const seen = new Set<string>()

  for (const match of matches) {
    if (!seen.has(match.target)) {
      seen.add(match.target)
      targets.push(match.target)
    }

    if (targets.length >= maxTargets) {
      break
    }
  }

  return targets
}
