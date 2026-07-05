import { normalize } from '@/lib/text'

const VALID_LANGUAGE_RE = /^[a-z0-9][a-z0-9+#-]*$/i
const NON_CODE_FENCE_LANGUAGES = new Set(['', 'text', 'plain', 'plaintext', 'md', 'markdown'])

const COMMON_CODE_LANGUAGES = new Set([
  'bash',
  'c',
  'cpp',
  'css',
  'diff',
  'go',
  'html',
  'java',
  'javascript',
  'js',
  'json',
  'jsx',
  'markdown',
  'md',
  'php',
  'python',
  'py',
  'ruby',
  'rust',
  'rs',
  'sh',
  'sql',
  'swift',
  'tsx',
  'ts',
  'typescript',
  'xml',
  'yaml',
  'yml'
])

interface CodeSignals {
  bulletLines: number
  codeSignals: number
  hasMarkdown: boolean
  proseLines: number
  trimmed: string
  urlLines: number
}

export function sanitizeLanguageTag(tag: string): string {
  const trimmed = tag.trim()
  const first = trimmed.split(/\s/, 1)[0] || ''

  return VALID_LANGUAGE_RE.test(first) && first.length <= 16 ? first.toLowerCase() : ''
}

// Sanitized language tag → codicon glyph. Anything not listed falls back to
// the generic `code` glyph, which matches what the tool-row icons use.
const CODICON_BY_LANGUAGE: Record<string, string> = {
  bash: 'terminal',
  cmd: 'terminal',
  console: 'terminal',
  fish: 'terminal',
  powershell: 'terminal',
  ps1: 'terminal',
  sh: 'terminal',
  shell: 'terminal',
  zsh: 'terminal',

  md: 'markdown',
  markdown: 'markdown',

  json: 'json',
  json5: 'json',

  ini: 'settings-gear',
  toml: 'settings-gear',
  yaml: 'settings-gear',
  yml: 'settings-gear',
  dotenv: 'settings-gear',
  env: 'settings-gear',

  graphql: 'database',
  gql: 'database',
  mysql: 'database',
  postgres: 'database',
  postgresql: 'database',
  sql: 'database',
  sqlite: 'database',

  diff: 'diff',
  patch: 'diff',

  css: 'symbol-color',
  less: 'symbol-color',
  sass: 'symbol-color',
  scss: 'symbol-color',
  svg: 'symbol-color',

  regex: 'regex',
  regexp: 'regex',

  curl: 'globe',
  http: 'globe',

  docker: 'package',
  dockerfile: 'package',

  mermaid: 'graph'
}

export function codiconForLanguage(language: string | undefined): string {
  return CODICON_BY_LANGUAGE[sanitizeLanguageTag(language || '')] || 'code'
}

// File extension → language tag, so a filename can resolve to the same icon a
// fenced code block of that language would get. Only extensions that map to a
// non-generic codicon need an entry; everything else falls through to `code`.
const LANGUAGE_BY_EXTENSION: Record<string, string> = {
  bash: 'bash',
  cfg: 'ini',
  conf: 'ini',
  css: 'css',
  dockerfile: 'dockerfile',
  env: 'env',
  gql: 'graphql',
  graphql: 'graphql',
  ini: 'ini',
  json: 'json',
  json5: 'json',
  less: 'less',
  markdown: 'markdown',
  md: 'markdown',
  mdx: 'markdown',
  mmd: 'mermaid',
  ps1: 'powershell',
  psql: 'sql',
  sass: 'sass',
  scss: 'scss',
  sh: 'bash',
  sql: 'sql',
  svg: 'svg',
  toml: 'toml',
  yaml: 'yaml',
  yml: 'yml',
  zsh: 'zsh'
}

// Pick an icon for a file path by its extension (or bare name like
// `Dockerfile`), reusing the language→codicon map so file-edit rows and code
// blocks share one visual vocabulary. Unknown / generic code files get `code`.
export function codiconForFilename(path: string | undefined): string {
  const token = filenameExtToken(path)
  const language = LANGUAGE_BY_EXTENSION[token] || token

  return codiconForLanguage(language)
}

// Last path segment's extension (or the bare lowercased name for `Dockerfile`,
// `Makefile`, …). Shared by the icon and Shiki-language resolvers.
function filenameExtToken(path: string | undefined): string {
  const base = normalize((path || '').replace(/\\/g, '/').split('/').pop())
  const dot = base.lastIndexOf('.')

  return dot > 0 ? base.slice(dot + 1) : base
}

// File extension → Shiki bundled-language id, for syntax-highlighting diffs in
// the editing tool's own language. Unknown extensions return '' so callers fall
// back to the plain color-only diff renderer.
const SHIKI_LANGUAGE_BY_EXTENSION: Record<string, string> = {
  astro: 'astro',
  bash: 'bash',
  c: 'c',
  cc: 'cpp',
  cjs: 'javascript',
  clj: 'clojure',
  cpp: 'cpp',
  cs: 'csharp',
  css: 'css',
  cxx: 'cpp',
  dart: 'dart',
  dockerfile: 'docker',
  ex: 'elixir',
  exs: 'elixir',
  fish: 'fish',
  go: 'go',
  gql: 'graphql',
  graphql: 'graphql',
  h: 'c',
  hpp: 'cpp',
  hs: 'haskell',
  htm: 'html',
  html: 'html',
  ini: 'ini',
  java: 'java',
  jl: 'julia',
  js: 'javascript',
  json: 'json',
  json5: 'json5',
  jsonc: 'jsonc',
  jsx: 'jsx',
  kt: 'kotlin',
  kts: 'kotlin',
  less: 'less',
  lua: 'lua',
  makefile: 'make',
  markdown: 'markdown',
  md: 'markdown',
  mdx: 'mdx',
  mjs: 'javascript',
  ml: 'ocaml',
  mts: 'typescript',
  nix: 'nix',
  php: 'php',
  pl: 'perl',
  proto: 'proto',
  ps1: 'powershell',
  py: 'python',
  pyi: 'python',
  r: 'r',
  rb: 'ruby',
  rs: 'rust',
  sass: 'sass',
  scala: 'scala',
  scss: 'scss',
  sh: 'bash',
  sql: 'sql',
  svelte: 'svelte',
  swift: 'swift',
  tf: 'terraform',
  toml: 'toml',
  ts: 'typescript',
  tsx: 'tsx',
  vue: 'vue',
  xml: 'xml',
  yaml: 'yaml',
  yml: 'yaml',
  zig: 'zig',
  zsh: 'bash'
}

export function shikiLanguageForFilename(path: string | undefined): string {
  return SHIKI_LANGUAGE_BY_EXTENSION[filenameExtToken(path)] || ''
}

function proseLineCount(body: string): number {
  return body.split('\n').filter(line => {
    const trimmed = line.trim()

    return Boolean(trimmed) && /^[A-Za-z0-9"'`*-]/.test(trimmed)
  }).length
}

const CODE_SIGNAL_RE = [
  /(^|\s)(const|let|var|function|class|import|export|return|if|for|while|switch)\b/gim,
  /=>|==|===|!=|!==|\{|\}|;|<\/?[a-z][^>]*>/gi,
  /^\s*(#include|SELECT|INSERT|UPDATE|DELETE|CREATE|DROP)\b/gim
]

function codeSignalCount(body: string): number {
  return CODE_SIGNAL_RE.reduce((total, pattern) => total + (body.match(pattern)?.length ?? 0), 0)
}

function codeSignals(body: string): CodeSignals {
  const trimmed = body.trim()
  const markdownSignals = (trimmed.match(/\*\*[^*]+\*\*/g) || []).length + (trimmed.match(/`[^`\n]+`/g) || []).length

  return {
    bulletLines: (trimmed.match(/^\s*[-*]\s+\S+/gm) || []).length,
    codeSignals: codeSignalCount(trimmed),
    hasMarkdown: markdownSignals > 0,
    proseLines: proseLineCount(trimmed),
    trimmed,
    urlLines: (trimmed.match(/^\s*https?:\/\/\S+\s*$/gim) || []).length
  }
}

export function isLikelyProseFence(info: string, body: string): boolean {
  const trimmedInfo = info.trim()
  const rawInfo = trimmedInfo.toLowerCase()
  const language = sanitizeLanguageTag(info)
  const infoToken = trimmedInfo.split(/\s+/, 1)[0] || ''
  const hasInfoTail = Boolean(trimmedInfo) && trimmedInfo !== infoToken

  if (/^[-*+]\s/.test(rawInfo) || /^https?:\/\//.test(rawInfo)) {
    return true
  }

  const signals = codeSignals(body)

  if (!signals.trimmed) {
    return false
  }

  if (
    hasInfoTail &&
    signals.codeSignals <= 2 &&
    (signals.proseLines >= 2 || signals.bulletLines >= 1 || signals.urlLines >= 1)
  ) {
    return true
  }

  if (!NON_CODE_FENCE_LANGUAGES.has(language)) {
    return false
  }

  return (
    (signals.bulletLines >= 2 && signals.hasMarkdown && signals.codeSignals <= 2) ||
    (signals.proseLines >= 3 && signals.codeSignals === 0)
  )
}

export function isLikelyProseCodeBlock(language: string | undefined, code: string | undefined): boolean {
  const cleanLanguage = sanitizeLanguageTag(language || '')
  const signals = codeSignals(code || '')

  if (!signals.trimmed || signals.codeSignals >= 3) {
    return false
  }

  if (signals.bulletLines >= 1 && (signals.hasMarkdown || signals.proseLines >= 2)) {
    return true
  }

  if (NON_CODE_FENCE_LANGUAGES.has(cleanLanguage)) {
    return signals.proseLines >= 3 && signals.codeSignals === 0
  }

  return !COMMON_CODE_LANGUAGES.has(cleanLanguage) && signals.proseLines >= 2 && signals.codeSignals <= 1
}
