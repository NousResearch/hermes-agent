/**
 * Obsidian wikilink + relative .md link resolver for Desktop preview pane.
 *
 * Transforms `[[Note Name]]`, `[[Note|alias]]`, `[[Note#Heading]]`, and
 * `[text](relative.md)` into `#preview/file:///abs/path` links that the
 * existing preview routing system already understands.
 */

interface VaultIndex {
  byBasename: Map<string, string>
}

const WIKILINK_RE = /\[\[([^\]|#]+?)(?:#([^\]|]+?))?(?:\|([^\]]+?))?\]\]/g
const RELATIVE_MD_LINK_RE = /(?<!!)\[([^\]]*)\]\(([^)]+?\.md(?:#[^)]*)?)\)/gi
const VAULT_ROOT_CACHE_KEY = 'hermes.desktop.obsidianVaultRoot.v1'
const VAULT_INDEX_TTL_MS = 5 * 60 * 1000

let cachedIndex: VaultIndex | null = null
let cachedIndexAt = 0
let cachedIndexVaultRoot = ''

export function dirname(filePath: string): string {
  const lastSep = Math.max(filePath.lastIndexOf('/'), filePath.lastIndexOf('\\'))
  return lastSep > 0 ? filePath.slice(0, lastSep) : filePath
}

export function basename(filePath: string, ext?: string): string {
  const lastSep = Math.max(filePath.lastIndexOf('/'), filePath.lastIndexOf('\\'))
  const name = filePath.slice(lastSep + 1)
  if (ext && name.endsWith(ext)) return name.slice(0, -ext.length)
  return name
}

function resolveRelative(base: string, rel: string): string {
  if (/^(?:\/|[a-zA-Z]:)/i.test(rel)) return rel

  const isAbsolute = base.startsWith('/')
  const parts = base.split(/[/\\]/).filter(Boolean)
  const relParts = rel.split(/[/\\]/)

  for (const part of relParts) {
    if (part === '..') parts.pop()
    else if (part !== '.') parts.push(part)
  }

  return (isAbsolute ? '/' : '') + parts.join('/')
}

/**
 * Get the stored Obsidian vault root override from localStorage.
 */
export function getStoredVaultRoot(): string | null {
  if (typeof window === 'undefined') return null
  try {
    return window.localStorage.getItem(VAULT_ROOT_CACHE_KEY) || null
  } catch {
    return null
  }
}

/**
 * Store an Obsidian vault root override in localStorage.
 */
export function setStoredVaultRoot(vaultRoot: string): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(VAULT_ROOT_CACHE_KEY, vaultRoot)
  } catch {
    // localStorage unavailable
  }
}

/**
 * Build a vault index from a list of absolute .md file paths.
 * Maps basename (without extension, lowercase) to the first matching full path.
 */
export function buildVaultIndex(files: string[]): VaultIndex {
  const byBasename = new Map<string, string>()

  for (const filePath of files) {
    const name = basename(filePath, '.md').toLowerCase()
    if (!byBasename.has(name)) {
      byBasename.set(name, filePath)
    }
  }

  return { byBasename }
}

/**
 * Get or build the vault index, using cache if available and fresh.
 */
export async function getVaultIndex(vaultRoot: string): Promise<VaultIndex | null> {
  const now = Date.now()

  if (cachedIndex && cachedIndexVaultRoot === vaultRoot && now - cachedIndexAt < VAULT_INDEX_TTL_MS) {
    return cachedIndex
  }

  const desktop = window.hermesDesktop
  if (!desktop?.scanVaultMd) return null

  try {
    const result = await desktop.scanVaultMd(vaultRoot)
    if (result.error || !result.files.length) return null

    cachedIndex = buildVaultIndex(result.files)
    cachedIndexAt = now
    cachedIndexVaultRoot = vaultRoot

    return cachedIndex
  } catch {
    return null
  }
}

/**
 * Invalidate the cached vault index (e.g., on window focus).
 */
export function invalidateVaultIndex(): void {
  cachedIndex = null
  cachedIndexAt = 0
}

function previewFileHref(absPath: string, heading?: string): string {
  const fileUrl = `file://${absPath}`
  const encoded = encodeURIComponent(fileUrl)
  const suffix = heading ? `#${heading.toLowerCase().replace(/\s+/g, '-')}` : ''
  return `#preview/${encoded}${suffix}`
}

/**
 * Resolve Obsidian wikilinks in markdown text.
 * Unresolved wikilinks become inline code (won't break markdown).
 */
export function resolveWikilinks(text: string, index: VaultIndex | null): string {
  return text.replace(WIKILINK_RE, (_match, name: string, heading?: string, alias?: string) => {
    const target = name.trim()
    const display = alias?.trim() || (heading ? `${target} > ${heading.trim()}` : target)

    if (!index) return `\`${display}\``

    const filePath = index.byBasename.get(target.toLowerCase())
    if (!filePath) return `\`${display}\``

    return `[${display}](${previewFileHref(filePath, heading)})`
  })
}

/**
 * Resolve relative .md links in markdown text.
 */
export function resolveRelativeMdLinks(text: string, currentDir: string): string {
  return text.replace(RELATIVE_MD_LINK_RE, (_match, linkText: string, href: string) => {
    if (/^(?:https?:|file:|mailto:|tel:|#)/i.test(href)) return _match

    const [relPath, heading] = href.split('#', 2)
    const absPath = resolveRelative(currentDir, relPath)

    if (!absPath.toLowerCase().endsWith('.md')) return _match

    return `[${linkText}](${previewFileHref(absPath, heading)})`
  })
}

/**
 * Main preprocessing function: resolve all Obsidian-style links in markdown text.
 * Call this before passing text to the markdown renderer.
 */
export function resolveObsidianLinks(
  text: string,
  currentFilePath: string,
  index: VaultIndex | null
): string {
  const currentDir = dirname(currentFilePath)
  let result = resolveWikilinks(text, index)
  result = resolveRelativeMdLinks(result, currentDir)
  return result
}
