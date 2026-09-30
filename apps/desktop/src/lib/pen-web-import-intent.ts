/**
 * Import-from-the-web intent in a chat draft, with no pen context required.
 *
 * Two shapes: a URL beside a design word ("copy https://x.com into a mockup"),
 * or a clone verb about a page while the preview pane is showing one
 * ("recreate this site in figma"). The caller supplies that page's URL.
 */

const URL_RE = /https?:\/\/[^\s<>"')]+/iu

const DESIGN_INTENT_RE =
  /\b(design|mockup|mock-up|wireframe|canvas|figma|pen\.dev|pencil|clone|copy|recreate|redesign|import|trace|rebuild)\b/iu

const CLONE_VERB_RE = /\b(clone|copy|recreate|redesign|import|trace|rebuild|replicate)\b/iu
const PAGE_NOUN_RE = /\b(site|website|page|landing|homepage|component|header|hero|nav|footer|this|it)\b/iu

export interface WebImportIntent {
  /** Short display name for the page — its host. */
  host: string
  /** Set when the draft named the page itself; absent means "the page on screen". */
  url?: string
}

export function webImportIntent(text: string, activePageUrl?: string): null | WebImportIntent {
  const url = URL_RE.exec(text)?.[0]

  if (url) {
    return DESIGN_INTENT_RE.test(text.replace(url, ' ')) ? { host: hostOf(url), url } : null
  }

  if (activePageUrl && /^https?:/iu.test(activePageUrl) && CLONE_VERB_RE.test(text) && PAGE_NOUN_RE.test(text)) {
    return { host: hostOf(activePageUrl) }
  }

  return null
}

export function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./u, '') || url
  } catch {
    return url
  }
}
