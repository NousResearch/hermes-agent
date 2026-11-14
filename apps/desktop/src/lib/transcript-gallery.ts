import { isElementInHiddenPane } from '@/components/pane-shell/pane-visibility'

/** Attribute every transcript `ZoomableImage` stamps with the source its lightbox opens. */
export const LIGHTBOX_SRC_ATTR = 'data-lightbox-src'

const GALLERY_SCOPE = '[data-chat-surface]'

/**
 * The lightbox sources around `currentSrc`, in reading order, among the images
 * rendered in the same chat surface as `anchor` (the open image's thumbnail).
 * Images outside that surface (another chat tile, the artifacts page) and in
 * hidden keep-alive panes are not part of the gallery; a source shown twice
 * counts once, so stepping never loops between two copies of one image.
 */
export function lightboxNeighbors(
  anchor: Element | null,
  currentSrc: string
): { next: string | null; previous: string | null } {
  const scope = anchor?.closest(GALLERY_SCOPE)

  if (!scope) {
    return { next: null, previous: null }
  }

  const sources: string[] = []

  for (const el of scope.querySelectorAll<HTMLElement>(`[${LIGHTBOX_SRC_ATTR}]`)) {
    const src = el.getAttribute(LIGHTBOX_SRC_ATTR)

    if (src && !isElementInHiddenPane(el) && !sources.includes(src)) {
      sources.push(src)
    }
  }

  const index = sources.indexOf(currentSrc)

  if (index === -1) {
    return { next: null, previous: null }
  }

  return { next: sources[index + 1] ?? null, previous: sources[index - 1] ?? null }
}

/** Scroll the thumbnail for `src` into view so closing the lightbox lands where the user stepped to. */
export function revealLightboxThumbnail(anchor: Element | null, src: string): void {
  const scope = anchor?.closest(GALLERY_SCOPE)

  const thumb = [...(scope?.querySelectorAll<HTMLElement>(`[${LIGHTBOX_SRC_ATTR}]`) ?? [])].find(
    el => el.getAttribute(LIGHTBOX_SRC_ATTR) === src
  )

  if (typeof thumb?.scrollIntoView === 'function') {
    thumb.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }
}
