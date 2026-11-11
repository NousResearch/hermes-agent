/**
 * Tab attention for the Webapp: while the tab is in the background and an
 * event that would raise an OS notification arrives, the tab title alternates
 * between that notification's title and the normal one, and the favicon wears
 * a dot, until the user comes back to the tab. The browser counterpart of
 * Electron's `flashFrame`; it never focuses or navigates on its own.
 */

import { atom } from 'nanostores'

import { hostHas } from '@/lib/host-capabilities'

const FLASH_INTERVAL_MS = 1000
const BADGE_COLOR = '#f59e0b'

/** The text shown in place of the normal title right now; null shows the normal title. */
export const $tabFlashText = atom<null | string>(null)

let label: null | string = null
let timer: null | ReturnType<typeof setInterval> = null
const originalIconHrefs = new Map<HTMLLinkElement, string>()

function userIsAway(): boolean {
  return document.hidden || (typeof document.hasFocus === 'function' && !document.hasFocus())
}

function iconLinks(): HTMLLinkElement[] {
  return Array.from(document.querySelectorAll<HTMLLinkElement>('link[rel~="icon"]'))
}

function badgeFavicon(): void {
  const links = iconLinks()
  const source = links[0]?.href

  if (!source || originalIconHrefs.size) {
    return
  }

  links.forEach(link => originalIconHrefs.set(link, link.href))

  const image = new Image()

  image.onload = () => {
    // The flash may have ended while the icon loaded.
    if (!originalIconHrefs.size) {
      return
    }

    const size = 64
    const canvas = document.createElement('canvas')

    canvas.width = size
    canvas.height = size
    const context = canvas.getContext('2d')

    if (!context) {
      return
    }

    context.drawImage(image, 0, 0, size, size)
    context.beginPath()
    context.arc(size * 0.75, size * 0.25, size * 0.22, 0, Math.PI * 2)
    context.fillStyle = BADGE_COLOR
    context.fill()
    context.lineWidth = size * 0.06
    context.strokeStyle = '#ffffff'
    context.stroke()
    const badged = canvas.toDataURL('image/png')

    originalIconHrefs.forEach((_, link) => (link.href = badged))
  }

  image.src = source
}

function restoreFavicon(): void {
  originalIconHrefs.forEach((href, link) => (link.href = href))
  originalIconHrefs.clear()
}

function stop(): void {
  if (timer) {
    clearInterval(timer)
  }

  timer = null
  label = null
  $tabFlashText.set(null)
  restoreFavicon()
  window.removeEventListener('focus', onReturn)
  document.removeEventListener('visibilitychange', onReturn)
}

function onReturn(): void {
  if (!userIsAway()) {
    stop()
  }
}

/** Start (or retarget) the flash for a notification titled `text`. A no-op
 *  while the user is looking at the tab, and outside a browser host. */
export function flagTabAttention(text: string): void {
  if (typeof document === 'undefined' || !hostHas('tabTitleStatus') || !userIsAway()) {
    return
  }

  label = text
  $tabFlashText.set(text)
  badgeFavicon()

  if (timer) {
    return
  }

  window.addEventListener('focus', onReturn)
  document.addEventListener('visibilitychange', onReturn)
  timer = setInterval(() => {
    if (!userIsAway()) {
      stop()

      return
    }

    $tabFlashText.set($tabFlashText.get() === null ? label : null)
  }, FLASH_INTERVAL_MS)
}
