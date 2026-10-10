import { withTimeout } from '@/lib/with-timeout'

import { decodeCapture, type LensCapture } from './model'

/** Self-contained: runs in the sandboxed page, with no host capabilities. */
export function captureLensPage(mode: 'page' | 'selection' | 'refresh', selector = '', tag = ''): unknown {
  const limit = 6000
  let element: Element | null = null

  if (mode === 'refresh') {
    element = document.querySelector(selector)

    if (!element || element.tagName !== tag) {
      return { error: 'sourceChanged' }
    }
  } else if (mode === 'selection') {
    const selection = window.getSelection()

    if (!selection || selection.isCollapsed || !selection.rangeCount) {
      return { error: 'selectText' }
    }

    const node = selection.getRangeAt(0).commonAncestorContainer
    element = node instanceof Element ? node : node.parentElement
    // Capture the enclosing block so refresh has an honest, stable unit to compare.
    element = element?.closest('p,li,article,section,td,th,blockquote,pre,h1,h2,h3,div') ?? element

    if (!element || element === document.body || element === document.documentElement) {
      return { error: 'selectText' }
    }
  } else {
    element = document.querySelector('main,[role="main"],article') ?? document.body
  }

  if (!element || element.closest('input,textarea,[contenteditable="true"]')) {
    return { error: 'selectText' }
  }

  const text = (element as HTMLElement).innerText?.trim() ?? ''

  if (!text) {
    return { error: 'emptyPage' }
  }

  if (mode !== 'refresh') {
    const parts: string[] = []
    let node: Element | null = element

    while (node && node !== document.documentElement) {
      if (node.id && document.querySelectorAll('#' + CSS.escape(node.id)).length === 1) {
        parts.unshift('#' + CSS.escape(node.id))

        break
      }

      const parent: Element | null = node.parentElement
      const siblings = parent ? Array.from(parent.children).filter(child => child.tagName === node!.tagName) : []
      parts.unshift(node.tagName.toLowerCase() + ':nth-of-type(' + (siblings.indexOf(node) + 1) + ')')
      node = parent
    }

    selector = parts.join(' > ')
  }

  return {
    url: location.href,
    title: document.title,
    text: text.slice(0, limit),
    selector,
    tag: element.tagName,
    truncated: text.length > limit
  }
}

export interface LensGuest {
  executeJavaScript?: (code: string) => Promise<unknown>
  getURL?: () => string
  reload?: () => void
  addEventListener: (name: string, fn: EventListener) => void
  removeEventListener: (name: string, fn: EventListener) => void
}

export async function readLensGuest(
  guest: LensGuest,
  mode: 'page' | 'selection' | 'refresh',
  source?: LensCapture
): Promise<LensCapture> {
  if (!guest.executeJavaScript) {
    throw new Error('unavailable')
  }

  if (source && guest.getURL?.() !== source.url) {
    throw new Error('openFirst')
  }

  const expectedUrl = guest.getURL?.()

  const script =
    '(' +
    captureLensPage.toString() +
    ')(' +
    JSON.stringify(mode) +
    ',' +
    JSON.stringify(source?.selector ?? '') +
    ',' +
    JSON.stringify(source?.tag ?? '') +
    ')'

  const result = await withTimeout(guest.executeJavaScript(script), 10000, 'unavailable')
  const capture = decodeCapture(result)

  if (!capture) {
    const reason = result && typeof result === 'object' && 'error' in result ? String(result.error) : 'unavailable'
    throw new Error(reason)
  }

  if (
    guest.getURL?.() !== expectedUrl ||
    (expectedUrl && capture.url !== expectedUrl) ||
    (source && capture.url !== source.url)
  ) {
    throw new Error('sourceChanged')
  }

  return capture
}

/** Reload only an already-open source. Never silently navigate another tab. */
export async function reloadLensGuest(guest: LensGuest, source: LensCapture): Promise<LensCapture> {
  if (!guest.reload || guest.getURL?.() !== source.url) {
    throw new Error('openFirst')
  }

  let cleanup = () => {}

  const loaded = new Promise<void>((resolve, reject) => {
    const done = () => {
      cleanup()
      resolve()
    }

    const failed = () => {
      cleanup()
      reject(new Error('unavailable'))
    }

    cleanup = () => {
      guest.removeEventListener('did-stop-loading', done)
      guest.removeEventListener('destroyed', failed)
    }

    guest.addEventListener('did-stop-loading', done)
    guest.addEventListener('destroyed', failed)
    guest.reload!()
  })

  try {
    await withTimeout(loaded, 20000, 'unavailable')

    return await readLensGuest(guest, 'refresh', source)
  } finally {
    cleanup()
  }
}
