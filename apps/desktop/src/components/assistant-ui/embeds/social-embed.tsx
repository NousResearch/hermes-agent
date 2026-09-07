'use client'

import { useEffect, useRef } from 'react'

import { escapeHtml } from './escape-html'
import type { EmbedDescriptor } from './providers/types'
import { useIsDark } from './use-is-dark'

// The provider embed scripts need a REAL origin to run (they touch
// cookies/storage/postMessage), so — exactly like react-social-media-embed — we
// render the official blockquote in this document and let the script swap it for
// a correctly-sized iframe. A sandboxed srcDoc iframe gives a null origin and
// the scripts silently bail (white / 2px). The container is height:auto, so it
// grows to whatever the provider renders. No measuring, no forced height.
type EmbedWindow = Window &
  typeof globalThis & {
    instgrm?: { Embeds?: { process?: () => void } }
    twttr?: { widgets?: { load?: (el?: HTMLElement) => void } }
  }

const SCRIPT: Record<string, { id: string; src: string }> = {
  instagram: { id: 'hermes-ig-embed', src: 'https://www.instagram.com/embed.js' },
  tiktok: { id: 'hermes-tt-embed', src: 'https://www.tiktok.com/embed.js' },
  twitter: { id: 'hermes-tw-embed', src: 'https://platform.twitter.com/widgets.js' }
}

const PROCESS_DELAYS_MS = [0, 300, 800, 1600, 3000]

function markup(descriptor: EmbedDescriptor, theme: 'dark' | 'light'): string {
  const url = escapeHtml(descriptor.sourceUrl)

  switch (descriptor.provider) {
    case 'instagram':
      return `<blockquote class="instagram-media" data-instgrm-permalink="${url}" data-instgrm-version="14" style="margin:0;width:100%;min-width:0;max-width:100%"></blockquote>`
    case 'tiktok': {
      const id = escapeHtml(descriptor.id.replace(/^tiktok:/, ''))

      return `<blockquote class="tiktok-embed" cite="${url}" data-video-id="${id}" style="margin:0;max-width:100%"><section></section></blockquote>`
    }

    case 'twitter':
      // data-chrome="transparent" drops the card background so the themed page
      // shows through instead of a white box.
      return `<blockquote class="twitter-tweet" data-dnt="true" data-theme="${theme}" data-chrome="transparent"><a href="${url}"></a></blockquote>`

    default:
      return ''
  }
}

function loadScript(provider: string): Promise<void> {
  const { id, src } = SCRIPT[provider]

  // TikTok exposes no re-process API; its script rescans the document each time
  // it runs, so we re-inject it. The others are loaded once and reused.
  if (provider === 'tiktok') {
    document.getElementById(id)?.remove()
  } else if (document.getElementById(id)) {
    return Promise.resolve()
  }

  return new Promise(resolve => {
    const script = document.createElement('script')

    script.async = true
    script.id = id
    script.onload = () => resolve()
    script.onerror = () => resolve()
    script.src = src
    document.body.appendChild(script)
  })
}

function processEmbed(provider: string, container: HTMLElement): void {
  const win = window as EmbedWindow

  if (provider === 'instagram') {
    win.instgrm?.Embeds?.process?.()
  } else if (provider === 'twitter') {
    win.twttr?.widgets?.load?.(container)
  }
  // TikTok auto-scans on (re)injection — no manual process call.
}

export default function SocialEmbedRenderer({ descriptor }: { descriptor: EmbedDescriptor }) {
  const isDark = useIsDark()
  const ref = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const container = ref.current

    if (!container) {
      return
    }

    let cancelled = false
    const timers: number[] = []

    container.innerHTML = markup(descriptor, isDark ? 'dark' : 'light')

    void loadScript(descriptor.provider).then(() => {
      // The script renders asynchronously; nudge a few times so the embed
      // settles whether the script was cached or freshly fetched.
      for (const delay of PROCESS_DELAYS_MS) {
        timers.push(window.setTimeout(() => !cancelled && processEmbed(descriptor.provider, container), delay))
      }
    })

    return () => {
      cancelled = true

      for (const timer of timers) {
        clearTimeout(timer)
      }

      container.innerHTML = ''
    }
  }, [descriptor, isDark])

  // The white corner/box on tweets is a color-scheme MISMATCH: when the iframe's
  // resolved scheme differs from ours, the browser paints an opaque (white)
  // Canvas behind it. Twitter's embed resolves to `light`, so we force the iframe
  // to `light` to match — no mismatch, no Canvas — and data-chrome=transparent
  // then lets the dark page show through. (Confirmed: mkdocs-material #6889.)
  return (
    <div
      className="w-full [&_.instagram-media]:!min-w-0 [&_iframe]:!m-0 [&_iframe]:!max-w-full [&_iframe]:[color-scheme:light]"
      ref={ref}
    />
  )
}
