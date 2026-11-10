'use client'

import { useEffect, useMemo, useRef, useState } from 'react'

import { EMBED_DEFAULT_H } from './embed-size'
import type { EmbedDescriptor, EmbedProvider } from './providers/types'
import { useIsDark } from './use-is-dark'

// X and Instagram render in the provider's own cross-origin iframe, like every
// other embed. Their widget scripts (widgets.js / embed.js) must never run in
// this document: it is the privileged app window, and any script here can call
// the preload bridge (window.hermesDesktop: terminal, files, backend API). The
// sandbox keeps only what the embed pages need (scripts, and their own
// origin's cookies and storage) and drops top navigation, popups (main's
// window-open policy denies them anyway), forms and modals. The frame is
// cross-origin, so allow-same-origin gives it nothing of ours, and Electron
// injects no preload into subframes.
const SOCIAL_FRAME_SANDBOX = 'allow-scripts allow-same-origin'

const FRAME_ORIGIN: Partial<Record<EmbedProvider, string>> = {
  instagram: 'https://www.instagram.com',
  twitter: 'https://platform.twitter.com'
}

function socialFrameSrc(descriptor: EmbedDescriptor, theme: 'dark' | 'light'): string {
  if (descriptor.renderer !== 'tweet') {
    return descriptor.embedUrl
  }

  const url = new URL('https://platform.twitter.com/embed/Tweet.html')

  url.searchParams.set('id', descriptor.tweetId)
  url.searchParams.set('theme', theme)
  url.searchParams.set('dnt', 'true')

  return url.toString()
}

interface EmbedMessage {
  details?: { height?: unknown }
  type?: unknown
  'twttr.embed'?: { method?: unknown; params?: { height?: unknown }[] }
}

// Both embed pages post their rendered height to the parent, the messages their
// own widget scripts listen for: X sends {"twttr.embed": {method:
// "twttr.private.resize", params: [{height}]}}, Instagram the JSON string
// {"type": "MEASURE", "details": {height}}. Only that number is read.
function reportedHeight(data: unknown): number | null {
  let message = data

  if (typeof message === 'string') {
    try {
      message = JSON.parse(message)
    } catch {
      return null
    }
  }

  if (!message || typeof message !== 'object') {
    return null
  }

  const { details, type, 'twttr.embed': tweet } = message as EmbedMessage

  const height =
    tweet?.method === 'twttr.private.resize'
      ? tweet.params?.[0]?.height
      : type === 'MEASURE'
        ? details?.height
        : undefined

  return typeof height === 'number' && Number.isFinite(height) && height > 0 ? Math.ceil(height) : null
}

export default function SocialEmbedRenderer({ descriptor }: { descriptor: EmbedDescriptor }) {
  const isDark = useIsDark()
  const ref = useRef<HTMLIFrameElement | null>(null)
  const [height, setHeight] = useState(descriptor.height ?? EMBED_DEFAULT_H)
  const src = useMemo(() => socialFrameSrc(descriptor, isDark ? 'dark' : 'light'), [descriptor, isDark])

  useEffect(() => {
    const onMessage = (event: MessageEvent) => {
      if (event.source !== ref.current?.contentWindow || event.origin !== FRAME_ORIGIN[descriptor.provider]) {
        return
      }

      const next = reportedHeight(event.data)

      if (next) {
        setHeight(next)
      }
    }

    window.addEventListener('message', onMessage)

    return () => window.removeEventListener('message', onMessage)
  }, [descriptor.provider])

  // The white corner/box on tweets is a color-scheme MISMATCH: when the iframe's
  // resolved scheme differs from ours, the browser paints an opaque (white)
  // Canvas behind it. The embed pages resolve to `light`, so the iframe is
  // forced to `light` to match; theme=dark still gives the dark tweet card.
  return (
    <iframe
      allowFullScreen
      className="block w-full border-0 bg-transparent"
      loading="lazy"
      ref={ref}
      referrerPolicy="strict-origin-when-cross-origin"
      sandbox={SOCIAL_FRAME_SANDBOX}
      scrolling="no"
      src={src}
      style={{ colorScheme: 'light', height }}
      title={`${descriptor.label} embed`}
    />
  )
}
