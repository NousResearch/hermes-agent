'use client'

import { useMemo } from 'react'

import type { FrameEmbed } from './providers/types'
import { useIsDark } from './use-is-dark'

const YOUTUBE_ALLOW =
  'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share; fullscreen'

function youtubeSrc(embedUrl: string): string {
  const url = new URL(embedUrl)

  // Only pass origin when it is an HTTP(S) origin; custom schemes (app://,
  // file://) can make the player reject otherwise embeddable videos.
  if (
    typeof window !== 'undefined' &&
    (window.location.protocol === 'http:' || window.location.protocol === 'https:') &&
    window.location.origin &&
    window.location.origin !== 'null'
  ) {
    url.searchParams.set('origin', window.location.origin)
  }

  return url.toString()
}

// Keep this as a plain iframe and let YouTube render its native player/error UI.
export default function YouTubeEmbedRenderer({ descriptor }: { descriptor: FrameEmbed }) {
  const isDark = useIsDark()
  const src = useMemo(() => youtubeSrc(descriptor.embedUrl), [descriptor.embedUrl])

  // Width is capped to the ratio by UrlEmbed, so aspect-video sizes height ≤ cap.
  return (
    <iframe
      allow={YOUTUBE_ALLOW}
      allowFullScreen
      className="block aspect-video w-full border-0 bg-transparent"
      loading="lazy"
      referrerPolicy="strict-origin-when-cross-origin"
      scrolling="no"
      src={src}
      style={{ colorScheme: isDark ? 'dark' : 'light' }}
      title="YouTube embed"
    />
  )
}
