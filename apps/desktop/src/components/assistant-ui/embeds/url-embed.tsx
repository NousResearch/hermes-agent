'use client'

import { type CSSProperties, lazy, Suspense } from 'react'

import { EMBED_MAX_H } from './embed-size'
import { EmbedFail } from './fail'
import type { EmbedDescriptor } from './providers/types'
import { RichBoundary } from './rich-boundary'

const FrameEmbedRenderer = lazy(() => import('./frame-embed'))
const SocialEmbedRenderer = lazy(() => import('./social-embed'))
const SpotifyEmbedRenderer = lazy(() => import('./spotify-embed'))
const YouTubeEmbedRenderer = lazy(() => import('./youtube-embed'))

function intrinsicHeight(descriptor: EmbedDescriptor): number {
  if (descriptor.aspectRatio) {
    return Math.round((descriptor.maxWidth ?? 640) / descriptor.aspectRatio)
  }

  return descriptor.height ?? 320
}

function LazyRenderer({ descriptor }: { descriptor: EmbedDescriptor }) {
  // X and Instagram load their official blockquote script in-document. The tweet
  // check also narrows the union to FrameEmbed for the iframe renderers below.
  if (descriptor.renderer === 'tweet' || descriptor.provider === 'instagram') {
    return <SocialEmbedRenderer descriptor={descriptor} />
  }

  if (descriptor.provider === 'youtube') {
    return <YouTubeEmbedRenderer descriptor={descriptor} />
  }

  if (descriptor.provider === 'spotify') {
    return <SpotifyEmbedRenderer descriptor={descriptor} />
  }

  return <FrameEmbedRenderer descriptor={descriptor} />
}

export function UrlEmbed({ descriptor }: { descriptor: EmbedDescriptor }) {
  const aspect = descriptor.aspectRatio

  // Ratio embeds cap WIDTH off the ratio so height tops out at the cap while
  // scaling. Non-ratio embeds own their own height (scale-to-fit / measured /
  // fixed) — no maxHeight here or it would clip what the renderer just sized.
  const style: CSSProperties = {
    containIntrinsicSize: `auto ${intrinsicHeight(descriptor)}px`,
    contentVisibility: 'auto',
    ...(aspect
      ? { width: `min(${descriptor.maxWidth ?? 640}px, 100%, calc(${EMBED_MAX_H} * ${aspect}))` }
      : { width: descriptor.maxWidth ? `min(${descriptor.maxWidth}px, 100%)` : '100%' })
  }

  return (
    <span className="group/embed my-2 block overflow-hidden rounded-lg" data-slot="aui_embed-card" style={style}>
      <RichBoundary fallback={<EmbedFail label={descriptor.label} />} resetKey={descriptor.id}>
        <Suspense fallback={null}>
          <LazyRenderer descriptor={descriptor} />
        </Suspense>
      </RichBoundary>
    </span>
  )
}
