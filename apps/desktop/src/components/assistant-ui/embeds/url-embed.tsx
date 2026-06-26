'use client'

import { useStore } from '@nanostores/react'
import { type CSSProperties, lazy, Suspense, useState } from 'react'

import { PrettyLink } from '@/lib/external-link'
import { $embedAllowed, $embedMode } from '@/store/embed-consent'

import { EmbedFacade } from './embed-consent'
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
  const mode = useStore($embedMode)
  const allowed = useStore($embedAllowed)
  const [loaded, setLoaded] = useState(false)

  // Privacy gate: don't reach out to the provider until consented. `off` keeps
  // it a plain link; otherwise the placeholder shows until "Load" (this embed)
  // or "Always allow" / global `always` permits the fetch.
  if (mode === 'off') {
    return <PrettyLink className="wrap-anywhere" href={descriptor.sourceUrl} />
  }

  const consented = mode === 'always' || loaded || allowed.includes(descriptor.provider)
  const aspect = descriptor.aspectRatio

  // Ratio embeds cap WIDTH off the ratio so height tops out at the cap while
  // scaling. Non-ratio embeds own their own height (measured / fixed).
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
        {consented ? (
          <Suspense fallback={null}>
            <LazyRenderer descriptor={descriptor} />
          </Suspense>
        ) : (
          <EmbedFacade descriptor={descriptor} onLoad={() => setLoaded(true)} />
        )}
      </RichBoundary>
    </span>
  )
}
