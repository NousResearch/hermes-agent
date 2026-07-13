'use client'

import { type CSSProperties, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { SplitButton } from '@/components/ui/split-button'
import { Play } from '@/lib/icons'
import { allowProvider } from '@/store/embed-consent'

import type { EmbedDescriptor } from './providers/types'

// Privacy placeholder shown before an embed reaches out to a third party. Sized
// to the embed's footprint (no layout shift). The split control mirrors the
// commit button: primary "Load" (this embed) with a caret for "Always allow
// <service>" (persisted). Global off lives in Appearance settings.
export function EmbedFacade({ descriptor, onLoad }: { descriptor: EmbedDescriptor; onLoad: () => void }) {
  const [choice, setChoice] = useState('once')

  const style: CSSProperties = descriptor.aspectRatio
    ? { aspectRatio: descriptor.aspectRatio }
    : { height: descriptor.height ?? 320 }

  const actions = [
    { id: 'once', label: `Load ${descriptor.label}` },
    { id: 'always', label: `Always allow ${descriptor.label}` }
  ]

  if (descriptor.provider === 'youtube' && descriptor.previewUrl) {
    return <YouTubeFacade descriptor={descriptor} onLoad={onLoad} style={style} />
  }

  return (
    <span
      className="relative flex size-full flex-col items-center justify-center gap-2 overflow-hidden rounded-lg border border-dashed border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary)/30"
      style={style}
    >
      {descriptor.previewUrl ? (
        <>
          <img
            alt=""
            aria-hidden="true"
            className="absolute inset-0 size-full object-cover"
            loading="lazy"
            referrerPolicy="no-referrer"
            src={descriptor.previewUrl}
          />
          <span className="absolute inset-0 bg-black/25" />
        </>
      ) : null}
      <span className="relative z-10 flex flex-col items-center gap-2">
        <SplitButton
          actions={actions}
          onTrigger={id => (id === 'always' ? allowProvider(descriptor.provider) : onLoad())}
          onValueChange={setChoice}
          primaryIcon={<Play className="size-3 translate-x-px fill-current" />}
          value={choice}
        />
        <span
          className={
            descriptor.previewUrl
              ? 'rounded-full bg-black/55 px-2 py-0.5 text-[0.6875rem] text-white/85'
              : 'text-[0.6875rem] text-(--ui-text-tertiary)'
          }
        >
          {hostOf(descriptor)}
        </span>
      </span>
    </span>
  )
}

function YouTubeFacade({
  descriptor,
  onLoad,
  style
}: {
  descriptor: EmbedDescriptor
  onLoad: () => void
  style: CSSProperties
}) {
  const allowAndLoad = () => {
    allowProvider(descriptor.provider)
    onLoad()
  }

  return (
    <span
      className="relative block size-full overflow-hidden rounded-lg border border-(--ui-stroke-tertiary) bg-black"
      style={style}
    >
      <button
        aria-label="Play YouTube video"
        className="group absolute inset-0 size-full cursor-pointer"
        onClick={onLoad}
        type="button"
      >
        <img
          alt=""
          aria-hidden="true"
          className="absolute inset-0 size-full object-cover"
          loading="lazy"
          referrerPolicy="no-referrer"
          src={descriptor.previewUrl}
        />
        <span className="absolute inset-0 bg-black/10 transition-colors group-hover:bg-black/20" />
        <svg
          aria-hidden="true"
          className="absolute top-1/2 left-1/2 h-12 w-[4.25rem] -translate-x-1/2 -translate-y-1/2 drop-shadow-lg transition-transform group-hover:scale-105"
          viewBox="0 0 68 48"
        >
          <rect fill="#ff0033" height="48" rx="12" width="68" />
          <path d="M28 14 46 24 28 34Z" fill="white" />
        </svg>
        <span className="absolute right-3 bottom-3 rounded-md bg-black/70 px-2 py-1 text-xs font-medium text-white">
          Watch on YouTube
        </span>
      </button>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            aria-label="YouTube embed options"
            className="absolute top-3 right-3 z-10 bg-black/65 text-white hover:bg-black/80 hover:text-white"
            size="icon-xs"
            variant="ghost"
          >
            <Codicon name="chevron-down" size="0.8rem" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="min-w-44">
          <DropdownMenuItem onSelect={allowAndLoad}>Always allow YouTube</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </span>
  )
}

function hostOf(descriptor: EmbedDescriptor): string {
  // x.com posts often arrive as twitter.com links — show the current brand.
  if (descriptor.provider === 'twitter') {
    return 'x.com'
  }

  try {
    return new URL(descriptor.sourceUrl).hostname.replace(/^www\./, '')
  } catch {
    return descriptor.label
  }
}
