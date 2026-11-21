'use client'

import { useStore } from '@nanostores/react'
import { type ReactNode, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { ImageIcon } from '@/lib/icons'
import { $remoteMediaMode } from '@/store/remote-media'

function hostOf(src: string): string {
  try {
    return new URL(src).hostname.replace(/^www\./, '')
  } catch {
    return src
  }
}

/** Renders `children` (the real <img>/<video>/<audio>) only once remote loading
 *  is allowed: always in the default mode, after a click in "ask" mode. The
 *  placeholder names the host so the user sees where the request would go. */
export function RemoteMediaGate({ alt, children, src }: { alt: string; children: ReactNode; src: string }) {
  const mode = useStore($remoteMediaMode)
  const [loaded, setLoaded] = useState(false)
  const { t } = useI18n()

  if (mode === 'always' || loaded) {
    return <>{children}</>
  }

  const host = hostOf(src)

  return (
    <span
      className="my-2 flex w-fit max-w-full items-center gap-2 rounded-lg border border-dashed border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary)/30 px-3 py-2 text-sm text-muted-foreground"
      data-slot="aui_remote-media-gate"
      title={src}
    >
      <ImageIcon className="size-4 shrink-0" />
      <span className="min-w-0 truncate">{alt || host}</span>
      <Button onClick={() => setLoaded(true)} size="xs" type="button" variant="textStrong">
        {t.settings.appearance.remoteMediaLoad(host)}
      </Button>
    </span>
  )
}
