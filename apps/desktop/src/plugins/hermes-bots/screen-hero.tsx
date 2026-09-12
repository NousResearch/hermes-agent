/**
 * Screen hero — the big preview of a bot's computer at the top of its pane.
 *
 * Shows a live thumbnail (one `display.thumbnail` JPEG every few seconds while
 * the screen runs and the hero is on screen), or an honest placeholder for
 * stopped / not installed / unsupported. Clicking it opens the full Screen pane
 * where the user can take over.
 */

import { Codicon } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { useBots } from './i18n'
import { displayRequest } from './screen-connection'
import { openBotScreen } from './screen-open'
import { type PortalTone, useScreenPortalState } from './screen-portal'
import type { BotMeta, RosterRow } from './types'

const REFRESH_MS = 4000
const STALE_AFTER = 3

function useLiveThumbnail(bot: RosterRow, running: boolean) {
  const [dataUrl, setDataUrl] = useState<string | null>(null)
  // Consecutive failed refreshes; past STALE_AFTER the frame is shown dimmed as "last seen" so a
  // dead gateway never keeps looking live. Success resets it.
  const [misses, setMisses] = useState(0)
  const boxRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    if (!running) {
      setDataUrl(null)
      setMisses(0)

      return
    }

    let cancelled = false
    let timer: number | null = null
    let visible = true

    const observer =
      typeof IntersectionObserver === 'undefined'
        ? null
        : new IntersectionObserver(entries => {
            visible = entries.some(entry => entry.isIntersecting)
          })

    if (observer && boxRef.current) {
      observer.observe(boxRef.current)
    }

    const tick = () => {
      if (cancelled) {
        return
      }

      if (!visible || document.hidden) {
        timer = window.setTimeout(tick, REFRESH_MS)

        return
      }

      void displayRequest<{ data_url: string | null }>(bot, 'display.thumbnail')
        .then(result => {
          if (!cancelled) {
            setDataUrl(result.data_url)
            setMisses(0)
          }
        })
        .catch(() => {
          if (!cancelled) {
            setMisses(prev => prev + 1) // the last good frame stays up, marked stale past STALE_AFTER
          }
        })
        .finally(() => {
          if (!cancelled) {
            timer = window.setTimeout(tick, REFRESH_MS)
          }
        })
    }

    tick()

    return () => {
      cancelled = true
      observer?.disconnect()

      if (timer !== null) {
        window.clearTimeout(timer)
      }
    }
  }, [bot, running])

  return { dataUrl, boxRef, stale: misses >= STALE_AFTER }
}

const TONE_RING: Partial<Record<PortalTone, string>> = {
  human: 'ring-2 ring-red-500/80',
  other: 'ring-2 ring-amber-500/70'
}

export function ScreenHero({ bot, meta }: { bot: RosterRow; meta?: BotMeta | null }) {
  const t = useBots()
  const { tone } = useScreenPortalState(bot)
  const running = tone === 'live' || tone === 'human' || tone === 'other'
  const { dataUrl, boxRef, stale } = useLiveThumbnail(bot, running)

  if (tone === 'unsupported') {
    return null
  }

  const caption = stale ? t.screen.heroStale : {
    live: t.screen.portalWatching,
    human: t.screen.portalYouControl,
    other: t.screen.portalOtherControls,
    off: t.screen.heroStopped,
    missing: t.screen.heroNotInstalled,
    unsupported: t.screen.portalUnsupported,
    unknown: t.screen.heroConnecting
  }[tone]

  const cta = running ? t.screen.heroOpenLive : tone === 'missing' ? t.screen.heroInstall : tone === 'off' ? t.screen.heroStart : ''

  return (
    <button
      aria-label={`${t.screen.portalTitle}: ${caption}`}
      className={`group relative block w-full overflow-hidden rounded-lg border border-(--ui-stroke-secondary) bg-black text-left ${TONE_RING[tone] ?? ''}`}
      onClick={() => openBotScreen(bot, meta ?? null)}
      ref={boxRef}
      style={{ aspectRatio: '16 / 10' }}
      type="button"
    >
      {dataUrl ? (
        <img alt="" className={stale ? 'absolute inset-0 size-full object-cover opacity-40 grayscale' : 'absolute inset-0 size-full object-cover'} draggable={false} src={dataUrl} />
      ) : (
        <span className="absolute inset-0 grid place-items-center bg-[radial-gradient(ellipse_at_center,rgba(255,255,255,0.08),transparent_70%)]">
          <Codicon className="text-[2.25rem] text-white/30" name={running ? 'loading' : tone === 'missing' ? 'cloud-download' : 'vm'} />
        </span>
      )}

      <span className="absolute inset-x-0 bottom-0 flex items-center gap-2 bg-gradient-to-t from-black/85 to-black/0 px-2.5 pb-2 pt-6 text-white">
        <span className={`size-2 shrink-0 rounded-full ${running && !stale ? (tone === 'live' ? 'bg-emerald-400' : tone === 'human' ? 'bg-red-400' : 'bg-amber-400') : 'bg-white/40'}`} />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-xs font-medium">{t.screen.portalTitle}</span>
          <span className="block truncate text-[0.65rem] text-white/70">{caption}</span>
        </span>
        {cta ? (
          <span className="flex shrink-0 items-center gap-1 rounded-md bg-white/15 px-2 py-1 text-[0.65rem] font-medium backdrop-blur-sm transition-colors group-hover:bg-white/25">
            {cta} <Codicon name="arrow-right" />
          </span>
        ) : null}
      </span>
    </button>
  )
}
