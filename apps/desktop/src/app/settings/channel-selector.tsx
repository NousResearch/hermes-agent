import { useEffect, useState } from 'react'

import { cn } from '@/lib/utils'

import type { Translations } from '@/i18n'

type AboutCopy = Translations['settings']['about']
export type UpdateChannel = 'stable' | 'beta'

/**
 * Channel selector for the About popup: "Stable" / "Beta (main)".
 *
 * Persist-only contract: choosing a channel writes the installation-scoped
 * record the CLI updater reads (update-channel.json). It never applies,
 * rebuilds, or restarts anything — checking and applying updates stay
 * separate, explicit actions that revalidate the exact target.
 */
export function ChannelSelector({ a }: { a: AboutCopy }) {
  const [channel, setChannel] = useState<UpdateChannel>('stable')
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let cancelled = false

    void window.hermesDesktop?.updates
      ?.getTrack()
      .then(result => {
        if (!cancelled && result?.channel) {
          setChannel(result.channel)
        }
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) {
          setLoaded(true)
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  const select = async (next: UpdateChannel) => {
    if (next === channel) {
      return
    }

    setChannel(next)

    try {
      const result = await window.hermesDesktop?.updates?.setTrack(next)

      if (result?.channel) {
        setChannel(result.channel)
      }
    } catch {
      // Persist failed; re-read the record so the UI reflects reality.
      try {
        const result = await window.hermesDesktop?.updates?.getTrack()

        if (result?.channel) {
          setChannel(result.channel)
        }
      } catch {
        // leave as-is; next mount re-syncs
      }
    }
  }

  const options: Array<{
    key: UpdateChannel
    label: string
    desc: string
  }> = [
    { key: 'stable', label: a.channelStable, desc: a.channelStableDesc },
    { key: 'beta', label: a.channelBeta, desc: a.channelBetaDesc }
  ]

  return (
    <div className="mt-3">
      <p className="text-sm font-medium">{a.channel}</p>
      <p className="mt-1 text-xs text-muted-foreground">{a.channelDesc}</p>
      <div className="mt-2 grid gap-2 sm:grid-cols-2" role="radiogroup" aria-label={a.channel}>
        {options.map(option => (
          <button
            aria-checked={channel === option.key}
            className={cn(
              'rounded-lg border px-3 py-2 text-left text-sm transition-colors',
              channel === option.key
                ? 'border-primary/50 bg-primary/5'
                : 'border-border/70 hover:bg-muted/30'
            )}
            disabled={!loaded}
            key={option.key}
            onClick={() => void select(option.key)}
            role="radio"
            type="button"
          >
            <span className="font-medium">{option.label}</span>
            <span className="mt-0.5 block text-xs text-muted-foreground">{option.desc}</span>
          </button>
        ))}
      </div>
      <p className="mt-2 text-xs text-muted-foreground">{a.channelPersistOnlyNote}</p>
    </div>
  )
}
