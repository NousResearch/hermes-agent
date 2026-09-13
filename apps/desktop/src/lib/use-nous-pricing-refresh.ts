import { useEffect } from 'react'

import type { ModelOptionProvider } from '@/types/hermes'

/** Catalog reads never run a model. The gateway warms cold prices off-thread
 * and expires the Nous catalog after five minutes. Keep open editors in sync. */
export function useNousPricingRefresh({
  providers,
  refetch,
  enabled = true,
  scope = ''
}: {
  providers?: Pick<ModelOptionProvider, 'slug' | 'free_tier_row' | 'pricing_pending' | 'free_tier_pending'>[]
  refetch: () => Promise<unknown>
  enabled?: boolean
  scope?: string
}) {
  const nous = providers?.find(provider => provider.slug === 'nous' && !provider.free_tier_row)
  const active = enabled && !!nous
  const pending = active && !!(nous?.pricing_pending || nous?.free_tier_pending)

  useEffect(() => {
    if (!active) {
      return
    }

    const timer = setInterval(() => void refetch().catch(() => undefined), 300_000)

    return () => clearInterval(timer)
  }, [active, refetch, scope])

  useEffect(() => {
    if (!pending) {
      return
    }

    let cancelled = false
    let attempts = 0
    let timer: ReturnType<typeof setTimeout>

    const poll = async () => {
      await refetch().catch(() => undefined)
      attempts += 1

      if (!cancelled && attempts < 10) {
        timer = setTimeout(() => void poll(), 1_500)
      }
    }

    timer = setTimeout(() => void poll(), 1_500)

    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [pending, refetch, scope])
}
