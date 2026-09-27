import './guide-loading.css'

import { BrandMark } from '@/components/brand-mark'
import { Loader } from '@/components/ui/loader'
import { useMediaQuery } from '@/hooks/use-media-query'
import { useI18n } from '@/i18n'

/** Stays independent of greeting discovery and transcript hydration. */
export function GuideLoading() {
  const { t } = useI18n()
  const reducedMotion = useMediaQuery('(prefers-reduced-motion: reduce)')
  const label = t.boot.steps.startingHermesDesktop

  return (
    <div
      aria-busy="true"
      aria-label={label}
      className="fixed inset-0 z-(--z-onboarding) grid place-items-center bg-(--ui-chat-surface-background) px-8"
      data-glass-opaque=""
      data-slot="guide-opening"
      role="status"
    >
      <div className="flex flex-col items-center gap-6 pb-12 text-center">
        {reducedMotion ? (
          <BrandMark aria-hidden="true" className="size-20" />
        ) : (
          <Loader aria-hidden="true" className="size-24" role="presentation" type="lemniscate-bloom" />
        )}
        <p className="text-sm text-muted-foreground">{label}</p>
      </div>
    </div>
  )
}
