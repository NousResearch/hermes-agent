import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { StatusPulse } from '@/components/ui/status-pulse'
import { useI18n } from '@/i18n'
import { sessionCompacting } from '@/store/compaction'

/**
 * "Compacting context — sent messages will queue" strip above the input while
 * a standalone compression runs. The gateway holds sends until the compress
 * lands, but the composer looked idle and sendable, so the message seemed to
 * vanish. Mid-turn compaction already lights the busy chrome (arc border,
 * busy Send), so this stays out of the way while a turn runs.
 */
export function CompactingNotice() {
  const { t } = useI18n()
  const view = useSessionView()
  const sessionId = useStore(view.$runtimeId)
  const busy = useStore(view.$busy)
  const compacting = useStore(useMemo(() => sessionCompacting(sessionId), [sessionId]))

  if (busy || !compacting) {
    return null
  }

  return (
    <div
      className="flex items-center gap-2 rounded-lg border border-[color-mix(in_srgb,var(--dt-composer-ring)_32%,transparent)] bg-accent/18 px-2 py-1"
      data-slot="composer-compacting"
      role="status"
    >
      <StatusPulse
        aria-hidden="true"
        className="dither inline-block size-3 shrink-0 rounded-[2px] text-midground/80"
        kind="opacity"
      />
      <div className="min-w-0 text-[0.7rem] text-muted-foreground/88">{t.composer.compactingNotice}</div>
    </div>
  )
}
