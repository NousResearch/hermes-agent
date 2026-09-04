import { useStore } from '@nanostores/react'

import { AlertCircle } from '@/lib/icons'
import { $storageStatus } from '@/store/storage-status'

/** Persistent recovery guidance for a backend with degraded state.db health. */
export function StorageDegradedBanner() {
  const storageStatus = useStore($storageStatus)

  if (storageStatus !== 'degraded') {
    return null
  }

  return (
    <div
      className="pointer-events-none fixed inset-x-0 top-0 z-50 flex justify-center px-3 pt-2"
      role="alert"
    >
      <div className="flex max-w-2xl items-start gap-2 rounded-md border border-destructive/45 bg-destructive/12 px-3 py-2 text-sm shadow-lg">
        <AlertCircle aria-hidden className="mt-0.5 size-4 shrink-0 text-destructive" />
        <div>
          <p className="font-medium">Session database needs repair</p>
          <p className="text-muted-foreground">
            Storage is operating in a degraded mode. For structural corruption, stop profile writers and preserve the
            {' '}database, then run <code>hermes sessions recover --source &lt;state.db&gt; --inspect-only</code> or
            {' '}restore a snapshot. Use <code>hermes sessions repair</code> only when that repair path is safe.
          </p>
        </div>
      </div>
    </div>
  )
}
