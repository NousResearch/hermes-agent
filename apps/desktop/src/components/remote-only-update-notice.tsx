import { useStore } from '@nanostores/react'
import type { ReactElement } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { ExternalLink } from '@/lib/icons'
import { $remoteOnlyUpdate } from '@/store/updates'

/**
 * About ▸ Updates, remote-only artifacts: name the newer official release and
 * link to it. Deliberately has no "Update now" action — this build can only be
 * moved by rebasing the fork, never by the app itself.
 */
export function RemoteOnlyUpdateNotice(): ReactElement | null {
  const { t } = useI18n()
  const status = useStore($remoteOnlyUpdate)

  if (!status?.supported || !status.newer || !status.latest) {
    return null
  }

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border/70 bg-muted/20 px-3 py-3 text-sm">
      <span className="min-w-0">{t.remoteOnly.updateAvailable(status.latest)}</span>
      <Button
        onClick={() => window.hermesDesktop?.openExternal?.(status.releaseUrl)}
        size="sm"
        variant="secondary"
      >
        <ExternalLink className="size-4" />
        {t.remoteOnly.viewRelease}
      </Button>
    </div>
  )
}
