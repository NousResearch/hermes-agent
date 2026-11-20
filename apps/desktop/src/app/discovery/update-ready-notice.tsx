import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { $newVersionNoticeShown, putNewVersionNoticeAway } from '@/store/new-version'
import { usePortalShell } from '@/store/shell-variant'

import { useDiscoveryCopy } from './discovery-copy'

/**
 * E4 (Figma 18677:29704): a Webapp tab mid-task learns a newer build is deployed and says so in the
 * chat, over the composer: a line that it loads with the next new chat, Refresh now and Later. It
 * never reloads by itself (store/new-version.ts); Later puts it away and the tray keeps the build.
 */
export function UpdateReadyNotice() {
  const portal = usePortalShell()
  const shown = useStore($newVersionNoticeShown)
  const copy = useDiscoveryCopy().newVersion

  if (!portal || !shown) {
    return null
  }

  return (
    <section aria-label={copy.title} className="discovery-update-notice" data-slot="update-ready-notice">
      <p className="discovery-update-notice-line">
        <Codicon aria-hidden className="discovery-welcome-glyph" name="info" />
        {copy.notice}
      </p>
      <div className="discovery-welcome-actions">
        <Button className="discovery-welcome-action" onClick={() => window.location.reload()}>
          {copy.refresh}
        </Button>
        <Button className="discovery-welcome-action" onClick={putNewVersionNoticeAway} variant="secondary">
          {copy.later}
        </Button>
      </div>
    </section>
  )
}
