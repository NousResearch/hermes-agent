import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { ExternalLink } from '@/lib/icons'
import { openPreviewTargetInBrowser } from '@/lib/local-preview'
import { notifyError } from '@/store/notifications'
import type { PreviewTarget } from '@/store/preview'

interface PreviewBrowserFrameHintProps {
  target: PreviewTarget
}

/** Above the browser-hosted preview iframe. Framing denials can still fire
 *  load, and cross-origin navigation is private to the frame, so the pane
 *  cannot tell a blocked page from a loaded one: always offer the original URL. */
export function PreviewBrowserFrameHint({ target }: PreviewBrowserFrameHintProps) {
  const { t } = useI18n()
  const copy = t.preview.web

  return (
    <div className="pointer-events-auto flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-2 text-xs text-(--ui-text-secondary)">
      <p className="min-w-0 flex-1">{copy.embeddedPreviewHint}</p>
      <Tip label={copy.openTarget(target.url)}>
        <Button
          onClick={() =>
            void openPreviewTargetInBrowser(target).catch(error => notifyError(error, t.preview.unavailable))
          }
          size="xs"
          type="button"
          variant="secondary"
        >
          <ExternalLink />
          {t.preview.openInBrowser}
        </Button>
      </Tip>
    </div>
  )
}
