import { useStore } from '@nanostores/react'
import { memo, useState } from 'react'

import { StatusRow } from '@/components/chat/status-row'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { ChevronRight, X } from '@/lib/icons'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { cn } from '@/lib/utils'
import { PREVIEW_PANE_ID } from '@/store/layout'
import { notifyError } from '@/store/notifications'
import { $paneOpen } from '@/store/panes'
import { $previewTarget, dismissPreviewTarget, setCurrentSessionPreviewTarget } from '@/store/preview'
import { type PreviewArtifact } from '@/store/preview-status'

interface PreviewStatusRowProps {
  item: PreviewArtifact
  onDismiss: (id: string) => void
}

/** One detected artifact, single line, always visible: filename + open + close. */
export const PreviewStatusRow = memo(function PreviewStatusRow({ item, onDismiss }: PreviewStatusRowProps) {
  const { t } = useI18n()
  const activePreview = useStore($previewTarget)
  const previewPaneOpen = useStore($paneOpen(PREVIEW_PANE_ID))
  const [opening, setOpening] = useState(false)
  const isOpen = activePreview?.source === item.target && previewPaneOpen

  const resolveTarget = async () => {
    const target = await normalizeOrLocalPreviewTarget(item.target, item.cwd || undefined)

    if (!target) {
      throw new Error(`Could not open preview target: ${item.target}`)
    }

    return target
  }

  const togglePreview = async () => {
    if (opening) {
      return
    }

    if (isOpen) {
      dismissPreviewTarget()

      return
    }

    setOpening(true)

    try {
      setCurrentSessionPreviewTarget(await resolveTarget(), 'tool-result', item.target)
    } catch (error) {
      notifyError(error, t.preview.unavailable)
    } finally {
      setOpening(false)
    }
  }

  const openInBrowser = async () => {
    try {
      const bridge = window.hermesDesktop?.openPreviewInBrowser

      if (!bridge) {
        throw new Error('Desktop preview browser bridge is unavailable')
      }

      await bridge((await resolveTarget()).url)
    } catch (error) {
      notifyError(error, t.preview.unavailable)
    }
  }

  return (
    <StatusRow
      leading={<ChevronRight aria-hidden className="size-3 text-muted-foreground/80" />}
      onActivate={() => void togglePreview()}
      trailing={
        <span className="-my-1 flex items-center gap-0.5">
          <Tip label={t.preview.openInBrowser}>
            <Button
              aria-label={t.preview.openInBrowser}
              className="size-4 rounded-md text-muted-foreground/60 hover:text-foreground/90"
              onClick={event => {
                event.stopPropagation()
                void openInBrowser()
              }}
              size="icon-xs"
              type="button"
              variant="ghost"
            >
              <Codicon name="link-external" size="0.75rem" />
            </Button>
          </Tip>
          <Tip label={t.statusStack.dismiss}>
            <Button
              aria-label={t.statusStack.dismiss}
              className="size-4 rounded-md text-muted-foreground/60 hover:text-foreground/90"
              onClick={event => {
                event.stopPropagation()
                onDismiss(item.id)
              }}
              size="icon-xs"
              type="button"
              variant="ghost"
            >
              <X size={12} />
            </Button>
          </Tip>
        </span>
      }
      trailingVisible
    >
      <span className="min-w-0 max-w-[18rem] truncate text-[0.73rem] leading-4 text-foreground/92" title={item.target}>
        {item.label}
      </span>
      <span className={cn('shrink-0 text-[0.62rem] leading-4 text-muted-foreground/70', opening && 'animate-pulse')}>
        {opening ? t.preview.opening : isOpen ? t.preview.hide : t.preview.openPreview}
      </span>
    </StatusRow>
  )
})
