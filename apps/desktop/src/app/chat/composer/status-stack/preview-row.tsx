import { useStore } from '@nanostores/react'
import { memo, useState } from 'react'

import { StatusRow } from '@/components/chat/status-row'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { isDesktopFsRemoteMode } from '@/lib/desktop-fs'
import { normalizeOrLocalPreviewTarget, openPreviewTargetInBrowser } from '@/lib/local-preview'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import { $previewTabSources, closePreviewForSource, openPreview } from '@/store/preview'
import { type PreviewArtifact } from '@/store/preview-status'

interface PreviewStatusRowProps {
  item: PreviewArtifact
  onDismiss: (id: string) => void
}

/** One detected artifact, single line, always visible: filename + open + close. */
export const PreviewStatusRow = memo(function PreviewStatusRow({ item, onDismiss }: PreviewStatusRowProps) {
  const { t } = useI18n()
  const openSources = useStore($previewTabSources)
  const [opening, setOpening] = useState(false)
  // A tab open IS a pane in the tree now, so its presence is the whole answer.
  const isOpen = openSources.includes(item.target)

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
      closePreviewForSource(item.target)

      return
    }

    setOpening(true)

    try {
      openPreview(await resolveTarget(), 'tool-result')
    } catch (error) {
      notifyError(error, t.preview.unavailable)
    } finally {
      setOpening(false)
    }
  }

  const openDefaultTarget = async () => {
    try {
      const target = await resolveTarget()

      // A file:// URL resolved in remote mode names a file on the backend
      // host, not on the machine running Electron. Keep local files and
      // ordinary URLs on the browser path, but route remote files through the
      // in-app preview pane so its filesystem adapter reads via the gateway.
      // (Remote HTML stays on openPreviewTargetInBrowser, which stages a
      // sanitized local copy before opening it.)
      if (target.kind === 'file' && target.previewKind !== 'html' && isDesktopFsRemoteMode()) {
        openPreview(target, 'tool-result')

        return
      }

      await openPreviewTargetInBrowser(target)
    } catch (error) {
      notifyError(error, t.preview.unavailable)
    }
  }

  return (
    <StatusRow
      leading={
        <Codicon
          aria-hidden
          className={cn('text-muted-foreground/70', opening && 'animate-pulse')}
          name="globe"
          size="0.8rem"
        />
      }
      // Plain click opens the link in the browser, except remote files which
      // only the in-app gateway-backed preview can read. ⌘/Ctrl-click always
      // uses the in-app preview pane. (isOpen still toggles the pane closed.)
      onActivate={event => {
        if (event.metaKey || event.ctrlKey) {
          void togglePreview()
        } else {
          void openDefaultTarget()
        }
      }}
      trailing={
        <Tip label={t.statusStack.dismiss}>
          <Button
            aria-label={t.statusStack.dismiss}
            className="-my-1 size-4 rounded-md text-muted-foreground/60 hover:text-foreground/90"
            onClick={event => {
              event.stopPropagation()
              onDismiss(item.id)
            }}
            size="icon-xs"
            type="button"
            variant="ghost"
          >
            <Codicon name="close" size="0.75rem" />
          </Button>
        </Tip>
      }
      trailingVisible
    >
      <Tip
        label={
          // Inline flow with a hard break, not a flex column: Tip's background
          // only wraps inline content, so a flex box would light the first
          // line and leave the rest dark-on-dark.
          <>
            {item.target}
            <br />
            <span className="opacity-70">{t.preview.linkHint}</span>
          </>
        }
      >
        <span className="min-w-0 truncate text-[0.73rem] leading-4 text-foreground/92">{item.label}</span>
      </Tip>
    </StatusRow>
  )
})
