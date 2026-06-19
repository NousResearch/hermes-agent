import { useStore } from '@nanostores/react'
import { useRef, useState } from 'react'

import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import { useI18n } from '@/i18n'
import { splitFilePathSuffix } from '@/lib/file-path-links'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import { setCurrentSessionPreviewTarget } from '@/store/preview'
import { $currentCwd } from '@/store/session'

// A bare file path the agent mentioned in chat, rendered inline and clickable.
// Left-click opens the file in the in-app sandboxed preview pane; the
// right-click menu offers the hardened OS actions (open in default app / reveal
// in file manager) plus copy. The `:line[:col]` suffix is shown but stripped
// before opening, since the preview pane has no line target in v1.
export function FilePathLink({ target }: { target: string }) {
  const { t } = useI18n()
  const cwd = useStore($currentCwd)
  const [opening, setOpening] = useState(false)
  const openingRef = useRef(false)
  const { display, path } = splitFilePathSuffix(target)
  const bridge = typeof window === 'undefined' ? undefined : window.hermesDesktop

  function openErrorMessage(reason?: string): string {
    if (reason === 'not-openable') {
      return t.filePath.openBlocked
    }

    if (reason === 'outside-workspace') {
      return t.filePath.outsideWorkspace
    }

    if (reason === 'sensitive-file') {
      return t.filePath.blockedSensitive
    }

    return t.filePath.openFailed
  }

  async function openPreview() {
    if (openingRef.current) {
      return
    }

    openingRef.current = true
    setOpening(true)

    try {
      const preview = await normalizeOrLocalPreviewTarget(path, cwd || undefined)

      if (!preview) {
        throw new Error(`Could not open preview: ${path}`)
      }

      setCurrentSessionPreviewTarget(preview, 'chat-path', path)
    } catch (error) {
      notifyError(error, t.preview.unavailable)
    } finally {
      openingRef.current = false
      setOpening(false)
    }
  }

  async function openInDefaultApp() {
    if (!bridge?.openPath) {
      return
    }

    const result = await bridge.openPath(path, {
      baseDir: cwd || undefined,
      confirm: {
        cancelLabel: t.common.cancel,
        confirmLabel: t.filePath.confirmOpen,
        detail: t.filePath.confirmDetail,
        title: t.filePath.confirmTitle
      }
    })

    if (!result?.ok && result?.reason && result.reason !== 'canceled') {
      notifyError(result.error || result.reason, openErrorMessage(result.reason))
    }
  }

  async function reveal() {
    if (!bridge?.revealPath) {
      return
    }

    const result = await bridge.revealPath(path, { baseDir: cwd || undefined })

    if (!result?.ok && result?.reason) {
      notifyError(result.error || result.reason, t.filePath.revealFailed)
    }
  }

  async function copyPath() {
    try {
      if (bridge?.writeClipboard) {
        await bridge.writeClipboard(path)
      } else {
        await navigator.clipboard?.writeText(path)
      }
    } catch (error) {
      notifyError(error, t.common.copyFailed)
    }
  }

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <a
          aria-busy={opening || undefined}
          className={cn(
            'cursor-pointer font-mono text-[0.9em] text-foreground underline decoration-current/25 underline-offset-2 transition-colors wrap-anywhere hover:decoration-current/60'
          )}
          href="#"
          onClick={event => {
            event.preventDefault()
            void openPreview()
          }}
          title={t.filePath.tooltip}
        >
          {display}
        </a>
      </ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onSelect={() => void openPreview()}>{t.preview.openPreview}</ContextMenuItem>
        {bridge?.openPath ? (
          <ContextMenuItem onSelect={() => void openInDefaultApp()}>{t.filePath.openInDefaultApp}</ContextMenuItem>
        ) : null}
        {bridge?.revealPath ? (
          <ContextMenuItem onSelect={() => void reveal()}>{t.filePath.revealInFolder}</ContextMenuItem>
        ) : null}
        <ContextMenuSeparator />
        <ContextMenuItem onSelect={() => void copyPath()}>{t.filePath.copyPath}</ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  )
}
