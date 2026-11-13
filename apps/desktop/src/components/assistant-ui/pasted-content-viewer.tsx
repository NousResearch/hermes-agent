'use client'

import { type FC, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { DesktopFileMissingError, readDesktopFileText } from '@/lib/desktop-fs'

/**
 * Reader for a large paste that was turned into a `.txt` attachment.
 *
 * The chip in a sent message only carries the file path, so the text a user
 * pasted is not part of the transcript. This dialog reads that file back off
 * disk and shows it read-only — the paste is never re-submitted, and the file
 * is only opened when someone asks to see it.
 *
 * The three terminal states are deliberately distinct: a missing file (the
 * temp paste was cleaned up) says so instead of painting an empty box, a read
 * failure keeps the error message, and a truncated read is labelled.
 */
type LoadState =
  | { message: string; status: 'error' }
  | { status: 'loading' }
  | { status: 'missing' }
  | { status: 'ready'; text: string; truncated: boolean }

export const PastedContentViewer: FC<{
  onOpenChange: (open: boolean) => void
  open: boolean
  path: string
}> = ({ onOpenChange, open, path }) => {
  const { t } = useI18n()
  const copy = t.desktop
  const [state, setState] = useState<LoadState>({ status: 'loading' })

  useEffect(() => {
    if (!open) {
      return
    }

    let live = true

    setState({ status: 'loading' })

    readDesktopFileText(path)
      .then(result => {
        if (live) {
          setState({ status: 'ready', text: result.text ?? '', truncated: Boolean(result.truncated) })
        }
      })
      .catch((error: unknown) => {
        if (!live) {
          return
        }

        if (error instanceof DesktopFileMissingError) {
          setState({ status: 'missing' })

          return
        }

        setState({ message: error instanceof Error ? error.message : String(error), status: 'error' })
      })

    return () => {
      live = false
    }
  }, [open, path])

  const ready = state.status === 'ready' ? state : null

  // Same bus every other out-of-composer producer uses, imported on demand so
  // the transcript's module graph never has to boot the composer stack.
  const insertIntoComposer = () => {
    if (!ready) {
      return
    }

    const text = ready.text

    void import('@/app/chat/composer/focus').then(({ requestComposerInsert }) => {
      requestComposerInsert(text, { mode: 'block' })
    })
    onOpenChange(false)
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-2xl" data-slot="pasted-content-viewer">
        <DialogHeader>
          <DialogTitle>{copy.pastedContent}</DialogTitle>
          <DialogDescription className="wrap-anywhere font-mono text-[0.6875rem]" title={path}>
            {path}
          </DialogDescription>
        </DialogHeader>

        {state.status === 'loading' && (
          <p className="text-[0.8125rem] text-(--ui-text-tertiary)" data-slot="pasted-content-loading">
            {copy.pastedContentLoading}
          </p>
        )}

        {state.status === 'missing' && (
          <p className="text-[0.8125rem] text-destructive" data-slot="pasted-content-missing" role="alert">
            {copy.pastedContentMissing}
          </p>
        )}

        {state.status === 'error' && (
          <p className="wrap-anywhere text-[0.8125rem] text-destructive" data-slot="pasted-content-error" role="alert">
            {copy.pastedContentReadFailed(state.message)}
          </p>
        )}

        {ready && (
          <pre
            className="max-h-[55vh] overflow-auto whitespace-pre-wrap wrap-anywhere rounded-md border border-(--ui-stroke-tertiary) bg-[color-mix(in_srgb,currentColor_4%,transparent)] p-3 font-mono text-[0.78rem] leading-relaxed select-text"
            data-slot="pasted-content-text"
          >
            {ready.text}
          </pre>
        )}

        {ready?.truncated && (
          <p className="text-[0.75rem] text-(--ui-text-tertiary)" data-slot="pasted-content-truncated">
            {copy.pastedContentTruncated}
          </p>
        )}

        <DialogFooter className="items-center">
          <Button onClick={() => onOpenChange(false)} variant="ghost">
            {t.common.close}
          </Button>
          <CopyButton disabled={!ready} label={copy.pastedContentCopy} text={() => ready?.text ?? ''} />
          <Button data-slot="pasted-content-insert" disabled={!ready} onClick={insertIntoComposer}>
            {copy.pastedContentInsert}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
