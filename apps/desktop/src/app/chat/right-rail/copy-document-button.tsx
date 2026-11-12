import { useEffect, useRef, useState } from 'react'

import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { useI18n } from '@/i18n'
import { copyTextToClipboard } from '@/lib/desktop-fs'
import { Check, Copy } from '@/lib/icons'
import { cn } from '@/lib/utils'

/** Cap the on-screen confirmation so a huge document can't flood the popover. */
const CONFIRMATION_PREVIEW_LIMIT = 4_000
const SUCCESS_RESET_MS = 6_000

/** Reads the editor's ref at click time, not the last rendered draft. */
export function CopyDocumentButton({ getText }: { getText: () => string }) {
  const { t } = useI18n()
  const [status, setStatus] = useState<'idle' | 'pending' | 'success' | 'error'>('idle')
  const [copiedText, setCopiedText] = useState('')
  const [open, setOpen] = useState(false)
  const pending = useRef(false)
  const mounted = useRef(true)

  // eslint-disable-next-line no-restricted-syntax -- component lifetime guard, not an atom mirror
  useEffect(() => {
    mounted.current = true

    return () => {
      mounted.current = false
    }
  }, [])

  useEffect(() => {
    if (status !== 'success') {return}

    const timer = setTimeout(() => {
      setStatus('idle')
      setOpen(false)
    }, SUCCESS_RESET_MS)

    return () => clearTimeout(timer)
  }, [status])

  const copy = async () => {
    if (pending.current) {return}
    pending.current = true
    setStatus('pending')

    try {
      const text = getText()
      await copyTextToClipboard(text)

      if (mounted.current) {
        setCopiedText(text)
        setStatus('success')
        setOpen(true)
      }
    } catch {
      if (mounted.current) {setStatus('error')}
    } finally {
      pending.current = false
    }
  }

  const truncated = copiedText.length > CONFIRMATION_PREVIEW_LIMIT
  const preview = truncated ? copiedText.slice(0, CONFIRMATION_PREVIEW_LIMIT) : copiedText

  return (
    <span className="flex items-center gap-2">
      <Popover onOpenChange={setOpen} open={open}>
        <PopoverTrigger asChild>
          <button
            className={cn(
              'flex items-center gap-1 whitespace-nowrap text-[0.625rem] font-bold transition-colors disabled:opacity-50',
              status === 'success'
                ? 'text-(--ui-success) hover:text-(--ui-success)'
                : 'text-muted-foreground hover:text-foreground'
            )}
            disabled={status === 'pending'}
            onClick={() => void copy()}
            type="button"
          >
            {status === 'success' ? (
              <Check aria-hidden="true" className="size-3" />
            ) : (
              <Copy aria-hidden="true" className="size-3" />
            )}
            {status === 'success' ? t.common.copied : t.preview.copyDocument}
          </button>
        </PopoverTrigger>
        {status === 'success' && (
          <PopoverContent align="end" className="w-80 p-0">
            <div
              aria-live="polite"
              className="max-h-72 overflow-auto rounded-md border border-(--ui-success)/40 bg-(--ui-success)/10 p-2"
              data-slot="copy-document-confirmation"
              role="status"
            >
              <div className="mb-1 flex items-center gap-1 text-[0.625rem] font-bold text-(--ui-success)">
                <Check aria-hidden="true" className="size-3" />
                {t.common.copied}
              </div>
              <pre className="max-h-56 overflow-auto whitespace-pre-wrap break-words font-mono text-[0.625rem] leading-relaxed text-foreground/90">
                {preview}
              </pre>
              {truncated && (
                <div className="mt-1 text-[0.625rem] text-muted-foreground">
                  {t.preview.copyDocumentTruncated(copiedText.length)}
                </div>
              )}
            </div>
          </PopoverContent>
        )}
      </Popover>
      <span
        aria-live="polite"
        className={status === 'error' ? 'text-[0.625rem] text-destructive' : 'sr-only'}
        role={status === 'error' ? 'alert' : undefined}
      >
        {status === 'error' ? t.common.copyFailed : ''}
      </span>
    </span>
  )
}
