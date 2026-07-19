import * as QRCode from 'qrcode'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { ErrorState } from '@/components/ui/error-state'
import { Loader } from '@/components/ui/loader'
import { useI18n } from '@/i18n'
import {
  type ContinueOnPhoneResult,
  resolveContinueOnPhoneUrl
} from '@/lib/continue-on-phone'
import { notifyError } from '@/store/notifications'

type ResolveUrl = (sessionId: string, profile?: string) => Promise<ContinueOnPhoneResult>
type GenerateQr = (url: string) => Promise<string>

interface ContinueOnPhoneDialogProps {
  generateQr?: GenerateQr
  onOpenChange: (open: boolean) => void
  open: boolean
  profile?: string
  resolveUrl?: ResolveUrl
  sessionId: string
}

type DialogState =
  | { phase: 'error' }
  | { phase: 'idle' | 'loading' }
  | { phase: 'ready'; qrDataUrl: string; url: string }

const generateQrDataUrl: GenerateQr = url =>
  QRCode.toDataURL(url, {
    errorCorrectionLevel: 'M',
    margin: 3,
    width: 240
  })

export function ContinueOnPhoneDialog({
  generateQr = generateQrDataUrl,
  onOpenChange,
  open,
  profile,
  resolveUrl = resolveContinueOnPhoneUrl,
  sessionId
}: ContinueOnPhoneDialogProps) {
  const { t } = useI18n()
  const copy = t.sidebar.row
  const [retry, setRetry] = useState(0)
  const [state, setState] = useState<DialogState>({ phase: 'idle' })

  useEffect(() => {
    if (!open) {
      return
    }

    let cancelled = false
    setState({ phase: 'loading' })

    void resolveUrl(sessionId, profile)
      .then(async result => {
        if (!result.ok) {
          return null
        }

        return {
          qrDataUrl: await generateQr(result.url),
          url: result.url
        }
      })
      .then(result => {
        if (cancelled) {
          return
        }

        setState(result ? { phase: 'ready', ...result } : { phase: 'error' })
      })
      .catch(() => {
        if (!cancelled) {
          setState({ phase: 'error' })
        }
      })

    return () => {
      cancelled = true
    }
  }, [generateQr, open, profile, resolveUrl, retry, sessionId])

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{copy.continueOnPhoneTitle}</DialogTitle>
          <DialogDescription>{copy.continueOnPhoneDesc}</DialogDescription>
        </DialogHeader>

        {state.phase === 'loading' && (
          <div className="grid min-h-72 place-items-center gap-3 py-8 text-center">
            <Loader className="size-12" label={copy.continueOnPhonePreparing} type="lemniscate-bloom" />
            <p className="text-sm text-(--ui-text-secondary)">{copy.continueOnPhonePreparing}</p>
          </div>
        )}

        {state.phase === 'error' && (
          <ErrorState
            className="py-8"
            description={copy.continueOnPhoneUnavailableDesc}
            title={copy.continueOnPhoneUnavailableTitle}
          >
            <Button onClick={() => setRetry(value => value + 1)} type="button" variant="secondary">
              {t.common.retry}
            </Button>
          </ErrorState>
        )}

        {state.phase === 'ready' && (
          <>
            <div className="grid justify-items-center gap-4 py-3">
              <img
                alt={copy.continueOnPhoneQrAlt}
                className="size-60 max-w-full"
                height={240}
                src={state.qrDataUrl}
                width={240}
              />
              <p className="max-w-full truncate text-xs text-(--ui-text-tertiary)">{state.url}</p>
            </div>
            <DialogFooter>
              <CopyButton
                buttonSize="sm"
                buttonVariant="secondary"
                label={copy.continueOnPhoneCopyLink}
                text={state.url}
              />
              <Button
                onClick={() => {
                  void window.hermesDesktop.openExternal(state.url).catch(error =>
                    notifyError(error, copy.continueOnPhoneOpenFailed)
                  )
                }}
                size="sm"
                type="button"
              >
                {copy.continueOnPhoneOpenBrowser}
              </Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}
