import { useEffect, useMemo, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ErrorBanner } from '@/components/ui/error-state'
import { Input } from '@/components/ui/input'
import {
  applyTelegramOnboarding,
  cancelTelegramOnboarding,
  getTelegramOnboardingStatus,
  type MessagingPlatformInfo,
  type ProfileScope,
  startTelegramOnboarding,
  type TelegramOnboardingApplyResponse,
  type TelegramOnboardingStartResponse
} from '@/hermes'
import { useI18n } from '@/i18n'
import { openExternalLink } from '@/lib/external-link'
import { Check, ExternalLink, QrCode, Save, X } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { CREDENTIAL_CONTROL_CLASS } from '../settings/credential-key-ui'

const TELEGRAM_USER_ID_RE = /^\d{3,20}$/

type Phase = 'applying' | 'idle' | 'ready' | 'starting' | 'waiting'

/** A 410 from the status endpoint means the pairing is gone for good (expired/claimed). */
export function isTerminalTelegramOnboardingError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /\b410\b/.test(message) && /\b(expired|claimed|gone)\b/i.test(message)
}

export function formatExpiry(expiresAt: string, now = Date.now()): null | string {
  const ms = Date.parse(expiresAt) - now

  if (!Number.isFinite(ms) || ms <= 0) {
    return null
  }

  const seconds = Math.ceil(ms / 1000)

  return `${Math.floor(seconds / 60)}:${(seconds % 60).toString().padStart(2, '0')}`
}

async function renderQr(payload: string): Promise<string> {
  // Lazy: the QR encoder is only needed while a pairing is on screen.
  const QRCode = await import('qrcode')

  return QRCode.toDataURL(payload, { errorCorrectionLevel: 'M', margin: 1, width: 224 })
}

export interface TelegramQrSetupProps {
  /** Called after the backend wrote the token + allowlist; receives the restart outcome. */
  onApplied: (result: TelegramOnboardingApplyResponse) => void
  platform: MessagingPlatformInfo
  /** Request-shaped profile scope (undefined → active profile). */
  scopeProfile: ProfileScope
}

/**
 * Telegram "Quick setup": the Nous pairing service mints a bot on the user's
 * behalf; we show its QR/deep link, poll until Telegram confirms, let the user
 * review the allowlist, and apply. Mirrors the dashboard's Channels flow so
 * the Desktop no longer sends users to @BotFather by hand.
 */
export function TelegramQrSetup({ onApplied, platform, scopeProfile }: TelegramQrSetupProps) {
  const { t } = useI18n()
  const q = t.messaging.telegramQr
  const [setup, setSetup] = useState<null | TelegramOnboardingStartResponse>(null)
  const [qrDataUrl, setQrDataUrl] = useState('')
  const [phase, setPhase] = useState<Phase>('idle')
  const [botUsername, setBotUsername] = useState<null | string>(null)
  const [allowedIds, setAllowedIds] = useState<string[]>([])
  const [detectedOwnerId, setDetectedOwnerId] = useState<null | string>(null)
  const [newAllowedId, setNewAllowedId] = useState('')
  const [error, setError] = useState('')
  const [tick, setTick] = useState(0)

  const reset = () => {
    setSetup(null)
    setQrDataUrl('')
    setPhase('idle')
    setBotUsername(null)
    setAllowedIds([])
    setDetectedOwnerId(null)
    setNewAllowedId('')
    setError('')
  }

  // Poll the pairing until Telegram confirms. A transient fetch error keeps
  // polling with a visible hint; a terminal 410 (or local expiry) resets.
  useEffect(() => {
    if (!setup || phase !== 'waiting') {
      return
    }

    let cancelled = false
    let timer: null | number = null

    const poll = async () => {
      try {
        const status = await getTelegramOnboardingStatus(setup.pairing_id, scopeProfile)

        if (cancelled) {
          return
        }

        if (status.status === 'ready') {
          setPhase('ready')
          setBotUsername(status.bot_username ?? null)
          setError('')

          if (status.owner_user_id && TELEGRAM_USER_ID_RE.test(status.owner_user_id)) {
            setDetectedOwnerId(status.owner_user_id)
            setAllowedIds([status.owner_user_id])
          }

          return
        }

        setError('')
        timer = window.setTimeout(() => void poll(), 2000)
      } catch (pollError) {
        if (cancelled) {
          return
        }

        const expiresAt = Date.parse(setup.expires_at)
        const expired = Number.isFinite(expiresAt) && Date.now() >= expiresAt

        if (isTerminalTelegramOnboardingError(pollError) || expired) {
          setSetup(null)
          setQrDataUrl('')
          setPhase('idle')
          setError(q.pairingExpired)

          return
        }

        setError(q.stillWaiting(String(pollError)))
        timer = window.setTimeout(() => void poll(), 2000)
      }
    }

    timer = window.setTimeout(() => void poll(), 1200)

    return () => {
      cancelled = true

      if (timer !== null) {
        window.clearTimeout(timer)
      }
    }
  }, [phase, q, scopeProfile, setup])

  // One-second tick keeps the expiry countdown honest while a pairing is shown.
  useEffect(() => {
    if (!setup) {
      return
    }

    const timer = window.setInterval(() => setTick(value => value + 1), 1000)

    return () => window.clearInterval(timer)
  }, [setup])

  const expiresIn = useMemo(
    () => (setup ? formatExpiry(setup.expires_at) : null),
    // tick refreshes the memo each second without recomputing on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [setup, tick]
  )

  const start = async () => {
    setPhase('starting')
    setError('')
    setBotUsername(null)
    setAllowedIds([])
    setDetectedOwnerId(null)
    setNewAllowedId('')

    try {
      const result = await startTelegramOnboarding(undefined, scopeProfile)
      const dataUrl = await renderQr(result.qr_payload)
      setSetup(result)
      setQrDataUrl(dataUrl)
      setPhase('waiting')
    } catch (startError) {
      setPhase('idle')
      setError(String(startError))
    }
  }

  const cancel = async () => {
    if (setup) {
      try {
        await cancelTelegramOnboarding(setup.pairing_id, scopeProfile)
      } catch {
        // Local cleanup still wins; the backend prunes expired pairings itself.
      }
    }

    reset()
  }

  const addAllowedId = () => {
    const trimmed = newAllowedId.trim()

    if (!TELEGRAM_USER_ID_RE.test(trimmed)) {
      setError(q.numericOnly)

      return
    }

    setError('')
    setAllowedIds(ids => (ids.includes(trimmed) ? ids : [...ids, trimmed]))
    setNewAllowedId('')
  }

  const apply = async () => {
    if (!setup) {
      return
    }

    if (allowedIds.length === 0) {
      setError(q.addAtLeastOne)

      return
    }

    setPhase('applying')
    setError('')

    try {
      const result = await applyTelegramOnboarding(setup.pairing_id, allowedIds, scopeProfile)
      reset()
      onApplied(result)
    } catch (applyError) {
      setPhase('ready')
      setError(String(applyError))
    }
  }

  return (
    <div className="rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-surface-secondary,transparent) p-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[length:var(--conversation-text-font-size)] font-medium">{q.quickSetup}</span>
            <Badge variant="success">{q.recommended}</Badge>
          </div>
          <p className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {q.quickHelp}
          </p>
        </div>
        {phase === 'idle' && (
          <Button onClick={() => void start()} size="sm">
            <QrCode />
            {q.createWithQr}
          </Button>
        )}
        {phase === 'starting' && (
          <Button disabled size="sm">
            {q.starting}
          </Button>
        )}
      </div>

      {platform.configured && phase === 'idle' && (
        <p className="mt-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-muted-foreground">
          {q.replaceWarning}
        </p>
      )}

      {error && <ErrorBanner className="mt-3">{error}</ErrorBanner>}

      {setup && qrDataUrl && (
        <div className="mt-3 grid gap-4 border-t border-(--ui-stroke-secondary) pt-3 lg:grid-cols-[minmax(0,1fr)_240px]">
          <div className="grid content-start gap-3">
            {phase === 'waiting' && (
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="warn">{q.waiting}</Badge>
                <span className="text-xs text-muted-foreground">{q.scanHint}</span>
              </div>
            )}

            {(phase === 'ready' || phase === 'applying') && (
              <>
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant="success">{q.ready}</Badge>
                  {botUsername && <span className="font-mono text-xs text-muted-foreground">@{botUsername}</span>}
                </div>

                <div className="grid gap-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[0.7rem] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                      {q.allowedUsers}
                    </span>
                    {detectedOwnerId && allowedIds.includes(detectedOwnerId) && (
                      <Badge variant="success">{q.ownerDetected}</Badge>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {allowedIds.map(id => (
                      <button
                        aria-label={`${t.common.remove} ${id}`}
                        className={cn(
                          'inline-flex items-center gap-1 rounded-md border border-(--ui-stroke-secondary) px-2 py-1 font-mono text-xs',
                          'hover:border-destructive/50 hover:text-destructive'
                        )}
                        key={id}
                        onClick={() => setAllowedIds(ids => ids.filter(existing => existing !== id))}
                        type="button"
                      >
                        {id}
                        <X className="size-3" />
                      </button>
                    ))}
                    {allowedIds.length === 0 && (
                      <span className="text-xs text-muted-foreground">{q.addAtLeastOne}</span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <Input
                      className={CREDENTIAL_CONTROL_CLASS}
                      onChange={event => setNewAllowedId(event.target.value)}
                      onKeyDown={event => {
                        if (event.key === 'Enter') {
                          event.preventDefault()
                          addAllowedId()
                        }
                      }}
                      placeholder={q.userIdPlaceholder}
                      value={newAllowedId}
                    />
                    <Button onClick={addAllowedId} size="sm" variant="secondary">
                      <Check />
                      {q.add}
                    </Button>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  <Button disabled={phase === 'applying'} onClick={() => void apply()} size="sm">
                    <Save />
                    {phase === 'applying' ? q.applying : q.saveAndRestart}
                  </Button>
                  <Button disabled={phase === 'applying'} onClick={() => void cancel()} size="sm" variant="ghost">
                    {t.common.cancel}
                  </Button>
                </div>
              </>
            )}
          </div>

          <div className="flex flex-col items-center gap-2">
            <img alt="Telegram setup QR code" className="size-56 rounded-md bg-white p-2" src={qrDataUrl} />
            <Badge variant={expiresIn ? 'outline' : 'destructive'}>
              {expiresIn ? q.expiresIn(expiresIn) : q.expired}
            </Badge>
            <div className="flex flex-wrap justify-center gap-2">
              <Button onClick={() => openExternalLink(setup.deep_link)} size="sm" variant="secondary">
                <ExternalLink />
                {q.openTelegram}
              </Button>
              {phase === 'waiting' && (
                <Button onClick={() => void cancel()} size="sm" variant="ghost">
                  {t.common.cancel}
                </Button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
