import { useCallback, useEffect, useRef, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ErrorBanner } from '@/components/ui/error-state'
import {
  getRelayLinks,
  getRelayLinkState,
  type MessagingPlatformInfo,
  type RelayLinkCode,
  type RelayLinkedAccount,
  requestRelayLink,
  unlinkRelayAccount
} from '@/hermes'
import { useI18n } from '@/i18n'
import { openExternalLink } from '@/lib/external-link'
import { Copy, ExternalLink, QrCode } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'

const POLL_MS = 3000
// Consecutive poll failures before the panel stops waiting and shows the error.
const MAX_POLL_ERRORS = 5

function hasExpired(link: RelayLinkCode): boolean {
  const expiresAt = link.expires_at ? Date.parse(link.expires_at) : Number.NaN

  return Number.isFinite(expiresAt) && Date.now() >= expiresAt
}

async function renderQr(payload: string): Promise<string> {
  // Lazy: the QR encoder is only needed while a code is on screen.
  const QRCode = await import('qrcode')

  return QRCode.toDataURL(payload, { errorCorrectionLevel: 'M', margin: 1, width: 192 })
}

// The IPC bridge hands a router refusal over as "Error invoking remote method 'hermes:api': Error:
// 409: {"detail":{"code":…,"message":…}}"; the card shows the connector's sentence, not the envelope.
function relayErrorText(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error)
  const text = (raw.match(/Error invoking remote method '[^']+': Error: (.+)$/s)?.[1] ?? raw).replace(/^Error:\s*/, '')
  const body = text.match(/^\d{3}:\s*(\{.*\})$/s)?.[1]

  if (!body) {
    return text
  }

  try {
    const detail = (JSON.parse(body) as { detail?: unknown }).detail
    const message = typeof detail === 'string' ? detail : (detail as undefined | { message?: unknown })?.message

    return typeof message === 'string' && message ? message : text
  } catch {
    return text
  }
}

function accountLabel(account: RelayLinkedAccount): string {
  return account.displayName || (account.handle ? `@${account.handle.replace(/^@/, '')}` : '') || account.platformUserId
}

/**
 * Team Gateway setup for a platform the relay's shared bot carries (#141546). On such an instance a
 * bot of the user's own is switched off at every gateway start, so setup is linking the user's chat
 * account to the shared bot: mint a one-time code, send `/link <code>` to the bot, done. A
 * Portal-managed agent takes links only from the Portal (a code minted here would bind nothing), so
 * there the card lists the linked accounts and points to the Portal.
 */
export function RelayLinkSetup({
  platform,
  scopeProfile
}: {
  platform: MessagingPlatformInfo
  scopeProfile?: null | string
}) {
  const { t } = useI18n()
  const r = t.messaging.relayLink
  const [accounts, setAccounts] = useState<null | RelayLinkedAccount[]>(null)
  const [botUrl, setBotUrl] = useState<null | string>(null)
  const [link, setLink] = useState<null | RelayLinkCode>(null)
  const [qrDataUrl, setQrDataUrl] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [portalUrl, setPortalUrl] = useState<null | string>(null)
  // Accounts linked when the code was minted: only an id not in this set proves a new link.
  const linkedBefore = useRef<Set<string>>(new Set())

  const loadAccounts = useCallback(async () => {
    try {
      const response = await getRelayLinks(scopeProfile)
      const row = response.platforms.find(entry => entry.platform === platform.id)
      setPortalUrl(response.managed_by === 'portal' ? (response.portal_url ?? null) : null)
      setAccounts(row?.accounts ?? [])
      setBotUrl(row?.bot_url ?? null)
      setError('')

      return row?.accounts ?? []
    } catch (loadError) {
      setAccounts(null)
      setError(relayErrorText(loadError))

      return null
    }
  }, [platform.id, scopeProfile])

  useEffect(() => {
    void loadAccounts()
  }, [loadAccounts])

  useEffect(() => {
    if (!link) {
      return
    }

    let cancelled = false
    let timer: null | number = null
    let failures = 0

    const poll = async () => {
      try {
        const state = await getRelayLinkState(link.code, scopeProfile)
        failures = 0

        if (cancelled) {
          return
        }

        if (state.state === 'pending') {
          timer = window.setTimeout(() => void poll(), POLL_MS)

          return
        }

        if (state.state === 'refused') {
          setLink(null)
          setQrDataUrl('')
          setError(r.refused(platform.name))

          return
        }

        // Reload before closing the code: closing re-runs this effect and cancels the rest of it.
        const after = await loadAccounts()

        if (cancelled) {
          return
        }

        setLink(null)
        setQrDataUrl('')

        if (!after) {
          return
        }

        // `inactive` alone means used, expired or unknown: only a new account id proves the link.
        if (after.some(account => !linkedBefore.current.has(account.platformUserId))) {
          notify({ kind: 'success', message: r.linked(platform.name) })
        } else if (hasExpired(link)) {
          setError(r.expired)
        } else {
          setNotice(r.noNewAccount(platform.name))
        }
      } catch (pollError) {
        if (cancelled) {
          return
        }

        failures += 1

        if (failures >= MAX_POLL_ERRORS || hasExpired(link)) {
          setLink(null)
          setQrDataUrl('')
          setError(relayErrorText(pollError))

          return
        }

        timer = window.setTimeout(() => void poll(), POLL_MS)
      }
    }

    timer = window.setTimeout(() => void poll(), POLL_MS)

    return () => {
      cancelled = true

      if (timer !== null) {
        window.clearTimeout(timer)
      }
    }
  }, [link, loadAccounts, platform.name, r, scopeProfile])

  const start = async () => {
    setBusy(true)
    setError('')
    setNotice('')

    try {
      const result = await requestRelayLink(platform.id, scopeProfile)
      linkedBefore.current = new Set((accounts ?? []).map(account => account.platformUserId))
      setLink(result)
      const target = result.bot_url ?? botUrl
      setQrDataUrl(target ? await renderQr(target) : '')
    } catch (startError) {
      setError(relayErrorText(startError))
    } finally {
      setBusy(false)
    }
  }

  const unlink = async (account: RelayLinkedAccount) => {
    try {
      await unlinkRelayAccount(platform.id, account.platformUserId, scopeProfile)
      await loadAccounts()
    } catch (unlinkError) {
      notifyError(unlinkError, r.unlinkFailed)
    }
  }

  const copyCommand = async () => {
    if (!link) {
      return
    }

    try {
      await navigator.clipboard.writeText(link.command)
      notify({ kind: 'success', message: r.copied })
    } catch (copyError) {
      notifyError(copyError, r.copyFailed(link.command))
    }
  }

  const openUrl = link?.bot_url ?? botUrl

  return (
    <div
      className="rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-surface-secondary,transparent) p-3"
      data-testid={`relay-link-${platform.id}`}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[length:var(--conversation-text-font-size)] font-medium">{r.title}</span>
            <Badge variant="success">{r.badge}</Badge>
          </div>
          <p className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {r.intro(platform.name)}
            {portalUrl && ` ${r.managedIntro}`}
          </p>
        </div>
        {portalUrl && (
          <Button onClick={() => openExternalLink(portalUrl)} size="sm">
            <ExternalLink />
            {r.openPortal}
          </Button>
        )}
        {!portalUrl && !link && (
          <Button disabled={busy} onClick={() => void start()} size="sm">
            <QrCode />
            {accounts && accounts.length > 0 ? r.linkAnother(platform.name) : r.link(platform.name)}
          </Button>
        )}
      </div>

      {accounts && accounts.length > 0 && (
        <div className="mt-3 grid gap-1">
          <span className="text-xs text-muted-foreground">{r.linkedAccounts}</span>
          {accounts.map(account => (
            <div className="flex items-center justify-between gap-2" key={account.platformUserId}>
              <span className="truncate text-sm">{accountLabel(account)}</span>
              {!portalUrl && (
                <Button onClick={() => void unlink(account)} size="sm" variant="ghost">
                  {r.unlink}
                </Button>
              )}
            </div>
          ))}
        </div>
      )}

      {notice && <p className="mt-3 text-xs text-muted-foreground">{notice}</p>}
      {error && <ErrorBanner className="mt-3">{error}</ErrorBanner>}

      {link && (
        <div className="mt-3 grid gap-4 border-t border-(--ui-stroke-secondary) pt-3 lg:grid-cols-[minmax(0,1fr)_200px]">
          <div className="grid content-start gap-2 text-sm">
            <span>{r.stepOpen(platform.name)}</span>
            <span>{r.stepSend}</span>
            <code className="w-fit rounded bg-(--ui-surface-tertiary,transparent) px-2 py-1 font-mono">
              {link.command}
            </code>
            <div className="flex flex-wrap gap-2">
              {openUrl && (
                <Button onClick={() => openExternalLink(openUrl)} size="sm">
                  <ExternalLink />
                  {r.open(platform.name)}
                </Button>
              )}
              <Button onClick={() => void copyCommand()} size="sm" variant="secondary">
                <Copy />
                {r.copy}
              </Button>
              <Button
                onClick={() => {
                  setLink(null)
                  setQrDataUrl('')
                }}
                size="sm"
                variant="ghost"
              >
                {t.common.cancel}
              </Button>
            </div>
            <span className="text-xs text-muted-foreground">{r.waiting}</span>
          </div>
          {qrDataUrl && (
            <img
              alt={r.qrAlt(platform.name)}
              className="size-48 rounded-lg border border-(--ui-stroke-secondary) bg-white p-1"
              src={qrDataUrl}
            />
          )}
        </div>
      )}
    </div>
  )
}
