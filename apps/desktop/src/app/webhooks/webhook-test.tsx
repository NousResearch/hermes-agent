import { webhookTestCurl } from '@hermes/shared'
import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import { sendWebhookTest, type WebhookRoute, type WebhookTestResponse } from '@/hermes'
import { useI18n } from '@/i18n'
import { Send } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'

import { PanelBlock, PanelSectionLabel } from '../overlays/panel'

interface WebhookTestCommandProps {
  /** Only the create result knows it; the detail pane shows a placeholder. */
  secret?: string
  sub: Pick<WebhookRoute, 'signature' | 'url'>
}

/** The copy-paste signed curl for one route. */
export function WebhookTestCommand({ secret, sub }: WebhookTestCommandProps) {
  const w = useI18n().t.webhooks.test
  const command = webhookTestCurl({ secret, signature: sub.signature, url: sub.url })

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <PanelSectionLabel>{w.title}</PanelSectionLabel>
        <CopyButton appearance="icon" buttonSize="icon-sm" label={w.copyCommand} text={command} />
      </div>
      <PanelBlock className="font-mono">{command}</PanelBlock>
      <p className="text-[0.68rem] leading-relaxed text-muted-foreground">
        {w.hint} {secret ? null : w.placeholderHint}
      </p>
    </div>
  )
}

type TestCopy = ReturnType<typeof useI18n>['t']['webhooks']['test']

function resultNotice(w: TestCopy, result: WebhookTestResponse) {
  if (result.ok) {
    return { kind: 'success' as const, message: w.accepted(result.status, result.delivery_id) }
  }

  return { kind: 'error' as const, message: result.outcome === 'ignored' ? w.ignored : w.rejected(result.status, result.response) }
}

/** Detail-pane test block: the curl plus a Send test the backend signs with the stored secret.
 *  The parent keys it by (profile, subscription), so a pending test belongs to the row that sent it. */
export function WebhookTestPanel({ profile, sub }: { profile: string; sub: WebhookRoute }) {
  const w = useI18n().t.webhooks.test
  const [sending, setSending] = useState(false)
  const ownerActive = useRef(true)

  // Same StrictMode-safe ownership flag as the page: a completion after this row unmounts (another
  // row selected, profile switched) must not toast into the new owner's view.
  // eslint-disable-next-line no-restricted-syntax -- lifecycle ownership flag, not an atom mirror
  useEffect(() => {
    ownerActive.current = true

    return () => {
      ownerActive.current = false
    }
  }, [])

  const send = useCallback(async () => {
    setSending(true)

    try {
      const result = await sendWebhookTest(sub.name, profile)

      if (ownerActive.current) {
        notify(resultNotice(w, result))
      }
    } catch (err) {
      if (ownerActive.current) {
        notifyError(err, w.failed)
      }
    } finally {
      if (ownerActive.current) {
        setSending(false)
      }
    }
  }, [profile, sub.name, w])

  return (
    <div className="space-y-2">
      <WebhookTestCommand sub={sub} />
      <Button disabled={sending || !sub.enabled} onClick={() => void send()} size="sm" variant="secondary">
        <Send />
        {sending ? w.sending : w.send}
      </Button>
    </div>
  )
}
