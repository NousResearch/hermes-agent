import { useStore } from '@nanostores/react'

import { StatusRow } from '@/components/chat/status-row'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { $billingBlock, billingCtaLabel, clearBillingBlock, runBillingRecovery } from '@/store/billing-block'

function firstLine(text: string): string {
  return (text || '').split('\n')[0]?.trim() ?? ''
}

/**
 * Persistent, in-stack billing wall for THIS session. Rendered as a shared
 * {@link StatusRow} — same chrome as its status-stack siblings, so it reads as
 * one piece with the composer card (no bordered alert-in-a-card). It never
 * disables the composer — slash commands (`/topup`, `/model`, `/login`) stay
 * usable — it only offers recovery: Nous opens Settings → Billing in-app, other
 * providers deep-link out. The sticky toast is the loud surface; this is the calm
 * reminder that outlives it.
 */
export function BillingBanner({ sessionId }: { sessionId: null | string }) {
  const active = useStore($billingBlock)
  const { t } = useI18n()

  if (!active || !sessionId || active.sessionId !== sessionId) {
    return null
  }

  const { block } = active
  const copy = t.billingBlock
  const title = block.is_nous ? copy.titleNous : copy.titleProvider(block.provider_label)
  const message = firstLine(block.message) || copy.fallbackMessage

  return (
    <StatusRow
      leading={<Codicon aria-hidden className="text-destructive/85" name="credit-card" size="0.8rem" />}
      trailing={
        <>
          <Button
            className="text-foreground/90 hover:text-foreground"
            onClick={() => runBillingRecovery(block)}
            size="micro"
            type="button"
            variant="text"
          >
            {billingCtaLabel(block, copy)}
          </Button>
          <Tip label={copy.dismiss}>
            <Button
              aria-label={copy.dismiss}
              className="size-4 rounded-md text-muted-foreground/60 hover:text-foreground/90"
              onClick={() => clearBillingBlock(sessionId)}
              size="icon-xs"
              type="button"
              variant="ghost"
            >
              <Codicon name="close" size="0.75rem" />
            </Button>
          </Tip>
        </>
      }
      trailingVisible
    >
      <span className="min-w-0 truncate text-[0.73rem] leading-4 text-foreground/92">
        <span className="font-medium">{title}</span>
        {message && <span className="text-muted-foreground/80"> · {message}</span>}
      </span>
    </StatusRow>
  )
}
