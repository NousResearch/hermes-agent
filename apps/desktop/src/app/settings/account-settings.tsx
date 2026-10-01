import { compactNumber } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Activity, User } from '@/lib/icons'
import { $brandSession } from '@/store/brand-session'
import { $activeConnectionId } from '@/store/connections'
import { requestGatewayForAgent } from '@/store/gateway'
import { $gatewayState } from '@/store/session'
import { $settingsScopeProfile } from '@/store/settings-scope'

import { ListRow, SectionHeading, SettingsContent } from './primitives'

interface UsageModelTotal {
  model: string
  total_tokens: number
  cost_usd: number
}

interface UsageSummary {
  input_tokens: number
  output_tokens: number
  cache_read_tokens: number
  cache_write_tokens: number
  reasoning_tokens: number
  total_tokens: number
  cost_usd: number
  chat_count: number
  models: UsageModelTotal[]
}

function formatCost(amount: number): string {
  return `$${amount.toFixed(2)}`
}

function TokenUsage() {
  const { t } = useI18n()
  const copy = t.settings.account
  const gatewayState = useStore($gatewayState)
  const connectionId = useStore($activeConnectionId)
  const scopeProfile = useStore($settingsScopeProfile)
  const [summary, setSummary] = useState<UsageSummary | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    let cancelled = false
    setFailed(false)

    void requestGatewayForAgent<UsageSummary>(
      connectionId,
      scopeProfile,
      'usage.summary',
      scopeProfile ? { profile: scopeProfile } : {},
      undefined,
      undefined,
      { spawnPriority: 'foreground' }
    )
      .then(next => {
        if (!cancelled) {
          setSummary(next)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setFailed(true)
        }
      })

    return () => {
      cancelled = true
    }
  }, [connectionId, gatewayState, scopeProfile])

  const cache = (summary?.cache_read_tokens ?? 0) + (summary?.cache_write_tokens ?? 0)

  return (
    <>
      <SectionHeading icon={Activity} title={copy.limitsTitle} />
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {copy.limitsBody}
      </p>
      {failed ? (
        <p className="mb-4 text-sm text-(--ui-text-tertiary)">{copy.usageUnavailable}</p>
      ) : summary && summary.total_tokens > 0 ? (
        <>
          <ListRow description={copy.usageChats(summary.chat_count)} title={copy.usageTotal} action={compactNumber(summary.total_tokens)} />
          <ListRow title={copy.usageInput} action={compactNumber(summary.input_tokens)} />
          <ListRow title={copy.usageOutput} action={compactNumber(summary.output_tokens)} />
          <ListRow title={copy.usageCache} action={compactNumber(cache)} />
          <ListRow title={copy.usageReasoning} action={compactNumber(summary.reasoning_tokens)} />
          <ListRow title={copy.usageCost} action={formatCost(summary.cost_usd)} />
          {summary.models.length > 0 ? (
            <>
              <p className="mt-3 mb-1 text-xs font-medium text-(--ui-text-secondary)">{copy.usageModels}</p>
              {summary.models.map(row => (
                <ListRow
                  action={compactNumber(row.total_tokens)}
                  description={formatCost(row.cost_usd)}
                  key={row.model}
                  title={row.model}
                />
              ))}
            </>
          ) : null}
        </>
      ) : (
        <p className="mb-4 text-sm text-(--ui-text-tertiary)">{summary ? copy.usageEmpty : ''}</p>
      )}
    </>
  )
}

export function AccountSettings() {
  const brand = useStore($brandSession)
  const { t } = useI18n()
  const copy = t.settings.account
  const api = typeof window !== 'undefined' ? window.hermesDesktop?.brand : undefined
  const [busy, setBusy] = useState(false)
  const [draft, setDraft] = useState('')

  const brandLabel = brand.activeAppId || (brand.isSuper ? copy.allBrands : copy.signedOut)

  return (
    <SettingsContent>
      <SectionHeading icon={User} title={copy.title} />
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {brand.signedIn ? copy.intro : copy.signedOut}
      </p>

      {brand.signedIn ? (
        <>
          <ListRow description={brand.email} title={copy.email} />
          <ListRow description={brandLabel} title={copy.brand} />
          <ListRow description={brand.isSuper ? copy.superAccess : copy.brandAccess} title={copy.access} />
          {brand.appIds.length > 1 ? (
            <ListRow
              action={
                <select
                  aria-label={copy.brands}
                  className="bg-transparent text-sm"
                  onChange={event => {
                    void api?.select(event.target.value).then(next => $brandSession.set(next))
                  }}
                  value={brand.activeAppId}
                >
                  {brand.appIds.map(appId => (
                    <option key={appId} value={appId}>
                      {appId}
                    </option>
                  ))}
                </select>
              }
              title={copy.brands}
            />
          ) : null}
          {brand.isSuper ? (
            <ListRow
              action={
                <form
                  className="flex items-center gap-2"
                  onSubmit={event => {
                    event.preventDefault()
                    void api?.select(draft.trim()).then(next => {
                      $brandSession.set(next)
                      setDraft('')
                    })
                  }}
                >
                  <input
                    aria-label={copy.brand}
                    className="w-32 bg-transparent text-sm outline-none"
                    onChange={event => setDraft(event.target.value)}
                    placeholder={copy.brand}
                    value={draft}
                  />
                  {brand.activeAppId ? (
                    <Button
                      onClick={() => {
                        void api?.select('').then(next => $brandSession.set(next))
                      }}
                      type="button"
                      variant="outline"
                    >
                      {copy.allBrands}
                    </Button>
                  ) : null}
                </form>
              }
              title={copy.brand}
            />
          ) : null}
          <Button
            disabled={busy || !api}
            onClick={() => {
              if (!api) {
                return
              }

              setBusy(true)
              void api.signOut().then(next => $brandSession.set(next)).finally(() => setBusy(false))
            }}
            type="button"
            variant="outline"
          >
            {copy.signOut}
          </Button>
        </>
      ) : (
        <Button
          disabled={busy || !api}
          onClick={() => {
            if (!api) {
              return
            }

            setBusy(true)
            void api.signIn().then(next => $brandSession.set(next)).finally(() => setBusy(false))
          }}
          type="button"
        >
          {copy.signIn}
        </Button>
      )}
      <TokenUsage />
    </SettingsContent>
  )
}
