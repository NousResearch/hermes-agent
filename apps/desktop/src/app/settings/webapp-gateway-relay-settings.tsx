import { type FormEvent, useCallback, useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { GatewayRelayGateway, GatewayRelayGrant } from '@/global'
import { useI18n } from '@/i18n'
import { Check, Loader2, Network } from '@/lib/icons'
import { relativeTime } from '@/lib/time'
import { confirm } from '@/store/confirm'
import { refreshConnectionsRegistry } from '@/store/connections'
import { notify, notifyError } from '@/store/notifications'

import { ListRow, Pill, SettingsSection } from './primitives'

type Reachability = Record<string, { error?: string; ok: boolean }>

const OUTCOME_PARAMS = ['relay', 'relay_detail'] as const

/**
 * Settings → Gateways in a Webapp whose gateway relays to others (`webapp.relay.enabled`): connect
 * another gateway (its consent page opens here, and approving it there returns here), remove or test
 * one, and revoke the gateways that reach this one. Hidden where the bridge has no relay.
 */
export function WebappGatewayRelaySettings() {
  const relay = window.hermesDesktop?.gatewayRelay

  return relay ? <RelaySections relay={relay} /> : null
}

function RelaySections({ relay }: { relay: NonNullable<Window['hermesDesktop']['gatewayRelay']> }) {
  const { t } = useI18n()
  const r = t.settings.gateway.relay
  const location = useLocation()
  const navigate = useNavigate()
  const [gateways, setGateways] = useState<GatewayRelayGateway[] | null>(null)
  const [grants, setGrants] = useState<GatewayRelayGrant[] | null>(null)
  const [reach, setReach] = useState<Reachability>({})
  const [url, setUrl] = useState('')
  const [label, setLabel] = useState('')
  const [connecting, setConnecting] = useState(false)
  const [busy, setBusy] = useState<null | string>(null)

  const load = useCallback(async () => {
    try {
      const [listed, granted] = await Promise.all([relay.gateways(), relay.grants()])
      setGateways(listed)
      setGrants(granted)
    } catch (error) {
      notifyError(error, r.loadFailed)
    }
  }, [r.loadFailed, relay])

  useEffect(() => void load(), [load])

  // The pairing callback lands here with its outcome; say it once, then drop it from the URL.
  useEffect(() => {
    const params = new URLSearchParams(location.search)
    const outcome = params.get('relay')

    if (!outcome) {
      return
    }

    const detail = params.get('relay_detail') || ''

    if (outcome === 'connected') {
      notify({ kind: 'success', message: r.connected(detail) })
      void refreshConnectionsRegistry()
    } else {
      notify({ kind: 'error', message: outcome === 'denied' ? r.denied : r.failed(detail) })
    }

    OUTCOME_PARAMS.forEach(key => params.delete(key))
    navigate({ hash: location.hash, pathname: location.pathname, search: `?${params}` }, { replace: true })
  }, [location.hash, location.pathname, location.search, navigate, r])

  const connect = async (event: FormEvent) => {
    event.preventDefault()
    setConnecting(true)

    try {
      // The other gateway's consent page: the user approves there, signed in as themselves.
      window.location.assign(await relay.startPairing(url.trim(), label.trim()))
    } catch (error) {
      notifyError(error, r.connectFailed)
      setConnecting(false)
    }
  }

  const remove = async (row: GatewayRelayGateway) => {
    if (!(await confirm({ confirmLabel: r.remove, destructive: true, title: r.removeConfirm(row.label) }))) {
      return
    }

    setBusy(row.id)

    try {
      await window.hermesDesktop.connections?.remove(row.id)
      await load()
      await refreshConnectionsRegistry()
    } catch (error) {
      notifyError(error, r.removeFailed)
    } finally {
      setBusy(null)
    }
  }

  const test = async (row: GatewayRelayGateway) => {
    setBusy(row.id)

    try {
      const result = await window.hermesDesktop.connections?.test(row.id)
      setReach(current => ({ ...current, [row.id]: { error: result?.error ?? undefined, ok: Boolean(result?.ok) } }))
    } finally {
      setBusy(null)
    }
  }

  const revoke = async (grant: GatewayRelayGrant) => {
    if (!(await confirm({ confirmLabel: r.revoke, destructive: true, title: r.revokeConfirm(grant.hub_name) }))) {
      return
    }

    setBusy(grant.id)

    try {
      await relay.revokeGrant(grant.id)
      await load()
    } catch (error) {
      notifyError(error, r.revokeFailed)
    } finally {
      setBusy(null)
    }
  }

  return (
    <>
      <SettingsSection icon={Network} title={r.title}>
        <p className="mb-2 max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {r.intro}
        </p>
        {gateways?.length === 0 ? <p className="py-2 text-sm text-muted-foreground">{r.empty}</p> : null}
        {(gateways ?? []).map(row => {
          const reached = reach[row.id]

          return (
            <ListRow
              action={
                <div className="flex items-center gap-2">
                  {reached ? (
                    <Pill tone={reached.ok ? 'primary' : 'muted'}>
                      {reached.ok ? <Check className="size-3" /> : null}
                      {reached.ok ? r.reachable : r.unreachable(reached.error || '')}
                    </Pill>
                  ) : null}
                  <Button disabled={busy === row.id} onClick={() => void test(row)} variant="outline">
                    {r.test}
                  </Button>
                  <Button disabled={busy === row.id} onClick={() => void remove(row)} variant="outline">
                    {r.remove}
                  </Button>
                </div>
              }
              description={row.url}
              hint={row.scopes.length ? r.scopes(row.scopes.join(', ')) : r.coreOnly}
              key={row.id}
              title={row.label}
            />
          )
        })}
        <form className="mt-3 grid max-w-2xl gap-2" data-slot="relay-connect" onSubmit={event => void connect(event)}>
          <div className="text-[length:var(--conversation-text-font-size)] font-medium">{r.addTitle}</div>
          <label className="grid gap-1 text-xs text-muted-foreground">
            {r.urlLabel}
            <Input
              inputMode="url"
              name="relay-url"
              onChange={event => setUrl(event.target.value)}
              placeholder="https://mini.example.ts.net"
              required
              type="url"
              value={url}
            />
          </label>
          <label className="grid gap-1 text-xs text-muted-foreground">
            {r.labelLabel}
            <Input
              name="relay-label"
              onChange={event => setLabel(event.target.value)}
              placeholder="Mini"
              value={label}
            />
          </label>
          <div>
            <Button disabled={connecting || !url.trim()} type="submit">
              {connecting ? <Loader2 className="animate-spin" /> : null}
              {connecting ? r.connecting : r.connect}
            </Button>
          </div>
        </form>
      </SettingsSection>
      <SettingsSection icon={Network} title={r.grantsTitle}>
        <p className="mb-2 max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {r.grantsIntro}
        </p>
        {grants?.length === 0 ? <p className="py-2 text-sm text-muted-foreground">{r.grantsEmpty}</p> : null}
        {(grants ?? []).map(grant => (
          <ListRow
            action={
              <Button disabled={busy === grant.id} onClick={() => void revoke(grant)} variant="outline">
                {r.revoke}
              </Button>
            }
            description={r.grantFrom(grant.hub_origin, relativeTime(grant.created_at * 1000))}
            hint={grant.scopes.length ? r.scopes(grant.scopes.join(', ')) : r.coreOnly}
            key={grant.id}
            title={grant.hub_name}
          />
        ))}
      </SettingsSection>
    </>
  )
}
