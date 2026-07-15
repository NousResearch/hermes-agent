import { JsonRpcGatewayClient, type ConnectionState } from '@hermes/shared'
import { useEffect, useRef, useState } from 'react'

import { logout } from '~bridge/auth'
import { getGatewayWsUrl } from '~bridge/connection'
import { currentTarget } from '~bridge/state'

import { Brand, Button, Card, ErrorNote, Screen } from '../ui'

/**
 * Phase 0 exit criterion: prove the full networking stack end-to-end from the
 * device — mint a ws-ticket, open /api/ws, receive `gateway.ready`, and make one
 * JSON-RPC request round-trip (session.list). Phase 1 replaces this with the real
 * chat shell.
 */
export function ConnectedSmokeTest({ onLogout }: { onLogout: () => void }) {
  const target = currentTarget()
  const clientRef = useRef<JsonRpcGatewayClient | null>(null)
  const [state, setState] = useState<ConnectionState>('idle')
  const [ready, setReady] = useState(false)
  const [events, setEvents] = useState<string[]>([])
  const [sessionCount, setSessionCount] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let disposed = false
    const client = new JsonRpcGatewayClient()
    clientRef.current = client

    const offState = client.onState((s) => !disposed && setState(s))
    const offEvent = client.onEvent((ev) => {
      if (disposed) return
      setEvents((prev) => (prev.includes(ev.type) ? prev : [...prev, ev.type]))
      if (ev.type === 'gateway.ready') setReady(true)
    })

    ;(async () => {
      try {
        const wsUrl = await getGatewayWsUrl()
        await client.connect(wsUrl)
        // One request round-trip to prove the JSON-RPC channel both ways.
        try {
          const res = await client.request<{ sessions?: unknown[] }>('session.list', {})
          if (!disposed) setSessionCount(Array.isArray(res?.sessions) ? res.sessions.length : 0)
        } catch {
          /* session.list shape may vary; gateway.ready already proves the channel */
        }
      } catch (e) {
        if (!disposed) setError((e as Error).message)
      }
    })()

    return () => {
      disposed = true
      offState()
      offEvent()
      client.close()
    }
  }, [])

  async function doLogout() {
    clientRef.current?.close()
    if (target) await logout(target.baseUrl)
    onLogout()
  }

  return (
    <Screen>
      <Brand subtitle={target ? hostLabel(target.baseUrl) : 'Not connected'} />
      <Card>
        <Row label="WebSocket" value={state} ok={state === 'open'} />
        <Row label="gateway.ready" value={ready ? 'received' : 'waiting…'} ok={ready} />
        <Row
          label="session.list"
          value={sessionCount === null ? '—' : `${sessionCount} sessions`}
          ok={sessionCount !== null}
        />
        <Row label="events" value={events.length ? events.join(', ') : 'none yet'} ok={events.length > 0} />
        <ErrorNote>{error}</ErrorNote>
      </Card>
      <div className="rounded-xl border border-border bg-card/40 px-4 py-3 text-center text-xs text-muted-foreground">
        Phase 0 smoke test. If WebSocket is <b>open</b> and <b>gateway.ready</b> arrived, the bridge
        works end-to-end. The real chat UI lands in Phase 1.
      </div>
      <Button variant="ghost" onClick={doLogout}>
        Log out
      </Button>
    </Screen>
  )
}

function Row({ label, value, ok }: { label: string; value: string; ok: boolean }) {
  return (
    <div className="flex items-center justify-between gap-3 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className={ok ? 'font-medium text-primary' : 'text-foreground/70'}>
        {ok ? '✓ ' : ''}
        {value}
      </span>
    </div>
  )
}

function hostLabel(baseUrl: string): string {
  try {
    return new URL(baseUrl).host
  } catch {
    return baseUrl
  }
}
