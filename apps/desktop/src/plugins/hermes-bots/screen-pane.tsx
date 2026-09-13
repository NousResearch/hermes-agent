/**
 * Bot Screen pane — live view of a bot's headless desktop with Take over / Hand back.
 *
 * State authority: the BACKEND owns runtime + lease (`display.status`, pushed
 * as `display.lease` events); this pane paints a cache of it. The RFB stream
 * is a sibling WebSocket handed to noVNC's RFB; `viewOnly` here is UX only —
 * the gateway drops input from anyone but the lease holder.
 *
 * Every attach spends a single-use ticket, so a lease flip that the bridge
 * answers with close 4000 (`control-taken`) simply re-attaches in watch mode.
 */

import { Button, Codicon, EmptyState, GlyphSpinner, host, useValue } from '@hermes/plugin-sdk'
import type { RpcEvent } from '@hermes/plugin-sdk'
import { useCallback, useEffect, useRef, useState } from 'react'

import { useBots } from './i18n'
import {
  type DisplayLease,
  type DisplayObserveResult,
  displayRequest,
  type DisplayStatus,
  isDisplayUnavailable,
  isEventForBotScreen,
  leaseHeldBy,
  rememberScreenViewerCapability,
  resolveScreenWsUrl,
  retainBotScreen,
  storedScreenViewerCapability,
  viewerHash
} from './screen-connection'
import { ScreenInstallCard } from './screen-install'
import { $screenState, screenStateFor, setScreenLease, setScreenStatus, setScreenUnavailable, setScreenViewer } from './screen-state'
import type { RosterRow } from './types'

type RfbLike = {
  viewOnly: boolean
  scaleViewport: boolean
  resizeSession: boolean
  focusOnClick: boolean
  background: string
  qualityLevel: number
  addEventListener: (type: string, handler: (event: { detail?: { clean?: boolean; reason?: string } }) => void) => void
  disconnect: () => void
  focus: () => void
}

type ConnState = 'idle' | 'attaching' | 'live' | 'control-taken' | 'error'

/** Bridge close code when another viewer took the lease (mirrors tui_gateway display bridge). */
const CLOSE_CONTROL_TAKEN = 4000

async function loadRfb(): Promise<new (target: HTMLElement, socket: WebSocket, options?: Record<string, unknown>) => RfbLike> {
  const mod = (await import('@novnc/novnc')) as unknown as { default: new (...args: never[]) => RfbLike }

  return mod.default as unknown as new (target: HTMLElement, socket: WebSocket, options?: Record<string, unknown>) => RfbLike
}

export function BotScreenPane({ bot }: { bot: RosterRow }) {
  const t = useBots()
  const screen = useValue($screenState)
  const state = screenStateFor(screen, bot)
  const status = state?.status ?? null
  const lease = state?.lease ?? status?.lease ?? null
  // The server mints this window's viewer id per attach (`display.observe`); the lease
  // names its holder by hash, so a reload can never inherit a stale holder's authority.
  const viewer = state?.viewer ?? null
  const iHold = leaseHeldBy(lease, viewer)

  const canvasHost = useRef<HTMLDivElement | null>(null)
  const rfb = useRef<RfbLike | null>(null)
  const socket = useRef<WebSocket | null>(null)
  // Pins the bot's pooled gateway socket for the attach lifetime so display.lease
  // events keep arriving for an inactive registry-routed bot.
  const retention = useRef<(() => void) | null>(null)
  const [conn, setConn] = useState<ConnState>('idle')
  const [error, setError] = useState<null | string>(null)
  const [busy, setBusy] = useState(false)
  const attachGeneration = useRef(0)

  const refresh = useCallback(async () => {
    try {
      const next = await displayRequest<DisplayStatus>(bot, 'display.status')
      setScreenStatus(bot, next)
      setError(null)
    } catch (err) {
      if (isDisplayUnavailable(err)) {
        setScreenUnavailable(bot)
      }

      setError(err instanceof Error ? err.message : String(err))
    }
  }, [bot])

  useEffect(() => {
    void refresh()

    return host.onEvent('display.lease', (event: RpcEvent) => {
      const payload = event.payload as { lease?: DisplayLease } | undefined

      if (payload?.lease && isEventForBotScreen(bot, event, status?.profile_key)) {
        setScreenLease(bot, payload.lease)
      }
    })
  }, [bot, refresh, status?.profile_key])

  const detach = useCallback((handBack = false) => {
    attachGeneration.current += 1

    // noVNC closes without a status. Intentional pane closure must send 1000
    // first; reconnect teardown must keep the human lease instead.
    if (handBack) {
      socket.current?.close(1000)
    }

    rfb.current?.disconnect()
    rfb.current = null
    socket.current?.close()
    socket.current = null
    retention.current?.()
    retention.current = null
  }, [])

  const attach = useCallback(async () => {
    if (!canvasHost.current) {
      return
    }

    detach()
    const generation = attachGeneration.current
    setConn('attaching')
    setError(null)

    try {
      // Load the client BEFORE dialing: noVNC's Websock installs its own `onopen`, so a socket that
      // opened while the dynamic import was still in flight never hands it the open event.
      const Rfb = await loadRfb()
      const retain = await retainBotScreen(bot)

      if (generation !== attachGeneration.current) {
        retain()

        return
      }

      retention.current = retain
      const recovery = viewer ?? storedScreenViewerCapability(bot)

      const observe = await displayRequest<DisplayObserveResult>(
        bot,
        'display.observe',
        recovery ? { viewer_id: recovery.id, resume_token: recovery.resumeToken } : {}
      )

      const minted = {
        id: observe.viewer_id,
        hash: await viewerHash(observe.viewer_id),
        resumeToken: observe.resume_token
      }

      rememberScreenViewerCapability(bot, minted)
      setScreenStatus(bot, observe)
      const url = await resolveScreenWsUrl(bot, observe.ticket)

      if (generation !== attachGeneration.current || !canvasHost.current) {
        return
      }

      setScreenViewer(bot, minted)
      const ws = new WebSocket(url)
      ws.binaryType = 'arraybuffer'
      socket.current = ws
      // noVNC 1.7's `disconnect` detail carries only {clean}; the bridge's verdict lives in
      // the raw close frame (4000 = control-taken). Listen here, before RFB installs its
      // own `onclose`, so the code is known by the time the disconnect event fires.
      let closeCode = 0
      ws.addEventListener('close', event => {
        closeCode = event.code
      })
      const client = new Rfb(canvasHost.current, ws, { shared: true })
      client.scaleViewport = true
      client.resizeSession = false
      client.focusOnClick = true
      client.background = 'transparent'
      client.qualityLevel = 7
      client.viewOnly = !leaseHeldBy(observe.lease, minted)
      client.addEventListener('connect', () => {
        if (generation === attachGeneration.current) {
          setConn('live')
        }
      })
      client.addEventListener('disconnect', event => {
        // noVNC logs "Tried changing state of a disconnected RFB object" if we later call
        // disconnect() on a client that already closed itself (eviction, stream loss).
        if (rfb.current === client) {
          rfb.current = null
        }

        if (generation !== attachGeneration.current) {
          return
        }

        const reason = event.detail?.reason ?? ''

        if (closeCode === CLOSE_CONTROL_TAKEN || reason.includes('control-taken')) {
          setConn('control-taken')
        } else if (event.detail?.clean) {
          setConn('idle')
        } else {
          setConn('error')
          setError(reason || t.screen.streamLost)
        }

        void refresh()
      })
      rfb.current = client
    } catch (err) {
      if (generation === attachGeneration.current) {
        setConn('error')
        setError(err instanceof Error ? err.message : String(err))
      }
    }
  }, [bot, detach, refresh, t.screen.streamLost, viewer])

  // Visibility is not lifecycle: the stream stays attached while the pane is
  // hidden; only unmount tears it down (and hands control back server-side).
  useEffect(() => () => detach(true), [detach])

  useEffect(() => {
    if (status?.running && conn === 'idle') {
      void attach()
    }
  }, [attach, conn, status?.running])

  useEffect(() => {
    if (rfb.current) {
      rfb.current.viewOnly = !iHold

      if (iHold) {
        rfb.current.focus()
      }
    }
  }, [iHold])

  const start = useCallback(async () => {
    setBusy(true)

    try {
      const next = await displayRequest<DisplayStatus>(bot, 'display.start')
      setScreenStatus(bot, next)
      setConn('idle')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [bot])

  const takeOver = useCallback(async () => {
    setBusy(true)

    try {
      const result = await displayRequest<{ lease: DisplayLease }>(bot, 'display.lease.acquire', {
        viewer_id: viewer?.id
      })

      setScreenLease(bot, result.lease)

      if (conn !== 'live') {
        void attach()
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [attach, bot, conn, viewer?.id])

  const handBack = useCallback(async () => {
    setBusy(true)

    try {
      const result = await displayRequest<{ lease: DisplayLease }>(bot, 'display.lease.release', {
        viewer_id: viewer?.id
      })

      setScreenLease(bot, result.lease)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [bot, viewer?.id])

  if (state?.unavailable) {
    return <EmptyState description={t.screen.portalUnavailable} title={t.screen.unavailableTitle} />
  }

  if (status && !status.supported) {
    return <EmptyState description={t.screen.unsupportedBody} title={t.screen.unsupportedTitle} />
  }

  if (status && !status.installed) {
    return <ScreenInstallCard bot={bot} onInstalled={next => setScreenStatus(bot, next)} status={status} />
  }

  if (status && !status.running) {
    return (
      <div className="grid min-h-48 place-items-center p-6 text-center">
        <div className="flex flex-col items-center gap-2">
          <div className="text-sm font-medium">{t.screen.stoppedTitle}</div>
          <div className="text-xs text-muted-foreground">{t.screen.stoppedBody}</div>
          <Button disabled={busy} onClick={() => void start()} size="sm">
            {busy ? <GlyphSpinner /> : <Codicon name="play" />}
            {t.screen.start}
          </Button>
          {error ? <div className="text-xs text-red-500">{error}</div> : null}
        </div>
      </div>
    )
  }

  const humanOther = lease?.holder === 'human' && !iHold

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center gap-2 border-b px-3 py-1.5 text-xs">
        <Codicon name="device-desktop" />
        <span className="font-medium">{t.screen.title}</span>
        {status?.display ? <span className="text-muted-foreground">{status.display} · {status.geometry}</span> : null}
        <span className="grow" />
        {lease?.pending_handoff ? (
          <span className="rounded bg-amber-500/15 px-2 py-0.5 text-amber-600 dark:text-amber-400" title={lease.pending_handoff}>
            <Codicon name="bell" /> {t.screen.handoffRequested}
          </span>
        ) : lease?.holder === 'human' && lease.reason ? (
          // The agent's ask stays readable WHILE the human acts, not only before Take over.
          <span className="max-w-[40%] truncate rounded bg-amber-500/15 px-2 py-0.5 text-amber-600 dark:text-amber-400" title={lease.reason}>
            <Codicon name="bell" /> {lease.reason}
          </span>
        ) : null}
        {iHold ? (
          <span className="rounded bg-red-500/15 px-2 py-0.5 font-medium text-red-600 dark:text-red-400">{t.screen.youControl}</span>
        ) : humanOther ? (
          <span className="rounded bg-muted px-2 py-0.5 text-muted-foreground">{t.screen.otherControls}</span>
        ) : (
          <span className="rounded bg-muted px-2 py-0.5 text-muted-foreground">{t.screen.agentControls}</span>
        )}
        {iHold ? (
          <Button disabled={busy} onClick={() => void handBack()} size="sm" variant="secondary">
            <Codicon name="debug-continue" /> {t.screen.handBack}
          </Button>
        ) : humanOther ? null : (
          <>
            <Button disabled={busy || conn === 'attaching'} onClick={() => void takeOver()} size="sm">
              <Codicon name="record-keys" /> {t.screen.takeOver}
            </Button>
          </>
        )}
        <Button disabled={conn === 'attaching'} onClick={() => void attach()} size="sm" title={t.screen.reconnect} variant="ghost">
          <Codicon name="refresh" />
        </Button>
      </div>
      <div className={iHold ? 'relative min-h-0 grow bg-black ring-2 ring-inset ring-red-500/70' : 'relative min-h-0 grow bg-black'}>
        {/* data-terminal: the same keyboard-ownership marker the terminal pane uses, so the app's
            type-to-focus / bare-key shortcuts never steal keystrokes meant for the remote screen.
            data-remote-screen: tells the ⌘W close-tab router this is NOT a local terminal tab —
            the chord belongs to the remote desktop, nothing local should close. */}
        <div className="absolute inset-0" data-remote-screen="" data-terminal="" ref={canvasHost} />
        {conn === 'attaching' ? (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-xs text-white/70">
            <GlyphSpinner /> {t.screen.attaching}
          </div>
        ) : null}
        {conn === 'control-taken' ? (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/60 text-xs text-white">{t.screen.controlTaken}</div>
        ) : null}
        {conn === 'error' && error ? (
          <div className="absolute inset-x-0 bottom-0 bg-red-950/80 px-3 py-1.5 text-xs text-red-200">{error}</div>
        ) : null}
      </div>
    </div>
  )
}
