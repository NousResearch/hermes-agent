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
import { type DisplayLease, type DisplayObserveResult, displayRequest, type DisplayStatus, resolveScreenWsUrl, VIEWER_ID } from './screen-connection'
import { $screenState, screenStateFor, setScreenLease, setScreenStatus } from './screen-state'
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
  const iHold = lease?.holder === 'human' && lease.viewer_id === VIEWER_ID

  const canvasHost = useRef<HTMLDivElement | null>(null)
  const rfb = useRef<RfbLike | null>(null)
  const socket = useRef<WebSocket | null>(null)
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
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [bot])

  useEffect(() => {
    void refresh()

    return host.onEvent('display.lease', (event: RpcEvent) => {
      const payload = event.payload as { profile_key?: string; lease?: DisplayLease } | undefined

      if (payload?.lease && payload.profile_key && payload.profile_key === status?.profile_key) {
        setScreenLease(bot, payload.lease)
      }
    })
  }, [bot, refresh, status?.profile_key])

  const detach = useCallback(() => {
    attachGeneration.current += 1
    rfb.current?.disconnect()
    rfb.current = null
    socket.current?.close()
    socket.current = null
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
      const observe = await displayRequest<DisplayObserveResult>(bot, 'display.observe', { viewer_id: VIEWER_ID })
      setScreenStatus(bot, observe)
      const url = await resolveScreenWsUrl(bot, observe.ticket)

      if (generation !== attachGeneration.current || !canvasHost.current) {
        return
      }

      const ws = new WebSocket(url)
      ws.binaryType = 'arraybuffer'
      socket.current = ws
      const client = new Rfb(canvasHost.current, ws, { shared: true })
      client.scaleViewport = true
      client.resizeSession = false
      client.focusOnClick = true
      client.background = 'transparent'
      client.qualityLevel = 7
      client.viewOnly = !(observe.lease.holder === 'human' && observe.lease.viewer_id === VIEWER_ID)
      client.addEventListener('connect', () => {
        if (generation === attachGeneration.current) {
          setConn('live')
        }
      })
      client.addEventListener('disconnect', event => {
        if (generation !== attachGeneration.current) {
          return
        }

        const reason = event.detail?.reason ?? ''

        if (reason.includes('control-taken')) {
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
  }, [bot, detach, refresh, t.screen.streamLost])

  // Visibility is not lifecycle: the stream stays attached while the pane is
  // hidden; only unmount tears it down (and hands control back server-side).
  useEffect(() => () => detach(), [detach])

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
      const result = await displayRequest<{ lease: DisplayLease }>(bot, 'display.lease.acquire', { viewer_id: VIEWER_ID })
      setScreenLease(bot, result.lease)

      if (conn !== 'live') {
        void attach()
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [attach, bot, conn])

  const handBack = useCallback(async () => {
    setBusy(true)

    try {
      const result = await displayRequest<{ lease: DisplayLease }>(bot, 'display.lease.release', { viewer_id: VIEWER_ID })
      setScreenLease(bot, result.lease)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [bot])

  if (status && !status.supported) {
    return <EmptyState description={t.screen.unsupportedBody} title={t.screen.unsupportedTitle} />
  }

  if (status && !status.installed) {
    return (
      <div className="grid min-h-48 place-items-center p-6 text-center">
        <div className="flex max-w-md flex-col gap-2">
          <div className="text-sm font-medium">{t.screen.notInstalledTitle}</div>
          <div className="text-xs text-muted-foreground">{t.screen.notInstalledBody}</div>
          {status.install_command ? (
            <code className="select-text break-all rounded bg-muted px-2 py-1 text-left text-xs">{status.install_command}</code>
          ) : null}
          <div className="text-xs text-muted-foreground">{t.screen.installHint}</div>
          <Button disabled={busy} onClick={() => void refresh()} size="sm" variant="secondary">
            <Codicon name="refresh" /> {t.screen.recheck}
          </Button>
        </div>
      </div>
    )
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
        ) : (
          <Button disabled={busy || conn === 'attaching'} onClick={() => void takeOver()} size="sm">
            <Codicon name="record-keys" /> {t.screen.takeOver}
          </Button>
        )}
        <Button disabled={conn === 'attaching'} onClick={() => void attach()} size="sm" title={t.screen.reconnect} variant="ghost">
          <Codicon name="refresh" />
        </Button>
      </div>
      <div className={iHold ? 'relative min-h-0 grow bg-black ring-2 ring-inset ring-red-500/70' : 'relative min-h-0 grow bg-black'}>
        {/* data-terminal: the same keyboard-ownership marker the terminal pane uses, so the app's
            type-to-focus / bare-key shortcuts never steal keystrokes meant for the remote screen. */}
        <div className="absolute inset-0" data-terminal="" ref={canvasHost} />
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
