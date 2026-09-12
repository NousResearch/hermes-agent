/**
 * Bot Screen install card — installs the TigerVNC + Xfce packages on the bot's
 * gateway host from inside Hermes Desktop.
 *
 * `display.install` starts the distro package command on the host; sudo, when
 * needed, arrives as the same masked password card the terminal tool uses
 * (`display.install.sudo.request`), so the password never touches this pane.
 * Output streams back as `display.install.log`; `display.install.done` carries
 * a fresh status the caller uses to flip the pane to "Start screen".
 */

import { Button, Codicon, GlyphSpinner, host } from '@hermes/plugin-sdk'
import type { RpcEvent } from '@hermes/plugin-sdk'
import { useCallback, useEffect, useRef, useState } from 'react'

import { useBots } from './i18n'
import { displayRequest, type DisplayStatus } from './screen-connection'
import type { RosterRow } from './types'

const LOG_KEEP = 200

interface ScreenInstallCardProps {
  bot: RosterRow
  status: DisplayStatus
  onInstalled: (status: DisplayStatus) => void
}

export function ScreenInstallCard({ bot, status, onInstalled }: ScreenInstallCardProps) {
  const t = useBots()
  const [phase, setPhase] = useState<'idle' | 'running' | 'failed'>('idle')
  const [log, setLog] = useState<string[]>([])
  const [error, setError] = useState<null | string>(null)
  const logEnd = useRef<HTMLDivElement>(null)

  useEffect(() => {
    logEnd.current?.scrollIntoView({ block: 'end' })
  }, [log])

  useEffect(() => {
    const offLog = host.onEvent('display.install.log', (event: RpcEvent) => {
      const payload = event.payload as { profile_key?: string; line?: string } | undefined

      if (payload?.profile_key === status.profile_key && typeof payload.line === 'string') {
        const line = payload.line
        setLog(prev => (prev.length >= LOG_KEEP ? [...prev.slice(1), line] : [...prev, line]))
      }
    })

    const offDone = host.onEvent('display.install.done', (event: RpcEvent) => {
      const payload = event.payload as { profile_key?: string; code?: number; status?: DisplayStatus } | undefined

      if (payload?.profile_key !== status.profile_key) {
        return
      }

      if (payload.code === 0 && payload.status?.installed) {
        setPhase('idle')
        onInstalled(payload.status)
      } else {
        setPhase('failed')
        setError(payload.code === -1 ? t.screen.installCancelled : t.screen.installFailed)
      }
    })

    return () => {
      offLog()
      offDone()
    }
  }, [onInstalled, status.profile_key, t.screen.installCancelled, t.screen.installFailed])

  const install = useCallback(async () => {
    setPhase('running')
    setLog([])
    setError(null)

    try {
      await displayRequest(bot, 'display.install')
    } catch (err) {
      setPhase('failed')
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [bot])

  return (
    <div className="grid min-h-48 place-items-center p-6 text-center">
      <div className="flex w-full max-w-lg flex-col gap-2">
        <div className="text-sm font-medium">{t.screen.notInstalledTitle}</div>
        <div className="text-xs text-muted-foreground">{t.screen.notInstalledBody}</div>
        {status.install_command ? (
          <code className="select-text break-all rounded bg-muted px-2 py-1 text-left text-xs">{status.install_command}</code>
        ) : (
          <div className="text-xs text-muted-foreground">{t.screen.noPackageManager}</div>
        )}
        {status.install_command ? (
          <Button disabled={phase === 'running'} onClick={() => void install()} size="sm">
            {phase === 'running' ? <GlyphSpinner /> : <Codicon name="cloud-download" />}
            {phase === 'running' ? t.screen.installing : t.screen.install}
          </Button>
        ) : null}
        {log.length > 0 ? (
          <pre className="max-h-48 overflow-auto rounded bg-black/80 p-2 text-left font-mono text-[0.65rem] leading-tight text-white/85">
            {log.join('\n')}
            <div ref={logEnd} />
          </pre>
        ) : null}
        {error ? <div className="text-xs text-red-500">{error}</div> : null}
        <div className="text-xs text-muted-foreground">{t.screen.installHint}</div>
      </div>
    </div>
  )
}
