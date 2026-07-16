interface RendererTerminalSession {
  webContentsId?: number
}

interface RendererCrashRecoveryOptions {
  disposeTerminalSession: (id: string) => unknown
  maxReloads: number
  now: number
  onDisposeError: (id: string, error: unknown) => void
  onReload: () => void
  onSuppress: (count: number) => void
  reloadTimes: number[]
  reloadWindowMs: number
  rendererWebContentsId: number | null | undefined
  terminalSessions: ReadonlyMap<string, RendererTerminalSession>
}

export function recoverRendererAfterCrash({
  disposeTerminalSession,
  maxReloads,
  now,
  onDisposeError,
  onReload,
  onSuppress,
  reloadTimes,
  reloadWindowMs,
  rendererWebContentsId,
  terminalSessions
}: RendererCrashRecoveryOptions): number[] {
  if (rendererWebContentsId != null) {
    for (const [terminalId, sessionInfo] of [...terminalSessions.entries()]) {
      if (sessionInfo.webContentsId === rendererWebContentsId) {
        try {
          disposeTerminalSession(terminalId)
        } catch (error) {
          try {
            onDisposeError(terminalId, error)
          } catch {
            // Renderer recovery must not depend on error-reporting success.
          }
        }
      }
    }
  }

  const recentReloadTimes = reloadTimes.filter(time => now - time < reloadWindowMs)

  if (recentReloadTimes.length >= maxReloads) {
    onSuppress(recentReloadTimes.length)

    return recentReloadTimes
  }

  recentReloadTimes.push(now)
  onReload()

  return recentReloadTimes
}
