import { type ReactNode, useEffect } from 'react'

interface StartupHealthReporterProps {
  children: ReactNode
  reportHealthy?: () => Promise<unknown> | unknown
  scheduleAfterPaint?: (callback: () => void) => () => void
}

export function scheduleAfterNextPaint(callback: () => void): () => void {
  let secondFrame: number | undefined

  const firstFrame = window.requestAnimationFrame(() => {
    secondFrame = window.requestAnimationFrame(callback)
  })

  return () => {
    window.cancelAnimationFrame(firstFrame)

    if (secondFrame !== undefined) {
      window.cancelAnimationFrame(secondFrame)
    }
  }
}

async function reportStartupHealthy(): Promise<void> {
  const signalStartupHealthy = window.hermesDesktop?.signalStartupHealthy

  if (typeof signalStartupHealthy === 'function') {
    await signalStartupHealthy()
  }
}

export function StartupHealthReporter({
  children,
  reportHealthy = reportStartupHealthy,
  scheduleAfterPaint = scheduleAfterNextPaint
}: StartupHealthReporterProps) {
  useEffect(() => {
    let active = true

    const cancel = scheduleAfterPaint(() => {
      if (active) {
        void Promise.resolve()
          .then(reportHealthy)
          .catch(() => undefined)
      }
    })

    return () => {
      active = false
      cancel()
    }
  }, [reportHealthy, scheduleAfterPaint])

  return children
}
