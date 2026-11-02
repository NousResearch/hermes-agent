import { type RefObject, useRef, useState } from 'react'

import type { DesktopConnectionConfigInput, DesktopConnectionProbeResult } from '@/global'
import { useI18n } from '@/i18n'
import type { NotificationInput } from '@/store/notifications'

import type { RemoteSetupHost } from './use-remote-setup'

interface RemoteConnectionTestOptions {
  host: RemoteSetupHost
  url: string
  payload: DesktopConnectionConfigInput
  canTest: boolean
  authResolved: boolean
  targetSeq: RefObject<number>
  acceptProbe: (result: DesktopConnectionProbeResult) => void
  notify: (notice: NotificationInput) => void
}

export interface RemoteConnectionTest {
  testing: boolean
  error: string | null
  success: string | null
  // The last passing test covered exactly the payload on screen.
  passed: boolean
  invalidateTest: () => void
  clearTestOutcome: () => void
  reportError: (err: unknown, title?: string, kind?: 'error' | 'warning') => void
  test: () => Promise<void>
}

/** Test/feedback leg of the remote editor: one generation per test, fenced by target and payload. */
export function useRemoteConnectionTest(options: RemoteConnectionTestOptions): RemoteConnectionTest {
  const { t } = useI18n()
  const g = t.settings.gateway
  const { host, url, payload, canTest, authResolved, targetSeq, acceptProbe, notify } = options
  const [testing, setTesting] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [testedKey, setTestedKey] = useState<string | null>(null)
  const testSeq = useRef<number>(0)
  const payloadKey: string = JSON.stringify(payload)
  const currentKey = useRef<string>(payloadKey)
  currentKey.current = payloadKey

  const invalidateTest = (): void => {
    testSeq.current += 1
    setTesting(false)
    setTestedKey(null)
    setError(null)
    setSuccess(null)
  }

  // A retargeted probe drops the verdict but keeps the last error visible.
  const clearTestOutcome = (): void => {
    setTesting(false)
    setTestedKey(null)
    setSuccess(null)
  }

  const reportError = (err: unknown, title: string = g.testFailed, kind: 'error' | 'warning' = 'error'): void => {
    const message: string = err instanceof Error ? err.message : String(err || g.testFailed)
    setError(message)
    notify({ kind, title, message })
  }

  const reportSuccess = (message: string): void => {
    setSuccess(message)
    notify({ kind: 'success', title: g.reachableTitle, message })
  }

  const test = async (): Promise<void> => {
    if (!canTest) {
      return
    }

    const target: number = targetSeq.current
    const seq: number = ++testSeq.current

    const current = (): boolean =>
      target === targetSeq.current && seq === testSeq.current && payloadKey === currentKey.current

    setTesting(true)
    setError(null)
    setSuccess(null)
    setTestedKey(null)

    try {
      if (!authResolved) {
        const result = await window.hermesDesktop.probeConnectionConfig(url)

        if (current()) {
          acceptProbe(result)

          if (!result.reachable || result.authMode === 'unknown') {
            reportError(result.error || g.probeError)
          }
        }

        return
      }

      const result = await window.hermesDesktop.testConnectionConfig(payload)

      if (!current()) {
        return
      }

      if (result.ok === false || result.reachable === false) {
        throw new Error(result.error || g.testFailed)
      }

      reportSuccess(
        (host === 'first-run' ? t.install.testSucceeded : g.connectedTo)(
          result.baseUrl || url,
          result.version ?? undefined
        )
      )
      setTestedKey(payloadKey)
    } catch (err) {
      if (current()) {
        reportError(err)
      }
    } finally {
      if (current()) {
        setTesting(false)
      }
    }
  }

  return {
    testing,
    error,
    success,
    passed: testedKey === payloadKey,
    invalidateTest,
    clearTestOutcome,
    reportError,
    test
  }
}
