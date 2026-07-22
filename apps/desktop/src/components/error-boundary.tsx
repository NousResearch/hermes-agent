import { Component, type ErrorInfo, type ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { ErrorState } from '@/components/ui/error-state'
import { useI18n } from '@/i18n'

export interface ErrorBoundaryFallbackProps {
  error: Error
  reset: () => void
}

interface ErrorBoundaryProps {
  children: ReactNode
  fallback?: (props: ErrorBoundaryFallbackProps) => ReactNode
  label?: string
  onError?: (error: Error, info: ErrorInfo) => void
  recoverClientLookup?: boolean
  resetKeys?: readonly unknown[]
}

interface ErrorBoundaryState {
  error: Error | null
}

const CLIENT_LOOKUP_ERROR = /^(?:tap|use)ClientLookup: Index \d+\s+out of bounds \(length:\s*\d+\)$/i
const AUTO_RECOVERY_DELAYS_MS = [250, 1_000, 3_000] as const
const STABLE_RECOVERY_WINDOW_MS = 30_000

const isTransientClientLookupError = (error: Error): boolean => CLIENT_LOOKUP_ERROR.test(error.message)

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null }
  private autoRecoveryCount = 0
  private autoRecoveryExhausted = false
  private autoRecoveryTimer: number | null = null
  private stableRecoveryTimer: number | null = null
  private pendingAutoRecoveryAttempt: number | null = null

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error }
  }

  componentDidMount() {
    // React StrictMode simulates an unmount/remount cycle while preserving the
    // component instance. Restore a pending retry that the simulated unmount
    // cleaned up without consuming another attempt from the episode budget.
    if (this.autoRecoveryTimer === null && this.pendingAutoRecoveryAttempt !== null) {
      this.scheduleAutoRecovery(this.pendingAutoRecoveryAttempt)
    }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    const tag = this.props.label ? `[error-boundary:${this.props.label}]` : '[error-boundary]'
    console.error(tag, error, info.componentStack)
    this.props.onError?.(error, info)

    this.clearStableRecoveryTimer()

    const recoveryEnabled = this.props.recoverClientLookup ?? this.props.label === 'root'

    if (!recoveryEnabled || !isTransientClientLookupError(error)) {
      return
    }

    const attempt = this.takeAutoRecoveryAttempt()

    if (attempt === null) {
      if (!this.autoRecoveryExhausted) {
        this.autoRecoveryExhausted = true
        console.warn(`${tag} client lookup recovery exhausted`)
      }

      return
    }

    console.warn(`${tag} client lookup recovery attempt ${attempt}/${AUTO_RECOVERY_DELAYS_MS.length}`)
    this.scheduleAutoRecovery(attempt)
  }

  componentDidUpdate(previousProps: ErrorBoundaryProps) {
    if (this.state.error && resetKeysChanged(previousProps.resetKeys, this.props.resetKeys)) {
      this.reset()
    }
  }

  componentWillUnmount() {
    this.clearAutoRecoveryTimer()
    this.clearStableRecoveryTimer()
  }

  reset = () => {
    this.clearAutoRecoveryTimer()
    this.clearStableRecoveryTimer()
    this.autoRecoveryCount = 0
    this.autoRecoveryExhausted = false
    this.pendingAutoRecoveryAttempt = null
    this.setState({ error: null })
  }

  private takeAutoRecoveryAttempt(): number | null {
    if (this.autoRecoveryCount >= AUTO_RECOVERY_DELAYS_MS.length) {
      return null
    }

    this.autoRecoveryCount += 1

    return this.autoRecoveryCount
  }

  private scheduleAutoRecovery(attempt: number) {
    this.clearAutoRecoveryTimer()
    this.pendingAutoRecoveryAttempt = attempt
    this.autoRecoveryTimer = window.setTimeout(this.autoRecover, AUTO_RECOVERY_DELAYS_MS[attempt - 1])
  }

  private autoRecover = () => {
    this.autoRecoveryTimer = null
    const attempt = this.pendingAutoRecoveryAttempt
    this.pendingAutoRecoveryAttempt = null

    this.setState({ error: null }, () => {
      if (this.state.error !== null || attempt === null) {
        return
      }

      const tag = this.props.label ? `[error-boundary:${this.props.label}]` : '[error-boundary]'
      console.info(`${tag} client lookup recovery recovered after attempt ${attempt}`)
      this.scheduleStableRecoveryReset()
    })
  }

  private scheduleStableRecoveryReset() {
    this.clearStableRecoveryTimer()
    this.stableRecoveryTimer = window.setTimeout(() => {
      this.stableRecoveryTimer = null

      if (this.state.error === null) {
        this.autoRecoveryCount = 0
        this.autoRecoveryExhausted = false
      }
    }, STABLE_RECOVERY_WINDOW_MS)
  }

  private clearAutoRecoveryTimer() {
    if (this.autoRecoveryTimer !== null) {
      window.clearTimeout(this.autoRecoveryTimer)
      this.autoRecoveryTimer = null
    }
  }

  private clearStableRecoveryTimer() {
    if (this.stableRecoveryTimer !== null) {
      window.clearTimeout(this.stableRecoveryTimer)
      this.stableRecoveryTimer = null
    }
  }

  render() {
    const { error } = this.state

    if (!error) {
      return this.props.children
    }

    if (this.props.fallback) {
      return this.props.fallback({ error, reset: this.reset })
    }

    return <RootErrorFallback error={error} reset={this.reset} />
  }
}

function RootErrorFallback({ error, reset }: ErrorBoundaryFallbackProps) {
  const { t } = useI18n()

  return (
    <div className="fixed inset-0 z-[1500] grid place-items-center bg-(--ui-chat-surface-background) p-6">
      <ErrorState
        className="w-full max-w-[28rem]"
        description={error.message || t.errors.boundaryDesc}
        title={t.errors.boundaryTitle}
      >
        <Button className="font-semibold" onClick={reset} size="lg">
          {t.common.retry}
        </Button>
        <Button onClick={() => window.location.reload()} variant="text">
          {t.errors.reloadWindow}
        </Button>
        <Button onClick={() => void window.hermesDesktop?.revealLogs()?.catch(() => undefined)} variant="text">
          {t.errors.openLogs}
        </Button>
      </ErrorState>
    </div>
  )
}

export function ScopedErrorFallback({ error, reset }: ErrorBoundaryFallbackProps) {
  const { t } = useI18n()

  return (
    <div className="grid h-full min-h-64 place-items-center p-6">
      <ErrorState
        className="w-full max-w-[28rem]"
        description={error.message || t.errors.boundaryDesc}
        title={t.errors.boundaryTitle}
      >
        <Button className="font-semibold" onClick={reset} size="lg">
          {t.common.retry}
        </Button>
        <Button onClick={() => void window.hermesDesktop?.revealLogs()?.catch(() => undefined)} variant="text">
          {t.errors.openLogs}
        </Button>
      </ErrorState>
    </div>
  )
}

function resetKeysChanged(previous: readonly unknown[] | undefined, next: readonly unknown[] | undefined): boolean {
  if (!previous || !next || previous.length !== next.length) {
    return Boolean(previous || next)
  }

  return previous.some((value, index) => !Object.is(value, next[index]))
}
