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
}

interface ErrorBoundaryState {
  error: Error | null
}

// assistant-ui can momentarily render a stale message index against a thread
// that just shrank (session switch / teardown), throwing a render-race error
// that latches the WHOLE app on the root "Reload window" screen. These throws
// clear themselves on the next render against fresh state, so the root boundary
// recovers itself once the storm settles instead of stranding the user.
const RECOVERABLE_ERROR_PATTERNS = [
  /tapClientLookup: Index \d+\s+out of bounds \(length:\s*\d+\)/i,
  /Cannot read properties of undefined \(reading 'type'\)/i,
  /Tried to unmount a fiber that is already unmounted/i
]

const isRecoverableDesktopRenderError = (error: Error): boolean =>
  RECOVERABLE_ERROR_PATTERNS.some(pattern => pattern.test(error.message))

// Bound auto-recovery so a *persistent* (non-transient) error can't spin the
// boundary in a reset -> throw -> reset loop: at most MAX_RECOVERIES attempts
// inside RECOVERY_WINDOW_MS, after which the fallback is left up for the user.
const MAX_RECOVERIES = 3
const RECOVERY_WINDOW_MS = 5_000

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null }
  private recoverTimer: null | number = null
  private recoverCount = 0
  private recoverWindowStart = 0

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    const tag = this.props.label ? `[error-boundary:${this.props.label}]` : '[error-boundary]'
    console.error(tag, error, info.componentStack)
    this.props.onError?.(error, info)

    if (this.props.label === 'root' && isRecoverableDesktopRenderError(error) && this.canRecover()) {
      console.warn(`${tag} auto-recovering from transient render error`, error.message)
      this.scheduleRecover()
    }
  }

  componentWillUnmount() {
    this.clearRecoverTimer()
  }

  reset = () => {
    this.clearRecoverTimer()
    // A manual retry (button) starts a clean recovery budget.
    this.recoverCount = 0
    this.recoverWindowStart = 0
    this.setState({ error: null })
  }

  // True while the boundary still has recovery budget. Each storm gets a fresh
  // window; auto-recovery (autoReset) deliberately does NOT reset the count, so
  // a tight reset -> throw loop is capped at MAX_RECOVERIES and then falls back.
  private canRecover(): boolean {
    const now = Date.now()

    if (now - this.recoverWindowStart > RECOVERY_WINDOW_MS) {
      this.recoverWindowStart = now
      this.recoverCount = 0
    }

    this.recoverCount += 1

    return this.recoverCount <= MAX_RECOVERIES
  }

  private clearRecoverTimer() {
    if (this.recoverTimer !== null) {
      window.clearTimeout(this.recoverTimer)
      this.recoverTimer = null
    }
  }

  private scheduleRecover() {
    this.clearRecoverTimer()
    this.recoverTimer = window.setTimeout(this.autoReset, 0)
  }

  private autoReset = () => {
    this.recoverTimer = null
    this.setState({ error: null })
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
