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

// Some assistant-ui lookup races escape the message-local boundary and reach
// the root. Retry only that exact transient error class, never arbitrary render
// failures, and cap retries so a persistent failure still exposes the fallback.
const ASSISTANT_UI_LOOKUP_ERROR = /(useClientLookup|tapClient(Lookup|Resource)).*out of bounds/
const MAX_AUTO_RECOVERIES = 3
const AUTO_RECOVERY_WINDOW_MS = 5_000

const isTransientAssistantUiLookupError = (error: Error): boolean => ASSISTANT_UI_LOOKUP_ERROR.test(error.message)

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null }
  private autoRecoveryCount = 0
  private autoRecoveryPending = false
  private autoRecoveryTimer: number | null = null
  private autoRecoveryWindowStart = 0

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error }
  }

  componentDidMount() {
    // StrictMode replays mount lifecycles in development. Its synthetic
    // componentWillUnmount clears the timer scheduled by componentDidCatch,
    // so restore the still-owned recovery on the matching remount.
    if (this.autoRecoveryPending && this.autoRecoveryTimer === null) {
      this.scheduleAutoRecovery()
    }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    const tag = this.props.label ? `[error-boundary:${this.props.label}]` : '[error-boundary]'
    console.error(tag, error, info.componentStack)
    this.props.onError?.(error, info)

    if (this.props.label === 'root' && isTransientAssistantUiLookupError(error) && this.takeAutoRecoveryAttempt()) {
      console.warn(`${tag} auto-recovering from assistant-ui lookup render race`, error.message)
      this.autoRecoveryPending = true
      this.scheduleAutoRecovery()
    }
  }

  componentWillUnmount() {
    this.clearAutoRecoveryTimer()
  }

  reset = () => {
    this.clearAutoRecoveryTimer()
    this.autoRecoveryPending = false
    this.autoRecoveryCount = 0
    this.autoRecoveryWindowStart = 0
    this.setState({ error: null })
  }

  private takeAutoRecoveryAttempt(): boolean {
    const now = Date.now()

    if (this.autoRecoveryCount === 0 || now - this.autoRecoveryWindowStart >= AUTO_RECOVERY_WINDOW_MS) {
      this.autoRecoveryWindowStart = now
      this.autoRecoveryCount = 0
    }

    this.autoRecoveryCount += 1

    return this.autoRecoveryCount <= MAX_AUTO_RECOVERIES
  }

  private clearAutoRecoveryTimer() {
    if (this.autoRecoveryTimer !== null) {
      window.clearTimeout(this.autoRecoveryTimer)
      this.autoRecoveryTimer = null
    }
  }

  private scheduleAutoRecovery() {
    this.clearAutoRecoveryTimer()
    this.autoRecoveryTimer = window.setTimeout(this.autoRecover, 0)
  }

  private autoRecover = () => {
    this.autoRecoveryTimer = null
    this.autoRecoveryPending = false
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

export function RootErrorBoundary({ children }: { children: ReactNode }) {
  return <ErrorBoundary label="root">{children}</ErrorBoundary>
}

function RootErrorFallback({ error, reset }: ErrorBoundaryFallbackProps) {
  const { t } = useI18n()

  return (
    <div className="fixed inset-0 z-(--z-crash) grid place-items-center bg-(--ui-chat-surface-background) p-6">
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
