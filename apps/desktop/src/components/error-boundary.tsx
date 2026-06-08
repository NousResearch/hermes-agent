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

const RECOVERABLE_ERROR_PATTERNS = [
  /tapClientLookup: Index \d+\s+out of bounds \(length:\s*\d+\)/i,
  /Cannot read properties of undefined \(reading 'type'\)/i,
  /Tried to unmount a fiber that is already unmounted/i
]

function isRecoverableDesktopRenderError(error: Error): boolean {
  return RECOVERABLE_ERROR_PATTERNS.some(pattern => pattern.test(error.message))
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null }
  private recoverTimer: null | ReturnType<typeof globalThis.setTimeout> = null

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    const tag = this.props.label ? `[error-boundary:${this.props.label}]` : '[error-boundary]'
    console.error(tag, error, info.componentStack)
    this.props.onError?.(error, info)

    if (this.props.label === 'root' && isRecoverableDesktopRenderError(error)) {
      console.warn(`${tag} auto-recovering from transient render error`, error.message)
      this.scheduleRecover()
    }
  }

  componentWillUnmount() {
    this.clearRecoverTimer()
  }

  reset = () => {
    this.clearRecoverTimer()
    this.setState({ error: null })
  }

  private clearRecoverTimer() {
    if (this.recoverTimer !== null) {
      window.clearTimeout(this.recoverTimer)
      this.recoverTimer = null
    }
  }

  private scheduleRecover() {
    this.clearRecoverTimer()
    this.recoverTimer = globalThis.setTimeout(() => {
      this.recoverTimer = null
      this.reset()
    }, 0)
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
