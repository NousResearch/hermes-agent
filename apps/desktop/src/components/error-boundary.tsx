import { Component, type ErrorInfo, type ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { AlertTriangle, RefreshCw } from '@/lib/icons'

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

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    const tag = this.props.label ? `[error-boundary:${this.props.label}]` : '[error-boundary]'
    console.error(tag, error, info.componentStack)
    this.props.onError?.(error, info)
  }

  reset = () => {
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
  return (
    <div className="fixed inset-0 z-[1500] flex items-center justify-center bg-(--ui-chat-surface-background) p-6">
      <div className="w-full max-w-[40rem] overflow-hidden rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-chat-bubble-background) shadow-sm">
        <div className="flex items-start gap-3 border-b border-(--ui-stroke-tertiary) px-5 py-4">
          <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-destructive/10 text-destructive">
            <AlertTriangle className="size-5" />
          </div>
          <div>
            <h2 className="text-[0.9375rem] font-semibold tracking-tight">Something broke in the interface</h2>
            <p className="mt-1 text-[0.8125rem] leading-5 text-(--ui-text-tertiary)">
              The view hit an unexpected error. Your chats and settings are safe - try again, or reload the window.
            </p>
          </div>
        </div>

        <div className="grid gap-4 p-5">
          <div className="rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 font-mono text-[0.7rem] leading-4 text-destructive">
            {error.message || String(error)}
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <Button onClick={reset}>
              <RefreshCw className="size-4" />
              Try again
            </Button>
            <Button onClick={() => window.location.reload()} variant="text">
              Reload window
            </Button>
            <Button
              className="ml-auto"
              onClick={() => void window.hermesDesktop?.revealLogs()?.catch(() => undefined)}
              variant="text"
            >
              Open logs
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
