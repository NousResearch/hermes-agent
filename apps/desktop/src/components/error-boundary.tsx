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

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    const tag = this.props.label ? `[error-boundary:${this.props.label}]` : '[error-boundary]'
    console.error(tag, error, info.componentStack)
    // Persist the crash + React component stack so a repeat is diagnosable
    // (the fallback's "Open logs" only surfaces main-process logs; renderer
    // crash stacks were previously lost to the devtools console). Best-effort;
    // never let logging throw inside the boundary.
    try {
      const record = {
        at: new Date().toISOString(),
        label: this.props.label ?? 'root',
        message: error.message,
        stack: error.stack ?? null,
        componentStack: info.componentStack ?? null
      }
      window.localStorage?.setItem('hermes:lastCrash', JSON.stringify(record))
    } catch {
      // ignore storage failures
    }
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
  const { t } = useI18n()

  // Clears corrupt persisted renderer state (which session is open, cached
  // view state) BEFORE reloading. Plain "Reload window" reloads into the same
  // poisoned state and re-crashes on mount — an unrecoverable loop. This is the
  // one-click escape: it wipes web storage + caches (chats/settings live on the
  // backend and in the separate auth partition, so they survive) then reloads.
  const resetAndRecover = () => {
    const done = () => window.location.reload()
    try {
      // Preserve the crash record across the flush — clearing localStorage would
      // otherwise delete the very diagnostic hermes:lastCrash that Reset & recover
      // is meant to survive (a dev opening devtools afterward would find it gone).
      let lastCrash: string | null = null
      try {
        lastCrash = window.localStorage?.getItem('hermes:lastCrash') ?? null
      } catch {
        lastCrash = null
      }
      window.localStorage?.clear()
      window.sessionStorage?.clear()
      if (lastCrash !== null) {
        try {
          window.localStorage?.setItem('hermes:lastCrash', lastCrash)
        } catch {
          // storage unavailable post-clear — best effort
        }
      }
      window.indexedDB?.databases?.().then(dbs => {
        for (const db of dbs) {
          if (db.name) window.indexedDB.deleteDatabase(db.name)
        }
      }).catch(() => undefined)
      if (window.caches?.keys) {
        void window.caches.keys().then(keys => Promise.all(keys.map(k => window.caches.delete(k)))).catch(() => undefined)
      }
    } catch {
      // ignore — still reload below
    }
    // Give the async deletes a beat to start, then reload regardless.
    window.setTimeout(done, 150)
  }

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
        <Button onClick={resetAndRecover} variant="text">
          {t.errors.resetAndRecover}
        </Button>
        <Button onClick={() => void window.hermesDesktop?.revealLogs()?.catch(() => undefined)} variant="text">
          {t.errors.openLogs}
        </Button>
      </ErrorState>
    </div>
  )
}
