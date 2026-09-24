import { Component, type ErrorInfo, type ReactNode } from 'react'

/**
 * Error boundary for the transparent `?win=` overlay windows.
 *
 * Same job as `ErrorBoundary`, none of its weight. That one renders the app's
 * rich fallback, so importing it pulls `Button`, `ErrorState` and the i18n
 * provider — the app shell these windows exist to avoid, and which their own
 * docs claim they don't load. Under the dev server that chain WAS most of the
 * intro cinematic's module graph, queued ahead of the surface trying to paint.
 *
 * An overlay also has nowhere to put a fallback and nobody to click Retry: it
 * is a see-through window over the user's desktop. So a failure renders
 * nothing and the window's own deadman takes it off screen, which is the
 * outcome a stuck overlay needs anyway.
 */
export class OverlayErrorBoundary extends Component<{ children: ReactNode; label: string }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[overlay:${this.props.label}]`, error, info.componentStack)

    // Persist to desktop.log via Electron (#79428): console.error only reaches
    // the main process for windows with a console hook, is minified, and loses
    // the component stack.
    try {
      window.hermesDesktop?.reportRendererError?.({
        boundary: this.props.label,
        componentStack: info.componentStack ?? '',
        label: new URLSearchParams(window.location.search).get('win') ?? 'overlay',
        message: error.message
      })
    } catch {
      // Logging must never take the boundary down with it.
    }
  }

  render() {
    return this.state.failed ? null : this.props.children
  }
}
