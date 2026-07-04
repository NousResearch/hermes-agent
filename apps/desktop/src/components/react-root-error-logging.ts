import type { RootOptions } from 'react-dom/client'

// createRoot error hooks that keep minified production errors identifiable.
// React's defaults report only the wrapper (e.g. "Minified React error #520",
// the recoverable concurrent-render wrapper) — the real culprit rides in
// `error.cause` and the component stack, which the defaults drop. Logging via
// console.error lands in the console-message pipeline Electron main persists
// to desktop.log.
function logRootError(tag: string, error: unknown, errorInfo: { componentStack?: string }): void {
  const cause = error instanceof Error ? error.cause : undefined

  if (cause !== undefined) {
    console.error(tag, error, 'cause:', cause, errorInfo.componentStack ?? '')
  } else {
    console.error(tag, error, errorInfo.componentStack ?? '')
  }
}

export function reactRootErrorOptions(): Pick<
  RootOptions,
  'onCaughtError' | 'onRecoverableError' | 'onUncaughtError'
> {
  return {
    onRecoverableError: (error, errorInfo) => logRootError('[react:recoverable]', error, errorInfo),
    onCaughtError: (error, errorInfo) => logRootError('[react:caught]', error, errorInfo),
    onUncaughtError: (error, errorInfo) => logRootError('[react:uncaught]', error, errorInfo)
  }
}
