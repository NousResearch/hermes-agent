// Safari/WebKit lacks requestIdleCallback. The Electron build never notices,
// but the browser webapp crashes wherever it is called unguarded
// (capabilities catalog prefetch, mermaid render cache). Install a
// setTimeout-based fallback before any app module evaluates.
// Only installs where the API is missing, so jsdom tests keep their eager path
// unless they define it themselves.
const g = globalThis as typeof globalThis & Record<string, unknown>

if (typeof window !== 'undefined' && typeof g.requestIdleCallback !== 'function') {
  g.requestIdleCallback = ((cb: IdleRequestCallback) => {
    const start = Date.now()

    return setTimeout(
      () => cb({ didTimeout: false, timeRemaining: () => Math.max(0, 50 - (Date.now() - start)) }),
      1
    ) as unknown as number
  }) as typeof requestIdleCallback
  g.cancelIdleCallback = ((id: number) => clearTimeout(id)) as typeof cancelIdleCallback
}

export {}
