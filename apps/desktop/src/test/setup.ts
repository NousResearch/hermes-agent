import { beforeEach, vi } from 'vitest'

const makeStorage = (): Storage => {
  const values = new Map<string, string>()

  return {
    get length() {
      return values.size
    },
    clear: vi.fn(() => values.clear()),
    getItem: vi.fn((key: string) => values.get(String(key)) ?? null),
    key: vi.fn((index: number) => [...values.keys()][index] ?? null),
    removeItem: vi.fn((key: string) => {
      values.delete(String(key))
    }),
    setItem: vi.fn((key: string, value: string) => {
      values.set(String(key), String(value))
    })
  }
}

// jsdom SHIPS a localStorage, so the stub is only a fallback for bare
// environments (Greptile #274 P2: the guard means jsdom uses its own real
// storage — which is fine; isolation comes from the beforeEach clear below,
// which works for BOTH the real jsdom storage and the stub).
if (typeof window !== 'undefined' && !window.localStorage) {
  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    value: makeStorage()
  })
}

if (!globalThis.requestAnimationFrame) {
  globalThis.requestAnimationFrame = (callback: FrameRequestCallback) =>
    setTimeout(() => callback(performance.now()), 16) as unknown as number
}

if (!globalThis.cancelAnimationFrame) {
  globalThis.cancelAnimationFrame = (handle: number) => clearTimeout(handle)
}

if (!globalThis.CSS) {
  globalThis.CSS = {} as typeof CSS
}

if (!globalThis.CSS.escape) {
  globalThis.CSS.escape = (value: string) =>
    String(value).replace(/[\0-\x1f\x7f]|^-?\d|^-$|[^\w-]/g, char => {
      if (char === '\0') {
        return '\uFFFD'
      }

      return `\\${char.codePointAt(0)?.toString(16) ?? ''} `
    })
}

const canvasContext = {
  arc: vi.fn(),
  beginPath: vi.fn(),
  bezierCurveTo: vi.fn(),
  clearRect: vi.fn(),
  closePath: vi.fn(),
  createLinearGradient: vi.fn(() => ({ addColorStop: vi.fn() })),
  createRadialGradient: vi.fn(() => ({ addColorStop: vi.fn() })),
  drawImage: vi.fn(),
  fill: vi.fn(),
  fillRect: vi.fn(),
  fillText: vi.fn(),
  getImageData: vi.fn(() => ({ data: new Uint8ClampedArray([0, 0, 0, 255]) })),
  lineTo: vi.fn(),
  measureText: vi.fn((text: string) => ({ width: text.length * 8 })),
  moveTo: vi.fn(),
  putImageData: vi.fn(),
  quadraticCurveTo: vi.fn(),
  rect: vi.fn(),
  restore: vi.fn(),
  rotate: vi.fn(),
  save: vi.fn(),
  scale: vi.fn(),
  setLineDash: vi.fn(),
  setTransform: vi.fn(),
  stroke: vi.fn(),
  strokeRect: vi.fn(),
  translate: vi.fn()
}

if (typeof HTMLCanvasElement !== 'undefined') {
  HTMLCanvasElement.prototype.getContext = vi.fn(
    () => canvasContext
  ) as unknown as typeof HTMLCanvasElement.prototype.getContext
  HTMLCanvasElement.prototype.toDataURL = vi.fn(
    () => 'data:image/png;base64,'
  ) as typeof HTMLCanvasElement.prototype.toDataURL
}

// Per-test isolation (Greptile #274 P2s): clear storage state and reset the
// module-scope canvas mock call counts so no test observes a sibling's calls.
beforeEach(() => {
  if (typeof window !== 'undefined' && window.localStorage) {
    window.localStorage.clear()
  }
  if (typeof window !== 'undefined' && window.sessionStorage) {
    window.sessionStorage.clear()
  }
  for (const fn of Object.values(canvasContext)) {
    if (typeof fn === 'function' && 'mockClear' in fn) {
      ;(fn as ReturnType<typeof vi.fn>).mockClear()
    }
  }
})
