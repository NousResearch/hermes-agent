// Whether the Webapp runs in a Chromium app window (Install / "Open as
// window", `--app=`), where the page gets Ctrl+N/T/W/Tab before the browser.
//
// Chromium only: `BrowserCommandController::IsReservedCommandOrKey` returns
// false for TYPE_APP windows. Other engines' web-app windows (Firefox taskbar
// tabs) can report `standalone` while the browser still acts on those chords.
//
// The window kind changes live: "Open in Chrome" moves an app window into a
// tab without reloading the page, so callers follow `onBrowserWindowKindChange`.

import { arraysEqual } from '@/lib/storage'

const APP_WINDOW_QUERIES = ['standalone', 'minimal-ui', 'window-controls-overlay'].map(
  mode => `(display-mode: ${mode})`
)

interface UserAgentBrands {
  brands?: readonly { brand: string }[]
}

function isChromium(): boolean {
  // userAgentData is secure-context only (absent on a plain-http LAN origin),
  // and absent in Firefox and Safari; their UA strings carry no Chrome/ token.
  const brands = (navigator as Navigator & { userAgentData?: UserAgentBrands }).userAgentData?.brands

  return brands ? brands.some(entry => entry.brand === 'Chromium') : /Chrom(e|ium)\//.test(navigator.userAgent)
}

// Node test environments import the keybind store with no window at all.
const canWatch = () => typeof window !== 'undefined' && typeof window.matchMedia === 'function'

export function inChromiumAppWindow(): boolean {
  return (
    canWatch() &&
    isChromium() &&
    APP_WINDOW_QUERIES.some(query => window.matchMedia(query).matches)
  )
}

/** Runs `listener` whenever the window's display mode changes. */
export function onBrowserWindowKindChange(listener: () => void): () => void {
  if (!canWatch()) {
    return () => undefined
  }

  const lists = APP_WINDOW_QUERIES.map(query => window.matchMedia(query))

  for (const list of lists) {
    list.addEventListener('change', listener)
  }

  return () => {
    for (const list of lists) {
      list.removeEventListener('change', listener)
    }
  }
}

type Bindings = Record<string, string[]>

/** Moves every binding still on its old shipped default onto the new one; a
 *  user's own rebind (anything else, including a cleared binding) stays, and
 *  so does any id `keep` claims. Value equality alone cannot tell a rebind
 *  from a default: a rebind to Alt+W in an app window IS the tab default. */
export function rebaseShippedDefaults(
  bindings: Bindings,
  previous: Bindings,
  next: Bindings,
  keep: (id: string, combos: string[]) => boolean = () => false
): Bindings {
  const out = { ...bindings }

  for (const [id, combos] of Object.entries(next)) {
    if (!arraysEqual(out[id] ?? [], previous[id] ?? []) || arraysEqual(out[id] ?? [], combos) || keep(id, combos)) {
      continue
    }

    out[id] = [...combos]
  }

  return out
}

/** Keeps `store`'s defaults on the current window kind. */
export function followBrowserWindowKind(
  store: { get(): Bindings; set(value: Bindings): void },
  defaults: () => Bindings,
  keep?: (id: string, combos: string[]) => boolean
): () => void {
  let shipped = defaults()

  return onBrowserWindowKindChange(() => {
    const next = defaults()
    const current = store.get()
    const rebased = rebaseShippedDefaults(current, shipped, next, keep)

    shipped = next

    if (Object.keys(rebased).some(id => rebased[id] !== current[id])) {
      store.set(rebased)
    }
  })
}
