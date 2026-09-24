import { beforeEach, describe, expect, it, vi } from 'vitest'

import { type WatchHolder, watchInPage } from './watch-in-page'

/** jsdom has no layout engine, so every rect is zero and nothing would be
 *  positioned. Give elements a plausible box the way the act-engine tests do. */
beforeEach(() => {
  // Never run the callback: the tracking loop re-arms itself every frame, so a
  // stub that invokes synchronously recurses until the stack gives out. The
  // first pass happens on the direct call, which is the one worth asserting on.
  vi.stubGlobal('requestAnimationFrame', () => 1)
  vi.stubGlobal('cancelAnimationFrame', () => {})

  Element.prototype.getBoundingClientRect = function () {
    return { bottom: 140, height: 40, left: 100, right: 300, toJSON: () => ({}), top: 100, width: 200, x: 100, y: 100 }
  }

  document.body.replaceChildren()
  // The host hangs off documentElement, so clearing body does not reach it.
  document.querySelector('hermes-watch')?.remove()
  delete (window as unknown as Record<string, unknown>).__hermesWatch
})

/** The overlay lives in a closed shadow root, so a test can only see the host. */
const host = () => document.querySelector('hermes-watch')

/** …except through the parts the engine parks on the window for its own reuse,
 *  which is the only way to inspect what the overlay actually drew. */
const drawn = () =>
  (window as unknown as { __hermesWatch: { parts: Record<string, HTMLElement> & { shadow: ShadowRoot } } })
    .__hermesWatch.parts

/** Everything the overlay drew for this action. The cursor is the only fixed
 *  layer — every box and pin is a mark that comes and goes. */
const marks = () => [...drawn().shadow.children].filter(child => child !== drawn().pointer)

describe('watchInPage', () => {
  it('mounts one host on documentElement and reuses it across stages', () => {
    const target = document.createElement('button')
    document.body.append(target)

    const holder: WatchHolder = { aimed: target }

    watchInPage(document, holder, 'aim')
    watchInPage(document, holder, 'strike')

    expect(document.querySelectorAll('hermes-watch')).toHaveLength(1)
    expect(host()?.parentElement).toBe(document.documentElement)
  })

  it('stays out of the page, out of the way, and out of the a11y tree', () => {
    const target = document.createElement('button')
    document.body.append(target)

    watchInPage(document, { aimed: target }, 'aim')

    const node = host() as HTMLElement

    // pointer-events keeps real clicks reaching the page; aria-hidden keeps a
    // screen reader from announcing decoration.
    expect(node.style.pointerEvents).toBe('none')
    expect(node.getAttribute('aria-hidden')).toBe('true')
    // The closed shadow root is what keeps the overlay out of the act engine's
    // own querySelectorAll inventory — there is nothing to exclude by hand.
    expect(node.shadowRoot).toBeNull()
  })

  it('does nothing when there is no target to draw on', () => {
    watchInPage(document, {}, 'aim')
    watchInPage(document, { aimed: document.createElement('div') }, 'strike')

    expect(host()).toBeNull()
  })

  // The inventory pass is the one moment the agent is reading the whole page
  // rather than aiming at one thing, so the overlay has to be able to hold many
  // marks at once — not just the single ring every other stage draws.
  it('sweep lights up every catalogued element and holds them well past the sweep', () => {
    vi.useFakeTimers()

    const found = [document.createElement('button'), document.createElement('a'), document.createElement('input')]
    document.body.append(...found)

    watchInPage(document, { nodes: found }, 'sweep')
    expect(marks()).toHaveLength(found.length)

    // The point of the dwell: the grid is still up long after the pass that drew
    // it finished, so there is something to actually look at. jsdom has no Web
    // Animations, so this exercises the timed fallback, which holds for the same
    // beat as the animated path.
    vi.advanceTimersByTime(600)
    expect(marks()).toHaveLength(found.length)

    // …and then drains scattered rather than blinking off as one block, which is
    // the whole reason each box carries its own offset. Stepping through the
    // exit must therefore see the count come down in more than one tick.
    const drain = new Set<number>()

    for (let t = 0; t < 7_000; t += 100) {
      vi.advanceTimersByTime(100)
      drain.add(marks().length)
    }

    expect(drain.size).toBeGreaterThan(2)
    expect(marks()).toHaveLength(0)

    vi.useRealTimers()
  })

  it('names each swept box after the element under it, not its ref', () => {
    const found = [document.createElement('button'), document.createElement('a'), document.createElement('input')]

    found[0].id = 'buy'
    found[1].className = 'nav wide'
    found[2].setAttribute('type', 'search')
    document.body.append(...found)

    // Every drawable box gets named, addressable or not — a ref number only ever
    // meant something for the subset that had one, and said nothing about what
    // the element actually is.
    watchInPage(document, { field: found, nodes: found.slice(0, 2) }, 'sweep')

    const { shadow } = drawn()
    const cells = [...shadow.children].slice(-found.length)

    expect(cells.map(cell => cell.textContent)).toEqual(['button#buy', 'a.nav', 'input[search]'])
  })

  // The case that made positions necessary: a list of links with no id, no
  // class and no type. Naming them by tag alone gives a whole page of boxes all
  // reading "a", which is what the ref numbers were replaced for.
  it('falls back to position so anonymous siblings are still told apart', () => {
    const row = document.createElement('td')

    row.className = 'subtext'
    row.innerHTML = '<a></a><a></a><a></a>'
    document.body.append(row)

    const found = [...row.children]

    watchInPage(document, { field: found }, 'sweep')

    const { shadow } = drawn()
    const cells = [...shadow.children].slice(-found.length)

    expect(cells.map(cell => cell.textContent)).toEqual(['a[1]', 'a[2]', 'a[3]'])
  })

  it('sweep draws nothing when the inventory came back empty', () => {
    watchInPage(document, { nodes: [] }, 'sweep')

    expect(host()).toBeNull()
  })

  // The field is swept after every action, and a sweep retires the lock. The
  // cursor must not go with it — a hand that blinks out a fraction of a second
  // after every single step reads as the agent having wandered off.
  it('keeps the cursor up when a sweep retires the lock', () => {
    const target = document.createElement('button')
    const found = [document.createElement('a')]
    document.body.append(target, ...found)

    watchInPage(document, { aimed: target }, 'aim')

    const { pointer } = drawn()
    const lock = marks()[0]

    expect(pointer.style.opacity).toBe('1')

    watchInPage(document, { field: found }, 'sweep')

    expect(lock.isConnected).toBe(false)
    expect(pointer.style.opacity).toBe('1')
  })

  // The chrome is memoized for the life of the page, so a style change to it
  // would otherwise be injected, run, and restyle nothing — the layers it
  // describes were built by the version before. Reads as "HMR isn't picking my
  // CSS up" in dev, and as a long-lived tab stuck on an old build in production.
  it('rebuilds its chrome when the injected source changes underneath it', () => {
    const target = document.createElement('button')
    document.body.append(target)

    const stamp = (value: number) => {
      ;(window as unknown as Record<string, unknown>).__hermesWatchTag = value
    }

    stamp(1)
    watchInPage(document, { aimed: target }, 'aim')

    const first = drawn().host

    stamp(1)
    watchInPage(document, { aimed: target }, 'aim')

    expect(drawn().host).toBe(first)

    stamp(2)
    watchInPage(document, { aimed: target }, 'aim')

    expect(drawn().host).not.toBe(first)
    // …and the old one goes with it, rather than stacking a second overlay.
    expect(document.querySelectorAll('hermes-watch')).toHaveLength(1)
  })

  it('clear releases the target so the tracking loop can stop', () => {
    const target = document.createElement('button')
    document.body.append(target)

    const holder: WatchHolder = { aimed: target }

    watchInPage(document, holder, 'aim')
    watchInPage(document, holder, 'clear')

    expect(holder.aimed).toBeNull()
  })

  // Same contract as the act engine: the pane injects `watchInPage.toString()`
  // into the guest page, where module scope does not exist. One free identifier
  // is a ReferenceError that Electron reports only as "Script failed to execute".
  it('runs after being stringified and eval’d with no module scope', () => {
    const target = document.createElement('button')
    document.body.append(target)

    const injected = new Function('return (' + watchInPage.toString() + ')')() as typeof watchInPage

    expect(() => injected(document, { aimed: target }, 'aim', 'Clicking Save')).not.toThrow()
    expect(host()).not.toBeNull()
  })
})
