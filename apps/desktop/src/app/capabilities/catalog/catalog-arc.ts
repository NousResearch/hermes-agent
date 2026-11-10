/** One sweep for every card's hover arc.
 *
 *  Each hovered (or focused) card runs the arc rotation on the document timeline
 *  with `startTime = 0`, so every card sits at the same angle at the same
 *  instant: moving from one card to the next carries the sweep on instead of
 *  starting it over. The animation outlives the hover by the CSS fade, then is
 *  cancelled, so idle cards cost nothing. */

const CARD = '[data-catalog-card]'
const TURN_MS = 2400
const FADE_MS = 400

const arcs = new WeakMap<HTMLElement, { animation: Animation; timer: number }>()
let installs = 0

const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches

function start(card: HTMLElement) {
  const current = arcs.get(card)

  if (current) {
    clearTimeout(current.timer)
    current.timer = 0

    return
  }

  if (reduced()) {
    return
  }

  const animation = card.animate({ '--catalog-arc': ['0deg', '360deg'] }, {
    duration: TURN_MS,
    iterations: Infinity,
    pseudoElement: '::after'
  })

  animation.startTime = 0
  arcs.set(card, { animation, timer: 0 })
}

function stop(card: HTMLElement) {
  const current = arcs.get(card)

  if (!current || current.timer) {
    return
  }

  current.timer = window.setTimeout(() => {
    if (card.matches(':hover, :focus-within')) {
      current.timer = 0

      return
    }

    current.animation.cancel()
    arcs.delete(card)
  }, FADE_MS)
}

const cardOf = (node: EventTarget | null) => (node instanceof Element ? node.closest<HTMLElement>(CARD) : null)

function enter(event: Event) {
  const card = cardOf(event.target)

  if (card) {
    start(card)
  }
}

function leave(event: Event) {
  const card = cardOf(event.target)
  const next = cardOf((event as PointerEvent | FocusEvent).relatedTarget)

  if (card && card !== next) {
    stop(card)
  }
}

/** Installs the shared arc once per document; the returned cleanup drops this caller's hold. */
export function syncCatalogArcs() {
  if (installs++ === 0) {
    document.addEventListener('pointerover', enter)
    document.addEventListener('pointerout', leave)
    document.addEventListener('focusin', enter)
    document.addEventListener('focusout', leave)
  }

  return () => {
    if (--installs === 0) {
      document.removeEventListener('pointerover', enter)
      document.removeEventListener('pointerout', leave)
      document.removeEventListener('focusin', enter)
      document.removeEventListener('focusout', leave)
    }
  }
}
