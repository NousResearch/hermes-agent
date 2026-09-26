/** Off for now: without it every card rests at its theme's gradient anchor. */
export const CATALOG_POINTER_ENABLED = false

const FALLOFF_PX = 480

/**
 * Aims every card's gradient at the pointer. The gradient center is the pointer
 * clamped to the card, so it is radial under the pointer and anchored to the
 * nearest edge otherwise; `--catalog-near` fades 1 → 0 over FALLOFF_PX of
 * distance from that edge point. Reads all rects, then writes, once per frame.
 */
export function trackCatalogPointer(root: HTMLElement) {
  let frame = 0
  let x = 0
  let y = 0

  const paint = () => {
    frame = 0
    const cards = [...root.querySelectorAll<HTMLElement>('[data-catalog-card]')]
    const rects = cards.map(card => card.getBoundingClientRect())

    cards.forEach((card, index) => {
      const { left, top, right, bottom, width, height } = rects[index]

      if (bottom < 0 || top > innerHeight) {
        return
      }

      const px = Math.min(Math.max(x, left), right)
      const py = Math.min(Math.max(y, top), bottom)
      card.style.setProperty('--catalog-x', `${((px - left) / width) * 100}%`)
      card.style.setProperty('--catalog-y', `${((py - top) / height) * 100}%`)
      card.style.setProperty('--catalog-near', Math.max(0, 1 - Math.hypot(x - px, y - py) / FALLOFF_PX).toFixed(3))
    })
  }

  const schedule = () => void (frame ||= requestAnimationFrame(paint))

  const move = (event: PointerEvent) => {
    x = event.clientX
    y = event.clientY
    schedule()
  }

  // Scroll re-aims only once the pointer has been seen; until then cards rest.
  const scroll = () => void (x || y ? schedule() : undefined)

  root.addEventListener('pointermove', move)
  root.addEventListener('scroll', scroll, true)

  return () => {
    cancelAnimationFrame(frame)
    root.removeEventListener('pointermove', move)
    root.removeEventListener('scroll', scroll, true)
  }
}
