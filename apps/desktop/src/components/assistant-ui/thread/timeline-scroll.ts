const cssEscape = (value: string): string => {
  if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') {
    return CSS.escape(value)
  }

  return value.replace(/[^a-zA-Z0-9_:-]/g, ch => `\\${ch}`)
}

/** The non-sticky outer containment box does not force offscreen descendants to render. */
export const timelineTarget = (node: HTMLElement): HTMLElement =>
  node.closest<HTMLElement>('[data-slot="aui_message-group"]') ??
  node.closest<HTMLElement>('[data-slot="aui_turn-pair"]') ??
  node

/** Re-measure the same identity while content-visibility and the runtime settle. */
export function scrollTimelineTarget(viewport: HTMLElement, id: string, signal: AbortSignal): Promise<boolean> {
  return new Promise(resolve => {
    let frame = 0
    let stable = 0
    let expanded: HTMLElement | null = null
    let originalVisibility = ''
    const start = viewport.scrollTop
    const began = performance.now()
    const duration = matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 170

    const finish = (success: boolean) => {
      cancelAnimationFrame(frame)
      signal.removeEventListener('abort', abort)

      if (expanded) {
        expanded.style.contentVisibility = originalVisibility
      }

      resolve(success)
    }

    const abort = () => finish(false)

    const step = (now: number) => {
      if (signal.aborted) {
        return finish(false)
      }

      const node = viewport.querySelector<HTMLElement>(`[data-message-id="${CSS.escape(id)}"]`)

      if (!node) {
        return finish(false)
      }

      const target = timelineTarget(node)

      if (expanded !== target) {
        if (expanded) {
          expanded.style.contentVisibility = originalVisibility
        }

        expanded = target
        originalVisibility = target.style.contentVisibility
        target.style.contentVisibility = 'visible'
      }

      const destination = Math.min(
        Math.max(0, viewport.scrollHeight - viewport.clientHeight),
        Math.max(0, viewport.scrollTop + target.getBoundingClientRect().top - viewport.getBoundingClientRect().top - 8)
      )

      const progress = duration ? Math.min(1, (now - began) / duration) : 1
      const settled = progress === 1 && Math.abs(viewport.scrollTop - destination) <= 1
      stable = settled ? stable + 1 : 0
      viewport.scrollTop = start + (destination - start) * (1 - (1 - progress) ** 3)

      if (stable >= 2 || now - began >= 750) {
        return finish(stable >= 2)
      }

      frame = requestAnimationFrame(step)
    }

    if (signal.aborted) {
      return finish(false)
    }

    signal.addEventListener('abort', abort, { once: true })
    frame = requestAnimationFrame(step)
  })
}
