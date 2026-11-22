// TESTING ONLY: how the discovery lab reaches the parts of the real UI a store cannot open (a popover's
// trigger, a splash's own step), and how it reloads the page for what is read once per load.

/** How long a press waits for its control to appear (a route change, a lazy chunk). */
const PRESS_WAIT_MS = 4000

/** A new chat puts focus in its composer as it lands, which would close a popover opened before it:
 *  the press waits for the landing to settle. */
const PRESS_SETTLE_MS = 700

/** Press the first visible control matching `selector` once the screen has settled, as the tester
 *  would; `prepare` first brings it on screen (a sheet), and the screen settles again. Gives up quietly after {@link PRESS_WAIT_MS}: the state is set either way, only the last
 *  step is left to the tester. */
export function pressWhenShown(selector: string, prepare?: () => void): void {
  const deadline = performance.now() + 2 * PRESS_SETTLE_MS + PRESS_WAIT_MS

  const attempt = () => {
    const control = [...document.querySelectorAll<HTMLElement>(selector)].find(
      element => element.getClientRects().length > 0
    )

    if (control) {
      control.click()
    } else if (performance.now() < deadline) {
      requestAnimationFrame(attempt)
    }
  }

  window.setTimeout(() => {
    if (!prepare) {
      attempt()

      return
    }

    prepare()
    window.setTimeout(attempt, PRESS_SETTLE_MS)
  }, PRESS_SETTLE_MS)
}

/** Load the app afresh at `url` (same origin), even when only its route differs. */
export function reloadAt(url: string): void {
  window.history.replaceState(window.history.state, '', url)
  window.location.reload()
}
