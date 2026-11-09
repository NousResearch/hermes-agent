import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'

/** A React root in a container attached to the document, for the handful of
 *  suites that cannot use Testing Library's `render`.
 *
 *  They need the component mounted in the real document — a canvas that must
 *  paint, an element measured by its parent, a portal that reads
 *  `document.body` — and they need to unmount inside `act` mid-test to assert
 *  on teardown. Cleanup is idempotent so `afterEach` can call it after a test
 *  already did. */
export function reactRoot() {
  let root: Root | null = null
  let container: HTMLDivElement | null = null

  return {
    get container() {
      return container
    },
    /** The root itself, for a re-render that has to share an `act` with
     *  something else — a mutation-observer callback, a timer flush. */
    get root() {
      return root
    },
    render(ui: ReactNode) {
      container = document.createElement('div')
      document.body.append(container)
      root = createRoot(container)

      act(() => {
        root!.render(ui)
      })
    },
    unmount() {
      if (root) {
        act(() => root!.unmount())
      }

      container?.remove()
      root = null
      container = null
    }
  }
}
