import { useEffect } from 'react'

/** Who holds each attribute on <html>, and the value each holder gives it. */
const holders = new Map<string, Map<symbol, string>>()

function apply(attribute: string) {
  const held = holders.get(attribute)
  const value = held?.size ? [...held.values()].at(-1) : undefined

  if (value === undefined) {
    document.documentElement.removeAttribute(attribute)
  } else {
    document.documentElement.setAttribute(attribute, value)
  }
}

/**
 * Names a fact about the page on <html> while this component holds it, for style that reaches across
 * the window (`:root[data-x] .far-away`). `true` sets the bare attribute, a string its value, and
 * `false` or `null` lets go; the attribute stays while any mounted holder keeps it, the newest
 * holder's value winning.
 *
 * This is what a `:root:has(.thing) .far-away` selector would say, without its cost: with :has()
 * anchored on the root, Chromium re-matches every :has() rule in the app across the whole document
 * on each DOM change (a keystroke in a long chat restyled thousands of elements).
 */
export function useRootAttribute(attribute: string, value: boolean | null | string): void {
  useEffect(() => {
    if (value === false || value === null) {
      return
    }

    const holder = Symbol(attribute)
    const held = holders.get(attribute) ?? new Map<symbol, string>()

    holders.set(attribute, held)
    held.set(holder, value === true ? '' : value)
    apply(attribute)

    return () => {
      held.delete(holder)
      apply(attribute)
    }
  }, [attribute, value])
}
