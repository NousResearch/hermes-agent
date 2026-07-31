/**
 * CSS selector generation for the preview annotation layer.
 *
 * The goal is NOT the theoretically-shortest selector but one that is:
 *  - unique (re-selects exactly the annotated element)
 *  - stable across re-renders (no CSS-in-JS hash classes)
 *  - readable to an LLM asked to fix the underlying code
 */

/** Matches typical generated class names: css-xxxx, sc-xxxx, or long lowercase-alnum strings containing digits (hashes). */
const PREFIXED_HASH_RE = /^(?:css-[a-z0-9]+|sc-[a-z0-9]+)$/i
const BARE_HASH_RE = /^[a-z0-9]{6,}$/
const MAX_CLASSES = 3
const MAX_DEPTH = 8

/** CSS.escape with a jsdom-safe fallback (jsdom doesn't implement CSS.escape). */
function escapeIdent(value: string): string {
  if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') {
    return CSS.escape(value)
  }

  return value.replace(/([^a-zA-Z0-9_-])/g, '\\$1')
}

/**
 * Heuristic: is this class name stable enough to appear in a selector?
 * Drops CSS-in-JS hashes and single-letter utilities.
 */
export function isStableClass(className: string): boolean {
  if (!className || className.length < 2) {
    return false
  }

  if (PREFIXED_HASH_RE.test(className)) {
    return false
  }

  // Bare hashes (e1a7b3c9, a1b2c3): 6+ lowercase-alnum chars that contain at
  // least one digit. Pure-letter words that long ("submit", "header") are
  // almost always semantic, so they stay.
  if (BARE_HASH_RE.test(className) && /\d/.test(className)) {
    return false
  }

  return true
}

function stableClassesOf(el: Element): string[] {
  return Array.from(el.classList).filter(isStableClass).slice(0, MAX_CLASSES)
}

/**
 * Describes one step of the selector path for `el`, e.g. `li:nth-of-type(2)`,
 * `button.submit`, or `#login-btn`.
 */
function stepFor(el: Element): string {
  if (el.id) {
    return `#${escapeIdent(el.id)}`
  }

  const tag = el.tagName.toLowerCase()
  const classes = stableClassesOf(el)
  let step = classes.length > 0 ? `${tag}.${classes.map(c => escapeIdent(c)).join('.')}` : tag

  // Add :nth-of-type when same-tag siblings make the step ambiguous.
  const parent = el.parentElement
  if (parent) {
    const sameTagSiblings = Array.from(parent.children).filter(sib => sib.tagName === el.tagName)
    if (sameTagSiblings.length > 1) {
      const index = sameTagSiblings.indexOf(el) + 1
      step = classes.length > 0
        ? `${tag}.${classes.map(c => escapeIdent(c)).join('.')}:nth-of-type(${index})`
        : `${tag}:nth-of-type(${index})`
    }
  }

  return step
}

/**
 * Builds a unique, stable-ish CSS selector for `el`.
 *
 * Strategy: walk from the element up to <body>, stopping early at the first
 * ancestor with an id (ids are treated as unique anchors). Each step is
 * `tag.stableClass` with `:nth-of-type` added when same-tag siblings exist.
 */
export function buildCssSelector(el: Element): string {
  if (el.id) {
    return `#${escapeIdent(el.id)}`
  }

  const steps: string[] = []
  let current: Element | null = el
  let depth = 0

  while (current && current.tagName.toLowerCase() !== 'html' && depth < MAX_DEPTH) {
    const step = stepFor(current)
    steps.unshift(step)

    if (current.id) {
      // id anchors the path — nothing above it is needed
      break
    }

    current = current.parentElement
    depth += 1
  }

  // Guarantee uniqueness: if the composed selector matches more than one
  // element, fall back to a fully-nth-qualified path from body.
  const selector = steps.join(' > ')
  try {
    const matches = document.querySelectorAll(selector)
    if (matches.length === 1 && matches[0] === el) {
      return selector
    }
  } catch {
    // fall through to the nth-qualified path
  }

  return fullyQualifiedPath(el)
}

/** `body > main > div:nth-of-type(1) > …` — verbose but always unique. */
function fullyQualifiedPath(el: Element): string {
  const steps: string[] = []
  let current: Element | null = el

  while (current && current.tagName.toLowerCase() !== 'html') {
    const tag = current.tagName.toLowerCase()
    if (current.id) {
      steps.unshift(`#${escapeIdent(current.id)}`)
      break
    }

    const parent: Element | null = current.parentElement
    if (parent) {
      const tagName = current.tagName
      const sameTagSiblings = Array.from(parent.children).filter((sib: Element) => sib.tagName === tagName)
      if (sameTagSiblings.length > 1) {
        steps.unshift(`${tag}:nth-of-type(${sameTagSiblings.indexOf(current) + 1})`)
      } else {
        steps.unshift(tag)
      }
    } else {
      steps.unshift(tag)
    }

    current = parent
  }

  return steps.join(' > ')
}
