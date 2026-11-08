import { beforeEach, describe, expect, it } from 'vitest'

import { identityKit } from './identity'
import { namingKit } from './naming'
import type { PreviewActBinding } from './types'

const naming = () => namingKit(document)

function el(html: string): Element {
  document.body.innerHTML = html

  return document.body.firstElementChild!
}

/** A remembered element, with only the fields a given test cares about set. */
function bound(over: Partial<PreviewActBinding>): PreviewActBinding {
  return {
    el: document.body,
    label: '',
    name: '',
    off: false,
    path: 'root',
    ref: 'btn-x',
    role: 'button',
    stable: '',
    value: '',
    ...over
  }
}

beforeEach(() => {
  document.body.innerHTML = ''
})

describe('slug', () => {
  it('reads as words a person would recognise', () => {
    expect(naming().slug('Sign In')).toBe('sign-in')
    expect(naming().slug('  Add to cart!  ')).toBe('add-to-cart')
  })

  it('collapses runs of punctuation rather than stacking hyphens', () => {
    expect(naming().slug('Save — and / continue')).toBe('save-and-continue')
  })

  it('stays short enough to read at a glance, and never ends on a hyphen', () => {
    const long = naming().slug('Subscribe to our weekly newsletter for updates')

    expect(long.length).toBeLessThanOrEqual(24)
    expect(long.endsWith('-')).toBe(false)
  })

  it('has nothing to say about an unlabelled element', () => {
    expect(naming().slug('!!!')).toBe('')
  })
})

describe('stemOf', () => {
  // The stem is the part of a handle that survives translation: whatever the
  // button says, `btn-` tells the agent it is a button.
  it('gives each interactive role its own prefix', () => {
    const { stemOf } = naming()

    expect(stemOf('button')).toBe('btn')
    expect(stemOf('a')).toBe('lnk')
    expect(stemOf('input:search')).toBe('srch')
    expect(stemOf('input:checkbox')).toBe('chk')
    expect(stemOf('select')).toBe('sel')
  })

  it('treats every other kind of text field as one thing', () => {
    const { stemOf } = naming()

    expect(stemOf('input:email')).toBe('inp')
    expect(stemOf('input:date')).toBe('inp')
    expect(stemOf('textbox')).toBe('inp')
  })

  it('falls back rather than inventing a prefix for something unknown', () => {
    expect(naming().stemOf('marquee')).toBe('el')
  })
})

describe('labelOf', () => {
  it('prefers what the page tells assistive clients', () => {
    expect(naming().labelOf(el('<button aria-label="Close dialog">×</button>'))).toBe('Close dialog')
  })

  it('follows aria-labelledby to the element that names it', () => {
    document.body.innerHTML = '<h2 id="t">Billing</h2><button aria-labelledby="t">→</button>'

    expect(naming().labelOf(document.querySelector('button')!)).toBe('Billing')
  })

  it('falls back through placeholder and title before giving up', () => {
    expect(naming().labelOf(el('<input placeholder="Your email" />'))).toBe('Your email')
    expect(naming().labelOf(el('<a href="/x" title="Help"></a>'))).toBe('Help')
    expect(naming().labelOf(el('<button></button>'))).toBe('')
  })

  it('flattens the whitespace a formatted template leaves behind', () => {
    expect(naming().labelOf(el('<button>\n  Save\n  changes\n</button>'))).toBe('Save changes')
  })
})

describe('selectorFor', () => {
  // Identity only. A positional chain was 74% of the inventory and wrong the
  // moment a sibling appeared.
  it('names the element only when the page gave it a durable name', () => {
    const { selectorFor } = naming()

    expect(selectorFor(el('<button id="save">S</button>'))).toBe('#save')
    expect(selectorFor(el('<button data-testid="save">S</button>'))).toBe('[data-testid="save"]')
    expect(selectorFor(el('<button class="btn primary">S</button>'))).toBe('')
  })
})

describe('valueOf', () => {
  it('reports what a control currently holds', () => {
    const input = el('<input value="shoes" />')

    expect(naming().valueOf(input)).toBe('shoes')
  })

  // The DOM hands back "on" for an unset checkbox value, so reading the value
  // first made both states identical to the agent.
  it('reports a checkbox as its state rather than the value the DOM invents', () => {
    const box = el('<input type="checkbox" />') as HTMLInputElement

    expect(naming().valueOf(box)).toBe('unchecked')

    box.checked = true

    expect(naming().valueOf(box)).toBe('checked')
  })

  it('still reports a checkbox state when the page did set a value', () => {
    const box = el('<input type="checkbox" value="weekly" />') as HTMLInputElement
    box.checked = true

    expect(naming().valueOf(box)).toBe('checked')
  })

  it('leaves a radio group readable the same way', () => {
    const radio = el('<input type="radio" name="plan" value="pro" />') as HTMLInputElement

    expect(naming().valueOf(radio)).toBe('unchecked')
  })
})

describe('anchorOf', () => {
  // Coarse on purpose: a wrapper div appearing in the chain must not change
  // this string, because that churn is exactly what a re-bind sees through.
  it('places an element by the landmark it sits in', () => {
    document.body.innerHTML = '<nav><div><div><button>Home</button></div></div></nav>'

    expect(naming().anchorOf(document.querySelector('button')!)).toBe('nav')
  })

  it('tells two same-tag landmarks apart by their label', () => {
    document.body.innerHTML = '<nav aria-label="Primary nav"><button>Home</button></nav>'

    expect(naming().anchorOf(document.querySelector('button')!)).toBe('nav#primary-nav')
  })

  it('says so when there is no landmark to place it in', () => {
    expect(naming().anchorOf(el('<button>Home</button>'))).toBe('root')
  })
})

describe('alike', () => {
  const { alike } = identityKit(naming())

  it('sees through a count badge and a copy edit', () => {
    expect(alike('Inbox', 'Inbox (3)')).toBe(true)
    expect(alike('Save changes', 'Save your changes')).toBe(true)
  })

  it('does not call two unrelated labels the same thing', () => {
    expect(alike('Sign in', 'Delete account')).toBe(false)
    expect(alike('Save', '')).toBe(false)
  })
})

describe('affinity', () => {
  const { affinity } = identityKit(naming())

  it('is certain when both sides carry the same stable attribute', () => {
    expect(affinity(bound({ stable: 'save' }), bound({ stable: 'save' }))).toBe(1)
  })

  // The page is telling us outright that these are different things.
  it('refuses outright when two stable attributes disagree', () => {
    expect(affinity(bound({ stable: 'save' }), bound({ stable: 'cancel' }))).toBe(0)
  })

  it('refuses to move a handle across roles, however alike the rest reads', () => {
    const was = bound({ name: 'Save', role: 'button' })
    const now = bound({ name: 'Save', role: 'a' })

    expect(affinity(was, now)).toBe(0)
  })

  it('clears the bar on name and place together, and not on place alone', () => {
    const named = affinity(bound({ name: 'Save', path: 'main' }), bound({ name: 'Save', path: 'main' }))
    const placed = affinity(bound({ name: 'Save', path: 'main' }), bound({ name: 'Publish', path: 'main' }))

    expect(named).toBeGreaterThanOrEqual(0.6)
    expect(placed).toBeLessThan(0.6)
  })
})

describe('coin', () => {
  const mint = () => {
    const { coin } = identityKit(naming())
    const coined: Record<string, number> = {}

    return (role: string, name: string) => coin(coined, role, name)
  }

  it('names a handle after what the thing is and what it says', () => {
    const coin = mint()

    expect(coin('button', 'Sign in')).toBe('btn-sign-in')
    expect(coin('input:email', 'Email')).toBe('inp-email')
  })

  it('tells repeats apart without renaming the first one', () => {
    const coin = mint()

    expect(coin('button', 'Edit')).toBe('btn-edit')
    expect(coin('button', 'Edit')).toBe('btn-edit-1')
    expect(coin('button', 'Edit')).toBe('btn-edit-2')
  })

  // Never rewound, so a retired handle's name is not later handed to a
  // different element on the same page.
  it('does not reissue a name after the element holding it goes away', () => {
    const coin = mint()

    coin('button', 'Edit')
    coin('button', 'Edit')

    expect(coin('button', 'Edit')).toBe('btn-edit-2')
  })

  it('still mints something for a role it has no name for', () => {
    expect(mint()('button', '')).toBe('btn')
  })
})
