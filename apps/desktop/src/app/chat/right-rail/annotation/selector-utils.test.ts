import { describe, expect, it } from 'vitest'

import { buildCssSelector, isStableClass } from './selector-utils'

/**
 * Build a small DOM helper around jsdom so tests read like the pages they
 * describe. `html` is mounted under a fresh <div id="root"> each time.
 */
function mount(html: string, withId = true): HTMLElement {
  const root = document.createElement('div')
  if (withId) {
    root.id = 'root'
  }
  root.innerHTML = html
  document.body.appendChild(root)
  return root
}

function cleanup() {
  document.body.innerHTML = ''
}

describe('isStableClass', () => {
  it('keeps semantic class names', () => {
    expect(isStableClass('submit')).toBe(true)
    expect(isStableClass('btn-primary')).toBe(true)
    expect(isStableClass('header_nav')).toBe(true)
    expect(isStableClass('card2')).toBe(true)
  })

  it('drops CSS-in-JS / tailwind-jit hash classes', () => {
    expect(isStableClass('css-1x2y3z')).toBe(false)
    expect(isStableClass('e1a7b3c9')).toBe(false) // 8-char lowercase alnum hash
    expect(isStableClass('sc-hash12')).toBe(false) // styled-components prefix + hash
    expect(isStableClass('')).toBe(false)
    expect(isStableClass('a')).toBe(false) // single-char utility
  })
})

describe('buildCssSelector', () => {
  it('returns #id immediately when element has an id', () => {
    const root = mount(`<div><span><button id="login-btn">登录</button></span></div>`)
    const btn = root.querySelector('button')!
    expect(buildCssSelector(btn)).toBe('#login-btn')
    cleanup()
  })

  it('walks up to the nearest ancestor with an id', () => {
    const root = mount(`<div id="app"><section><p class="txt">hi</p></section></div>`)
    const p = root.querySelector('p')!
    expect(buildCssSelector(p)).toBe('#app > section > p.txt')
    cleanup()
  })

  it('uses tag + stable classes, dropping hash classes', () => {
    const root = mount(`<div id="app"><button class="submit css-1x2y3z primary">OK</button></div>`)
    const btn = root.querySelector('button')!
    expect(buildCssSelector(btn)).toBe('#app > button.submit.primary')
    cleanup()
  })

  it('disambiguates same-tag siblings with :nth-of-type', () => {
    const root = mount(`
      <div id="app">
        <ul>
          <li>one</li>
          <li><a href="#">two</a></li>
        </ul>
      </div>`)
    const a = root.querySelector('a')!
    // a is the only <a> in its <li>, so no nth needed there; li is 2nd <li>
    expect(buildCssSelector(a)).toBe('#app > ul > li:nth-of-type(2) > a')
    cleanup()
  })

  it('falls back to body path when no id exists anywhere', () => {
    const root = mount(`<main><div><span class="deep">deep</span></div></main>`, false)
    const span = root.querySelector('span')!
    expect(buildCssSelector(span)).toBe('body > div > main > div > span.deep')
    cleanup()
  })

  it('produces a selector that actually re-selects the element', () => {
    const root = mount(`
      <div id="form">
        <div class="row"><input type="text" placeholder="a"></div>
        <div class="row"><input type="password" placeholder="b"></div>
      </div>`)
    const pwd = root.querySelector('input[type="password"]')!
    const selector = buildCssSelector(pwd)
    expect(document.querySelector(selector)).toBe(pwd)
    cleanup()
  })

  it('caps class list to keep selector readable', () => {
    const root = mount(
      `<div id="app"><p class="a b c d e f g h">many</p></div>`
    )
    const p = root.querySelector('p')!
    const selector = buildCssSelector(p)
    // at most 3 classes retained
    expect((selector.match(/\./g) ?? []).length).toBeLessThanOrEqual(3)
    expect(document.querySelector(selector)).toBe(p)
    cleanup()
  })
})
