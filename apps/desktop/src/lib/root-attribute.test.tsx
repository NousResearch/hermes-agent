// @vitest-environment jsdom
import { readdirSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { useRootAttribute } from './root-attribute'

function Holder({ name, value }: { name: string; value: boolean | null | string }) {
  useRootAttribute(name, value)

  return null
}

const root = window.document.documentElement

describe('useRootAttribute', () => {
  afterEach(cleanup)

  it('names the fact on <html> while held and lets go on unmount', () => {
    const view = render(<Holder name="data-fact" value />)

    expect(root.getAttribute('data-fact')).toBe('')
    view.rerender(<Holder name="data-fact" value={false} />)
    expect(root.hasAttribute('data-fact')).toBe(false)
    view.rerender(<Holder name="data-fact" value="chats" />)
    expect(root.getAttribute('data-fact')).toBe('chats')
    view.unmount()
    expect(root.hasAttribute('data-fact')).toBe(false)
  })

  it('keeps the attribute while any holder remains', () => {
    const first = render(<Holder name="data-fact" value />)
    const second = render(<Holder name="data-fact" value />)

    first.unmount()
    expect(root.hasAttribute('data-fact')).toBe(true)
    second.unmount()
    expect(root.hasAttribute('data-fact')).toBe(false)
  })
})

function stylesheets(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap(entry =>
    entry.isDirectory()
      ? stylesheets(join(dir, entry.name))
      : entry.name.endsWith('.css')
        ? [join(dir, entry.name)]
        : []
  )
}

// A source scan, the same category as a lint rule (see no-native-title.test.ts). With :has() on the
// root compound, Chromium re-matches every :has() rule in the app across the whole document on each
// DOM change: one such rule made a keystroke in a long chat restyle thousands of elements. Name the
// fact on the root with useRootAttribute instead.
describe('stylesheets', () => {
  it('never anchor :has() on the document root', () => {
    const src = resolve(import.meta.dirname, '..')

    const offenders = stylesheets(src).flatMap(file => {
      const css = readFileSync(file, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '')
      const selectors = css.match(/(?:^|[},])\s*(?::root|html)(?:\[[^\]]*\]|\.[\w-]+|:not\((?:[^()]|\([^()]*\))*\))*:has\(/gm)

      return selectors ? [`${file.slice(src.length + 1)} (${selectors.length})`] : []
    })

    expect(offenders).toEqual([])
  })
})
