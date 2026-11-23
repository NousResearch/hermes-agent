import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { Input } from '@/components/ui/input'
import { SearchField } from '@/components/ui/search-field'
import { Select, SelectTrigger } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'

import stylesheet from './one-dark.css?raw'

let style: HTMLStyleElement

beforeEach(() => {
  style = document.createElement('style')
  style.textContent = stylesheet
  document.head.appendChild(style)
  document.documentElement.dataset.hermesPalette = 'one-dark'
})

afterEach(() => {
  cleanup()
  style.remove()
  delete document.documentElement.dataset.hermesPalette
})

describe('One Dark field surfaces (CSSOM)', () => {
  it('paints bare fields, adorned wrappers, search and composer without affecting another palette', () => {
    const { container } = render(
      <>
        <Input aria-label="bare" />
        <Input aria-label="adorned" prefix="$" />
        <Textarea aria-label="textarea" />
        <Select>
          <SelectTrigger aria-label="select" />
        </Select>
        <SearchField onChange={() => {}} placeholder="Search" value="" />
        <div data-slot="command-input-wrapper">
          <input />
        </div>
        <div data-slot="composer-root" data-thread-scrolled-up="">
          <div data-slot="composer-surface" />
        </div>
      </>
    )
    const selectors = [
      '.desktop-input-chrome',
      '[data-slot="input-group"]',
      '[data-slot="command-input-wrapper"]',
      '[data-slot="composer-surface"]'
    ]
    for (const selector of selectors) {
      for (const element of container.querySelectorAll(selector)) {
        expect(getComputedStyle(element).backgroundColor, selector).toBe('rgb(29, 31, 35)')
      }
    }
    const search = container.querySelector('input[placeholder="Search"]')!.parentElement!
    expect(getComputedStyle(search).backgroundColor).toBe('rgb(29, 31, 35)')
    expect(getComputedStyle(search).opacity).toBe('1')
    expect(
      getComputedStyle(container.querySelector('[data-slot="composer-root"]')!)
        .getPropertyValue('--composer-fill')
        .trim()
    ).toBe('#1d1f23')

    delete document.documentElement.dataset.hermesPalette
    expect(getComputedStyle(container.querySelector('[data-slot="input-group"]')!).backgroundColor).not.toBe(
      'rgb(29, 31, 35)'
    )
    expect(getComputedStyle(search).backgroundColor).not.toBe('rgb(29, 31, 35)')
  })

  it('keeps invalid borders destructive, not the neutral ring', () => {
    const { getByRole } = render(<Input aria-invalid="true" aria-label="invalid" />)
    expect(getComputedStyle(getByRole('textbox')).borderColor).toBe('rgb(224, 108, 117)')
  })

  it('keeps floating lists opaque rather than applying their usual translucent fill', () => {
    const { container } = render(<div data-slot="popover-content" />)
    expect(getComputedStyle(container.firstElementChild!).backgroundColor).toBe('rgb(33, 37, 43)')
    delete document.documentElement.dataset.hermesPalette
    expect(getComputedStyle(container.firstElementChild!).backgroundColor).not.toBe('rgb(33, 37, 43)')
  })
})
