import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { type LogSearch, LogSearchField } from './log-search'

function renderField() {
  const search: LogSearch = { active: 0, hits: [], next: vi.fn(), previous: vi.fn(), query: '회의' }
  const onChange = vi.fn()

  render(
    <I18nProvider>
      <LogSearchField onChange={onChange} placeholder="Search" search={search} value="회의" />
    </I18nProvider>
  )

  return { input: screen.getByPlaceholderText('Search'), onChange, search }
}

afterEach(cleanup)

describe('LogSearchField', () => {
  it('steps matches with Enter and Shift+Enter, and clears with Escape', () => {
    const { input, onChange, search } = renderField()

    fireEvent.keyDown(input, { key: 'Enter' })
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: true })
    fireEvent.keyDown(input, { key: 'Escape' })

    expect(search.next).toHaveBeenCalledTimes(1)
    expect(search.previous).toHaveBeenCalledTimes(1)
    expect(onChange).toHaveBeenCalledWith('')
  })

  // macOS delivers a Korean Enter twice (keyCode 229 while composing, then 13), and a
  // Japanese Enter or Escape confirms or cancels a conversion: none of these is a search key.
  it('leaves Enter and Escape to an IME that is still composing', () => {
    const { input, onChange, search } = renderField()

    fireEvent.keyDown(input, { isComposing: true, key: 'Enter', keyCode: 229 })
    fireEvent.keyDown(input, { key: 'Enter', keyCode: 229 })
    fireEvent.keyDown(input, { isComposing: true, key: 'Escape' })

    expect(search.next).not.toHaveBeenCalled()
    expect(onChange).not.toHaveBeenCalled()
  })
})
