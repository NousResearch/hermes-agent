import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import { ZoomNumericEntry, sanitizeZoomInput } from './appearance-settings'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('sanitizeZoomInput', () => {
  it.each([
    ['200', 200],
    ['200%', 200],
    ['  175  ', 175],
    ['90%', 90],
    ['500', 500],
    ['25', 25],
    ['499.6', 500]
  ])('accepts %s as %i', (raw, expected) => {
    expect(sanitizeZoomInput(raw)).toBe(expected)
  })

  it.each([
    ['abc'],
    [''],
    ['0'],
    ['24'],
    ['501'],
    ['999'],
    ['-50'],
    ['NaN'],
    ['undefined'],
    ['%']
  ])('rejects %s as null', raw => {
    expect(sanitizeZoomInput(raw)).toBeNull()
  })
})

function renderNumericEntry(percent: number, onCommit = vi.fn()) {
  render(
    <I18nProvider configClient={null}>
      <ZoomNumericEntry percent={percent} onCommit={onCommit} />
    </I18nProvider>
  )

  return screen.getByLabelText('UI Scale') as HTMLInputElement
}

describe('ZoomNumericEntry — valid Enter/blur commit', () => {
  it('commits a valid value on Enter', () => {
    const onCommit = vi.fn()
    const input = renderNumericEntry(150, onCommit)

    fireEvent.change(input, { target: { value: '200' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    // Enter triggers blur, which triggers commit.
    fireEvent.blur(input)

    expect(onCommit).toHaveBeenCalledWith(200)
    expect(onCommit).toHaveBeenCalledTimes(1)
  })

  it('commits a valid value on blur without Enter', () => {
    const onCommit = vi.fn()
    const input = renderNumericEntry(100, onCommit)

    fireEvent.change(input, { target: { value: '300' } })
    fireEvent.blur(input)

    expect(onCommit).toHaveBeenCalledWith(300)
  })

  it('does not commit when the value is unchanged', () => {
    const onCommit = vi.fn()
    const input = renderNumericEntry(150, onCommit)

    // Type the same value.
    fireEvent.change(input, { target: { value: '150' } })
    fireEvent.blur(input)

    expect(onCommit).not.toHaveBeenCalled()
  })
})

describe('ZoomNumericEntry — invalid rollback', () => {
  it('reverts to current zoom on invalid text', () => {
    const onCommit = vi.fn()
    const input = renderNumericEntry(175, onCommit)

    fireEvent.change(input, { target: { value: 'abc' } })
    fireEvent.blur(input)

    expect(onCommit).not.toHaveBeenCalled()
    expect(input.value).toBe('175')
  })

  it('reverts to current zoom on out-of-range value', () => {
    const onCommit = vi.fn()
    const input = renderNumericEntry(150, onCommit)

    fireEvent.change(input, { target: { value: '999' } })
    fireEvent.blur(input)

    expect(onCommit).not.toHaveBeenCalled()
    expect(input.value).toBe('150')
  })

  it('reverts on value below minimum', () => {
    const onCommit = vi.fn()
    const input = renderNumericEntry(100, onCommit)

    fireEvent.change(input, { target: { value: '10' } })
    fireEvent.blur(input)

    expect(onCommit).not.toHaveBeenCalled()
    expect(input.value).toBe('100')
  })

  it('reverts on empty input', () => {
    const onCommit = vi.fn()
    const input = renderNumericEntry(125, onCommit)

    fireEvent.change(input, { target: { value: '' } })
    fireEvent.blur(input)

    expect(onCommit).not.toHaveBeenCalled()
    expect(input.value).toBe('125')
  })
})

describe('ZoomNumericEntry — external zoom synchronization', () => {
  it('updates the displayed value when the percent prop changes', () => {
    const onCommit = vi.fn()
    const { rerender } = render(
      <I18nProvider configClient={null}>
        <ZoomNumericEntry percent={100} onCommit={onCommit} />
      </I18nProvider>
    )

    const input = screen.getByLabelText('UI Scale') as HTMLInputElement
    expect(input.value).toBe('100')

    // Simulate an external zoom change (preset click, Ctrl+/-, restore-on-focus).
    rerender(
      <I18nProvider configClient={null}>
        <ZoomNumericEntry percent={200} onCommit={onCommit} />
      </I18nProvider>
    )

    expect(input.value).toBe('200')
  })

  it('does not fire onCommit when the prop changes externally', () => {
    const onCommit = vi.fn()
    const { rerender } = render(
      <I18nProvider configClient={null}>
        <ZoomNumericEntry percent={150} onCommit={onCommit} />
      </I18nProvider>
    )

    rerender(
      <I18nProvider configClient={null}>
        <ZoomNumericEntry percent={175} onCommit={onCommit} />
      </I18nProvider>
    )

    expect(onCommit).not.toHaveBeenCalled()
  })
})
