/* eslint-disable no-restricted-globals -- the fixture IS the DOM: this test
   builds a transcript and sets a real Selection, which no window-free helper
   can express. */
import { cleanup, fireEvent, render } from '@testing-library/react'
import type { PropsWithChildren } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { type ComposerFollowUp, createComposerFollowUpScope } from '@/store/composer'

import { type ComposerScope, ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { FollowUpPill } from './pill'

/** jsdom lays nothing out, so a real (non-collapsed) selection still measures
 *  0×0 and the pill would read it as nothing to point at. Give the range a
 *  rect the way a laid-out transcript would. */
const stubRangeRects = () => {
  const rect = { bottom: 40, height: 20, left: 20, right: 120, top: 20, width: 100, x: 20, y: 20 }
  const original = Range.prototype.getBoundingClientRect

  Range.prototype.getBoundingClientRect = () => ({ ...rect, toJSON: () => rect }) as DOMRect

  return () => {
    Range.prototype.getBoundingClientRect = original
  }
}

let restoreRects: () => void

describe('FollowUpPill', () => {
  let scope: ComposerScope
  let followUp: ReturnType<typeof createComposerFollowUpScope>
  let viewport: HTMLDivElement
  let body: HTMLDivElement

  beforeEach(() => {
    followUp = createComposerFollowUpScope()
    scope = { ...MAIN_COMPOSER_SCOPE, followUp }
    viewport = document.createElement('div')
    viewport.innerHTML =
      '<div data-role="assistant" data-slot="aui_assistant-message-root"><div>the quoted answer</div></div>'
    document.body.append(viewport)
    body = viewport.querySelector('[data-role="assistant"] div') as HTMLDivElement
    restoreRects = stubRangeRects()
  })

  afterEach(() => {
    restoreRects()
    window.getSelection()?.removeAllRanges()
    viewport.remove()
    cleanup()
  })

  const mount = () => {
    const viewportRef = { current: viewport }

    const Wrapper = ({ children }: PropsWithChildren) => (
      <I18nProvider configClient={null}>
        <ComposerScopeProvider value={scope}>{children}</ComposerScopeProvider>
      </I18nProvider>
    )

    return render(<FollowUpPill viewportRef={viewportRef} />, { wrapper: Wrapper })
  }

  const selectBody = () => {
    const range = document.createRange()
    range.selectNodeContents(body)
    window.getSelection()?.addRange(range)
  }

  const pill = () => document.querySelector('[data-slot="aui_follow-up-pill"]')

  it('offers the pill once a passage is selected, and attaches it on press', () => {
    mount()
    expect(pill()).toBeNull()

    selectBody()
    fireEvent.keyUp(window, { key: 'Shift' })

    const button = pill()
    expect(button?.textContent).toContain('Follow-up')

    fireEvent.mouseDown(button!)

    expect(followUp.$followUp.get()).toEqual({ passage: 'the quoted answer', source: 'assistant' })
    // The passage moved into the composer card, so the transcript selection is
    // released and the pill is gone.
    expect(window.getSelection()?.rangeCount).toBe(0)
    expect(pill()).toBeNull()
  })

  it('waits for the gesture to end before offering anything', () => {
    mount()
    // A drag in progress: the selection already exists, the gesture does not.
    selectBody()
    fireEvent(document, new Event('selectionchange'))

    expect(pill()).toBeNull()

    fireEvent.pointerUp(window, { button: 0 })

    expect(pill()).not.toBeNull()
  })

  it('retires the offer when a new gesture starts', () => {
    mount()
    selectBody()
    fireEvent.pointerUp(window, { button: 0 })
    expect(pill()).not.toBeNull()

    fireEvent.pointerDown(document.body, { button: 0 })

    expect(pill()).toBeNull()
  })

  it('is offered for a selection out of the reader own message too', () => {
    mount()
    selectBody()
    fireEvent.keyUp(window, { key: 'Shift' })
    fireEvent.mouseDown(pill()!)

    expect((followUp.$followUp.get() as ComposerFollowUp).source).toBe('assistant')
  })

  it('sends the selection to the composer on ⌘/Ctrl+L', () => {
    mount()
    selectBody()

    fireEvent.keyDown(window, { key: 'l', ctrlKey: true })

    expect(followUp.$followUp.get()).toEqual({ passage: 'the quoted answer', source: 'assistant' })
  })

  it('stays hidden for a collapsed caret and for a selection outside the transcript', () => {
    mount()

    const range = document.createRange()
    range.setStart(body.firstChild!, 2)
    range.collapse(true)
    window.getSelection()?.addRange(range)
    fireEvent.keyUp(window, { key: 'Shift' })

    expect(pill()).toBeNull()

    const outside = document.createElement('div')
    outside.textContent = 'composer draft'
    document.body.append(outside)
    const outsideRange = document.createRange()
    outsideRange.selectNodeContents(outside)
    window.getSelection()?.removeAllRanges()
    window.getSelection()?.addRange(outsideRange)
    fireEvent.keyUp(window, { key: 'Shift' })

    expect(pill()).toBeNull()

    outside.remove()
  })
})
