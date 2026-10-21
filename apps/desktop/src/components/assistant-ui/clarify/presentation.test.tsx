// Exercise the real pending card with deterministic browser visibility signals.
// Presentation must not require enabling an untouched card's Confirm button.
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { I18nProvider } from '@/i18n'
import { rememberServerRequest, resetServerRequestsForTests } from '@/store/server-requests'

import { ClarifyToolPending } from './pending'

let frames: FrameRequestCallback[]
let hit: Element | null
let measured: Element | null
const report = vi.fn(async () => undefined)
const request = {
  questions: [{ choices: ['a', 'b'], multiSelect: false, qid: 'c', question: 'Choose one' }],
  requestId: 'c', sessionId: 's'
}

beforeEach(() => {
  frames = []
  hit = null
  report.mockClear()
  resetServerRequestsForTests()
  rememberServerRequest({ id: 'c', method: 'clarify', params: {}, respond: vi.fn(), fail: vi.fn(), reportClarifyPresentation: report } as never)
  vi.spyOn(document, 'hasFocus').mockReturnValue(true)
  vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
  Object.defineProperty(HTMLElement.prototype, 'checkVisibility', { configurable: true, value: () => true })
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
    measured = this
    return { x: 10, y: 10, top: 10, left: 10, bottom: 50, right: 110, width: 100, height: 40, toJSON: () => ({}) }
  })
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => frames.push(callback))
  vi.stubGlobal('cancelAnimationFrame', vi.fn())
  vi.stubGlobal('IntersectionObserver', class {
    constructor(_callback: IntersectionObserverCallback) {}
    observe() {}
    disconnect() {}
  })
  Object.defineProperty(document, 'elementFromPoint', { configurable: true, value: () => hit ?? measured })
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  resetServerRequestsForTests()
})

function card(props = {}, visible = true) {
  return <I18nProvider configClient={null} initialLocale="en"><PaneVisibleContext value={visible}>
    <ClarifyToolPending onAnswered={vi.fn()} request={request} undelivered={false} {...props} />
  </PaneVisibleContext></I18nProvider>
}

function paint() {
  act(() => { const pending = frames; frames = []; pending.forEach(callback => callback(0)) })
  act(() => { const pending = frames; frames = []; pending.forEach(callback => callback(1)) })
}

test('fresh answerable card reports with Confirm disabled, then a pick invalidates without answering', () => {
  render(card())
  expect((screen.getByRole('button', { name: /Confirm and continue/ }) as HTMLButtonElement).disabled).toBe(true)
  paint()
  expect(report).toHaveBeenCalledWith('presented')
  fireEvent.click(screen.getByRole('button', { name: 'Aa' }))
  expect(report).toHaveBeenCalledWith('interaction')
  expect((screen.getByRole('button', { name: /Confirm and continue/ }) as HTMLButtonElement).disabled).toBe(false)
})

test.each(['preview', 'hidden', 'blurred', 'offscreen', 'occluded', 'locked', 'undelivered'])('%s card supplies no presentation', state => {
  if (state === 'blurred') { vi.mocked(document.hasFocus).mockReturnValue(false) }
  if (state === 'offscreen') { vi.mocked(HTMLElement.prototype.getBoundingClientRect).mockReturnValue({ top: -1 } as DOMRect) }
  if (state === 'occluded') { hit = document.body }
  const props = state === 'preview' ? { request: null, fromArgs: { questions: [{ question: 'Choose', choices: ['a'] }] } }
    : state === 'locked' ? { request: { ...request, lockedAnswers: { c: null } } }
    : state === 'undelivered' ? { undelivered: true } : {}
  render(card(props, state !== 'hidden'))
  paint()
  expect(report).not.toHaveBeenCalledWith('presented')
})

test('blur and unmount withdraw presentation permanently on the mounted card', () => {
  const view = render(card())
  paint()
  expect(report).toHaveBeenCalledWith('presented')
  fireEvent(window, new Event('blur'))
  expect(report).toHaveBeenCalledWith('hidden')
  report.mockClear()
  fireEvent(window, new Event('focus'))
  paint()
  expect(report).not.toHaveBeenCalledWith('presented')
  view.unmount()
})

test('draft changes and clearing do not submit a fabricated human answer', () => {
  render(card())
  paint()
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'hold' } })
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '' } })
  expect(report).toHaveBeenCalledWith('interaction')
  expect((screen.getByRole('button', { name: /Confirm and continue/ }) as HTMLButtonElement).disabled).toBe(true)
})

test('disabled Confirm remains visible when its pointer events pass through to the bar', () => {
  render(card())
  const confirm = screen.getByRole('button', { name: /Confirm and continue/ })
  confirm.style.pointerEvents = 'none'
  Object.defineProperty(document, 'elementFromPoint', { configurable: true,
    value: () => measured === confirm ? confirm.parentElement : measured })
  paint()
  expect(report).toHaveBeenCalledWith('presented')
})

test('a same-card reconnect keeps unsubmitted interaction ineligible', () => {
  vi.mocked(document.hasFocus).mockReturnValue(false)
  const view = render(card())
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'hold' } })
  const replayReport = vi.fn()
  rememberServerRequest({ id: 'c', method: 'clarify', params: {}, respond: vi.fn(), fail: vi.fn(),
    reportClarifyPresentation: replayReport })
  vi.mocked(document.hasFocus).mockReturnValue(true)
  view.rerender(card())
  paint()
  expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('hold')
  expect(replayReport).not.toHaveBeenCalledWith('presented')
})
