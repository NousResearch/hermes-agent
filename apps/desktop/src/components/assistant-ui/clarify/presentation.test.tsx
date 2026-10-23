// A live card reports once when it is mounted and usable.
// Focus, scroll, size, and clicks do not gate or cancel that receipt.
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { I18nProvider } from '@/i18n'
import { rememberServerRequest, resetServerRequestsForTests } from '@/store/server-requests'

import { ClarifyToolPending } from './pending'

const report = vi.fn()
const request = {
  questions: [{ choices: ['a', 'b'], multiSelect: false, qid: 'c', question: 'Choose one' }],
  requestId: 'c', sessionId: 's'
}

beforeEach(() => {
  report.mockClear()
  resetServerRequestsForTests()
  rememberServerRequest({ id: 'c', method: 'clarify', params: {}, respond: vi.fn(), fail: vi.fn(), reportClarifyPresentation: report } as never)
})

afterEach(() => {
  cleanup()
  resetServerRequestsForTests()
})

function card(props = {}, visible = true) {
  return <I18nProvider configClient={null} initialLocale="en"><PaneVisibleContext value={visible}>
    <ClarifyToolPending onAnswered={vi.fn()} request={request} undelivered={false} {...props} />
  </PaneVisibleContext></I18nProvider>
}

test('a usable card reports once while Confirm stays disabled', () => {
  const view = render(card())

  expect((screen.getByRole('button', { name: /Confirm and continue/ }) as HTMLButtonElement).disabled).toBe(true)
  expect(report).toHaveBeenCalledTimes(1)
  expect(report).toHaveBeenCalledWith('presented')
  view.rerender(card())
  expect(report).toHaveBeenCalledTimes(1)
})

test('a hidden pane waits, then one receipt starts when that pane is shown', () => {
  const view = render(card({}, false))

  expect(report).not.toHaveBeenCalled()
  view.rerender(card({}, true))
  expect(report).toHaveBeenCalledTimes(1)
  expect(report).toHaveBeenCalledWith('presented')
})

test.each(['preview', 'undelivered'])('%s card supplies no presentation', state => {
  const props = state === 'preview'
    ? { request: null, fromArgs: { questions: [{ question: 'Choose', choices: ['a'] }] } }
    : { undelivered: true }

  render(card(props))
  expect(report).not.toHaveBeenCalled()
})

test('a prefilled card and later clicks still count as one usable receipt', () => {
  render(card({ request: { ...request, lockedAnswers: { c: 'a' } } }))
  expect(report).toHaveBeenCalledTimes(1)
  expect(report).toHaveBeenCalledWith('presented')
  fireEvent.click(screen.getByRole('button', { name: 'Aa' }))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'hold' } })
  expect(report).toHaveBeenCalledTimes(1)
  expect((screen.getByRole('button', { name: /Confirm and continue/ }) as HTMLButtonElement).disabled).toBe(false)
})

test('a replaced reporter cannot restart the receipt', () => {
  const view = render(card())
  const replayReport = vi.fn()

  rememberServerRequest({ id: 'c', method: 'clarify', params: {}, respond: vi.fn(), fail: vi.fn(),
    reportClarifyPresentation: replayReport })
  view.rerender(card())
  expect(report).toHaveBeenCalledTimes(1)
  expect(replayReport).not.toHaveBeenCalled()
})
