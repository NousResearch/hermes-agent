import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $clarifyInputs, $clarifyRequests, setClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { $activeSessionId } from '@/store/session'

import { ClarifyTool, readClarifyResult } from './clarify-tool'

vi.mock('@assistant-ui/react', () => ({
  useAuiState: () => true
}))

afterEach(() => {
  cleanup()
  $clarifyInputs.set({})
  $clarifyRequests.set({})
  $activeSessionId.set(null)
  $gateway.set(null)
})

function renderClarify(ui: ReactNode) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      {ui}
    </I18nProvider>
  )
}

function settledClarifyProps(
  args: ToolCallMessagePartProps['args'],
  result: ToolCallMessagePartProps['result'],
  toolCallId: string
): ToolCallMessagePartProps {
  return {
    addResult: vi.fn(),
    args,
    argsText: JSON.stringify(args),
    isError: false,
    result,
    resume: vi.fn(),
    status: { type: 'complete' },
    toolCallId,
    toolName: 'clarify',
    type: 'tool-call'
  }
}

function pendingClarifyProps(args: ToolCallMessagePartProps['args'], toolCallId: string): ToolCallMessagePartProps {
  return {
    ...settledClarifyProps(args, undefined, toolCallId),
    status: { type: 'running' }
  }
}

describe('readClarifyResult', () => {
  it('reads question + user_response from the tool JSON payload', () => {
    expect(
      readClarifyResult({
        question: 'Which target?',
        choices_offered: ['staging', 'prod'],
        user_response: 'staging'
      })
    ).toEqual({
      question: 'Which target?',
      answer: 'staging',
      error: undefined
    })
  })

  it('parses a JSON string result the same way as an object', () => {
    expect(
      readClarifyResult(
        JSON.stringify({
          question: 'Ship it?',
          user_response: 'yes'
        })
      )
    ).toEqual({
      question: 'Ship it?',
      answer: 'yes',
      error: undefined
    })
  })

  it('keeps an empty user_response so Skip can render as skipped', () => {
    expect(readClarifyResult({ question: 'Ok?', user_response: '' })).toEqual({
      question: 'Ok?',
      answer: '',
      error: undefined
    })
  })
})

describe('ClarifyTool pending view', () => {
  it('restores the draft, focus, selection, and scroll position after a remount', async () => {
    const question = 'What deployment context should I use?'
    const answer = Array.from({ length: 12 }, (_, index) => `Line ${index + 1}: preserve this context.`).join('\n')

    $activeSessionId.set('session-a')
    setClarifyRequest({ choices: null, question, requestId: 'req-a', sessionId: 'session-a' })

    const props = pendingClarifyProps({ question }, 'clarify-pending')
    const first = renderClarify(<ClarifyTool {...props} />)
    const textarea = screen.getByPlaceholderText(/type your answer/i) as HTMLTextAreaElement

    textarea.focus()
    fireEvent.change(textarea, { target: { value: answer } })
    textarea.setSelectionRange(answer.length - 8, answer.length - 3)
    textarea.scrollTop = 48
    fireEvent.select(textarea)
    fireEvent.scroll(textarea)

    first.unmount()
    renderClarify(<ClarifyTool {...props} />)

    const restored = screen.getByPlaceholderText(/type your answer/i) as HTMLTextAreaElement

    await waitFor(() => {
      expect(restored.value).toBe(answer)
      expect(globalThis.document.activeElement).toBe(restored)
      expect(restored.selectionStart).toBe(answer.length - 8)
      expect(restored.selectionEnd).toBe(answer.length - 3)
      expect(restored.scrollTop).toBe(48)
    })
  })
})

describe('ClarifyTool settled view', () => {
  it('keeps the question and answer visible after the tool completes', () => {
    renderClarify(
      <ClarifyTool
        {...settledClarifyProps(
          { question: 'Which deployment target?', choices: ['staging', 'prod'] },
          {
            question: 'Which deployment target?',
            choices_offered: ['staging', 'prod'],
            user_response: 'staging'
          },
          'clarify-1'
        )}
      />
    )

    expect(screen.getByText('Which deployment target?')).toBeTruthy()
    expect(screen.getByText('staging')).toBeTruthy()
    expect(globalThis.document.querySelector('[data-clarify-settled]')).toBeTruthy()
    expect(globalThis.document.querySelector('[data-clarify-answer]')?.textContent).toBe('staging')
  })

  it('labels an empty response as Skipped', () => {
    renderClarify(
      <ClarifyTool
        {...settledClarifyProps(
          { question: 'Anything else?' },
          { question: 'Anything else?', user_response: '' },
          'clarify-2'
        )}
      />
    )

    expect(screen.getByText('Anything else?')).toBeTruthy()
    expect(screen.getByText('Skipped')).toBeTruthy()
  })
})
