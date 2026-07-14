import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $clarifyRequests, setClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { $activeSessionId } from '@/store/session'

import { ClarifyTool, readClarifyResult } from './clarify-tool'

vi.mock('@assistant-ui/react', async importOriginal => ({
  ...(await importOriginal<typeof import('@assistant-ui/react')>()),
  useAuiState: () => true
}))

afterEach(() => {
  cleanup()
  $activeSessionId.set(null)
  $clarifyRequests.set({})
  $gateway.set(null)
})

function renderClarify(ui: ReactNode) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      {ui}
    </I18nProvider>
  )
}

function clarifyProps(
  args: ToolCallMessagePartProps['args'],
  result: ToolCallMessagePartProps['result'],
  toolCallId: string,
  status: ToolCallMessagePartProps['status'] = { type: 'complete' }
): ToolCallMessagePartProps {
  return {
    addResult: vi.fn(),
    args,
    argsText: JSON.stringify(args),
    isError: false,
    result,
    resume: vi.fn(),
    status,
    toolCallId,
    toolName: 'clarify',
    type: 'tool-call'
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
  it('keeps a typed answer when the pending panel remounts', () => {
    const question = 'What deployment context should I use?'
    $activeSessionId.set('session-a')
    setClarifyRequest({ choices: null, question, requestId: 'req-a', sessionId: 'session-a' })

    const props = clarifyProps({ question }, undefined, 'clarify-pending', { type: 'running' })
    const first = renderClarify(<ClarifyTool {...props} />)

    fireEvent.change(screen.getByPlaceholderText(/Type your answer/), {
      target: { value: 'Use staging first, then promote to production after smoke tests.' }
    })

    first.unmount()
    renderClarify(<ClarifyTool {...props} />)

    expect(screen.getByDisplayValue('Use staging first, then promote to production after smoke tests.')).toBeTruthy()
  })
})

describe('ClarifyTool settled view', () => {
  it('keeps the question and answer visible after the tool completes', () => {
    renderClarify(
      <ClarifyTool
        {...clarifyProps(
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
    expect(document.querySelector('[data-clarify-settled]')).toBeTruthy()
    expect(document.querySelector('[data-clarify-answer]')?.textContent).toBe('staging')
  })

  it('labels an empty response as Skipped', () => {
    renderClarify(
      <ClarifyTool
        {...clarifyProps({ question: 'Anything else?' }, { question: 'Anything else?', user_response: '' }, 'clarify-2')}
      />
    )

    expect(screen.getByText('Anything else?')).toBeTruthy()
    expect(screen.getByText('Skipped')).toBeTruthy()
  })
})
