import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { I18nProvider } from '@/i18n'
import { $clarifyRequests, setClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { $activeSessionId } from '@/store/session'

import { ClarifyTool, readClarifyResult } from './clarify-tool'

vi.mock('@assistant-ui/react', () => ({
  useAuiState: (selector: (state: unknown) => unknown) =>
    selector({ thread: { isRunning: true }, message: { status: { type: 'running' } } })
}))

afterEach(() => {
  cleanup()
  $clarifyRequests.set({})
  $gateway.set(null)
  $activeSessionId.set(null)
  vi.clearAllMocks()
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
    addResult: vi.fn(),
    args,
    argsText: JSON.stringify(args),
    isError: false,
    result: undefined,
    resume: vi.fn(),
    status: { type: 'running' },
    toolCallId,
    toolName: 'clarify',
    type: 'tool-call'
  }
}

function attachClarifyRequest(question: string) {
  const request = vi.fn().mockResolvedValue({ ok: true })

  $activeSessionId.set(null)
  setClarifyRequest({
    choices: null,
    question,
    requestId: 'clarify-request-1',
    sessionId: null
  })
  $gateway.set({ request } as unknown as HermesGateway)

  return request
}

const structuredArgs = {
  question: 'Which rollout should we use?',
  context: 'Production currently has no canary traffic.',
  recommendation: 'Choose Safe because it preserves rollback capacity.',
  choices: [
    {
      id: 'fast',
      label: 'Fast',
      description: 'Deploy to every instance immediately.',
      value: 'fast-rollout'
    },
    {
      id: 'safe',
      label: 'Safe',
      description: 'Canary first, then expand after verification.',
      value: 'safe-rollout'
    }
  ]
}

describe('readClarifyResult', () => {
  it('reads question + user_response from the tool JSON payload', () => {
    const result = readClarifyResult({
      question: 'Which target?',
      choices_offered: ['staging', 'prod'],
      user_response: 'staging'
    })

    expect(result).toMatchObject({
      question: 'Which target?',
      answer: 'staging',
      error: undefined
    })
    expect(result.options?.map(option => option.label)).toEqual(['staging', 'prod'])
  })

  it('parses a JSON string result the same way as an object', () => {
    expect(
      readClarifyResult(
        JSON.stringify({
          question: 'Ship it?',
          user_response: 'yes'
        })
      )
    ).toMatchObject({
      question: 'Ship it?',
      answer: 'yes',
      error: undefined
    })
  })

  it('keeps an empty user_response so Skip can render as skipped', () => {
    expect(readClarifyResult({ question: 'Ok?', user_response: '' })).toMatchObject({
      question: 'Ok?',
      answer: '',
      error: undefined
    })
  })

  it('reads structured result metadata', () => {
    const result = readClarifyResult({
      question: 'Which rollout should we use?',
      context: 'Production currently has no canary traffic.',
      recommendation: 'Choose Safe because it preserves rollback capacity.',
      options: structuredArgs.choices,
      selected_option: {
        id: 'safe',
        index: 2,
        label: 'Safe',
        description: 'Canary first, then expand after verification.',
        value: 'safe-rollout'
      },
      user_response: 'safe-rollout'
    })

    expect(result.context).toBe('Production currently has no canary traffic.')
    expect(result.recommendation).toBe('Choose Safe because it preserves rollback capacity.')
    expect(result.options?.[1]).toMatchObject({
      id: 'safe',
      index: 2,
      label: 'Safe',
      description: 'Canary first, then expand after verification.',
      value: 'safe-rollout'
    })
    expect(result.selectedOption).toMatchObject({
      id: 'safe',
      label: 'Safe',
      description: 'Canary first, then expand after verification.',
      value: 'safe-rollout'
    })
  })
})

describe('ClarifyTool pending view', () => {
  it('renders structured choices with numeric badges and sends the canonical value', async () => {
    const request = attachClarifyRequest(
      'Which rollout should we use?\n\nContext: Production currently has no canary traffic.'
    )

    renderClarify(<ClarifyTool {...pendingClarifyProps(structuredArgs, 'clarify-pending-1')} />)

    expect(screen.getByText('Which rollout should we use?')).toBeTruthy()
    expect(screen.getByText('Production currently has no canary traffic.')).toBeTruthy()
    expect(screen.getByText('Choose Safe because it preserves rollback capacity.')).toBeTruthy()
    expect(screen.getByText('Fast')).toBeTruthy()
    expect(screen.getByText('Deploy to every instance immediately.')).toBeTruthy()
    expect(screen.getByText('Safe')).toBeTruthy()
    expect(screen.getByText('Canary first, then expand after verification.')).toBeTruthy()
    expect([...globalThis.document.querySelectorAll('[data-slot="kbd"]')].map(node => node.textContent)).toEqual([
      '1',
      '2',
      '3'
    ])

    fireEvent.click(screen.getByRole('button', { name: /Safe/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }))

    await waitFor(() =>
      expect(request).toHaveBeenCalledWith('clarify.respond', {
        request_id: 'clarify-request-1',
        answer: 'safe-rollout',
        option_id: 'safe'
      })
    )
  })

  it('supports numeric choice shortcuts and Enter confirmation', async () => {
    const request = attachClarifyRequest('Which rollout should we use?')

    renderClarify(<ClarifyTool {...pendingClarifyProps(structuredArgs, 'clarify-pending-2')} />)

    fireEvent.keyDown(window, { key: '2' })
    fireEvent.keyDown(window, { key: 'Enter' })

    await waitFor(() =>
      expect(request).toHaveBeenCalledWith('clarify.respond', {
        request_id: 'clarify-request-1',
        answer: 'safe-rollout',
        option_id: 'safe'
      })
    )
  })

  it('focuses Other with the trailing numeric shortcut', () => {
    attachClarifyRequest('Which rollout should we use?')

    renderClarify(<ClarifyTool {...pendingClarifyProps(structuredArgs, 'clarify-pending-3')} />)

    fireEvent.keyDown(window, { key: '3' })

    expect(globalThis.document.activeElement).toBe(screen.getByPlaceholderText('Other (type your answer)'))
  })

  it('marks Other text as custom even when it matches an option label', async () => {
    const request = attachClarifyRequest('Which rollout should we use?')

    renderClarify(<ClarifyTool {...pendingClarifyProps(structuredArgs, 'clarify-pending-custom')} />)

    fireEvent.change(screen.getByPlaceholderText('Other (type your answer)'), {
      target: { value: 'Safe' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }))

    await waitFor(() =>
      expect(request).toHaveBeenCalledWith('clarify.respond', {
        request_id: 'clarify-request-1',
        answer: 'Safe',
        response_kind: 'custom'
      })
    )
  })

  it('renders at most four choices plus Other', () => {
    attachClarifyRequest('Pick one')

    renderClarify(
      <ClarifyTool
        {...pendingClarifyProps(
          {
            question: 'Pick one',
            choices: ['One', 'Two', 'Three', 'Four', 'Five']
          },
          'clarify-pending-max'
        )}
      />
    )

    expect(screen.getByText('Four')).toBeTruthy()
    expect(screen.queryByText('Five')).toBeNull()
    expect([...globalThis.document.querySelectorAll('[data-slot="kbd"]')].map(node => node.textContent)).toEqual([
      '1',
      '2',
      '3',
      '4',
      '5'
    ])
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
    expect(screen.getByText('prod')).toBeTruthy()
    expect(globalThis.document.querySelector('[data-clarify-settled]')).toBeTruthy()
    expect(globalThis.document.querySelector('[data-selected="true"]')?.textContent).toContain('staging')
    expect(globalThis.document.querySelector('[data-clarify-answer]')?.textContent).toBe('1. staging')
  })

  it('marks the selected structured option from selected_option', () => {
    renderClarify(
      <ClarifyTool
        {...settledClarifyProps(
          structuredArgs,
          {
            question: 'Which rollout should we use?',
            context: 'Production currently has no canary traffic.',
            recommendation: 'Choose Safe because it preserves rollback capacity.',
            options: structuredArgs.choices,
            selected_option: {
              id: 'safe',
              index: 2,
              label: 'Safe',
              description: 'Canary first, then expand after verification.',
              value: 'safe-rollout'
            },
            user_response: 'safe-rollout'
          },
          'clarify-structured-settled'
        )}
      />
    )

    expect(screen.getByText('Which rollout should we use?')).toBeTruthy()
    expect(screen.getByText('Production currently has no canary traffic.')).toBeTruthy()
    expect(screen.getByText('Choose Safe because it preserves rollback capacity.')).toBeTruthy()
    expect(screen.getByText('Fast')).toBeTruthy()
    expect(screen.getByText('Deploy to every instance immediately.')).toBeTruthy()
    expect(screen.getByText('Safe')).toBeTruthy()
    expect(screen.getByText('Canary first, then expand after verification.')).toBeTruthy()
    expect(globalThis.document.querySelector('[data-selected="true"]')?.textContent).toContain('Safe')
    expect(globalThis.document.querySelector('[data-clarify-answer]')?.textContent).toBe('2. Safe')
  })

  it('uses selected_option id before an ambiguous answer value', () => {
    const ambiguousOptions = [
      {
        id: 'first',
        index: 1,
        label: 'First',
        description: 'Shares a value.',
        value: 'shared'
      },
      {
        id: 'second',
        index: 2,
        label: 'Second',
        description: 'Was actually selected.',
        value: 'shared'
      }
    ]

    renderClarify(
      <ClarifyTool
        {...settledClarifyProps(
          { question: 'Which duplicate?', choices: ambiguousOptions },
          {
            question: 'Which duplicate?',
            options: ambiguousOptions,
            selected_option: ambiguousOptions[1],
            user_response: 'shared'
          },
          'clarify-ambiguous-value'
        )}
      />
    )

    expect(globalThis.document.querySelector('[data-clarify-option="first"]')?.getAttribute('data-selected')).toBeNull()
    expect(globalThis.document.querySelector('[data-clarify-option="second"]')?.getAttribute('data-selected')).toBe('true')
    expect(globalThis.document.querySelector('[data-clarify-answer]')?.textContent).toBe('2. Second')
  })

  it('keeps legacy user_response-only results readable', () => {
    renderClarify(
      <ClarifyTool
        {...settledClarifyProps(
          { question: 'Which deployment target?', choices: ['staging', 'prod'] },
          {
            question: 'Which deployment target?',
            user_response: 'staging'
          },
          'clarify-legacy-result'
        )}
      />
    )

    expect(screen.getByText('Which deployment target?')).toBeTruthy()
    expect(screen.getByText('staging')).toBeTruthy()
    expect(screen.getByText('prod')).toBeTruthy()
  })

  it('labels an empty response as Skipped', () => {
    renderClarify(
      <ClarifyTool
        {...settledClarifyProps({ question: 'Anything else?' }, { question: 'Anything else?', user_response: '' }, 'clarify-2')}
      />
    )

    expect(screen.getByText('Anything else?')).toBeTruthy()
    expect(screen.getByText('Skipped')).toBeTruthy()
  })
})
