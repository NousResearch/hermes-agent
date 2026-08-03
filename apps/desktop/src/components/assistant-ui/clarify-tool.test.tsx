import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { onComposerInsertRequest } from '@/app/chat/composer/focus'
import { I18nProvider } from '@/i18n'

import { ClarifyTool, readClarifyResult } from './clarify-tool'

afterEach(() => {
  cleanup()
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
    respondToApproval: vi.fn(),
    result,
    resume: vi.fn(),
    status: { type: 'complete' },
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
    expect(document.querySelector('[data-clarify-settled]')).toBeTruthy()
    expect(document.querySelector('[data-clarify-answer]')?.textContent).toBe('staging')
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

  it('keeps the original choices visible and clickable after a skip', async () => {
    const inserts: string[] = []

    const stop = onComposerInsertRequest(detail => {
      inserts.push(detail.text)
    })

    try {
      renderClarify(
        <ClarifyTool
          {...settledClarifyProps(
            { question: 'Which deployment target?', choices: ['staging', 'prod'] },
            { question: 'Which deployment target?', user_response: '' },
            'clarify-3'
          )}
        />
      )

      // The skip label renders AND the original options are still on screen.
      expect(screen.getByText('Skipped')).toBeTruthy()
      const group = document.querySelector('[data-clarify-late-choices]')
      expect(group).toBeTruthy()
      expect(screen.getByText('staging')).toBeTruthy()
      expect(screen.getByText('prod')).toBeTruthy()

      // Picking one drafts a quoted follow-up into the composer. The insert
      // bus defers dispatch by a macrotask, so flush one tick.
      fireEvent.click(screen.getByText('prod'))
      await new Promise(resolve => window.setTimeout(resolve, 0))

      expect(inserts).toHaveLength(1)
      expect(inserts[0]).toContain('Which deployment target?')
      expect(inserts[0]).toContain('prod')
    } finally {
      stop()
    }
  })

  it('does not render late choices on an answered clarify', () => {
    renderClarify(
      <ClarifyTool
        {...settledClarifyProps(
          { question: 'Which deployment target?', choices: ['staging', 'prod'] },
          { question: 'Which deployment target?', user_response: 'staging' },
          'clarify-4'
        )}
      />
    )

    expect(document.querySelector('[data-clarify-late-choices]')).toBeNull()
  })

  it('does not render late choices for a free-text (no-choice) skip', () => {
    renderClarify(
      <ClarifyTool
        {...settledClarifyProps(
          { question: 'Anything else?' },
          { question: 'Anything else?', user_response: '' },
          'clarify-5'
        )}
      />
    )

    expect(document.querySelector('[data-clarify-late-choices]')).toBeNull()
  })
})
