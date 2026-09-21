import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { type ReactNode, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import {
  TRANSCRIPT_MESSAGE_AREA,
  type TranscriptMessageContribution,
  type TranscriptMessageProps
} from '@/lib/transcript-message'
import { $displayTimestamps } from '@/store/display-timestamps'
import { $activeSessionId } from '@/store/session'

import { stubThreadEnvironment } from '../test-utils'

import { Thread } from '.'

// Timeline timestamps render only when `display.timestamps` is enabled.
$displayTimestamps.set(true)

const timestamp = new Date('2026-05-01T00:00:00.000Z')
stubThreadEnvironment()

function Harness({
  text,
  asyncResult,
  sessionId = null
}: {
  text: string
  asyncResult?: string
  sessionId?: string | null
}) {
  const message = {
    id: 'system-1',
    role: 'system',
    content: [{ type: 'text', text }],
    createdAt: timestamp,
    metadata: { custom: { timelineTimestamp: timestamp.getTime() / 1000, asyncResult } }
  } as unknown as ThreadMessage

  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [message],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread sessionId={sessionId} />
    </AssistantRuntimeProvider>
  )
}

function expectTimestampSeparated(container: HTMLElement, precedingText: string) {
  const row = container.querySelector('[data-role="system"]')
  const stamp = row?.querySelector('[data-slot="timeline-timestamp"]')?.textContent

  expect(stamp).toBeTruthy()
  expect(row?.textContent).toContain(`${precedingText} ${stamp}`)
}

const disposers: Array<() => void> = []

function addSlashResult(
  id: string,
  match: (props: TranscriptMessageProps) => boolean,
  renderResult: (props: TranscriptMessageProps) => ReactNode,
  order?: number
) {
  disposers.push(
    registry.register({
      id,
      area: TRANSCRIPT_MESSAGE_AREA,
      data: { match },
      order,
      render: renderResult
    } satisfies TranscriptMessageContribution)
  )
}

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  $activeSessionId.set(null)
  vi.restoreAllMocks()
})

describe('slash-result contributions', () => {
  it('renders the first ordered match inside the system message root with stable containing-thread identity', () => {
    $activeSessionId.set('focused-owner')
    const seen: TranscriptMessageProps[] = []
    let mounts = 0

    function IdentityCard(props: TranscriptMessageProps) {
      const [mount] = useState(() => {
        mounts += 1

        return mounts
      })

      seen.push(props)

      return <div data-testid="slash-card">inline card {mount}</div>
    }

    addSlashResult(
      'later',
      () => true,
      () => <div>later</div>,
      20
    )
    addSlashResult('winner', props => props.command === '/wisdom browse featured', IdentityCard, 10)

    const view = render(
      <Harness sessionId="containing-thread" text={'slash:/wisdom browse featured\nexisting output'} />
    )

    const card = screen.getByTestId('slash-card')
    expect(card.closest('[data-slot="aui_system-message-root"]')).not.toBeNull()
    expect(screen.queryByText('later')).toBeNull()
    expect(seen.at(-1)).toEqual({
      command: '/wisdom browse featured',
      isLast: true,
      kind: 'slash-result',
      messageId: 'system-1',
      output: 'existing output',
      sessionId: 'containing-thread'
    })

    $activeSessionId.set('other-focused-owner')
    view.rerender(
      <Harness sessionId="other-containing-thread" text={'slash:/wisdom browse featured\nexisting output'} />
    )

    expect(mounts).toBe(2)
    expect(seen.at(-1)?.sessionId).toBe('other-containing-thread')
  })

  it('keeps ordinary slash text when no contribution matches or matching fails', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    addSlashResult(
      'no-match',
      () => false,
      () => <div>wrong card</div>
    )
    addSlashResult(
      'broken-match',
      () => {
        throw new Error('broken matcher')
      },
      () => <div>broken card</div>
    )

    const { container } = render(<Harness text={'slash:/model fast\nmodel changed'} />)

    expect(container.textContent).toContain('/model fast')
    expect(container.textContent).toContain('model changed')
    expect(screen.queryByText('wrong card')).toBeNull()
    expect(screen.queryByText('broken card')).toBeNull()
  })

  it('falls back to ordinary slash text when a matched renderer throws', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    addSlashResult(
      'broken-render',
      () => true,
      () => {
        throw new Error('broken renderer')
      }
    )

    const { container, rerender } = render(<Harness text={'slash:/model fast\nmodel changed'} />)

    expect(container.textContent).toContain('/model fast')
    expect(container.textContent).toContain('model changed')

    disposers.splice(0).forEach(dispose => dispose())
    addSlashResult(
      'healthy-render',
      () => true,
      () => <div>replacement card</div>
    )
    rerender(<Harness text={'slash:/model fast\nmodel changed'} />)
    expect(screen.getByText('replacement card')).toBeTruthy()
  })

  it('removes the renderer when its registration is disposed', () => {
    addSlashResult(
      'temporary',
      () => true,
      () => <div>temporary card</div>
    )
    const { container, rerender } = render(<Harness text={'slash:/wisdom browse\nnative output'} />)

    expect(screen.getByText('temporary card')).toBeTruthy()
    disposers.splice(0).forEach(dispose => dispose())
    rerender(<Harness text={'slash:/wisdom browse\nnative output'} />)

    expect(screen.queryByText('temporary card')).toBeNull()
    expect(container.textContent).toContain('native output')
  })
})

describe('background report inline code', () => {
  // The report body is the same `aui-md prose` markdown renderer the
  // assistant turn uses, rendered with no assistant/room slot ancestor. The
  // real stylesheet's cascade must still reach its `<code>`, or Tailwind
  // Typography's fixed near-black ink wins on every dark theme (#107486).
  const stylesheet = readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), '../../../styles.css'), 'utf8')

  it('themes inline code in an opened report with the chat inline-code tokens', () => {
    const { container, getByRole } = render(
      <>
        <style>{stylesheet}</style>
        <Harness asyncResult="run `discover_models` first" text="1 background agent finished" />
      </>
    )

    fireEvent.click(getByRole('button', { name: '1 background agent finished' }))
    const code = container.querySelector('[data-role="system"] :not(pre) > code')
    expect(code).toBeTruthy()
    const style = getComputedStyle(code as Element)

    expect(style.color).toBe('var(--ui-inline-code-foreground)')
    expect(style.background).toBe('var(--ui-inline-code-background)')
  })
})

describe('background report disclosure', () => {
  it('keeps result bodies out of the transcript until opened and removes them when collapsed', () => {
    const report = '{"blockers":[{"title":"Local-model readiness uses the wrong endpoint"}]}'
    const { container, getByRole } = render(<Harness asyncResult={report} text="2 background agents finished" />)

    expect(container.textContent).not.toContain('blockers')
    expectTimestampSeparated(container, '2 background agents finished')
    const toggle = getByRole('button', { name: '2 background agents finished' })
    expect(toggle.getAttribute('aria-expanded')).toBe('false')

    fireEvent.click(toggle)
    expect(toggle.getAttribute('aria-expanded')).toBe('true')
    expect(container.textContent).toContain(report)

    fireEvent.click(toggle)
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    expect(container.textContent).not.toContain('blockers')
  })
})

describe('system message timestamp text separation', () => {
  it('separates an ordinary system row timestamp in accessible and copied text', () => {
    const { container } = render(<Harness text="Review saved." />)

    expectTimestampSeparated(container, 'Review saved.')
  })

  it('separates a slash-status timestamp in accessible and copied text', () => {
    const { container } = render(<Harness text={'slash:/model\nmodel changed'} />)

    expectTimestampSeparated(container, 'model changed')
  })

  it('separates a steer timestamp in accessible and copied text', () => {
    const { container } = render(<Harness text="steer:rerun tests" />)

    expectTimestampSeparated(container, 'rerun tests')
  })
})
