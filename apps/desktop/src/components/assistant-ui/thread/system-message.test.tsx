import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $displayTimestamps } from '@/store/display-timestamps'

import { stubThreadEnvironment } from '../test-utils'

import { Thread } from '.'

// Timeline timestamps render only when `display.timestamps` is enabled.
$displayTimestamps.set(true)

const timestamp = new Date('2026-05-01T00:00:00.000Z')
stubThreadEnvironment()

function Harness({
  text,
  asyncResult,
  adviser
}: {
  text: string
  asyncResult?: string
  adviser?: { severity: string; label: string; body: string; items: { label: string; severity: string; text: string }[] }
}) {
  const message = {
    id: 'system-1',
    role: 'system',
    content: [{ type: 'text', text }],
    createdAt: timestamp,
    metadata: { custom: { timelineTimestamp: timestamp.getTime() / 1000, asyncResult, adviser } }
  } as unknown as ThreadMessage

  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [message],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

function expectTimestampSeparated(container: HTMLElement, precedingText: string) {
  const row = container.querySelector('[data-role="system"]')
  const stamp = row?.querySelector('[data-slot="timeline-timestamp"]')?.textContent

  expect(stamp).toBeTruthy()
  expect(row?.textContent).toContain(`${precedingText} ${stamp}`)
}

afterEach(cleanup)

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

describe('adviser card', () => {
  it('renders the label and advice as a card, not a user prompt', () => {
    const { container } = render(
      <Harness
        adviser={{
          severity: 'blocker',
          label: 'blocker · tool_choice · Architecture',
          body: 'stop **now** and re-read the diff',
          items: []
        }}
        text="[adviser — advisory only] stop now [end adviser]"
      />
    )

    const card = container.querySelector('[data-slot="aui_adviser-message-root"]')
    expect(card).toBeTruthy()
    expect(card?.getAttribute('data-severity')).toBe('blocker')
    expect(card?.textContent).toContain('Adviser')
    expect(card?.textContent).toContain('blocker · tool_choice · Architecture')
    expect(card?.textContent).toContain('stop now and re-read the diff')
    expect(card?.textContent).not.toContain('advisory only')
  })

  it('paints each advice row with its own severity in a mixed block', () => {
    const { container } = render(
      <Harness
        adviser={{
          severity: 'blocker',
          label: 'blocker · ai_self_representation_check · default',
          body: '',
          items: [
            { label: 'concern · tool_choice · default', severity: 'concern', text: 'read the file first' },
            { label: 'blocker · ai_self_representation_check · default', severity: 'blocker', text: 'the test never ran' }
          ]
        }}
        text="[adviser — advisory only] x [end adviser]"
      />
    )

    const rows = container.querySelectorAll('[data-slot="aui_adviser-message-root"] div[data-severity]')
    expect(rows).toHaveLength(2)
    expect(rows[0]?.getAttribute('data-severity')).toBe('concern')
    expect(rows[1]?.getAttribute('data-severity')).toBe('blocker')
    expect(rows[0]?.textContent).toContain('concern · tool_choice · default')
    expect(rows[1]?.textContent).toContain('the test never ran')
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

describe('slash output face', () => {
  const body = (container: HTMLElement) => container.querySelector('[data-role="system"] .whitespace-pre-wrap')

  it('keeps column-padded output without a plan gauge in the default face', () => {
    const { container } = render(
      <Harness text={'slash:/usage\nSession Token Usage\nInput tokens:                 13,527'} />
    )

    expect(body(container)?.className).not.toContain('font-mono')
  })

  it('renders a usage-window /usage block (plan gauges) monospace', () => {
    const output =
      'Session Token Usage\nInput tokens:                 13,527\n\nNous Plan\nToday  [||        ]  20% used'
    const { container } = render(<Harness text={`slash:/usage\n${output}`} />)

    expect(body(container)?.className).toContain('font-mono')
  })
})
