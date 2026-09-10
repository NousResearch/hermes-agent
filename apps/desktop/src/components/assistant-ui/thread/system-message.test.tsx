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

function Harness({ text, asyncResult }: { text: string; asyncResult?: string }) {
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

describe('slash report presentation', () => {
  it('renders headed reports without changing escaped values or treating provider labels as markup', async () => {
    const output = [
      '## Analysis receipt',
      'Status: ANALYSIS\\_ONLY\\_COMPLETE',
      'Decision: NO\\_BET; reason: edge\\_insufficient',
      'Report: /tmp/home\\_vs\\_away/report\\_v38.json',
      'Team: \\<script\\> FC \\*United\\*',
      '',
      '### Markets',
      '- H2H: SILVER',
      '- Totals: BET\\_CANDIDATE',
      '',
      '`literal\\_value`'
    ].join('\n')

    const { container, findByRole } = render(<Harness text={`slash:/report\n${output}`} />)

    await findByRole('heading', { name: 'Analysis receipt', level: 2 })
    await findByRole('heading', { name: 'Markets', level: 3 })
    const row = container.querySelector('[data-role="system"]')!

    expect(row.textContent).toContain('ANALYSIS_ONLY_COMPLETE')
    expect(row.textContent).toContain('NO_BET; reason: edge_insufficient')
    expect(row.textContent).toContain('/tmp/home_vs_away/report_v38.json')
    expect(row.textContent).toContain('Team: <script> FC *United*')
    expect(row.querySelector('script')).toBeNull()
    expect(row.querySelector('em')).toBeNull()
    expect(row.querySelectorAll('li')).toHaveLength(2)
    expect(row.querySelector('code')?.textContent).toBe('literal\\_value')
    expect(row.textContent).not.toContain('## Analysis receipt')
  })

  it('preserves unheaded status tables and literal punctuation as plain text', () => {
    const output = 'Role      State\nworker_a  NO_BET\nworker_b  *idle*\npath      C:\\reports\\_current'
    const { container } = render(<Harness text={`slash:/status\n${output}`} />)
    const row = container.querySelector('[data-role="system"]')!

    expect(row.textContent).toContain(output)
    expect(row.querySelector('h1, h2, em, strong, table')).toBeNull()
    expect(row.querySelector('.whitespace-pre-wrap')?.textContent).toBe(output)
  })
})
