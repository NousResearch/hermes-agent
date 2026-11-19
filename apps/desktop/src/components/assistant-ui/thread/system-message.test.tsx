import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { $displayTimestamps } from '@/store/display-timestamps'

import { stubThreadEnvironment } from '../test-utils'

import { Thread } from '.'

// Timeline timestamps render only when `display.timestamps` is enabled.
$displayTimestamps.set(true)

const timestamp = new Date('2026-05-01T00:00:00.000Z')
stubThreadEnvironment()

function Harness({ text, asyncResult, profile }: { text: string; asyncResult?: string; profile?: string }) {
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
    <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, profile }}>
      <AssistantRuntimeProvider runtime={runtime}>
        <Thread />
      </AssistantRuntimeProvider>
    </ComposerScopeProvider>
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

describe('pending skill diff rendering', () => {
  it('renders a /skills diff response through the Markdown diff panel', () => {
    const output = [
      '# Pending skill write f16a8190: batch(1 ops: patch) on systematic-debugging',
      '',
      "## op 1/1: patch 'systematic-debugging' SKILL.md (+1/-1 lines)",
      '',
      '--- a/SKILL.md',
      '+++ b/SKILL.md',
      '@@ -44,1 +44,1 @@',
      ' before',
      '-old instruction',
      '+new instruction'
    ].join('\n')

    const { container } = render(<Harness profile="default-2" text={`slash:/skills\n${output}`} />)
    const panel = container.querySelector('[data-slot="file-diff-panel"]')

    expect(panel).not.toBeNull()
    expect(container.querySelector('[data-slot="pending-skill-profile"]')?.textContent).toContain('default-2')
    expect(panel?.textContent).toContain('old instruction')
    expect(panel?.textContent).toContain('new instruction')
    expect(panel?.className).toContain('overflow-auto')
  })

  it('labels an empty quoted skill name in a patch operation heading', () => {
    const output = [
      '# Pending skill write abc12345: batch(1 ops: patch)',
      '',
      "## op 1/1: patch '' SKILL.md (+1/-1 lines)",
      '',
      '--- a/SKILL.md',
      '+++ b/SKILL.md',
      '@@ -1,1 +1,1 @@',
      '-old instruction',
      '+new instruction'
    ].join('\n')

    const { container } = render(<Harness profile="default-2" text={`slash:/skills\n${output}`} />)
    const row = container.querySelector('[data-role="system"]')

    expect(row?.textContent).toContain('patch [missing skill name] SKILL.md (+1/-1 lines)')
    expect(row?.textContent).not.toContain("patch '' SKILL.md")
    expect(container.querySelector('[data-slot="file-diff-panel"]')?.textContent).toContain('new instruction')
  })

  it('renders a profile-scoped pending list with readable IDs and preserves command guidance', () => {
    const output = [
      'Pending skills writes (2):',
      '',
      '328756a9  batch(1 ops: create) on wildlife-card-prompt-engineering',
      'f16a8190  [auto] batch(1 ops: patch) on systematic-debugging',
      '',
      'Apply: /skills approve <id>   Reject: /skills reject <id>'
    ].join('\n')

    const { container } = render(<Harness profile="default-2" text={`slash:/skills\n${output}`} />)

    expect(container.querySelector('[data-slot="pending-skill-profile"]')?.textContent).toContain('default-2')
    expect(container.querySelectorAll('[data-slot="pending-skill-entry"]')).toHaveLength(2)
    expect(container.querySelector('[data-slot="pending-skill-entry"]')?.textContent).toContain('328756a9')
    expect(container.querySelectorAll('[data-slot="pending-skill-entry"]')[1]?.textContent).toContain('auto')
    expect(container.textContent).toContain('/skills approve <id>')
    expect(container.querySelector('[data-slot="file-diff-panel"]')).toBeNull()
  })

  it('keeps other slash responses in their existing compact rendering', () => {
    const { container } = render(<Harness text={'slash:/skills diff absent-id\nNo pending skill write with id'} />)

    expect(container.querySelector('[data-slot="file-diff-panel"]')).toBeNull()
    expect(container.querySelector('[data-role="system"] .whitespace-pre-wrap')?.textContent).toContain(
      'No pending skill write'
    )
  })
})
