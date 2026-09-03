import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $displayTimestamps } from '@/store/display-timestamps'

import { stubThreadEnvironment } from '../test-utils'

import { Thread } from '.'

// Timeline timestamps render only when `display.timestamps` is enabled.
$displayTimestamps.set(true)

const timestamp = new Date('2026-05-01T00:00:00.000Z')
stubThreadEnvironment()

const securityCheck = {
  schema_version: 1,
  status: 'pass',
  summary: 'No known matches detected.',
  checks: [
    {
      key: 'private_keys',
      label: 'Private keys',
      status: 'pass',
      finding_count: 0,
      details: []
    }
  ]
}

const professionalismCheck = {
  schema_version: 1,
  status: 'advisory',
  summary: 'One phrase may read as spam-like.',
  checks: [
    {
      key: 'manipulative_or_spam',
      status: 'advisory',
      finding_count: 1,
      details: ['Avoid urgency-based promotional wording.']
    }
  ]
}

function Harness({ text }: { text: string }) {
  const message = {
    id: 'system-1',
    role: 'system',
    content: [{ type: 'text', text }],
    createdAt: timestamp,
    metadata: { custom: { timelineTimestamp: timestamp.getTime() / 1000 } }
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

afterEach(() => {
  cleanup()
  Reflect.deleteProperty(window, 'hermesDesktop')
  window.location.hash = ''
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

  it('renders Wisdom slash output as a readable command result', () => {
    const { container } = render(
      <Harness text={'slash:/wisdom\nCollective Wisdom commands\n\n/wisdom browse — Search team skills'} />
    )
    const row = container.querySelector('[data-role="system"]')

    expect(row?.className).toContain('w-[min(92%,56rem)]')
    expect(row?.className).toContain('text-(--ui-text-secondary)')
    expect(row?.textContent).toContain('Collective Wisdom commands')
    expect(row?.textContent).toContain('/wisdom browse — Search team skills')
  })

  it('opens Wisdom browse results in an in-app skill preview', async () => {
    const openExternal = vi.fn().mockResolvedValue(undefined)
    const api = vi.fn().mockImplementation(({ path }: { path: string }) => {
      if (path.endsWith('/versions/1')) {
        return Promise.resolve({
          local_compatibility: { outcome: 'compatible' },
          portal_url: 'https://portal.example/orgs/team/wisdom/skills/skill-1?version=1',
          skill: { id: 'skill-1', slug: 'collective-wisdom-canary' },
          version: {
            author_description: 'The first canary release.',
            commit: 'sha256:commit-1',
            content_hash: 'sha256:content-1',
            explanation: 'Validated as a dependency-free canary.',
            package_manifest_hash: 'sha256:manifest-1',
            professionalism_check: professionalismCheck,
            published_at: '2026-01-02T03:04:05Z',
            scan: { verdict: 'pass', findings: [] },
            security_check: securityCheck,
            system_spec: {
              hermes: { minimum_version: '0.20.5' },
              platforms: ['macOS'],
              runtime: { shell: true }
            },
            verified_facts: { scan_verdict: 'pass' },
            version: 1
          }
        })
      }

      return Promise.resolve({
        latest_version_detail: {
          version: {
            author_description: 'Verify the Collective Wisdom canary flow.',
            professionalism_check: professionalismCheck,
            scan: { verdict: 'pass' },
            security_check: securityCheck,
            system_spec: {
              hermes: { minimum_version: '0.20.5' },
              platforms: ['macOS'],
              runtime: { shell: true }
            },
            version: 2
          }
        },
        local_compatibility: { outcome: 'compatible' },
        portal_url: 'https://portal.example/orgs/team/wisdom/skills/skill-1',
        skill: { id: 'skill-1', slug: 'collective-wisdom-canary' },
        versions: [
          {
            author_description: 'Current canary release.',
            professionalism_check: professionalismCheck,
            published_at: '2026-02-03T04:05:06Z',
            security_check: securityCheck,
            verified_facts: { scan_verdict: 'pass' },
            version: 2
          },
          {
            author_description: 'The first canary release.',
            professionalism_check: professionalismCheck,
            published_at: '2026-01-02T03:04:05Z',
            security_check: securityCheck,
            verified_facts: { scan_verdict: 'pass' },
            version: 1
          }
        ]
      })
    })

    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api, openExternal } })
    const { container } = render(
      <Harness
        text={
          'slash:/wisdom browse\nShared skills\n\ncollective-wisdom-canary\nv2 · Verify the canary.\nView: /wisdom show collective-wisdom-canary'
        }
      />
    )

    const preview = screen.getByRole('button', { name: 'Preview: collective-wisdom-canary' })

    expect(container.textContent).not.toContain('/wisdom show collective-wisdom-canary')
    expect(container.textContent).not.toContain('View:')
    fireEvent.click(preview)

    const dialog = await screen.findByRole('dialog', { name: /collective-wisdom-canary/ })

    expect(dialog).toBeTruthy()
    expect(await screen.findByText('Verify the Collective Wisdom canary flow.')).toBeTruthy()
    expect(screen.getByText('Hermes ≥ 0.20.5')).toBeTruthy()
    expect(screen.getByText('macOS')).toBeTruthy()
    expect(screen.getByText('shell')).toBeTruthy()
    expect(screen.getByText('compatible')).toBeTruthy()
    expect(within(dialog).getByRole('region', { name: 'Security check' })).toBeTruthy()
    expect(within(dialog).getByRole('region', { name: 'Professionalism check' })).toBeTruthy()
    expect(within(dialog).getByText('Private keys')).toBeTruthy()
    expect(within(dialog).getByText('Manipulative, deceptive, or spam-like wording')).toBeTruthy()
    expect(
      within(dialog).getByText('Agent-assessed and advisory. It does not block publication or installation.')
    ).toBeTruthy()
    expect(within(dialog).getByRole('button', { name: 'Versions' })).toBeTruthy()
    expect(within(dialog).getByRole('button', { name: 'View in Portal' })).toBeTruthy()
    await waitFor(() =>
      expect(api).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/wisdom/skills/collective-wisdom-canary' }))
    )

    fireEvent.click(within(dialog).getByRole('button', { name: 'View in Portal' }))
    expect(openExternal).toHaveBeenCalledWith('https://portal.example/orgs/team/wisdom/skills/skill-1')

    fireEvent.click(within(dialog).getByRole('button', { name: 'Versions' }))
    const firstVersion = within(dialog).getByRole('button', { name: /v1.*The first canary release/ })

    expect(firstVersion).toBeTruthy()
    fireEvent.click(firstVersion)

    expect(await within(dialog).findByText('Validated as a dependency-free canary.')).toBeTruthy()
    expect(within(dialog).getByText('sha256:content-1')).toBeTruthy()
    expect(within(dialog).getByText('sha256:manifest-1')).toBeTruthy()
    await waitFor(() =>
      expect(api).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/wisdom/skills/skill-1/versions/1' }))
    )

    fireEvent.click(within(dialog).getByRole('button', { name: 'View in Portal' }))
    expect(openExternal).toHaveBeenLastCalledWith('https://portal.example/orgs/team/wisdom/skills/skill-1?version=1')

    fireEvent.click(within(dialog).getByRole('button', { name: /^Install/ }))
    expect(window.location.hash).toBe('#/skills?tab=collective&wisdomAction=install&wisdomSkillId=skill-1%40v1')
  })

  it('keeps non-Wisdom multiline slash output on the compact system-row treatment', () => {
    const { container } = render(<Harness text={'slash:/status\nGateway status\nReady'} />)
    const row = container.querySelector('[data-role="system"]')

    expect(row?.className).toContain('w-[60%]')
    expect(row?.className).toContain('text-muted-foreground/60')
    expect(row?.className).not.toContain('w-[min(92%,56rem)]')
    expect(row?.className).not.toContain('text-(--ui-text-secondary)')
  })
})
