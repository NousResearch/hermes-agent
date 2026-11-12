import { cleanup, render, screen } from '@testing-library/react'
import type { ComponentProps, ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

vi.mock('@hermes/plugin-sdk', async () => {
  const { Streamdown } = await import('streamdown')
  const { en } = await import('@/i18n/en')
  const { relativeTime } = await import('@/lib/time')

  const MessageTextContent = ({ text }: { text: string }) => <div data-testid="message-text"><Streamdown>{text}</Streamdown></div>

  return { ...await (await import('./group-test-utils')).pluginSdkMock({}), MessageTextContent, Streamdown, relativeTime, useI18n: () => ({ locale: 'en', t: en }),
    Button: (props: ComponentProps<'button'>) => <button {...props} />,
    Codicon: () => <span />, Tip: ({ children }: { children: ReactNode }) => <>{children}</> }
})
vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { useCanonicalGroupLabels: () => ({ ...CANONICAL_GROUP_LOCALES.en, download: 'Download' }) }
})

import { CanonicalGroupHistory } from './canonical-group-history'

const binding = { connectionId: 'local', profile: 'reviewer', roomId: 'room-one' }

afterEach(cleanup)

it('marks only gateway-minted cron posts as Scheduled, under the job name', () => {
  const events = [
    { seq: 1, kind: 'message.user', actor: { kind: 'user', id: 'cron:job-1', display_name: 'Nightly digest' }, payload: { text: 'Digest body' } },
    { seq: 2, kind: 'message.user', actor: { kind: 'user', id: 'uid:42', display_name: 'cron:job-1' }, payload: { text: 'Lookalike' } }
  ]

  render(<CanonicalGroupHistory binding={binding} events={events} members={[]} />)

  expect(screen.getByText('Nightly digest')).toBeTruthy()
  expect(screen.getAllByText('Scheduled')).toHaveLength(1)
  const digest = screen.getByText('Digest body').closest('[data-room-message]')
  expect(digest?.textContent).toContain('Scheduled')
})
