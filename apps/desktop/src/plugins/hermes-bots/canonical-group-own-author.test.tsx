import { cleanup, render, screen } from '@testing-library/react'
import type { ComponentProps, ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

vi.mock('@hermes/plugin-sdk', async () => {
  const { Streamdown } = await import('streamdown')
  const { en } = await import('@/i18n/en')
  const { relativeTime } = await import('@/lib/time')

  // The shell's shared message renderer; media marks whether MEDIA: paths would resolve locally.
  const MessageTextContent = ({ media = true, text }: { media?: boolean; text: string }) => (
    <div data-media={String(media)} data-testid="message-text">
      <Streamdown>{text}</Streamdown>
    </div>
  )

  return {
    ...(await (await import('./group-test-utils')).pluginSdkMock({})),
    MessageTextContent,
    Streamdown,
    relativeTime,
    useI18n: () => ({ locale: 'en', t: en }),
    Button: (props: ComponentProps<'button'>) => <button {...props} />,
    Codicon: () => <span />,
    Tip: ({ children }: { children: ReactNode }) => <>{children}</>
  }
})
vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { useCanonicalGroupLabels: () => ({ ...CANONICAL_GROUP_LOCALES.en, download: 'Download' }) }
})

import { CanonicalGroupHistory } from './canonical-group-history'

const binding = { connectionId: 'local', profile: 'reviewer', roomId: 'room-one' }

afterEach(cleanup)

it('labels this Desktop’s own messages You and keeps every other user’s name or id', () => {
  const events = [
    { seq: 1, kind: 'message.user', actor: { kind: 'user', id: 'desktop' }, payload: { text: 'Mine' } },
    { seq: 2, kind: 'message.user', actor: { kind: 'user', id: 'uid:42' }, payload: { text: 'Theirs' } },
    {
      seq: 3,
      kind: 'message.user',
      actor: { kind: 'user', id: 'desktop', display_name: 'Morgan' },
      payload: { text: 'Named' }
    }
  ]

  render(<CanonicalGroupHistory binding={binding} events={events} members={[]} />)

  expect(screen.getByText('You')).toBeTruthy()
  expect(screen.getByText('uid:42')).toBeTruthy()
  expect(screen.getByText('Morgan')).toBeTruthy()
  expect(screen.queryByText('desktop')).toBeNull()
})
