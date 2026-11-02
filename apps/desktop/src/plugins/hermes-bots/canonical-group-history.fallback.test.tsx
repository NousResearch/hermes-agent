import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

// An SDK without MessageTextContent (older shells) still exports Streamdown, which renders links,
// images and fences with none of previewOnly's limits.
const streamdown = vi.hoisted(() => vi.fn(({ children }: { children: ReactNode }) => <div data-testid="streamdown">{children}</div>))
vi.mock('@hermes/plugin-sdk', () => ({ Button: () => null, MessageTextContent: undefined, Streamdown: streamdown, useI18n: () => ({ locale: 'en', t: {} }) }))
vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { useCanonicalGroupLabels: () => ({ ...CANONICAL_GROUP_LOCALES.en, you: 'You' }) }
})

import { CanonicalGroupHistory } from './canonical-group-history'

afterEach(cleanup)

it('shows gateway room text as plain text when the shell has no message renderer', () => {
  const text = 'See [the panel](http://127.0.0.1:9119/) and ![x](/home/user/.ssh/id_rsa)'

  render(<CanonicalGroupHistory binding={{ connectionId: 'local', profile: 'default', roomId: 'room-1' }}
    events={[{ seq: 1, room_id: 'room-1', kind: 'message.member', payload: { text }, actor: { kind: 'member', id: 'alpha' } }]} />)

  expect(screen.getByText(text)).toBeTruthy()
  expect(streamdown).not.toHaveBeenCalled()
})
