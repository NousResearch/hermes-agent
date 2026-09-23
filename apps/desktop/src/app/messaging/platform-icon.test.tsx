import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { PlatformAvatar } from './platform-icon'

const PLUGIN_BACKED_PLATFORMS = [
  { id: 'google_chat', name: 'Google Chat' },
  { id: 'irc', name: 'IRC' },
  { id: 'line', name: 'LINE' },
  { id: 'ntfy', name: 'ntfy' },
  { id: 'raft', name: 'Raft' },
  { id: 'simplex', name: 'SimpleX Chat' },
  { id: 'teams', name: 'Microsoft Teams' }
] as const

describe('plugin-backed platform icons (#79836)', () => {
  it.each(PLUGIN_BACKED_PLATFORMS)('$id uses a registered platform treatment', ({ id, name }) => {
    const { container } = render(<PlatformAvatar platformId={id} platformName={name} />)
    const avatar = container.firstElementChild

    expect(avatar).toBeInstanceOf(HTMLSpanElement)
    expect(avatar?.classList.contains('bg-(--ui-bg-tertiary)')).toBe(false)
  })
})
