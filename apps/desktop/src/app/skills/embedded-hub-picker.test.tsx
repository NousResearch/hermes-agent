// @vitest-environment jsdom
import { render } from '@testing-library/react'
import type { ComponentProps, ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'

type MockButtonProps = Omit<ComponentProps<'button'>, 'size'> & {
  children?: ReactNode
  size?: string
  variant?: string
}

vi.mock('@nanostores/react', () => ({
  useStore: () => undefined
}))

vi.mock('@/components/ui/button', () => ({
  Button: ({ children, size: _size, variant: _variant, ...props }: MockButtonProps) => (
    <button {...props}>{children}</button>
  )
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      skills: {
        hub: {
          actionFailed: 'Action failed',
          actionLog: 'Action log',
          alreadyInstalled: (name: string) => `${name} is already installed`,
          installStarted: (name: string) => `Installing ${name}`,
          pickerBrowse: 'Browse',
          pickerHide: 'Hide',
          pickerHint: 'Browse skills',
          pickerTitle: 'Skills Hub',
          updateAll: 'Update all',
          updateStarted: 'Update started',
          updating: 'Updating'
        }
      }
    }
  })
}))

vi.mock('@/lib/icons', () => ({
  Loader2: () => <span />
}))

vi.mock('@/lib/use-session-slice', () => ({
  useStoreSelector: () => false
}))

vi.mock('@/store/hub-actions', () => ({
  $hubActions: {},
  installHubSkill: vi.fn(),
  UPDATE_ALL_KEY: 'update-all',
  updateHubSkills: vi.fn()
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/store/panes', () => ({
  $paneHeightOverride: () => ({}),
  setPaneHeightOverride: vi.fn()
}))

import { EmbeddedHubPicker } from './embedded-hub-picker'

describe('EmbeddedHubPicker iframe policy', () => {
  it('allows user-initiated hub links and clipboard writes without broader iframe privileges', () => {
    const { container } = render(<EmbeddedHubPicker installedNames={new Set<string>()} />)
    const iframe = container.querySelector('iframe')

    expect(iframe).not.toBeNull()
    expect(iframe?.getAttribute('allow')).toBe('clipboard-write')
    expect(iframe?.getAttribute('sandbox')).toBe(
      'allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox'
    )
  })
})
