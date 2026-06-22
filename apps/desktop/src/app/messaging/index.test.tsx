import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { MessagingPlatformInfo } from '@/types/hermes'

import { PlatformDetail } from './index'

function platform(patch: Partial<MessagingPlatformInfo> = {}): MessagingPlatformInfo {
  return {
    configured: false,
    description: 'Connect Hermes to Tencent Yuanbao.',
    docs_url: '',
    enabled: false,
    env_vars: [],
    gateway_running: true,
    id: 'yuanbao',
    name: 'Yuanbao (元宝)',
    state: 'not_configured',
    ...patch
  }
}

function renderPlatformDetail(value: MessagingPlatformInfo) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <PlatformDetail
        edits={{}}
        onClear={vi.fn()}
        onEdit={vi.fn()}
        onSave={vi.fn()}
        onToggle={vi.fn()}
        platform={value}
        saving={null}
      />
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
})

describe('PlatformDetail', () => {
  it('hides the setup guide button when no docs URL is configured', () => {
    renderPlatformDetail(platform({ docs_url: '   ' }))

    expect(screen.queryByRole('link', { name: 'Open setup guide' })).toBeNull()
  })

  it('renders the setup guide button with the configured docs URL', () => {
    renderPlatformDetail(
      platform({ docs_url: 'https://hermes-agent.nousresearch.com/docs/user-guide/messaging/yuanbao/' })
    )

    const link = screen.getByRole('link', { name: 'Open setup guide' })
    expect(link.getAttribute('href')).toBe('https://hermes-agent.nousresearch.com/docs/user-guide/messaging/yuanbao/')
  })
})
