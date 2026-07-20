import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import type { GeneratedViewDocument } from '../generated-views/store'

import { GeneratedViewsSection } from './index'

const usageMonitor: GeneratedViewDocument = {
  connectionKey: 'local:',
  digest: 'a'.repeat(64),
  directory: '/Users/me/.hermes/generated-views/usage-monitor',
  entryPath: '/Users/me/.hermes/generated-views/usage-monitor/index.html',
  html: '<h1>Usage monitor</h1>',
  manifest: {
    bindings: ['hermes:status', 'hermes:usage-30d'],
    capabilities: ['theme:read', 'state:persist'],
    entry: 'index.html',
    id: 'usage-monitor',
    title: 'Usage Monitor',
    version: 1
  },
  manifestPath: '/Users/me/.hermes/generated-views/usage-monitor/view.json'
}

describe('GeneratedViewsSection', () => {
  afterEach(cleanup)

  it('explains view authority and opens the selected pane contribution', () => {
    const onOpen = vi.fn()

    render(
      <I18nProvider configClient={null}>
        <GeneratedViewsSection onOpen={onOpen} views={[usageMonitor]} />
      </I18nProvider>
    )

    expect(screen.getByText('Agent-authored views')).not.toBeNull()
    expect(screen.getByText('Usage Monitor')).not.toBeNull()
    expect(screen.getByText('usage-monitor')).not.toBeNull()
    expect(screen.getByText('theme:read, state:persist')).not.toBeNull()
    expect(screen.getByText('hermes:status, hermes:usage-30d')).not.toBeNull()
    expect(screen.getByText('sha256:aaaaaaaa')).not.toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Open view' }))
    expect(onOpen).toHaveBeenCalledWith('usage-monitor')
  })
})
