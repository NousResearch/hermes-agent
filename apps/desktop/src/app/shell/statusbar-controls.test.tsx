import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it } from 'vitest'

import { StatusbarControls } from './statusbar-controls'

describe('StatusbarControls', () => {
  afterEach(() => {
    cleanup()
  })

  it('applies item titles to truncated action and text controls', () => {
    render(
      <MemoryRouter>
        <StatusbarControls
          items={[
            {
              id: 'model',
              label: 'Grok Composer 2.5 · F...',
              title: 'Model · xai: grok/composer-2.5-fast',
              variant: 'action'
            }
          ]}
          leftItems={[
            {
              id: 'session',
              label: 'Session name that may truncate',
              title: 'Session name that may truncate',
              variant: 'text'
            }
          ]}
        />
      </MemoryRouter>
    )

    expect(screen.getByRole('button', { name: 'Grok Composer 2.5 · F...' }).getAttribute('title')).toBe(
      'Model · xai: grok/composer-2.5-fast'
    )
    expect(screen.getByText('Session name that may truncate').closest('div')?.getAttribute('title')).toBe(
      'Session name that may truncate'
    )
  })

  it('applies item titles to menu and link controls', async () => {
    render(
      <MemoryRouter>
        <StatusbarControls
          items={[
            {
              id: 'menu-model',
              label: 'Grok Composer 2.5 · F...',
              menuItems: [{ id: 'change', label: 'Change model', title: 'Change the active model' }],
              title: 'Model · xai: grok/composer-2.5-fast',
              variant: 'menu'
            },
            {
              href: 'https://example.test',
              id: 'docs',
              label: 'Docs',
              title: 'Open documentation',
              variant: 'link'
            }
          ]}
        />
      </MemoryRouter>
    )

    expect(screen.getByRole('button', { name: 'Grok Composer 2.5 · F...' }).getAttribute('title')).toBe(
      'Model · xai: grok/composer-2.5-fast'
    )
    expect(screen.getByRole('link', { name: 'Docs' }).getAttribute('title')).toBe('Open documentation')

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Grok Composer 2.5 · F...' }), {
      button: 0,
      ctrlKey: false
    })

    expect((await screen.findByText('Change model')).closest('[role="menuitem"]')?.getAttribute('title')).toBe(
      'Change the active model'
    )
  })
})
