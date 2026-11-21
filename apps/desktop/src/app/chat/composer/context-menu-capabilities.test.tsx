import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { capabilityScopeForChat, ContextMenu } from './context-menu'

vi.mock('./capability-menu', () => ({
  CapabilityMenu: ({ kind }: { kind: string }) => <output data-testid="capability-list">{kind}</output>
}))

afterEach(cleanup)

function Location() {
  const location = useLocation()

  return <output data-testid="route">{location.pathname + location.search}</output>
}

describe('composer + menu capabilities', () => {
  it('keeps a session owner’s backend connection when choosing the capability scope', () => {
    expect(capabilityScopeForChat({ connectionId: 'remote-1', profile: 'byte' }, 'byte')).toEqual({
      connectionId: 'remote-1',
      profile: 'byte'
    })
    expect(capabilityScopeForChat(undefined, 'byte')).toBe('byte')
  })

  it.each([
    ['Skills', 'skills'],
    ['Connectors', 'connectors'],
    ['Plugins', 'plugins']
  ])('opens the installed %s list inside the chat menu', (label, kind) => {
    const onInsertText = vi.fn()

    render(
      <MemoryRouter initialEntries={['/chat']}>
        <I18nProvider configClient={null} initialLocale="en">
          <ContextMenu
            onInsertSkill={vi.fn()}
            onInsertText={onInsertText}
            onOpenUrlDialog={vi.fn()}
            state={{
              model: { model: '', provider: '', canSwitch: false },
              tools: { enabled: true, label: 'Add to chat' },
              voice: { enabled: false, active: false }
            }}
          />
          <Location />
        </I18nProvider>
      </MemoryRouter>
    )

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Add to chat' }), { button: 0, ctrlKey: false })
    fireEvent.click(screen.getByRole('menuitem', { name: label }))

    expect(screen.getByTestId('capability-list').textContent).toBe(kind)
    expect(screen.getByTestId('route').textContent).toBe('/chat')
    expect(onInsertText).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('menuitem', { name: new RegExp('Back') }))
    expect(screen.getByRole('menuitem', { name: label })).toBeTruthy()
  })
})
