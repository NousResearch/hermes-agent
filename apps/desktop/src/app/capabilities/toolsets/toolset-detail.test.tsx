// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { ToolsetDetail } from './toolset-detail'

vi.mock('react-router', () => ({ useNavigate: () => vi.fn() }))
vi.mock('@/hermes', () => ({ profileScopeKey: (profile?: { connectionId?: string }) => profile?.connectionId ?? 'active' }))
vi.mock('@/i18n', () => ({ useI18n: () => ({ t: { skills: { needsKeys: 'Needs keys', noDescription: 'No description' } } }) }))
vi.mock('../../master-detail', () => ({ ToolChip: ({ children }: { children: React.ReactNode }) => <span>{children}</span> }))
vi.mock('../../overlays/panel', () => ({ PanelPill: ({ children }: { children: React.ReactNode }) => <span>{children}</span> }))
vi.mock('../primitives', () => ({ DetailHeader: () => null }))
vi.mock('../../settings/browser-real-profile-panel', () => ({ BrowserRealProfilePanel: () => null }))
vi.mock('../../settings/terminal-backend-panel', () => ({ TerminalBackendPanel: () => null }))
vi.mock('../../settings/computer-use-panel', () => ({
  ComputerUsePanel: ({ onTargetChange, target }: { onTargetChange: (target: 'windows-host') => void; target: string }) => (
    <button onClick={() => onTargetChange('windows-host')}>Select {target}</button>
  )
}))
vi.mock('../../settings/toolset-config-panel', () => ({
  ToolsetConfigPanel: ({ profile }: { profile?: { connectionId?: string } }) => <div>Setup scope: {profile?.connectionId ?? 'active'}</div>
}))

describe('ToolsetDetail', () => {
  it('moves the Computer Use setup panel to the local owner when Windows host is selected', () => {
    render(
      <ToolsetDetail
        onConfiguredChange={vi.fn()}
        profile={{ connectionId: 'remote-wsl', profile: 'default' }}
        toolCalls={{}}
        toolset={{ configured: true, description: '', enabled: true, label: 'Computer Use', name: 'computer_use', tools: [] }}
      />
    )

    expect(screen.getByText('Setup scope: remote-wsl')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Select guest' }))
    expect(screen.getByText('Setup scope: local')).toBeTruthy()
  })
})
