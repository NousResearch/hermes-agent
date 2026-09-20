// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

vi.mock('@/app/right-sidebar/files/tree', () => ({
  ProjectTree: ({ cwd, data }: { cwd: string; data: unknown[] }) => (
    <div data-cwd={cwd} data-testid="project-tree">
      {data.length} entries
    </div>
  )
}))

const projectTreeState = {
  collapseNonce: 0,
  collapseAll: () => {},
  data: [{ id: '/repo/a.ts', isDirectory: false, name: 'a.ts' }],
  effectiveCwd: '/repo',
  loadChildren: async () => {},
  openState: {},
  refreshRoot: async () => {},
  rootError: null as null | string,
  rootLoading: false,
  setNodeOpen: () => {},
  setShowIgnored: () => {},
  showIgnored: false
}

const useProjectTreeMock = vi.fn((_cwd: string) => projectTreeState)

vi.mock('@/app/right-sidebar/files/use-project-tree', () => ({
  useProjectTree: (cwd: string) => useProjectTreeMock(cwd)
}))

vi.mock('../../workspace', () => ({
  openIdeFolder: vi.fn()
}))

import { setIdeWorkspaceRoot } from '../../state'
import { openIdeFolder } from '../../workspace'

import { ExplorerRegion } from './index'

function renderRegion() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <ExplorerRegion />
    </I18nProvider>
  )
}

beforeEach(() => {
  useProjectTreeMock.mockClear()
  vi.mocked(openIdeFolder).mockReset()
  setIdeWorkspaceRoot(null)
})

afterEach(() => {
  cleanup()
})

describe('ExplorerRegion', () => {
  it('shows the honest empty state without a workspace', () => {
    renderRegion()

    expect(screen.getByText('No workspace open')).toBeTruthy()
    expect(useProjectTreeMock).toHaveBeenCalledWith('')
  })

  it('binds the tree to the IDE workspace root and shows its name', () => {
    setIdeWorkspaceRoot('D:\\My apps\\COAI')
    useProjectTreeMock.mockReturnValueOnce({ ...projectTreeState, effectiveCwd: 'D:\\My apps\\COAI' })

    renderRegion()

    expect(useProjectTreeMock).toHaveBeenCalledWith('D:\\My apps\\COAI')
    expect(screen.getByText('COAI')).toBeTruthy()
    expect(screen.getByTestId('project-tree').getAttribute('data-cwd')).toBe('D:\\My apps\\COAI')
  })

  it('surfaces an unreadable root instead of an empty tree', () => {
    setIdeWorkspaceRoot('/repo')
    useProjectTreeMock.mockReturnValueOnce({ ...projectTreeState, data: [], rootError: 'EACCES' })

    renderRegion()

    expect(screen.getByText('Unable to read this folder (EACCES).')).toBeTruthy()
  })

  it('offers Open folder in the empty state', () => {
    renderRegion()

    fireEvent.click(screen.getByText('Open folder…'))

    expect(vi.mocked(openIdeFolder)).toHaveBeenCalledWith('Could not open that folder')
  })

  it('keeps the Open folder affordance in the header with a workspace open', () => {
    setIdeWorkspaceRoot('/repo')

    renderRegion()
    fireEvent.click(screen.getByLabelText('Open folder…'))

    expect(vi.mocked(openIdeFolder)).toHaveBeenCalledTimes(1)
  })
})
