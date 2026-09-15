import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { NO_PROJECT_ID, type SidebarProjectTree } from '@/app/chat/sidebar/projects/workspace-groups'
import { I18nProvider } from '@/i18n'
import { $commandPaletteOpen } from '@/store/command-palette'
import {
  $activeProjectId,
  $projectScope,
  $projectTree,
  ALL_PROJECTS,
  enterProject
} from '@/store/projects'
import { $newChatWorkspaceTarget, applyConfiguredDefaultProjectDir } from '@/store/session'

import { Intro } from './intro'

function treeNode(
  over: Partial<SidebarProjectTree> & Pick<SidebarProjectTree, 'id' | 'label'>
): SidebarProjectTree {
  return { path: null, repos: [], sessionCount: 0, ...over }
}

function renderIntro() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <Intro seed={0} />
    </I18nProvider>
  )
}

describe('Intro landing', () => {
  beforeEach(() => {
    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([])
    $newChatWorkspaceTarget.set(undefined)
    $activeProjectId.set(null)
    applyConfiguredDefaultProjectDir(null)
    $commandPaletteOpen.set(false)
  })

  afterEach(() => {
    cleanup()
    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([])
    $newChatWorkspaceTarget.set(undefined)
    $activeProjectId.set(null)
    applyConfiguredDefaultProjectDir(null)
    $commandPaletteOpen.set(false)
  })

  it('names the entered project under the wordmark', () => {
    $projectTree.set([treeNode({ id: 'p_ws', label: 'Warsongs', path: '/repos/warsongs' })])
    enterProject('p_ws')
    renderIntro()
    expect(screen.getByRole('button', { name: 'New session in Warsongs' })).toBeTruthy()
  })

  it('says Detached when All projects has no cwd', () => {
    renderIntro()
    expect(screen.getByRole('button', { name: 'Detached — no project' })).toBeTruthy()
  })

  it('names Home when the no-folder bucket is scoped', () => {
    enterProject(NO_PROJECT_ID)
    renderIntro()
    expect(screen.getByRole('button', { name: 'New session in Home' })).toBeTruthy()
  })

  it('opens the command palette from a detached landing', () => {
    renderIntro()
    fireEvent.click(screen.getByRole('button', { name: 'Detached — no project' }))
    expect($commandPaletteOpen.get()).toBe(true)
  })
})
