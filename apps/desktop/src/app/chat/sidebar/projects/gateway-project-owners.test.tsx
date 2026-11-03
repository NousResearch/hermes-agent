import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SidebarProvider } from '@/components/ui/sidebar'
import { makeSessionInfo } from '@/test/session-info'
import type { SessionInfo } from '@/types/hermes'

import { EnteredProjectContent } from './entered-content'
import { enteredProjectLiveRows, withGatewayProjectRows } from './gateway-project-rows'
import { ProjectOverviewRow } from './overview-row'
import {
  NO_PROJECT_ID,
  overlayLiveLanes,
  overlayLivePreviews,
  projectOwnerBySessionId,
  type SidebarProjectTree
} from './workspace-groups'

// The window is on This device (rows bare or `local`); Studio is another registered gateway.
const ownerOf = (session: SessionInfo) => session.connection_id?.trim() || 'local'
const foreignOwner = (session: SessionInfo) => (ownerOf(session) === 'local' ? null : ownerOf(session))
const ownerKey = (session: SessionInfo) => JSON.stringify([ownerOf(session), session.profile || 'default'])
const owners = { foreignOwner, ownerKey }
const label = (id: string) => (id === 'studio' ? 'Studio' : id)

const localChat = makeSessionInfo({
  id: 'local-chat',
  title: 'LOCAL',
  connection_id: 'local',
  cwd: '/code/app',
  git_repo_root: '/code/app'
})

const studioChat = makeSessionInfo({
  id: 'studio-chat',
  title: 'REMOTE',
  connection_id: 'studio',
  profile: 'research',
  cwd: '/code/app',
  git_repo_root: '/code/app'
})

const localApp: SidebarProjectTree = {
  id: '/code/app',
  label: 'app',
  path: '/code/app',
  repos: [
    {
      id: '/code/app',
      label: 'app',
      path: '/code/app',
      groups: [
        { id: '/code/app::branch::main', isMain: true, label: 'main', path: '/code/app', sessions: [localChat] }
      ],
      sessionCount: 1
    }
  ],
  sessionCount: 1,
  sessionIds: [localChat.id],
  previewSessions: [localChat]
}

afterEach(cleanup)

function renderEntered(project: SidebarProjectTree, tree: SidebarProjectTree[], live: SessionInfo[]) {
  const liveSessions = enteredProjectLiveRows(project, live, owners)

  render(
    <SidebarProvider>
      <EnteredProjectContent
        liveSessions={liveSessions}
        project={overlayLiveLanes(project, liveSessions, new Set(), projectOwnerBySessionId(tree), ownerKey)}
        renderRows={sessions => sessions.map(session => <span key={session.id}>{session.title}</span>)}
      />
    </SidebarProvider>
  )
}

describe('projects another gateway owns', () => {
  it('start a new session on the owning gateway profile', () => {
    const tree = withGatewayProjectRows([localApp], [localChat, studioChat], foreignOwner, label)
    const onNewSession = vi.fn()

    for (const project of tree) {
      render(
        <SidebarProvider>
          <ProjectOverviewRow onNewSession={onNewSession} project={project} />
        </SidebarProvider>
      )
    }

    fireEvent.click(screen.getByRole('button', { name: /new session in app · studio/i }))
    fireEvent.click(screen.getByRole('button', { name: /new session in app$/i }))

    expect(onNewSession.mock.calls).toEqual([
      ['/code/app', { connectionId: 'studio', profile: 'research' }],
      ['/code/app']
    ])
  })

  it('list only their own chats when entered, and the local project only its own', () => {
    const tree = withGatewayProjectRows([localApp], [localChat, studioChat], foreignOwner, label)

    renderEntered(
      tree.find(project => project.connectionId === 'studio')!,
      tree,
      [localChat, studioChat]
    )
    expect(screen.queryByText('LOCAL')).toBeNull()
    expect(screen.getByText('REMOTE')).toBeTruthy()
    cleanup()

    renderEntered(localApp, tree, [localChat, studioChat])
    expect(screen.getByText('LOCAL')).toBeTruthy()
    expect(screen.queryByText('REMOTE')).toBeNull()
  })

  it('keep a folder-less chat sharing a durable id with a local one as its own Home row', () => {
    const local = makeSessionInfo({ id: 'same-id', title: 'LOCAL SAME', connection_id: 'local' })
    const remote = makeSessionInfo({ id: 'same-id', title: 'REMOTE SAME', connection_id: 'studio' })

    const home: SidebarProjectTree = {
      id: NO_PROJECT_ID,
      isNoProject: true,
      label: 'Home',
      path: null,
      repos: [],
      sessionCount: 1,
      sessionIds: [local.id],
      previewSessions: [local]
    }

    const tree = withGatewayProjectRows([home], [local, remote], foreignOwner, label)
    const previews = overlayLivePreviews(tree, [local, remote], [], 10, { ownerKey })

    expect(tree[0].sessionCount).toBe(2)
    expect(previews[NO_PROJECT_ID].map(session => session.title).sort()).toEqual(['LOCAL SAME', 'REMOTE SAME'])

    renderEntered(tree[0], tree, [local, remote])
    expect(screen.getByText('LOCAL SAME')).toBeTruthy()
    expect(screen.getByText('REMOTE SAME')).toBeTruthy()
  })
})
