import { describe, expect, it } from 'vitest'

import { makeSessionInfo } from '@/test/session-info'

import { gatewayProjectId, withGatewayProjectRows } from './gateway-project-rows'
import { NO_PROJECT_ID, overlayLivePreviews, type SidebarProjectTree } from './workspace-groups'

const home: SidebarProjectTree = {
  id: NO_PROJECT_ID,
  isNoProject: true,
  label: 'Home',
  path: null,
  repos: [],
  sessionCount: 1,
  sessionIds: ['local-1'],
  previewSessions: [makeSessionInfo({ id: 'local-1' })]
}

const localApp: SidebarProjectTree = { id: '/code/app', label: 'app', path: '/code/app', repos: [], sessionCount: 0 }

const rows = [
  makeSessionInfo({ id: 'local-1' }),
  makeSessionInfo({ id: 'lab-loose', connection_id: 'homelab' }),
  makeSessionInfo({ id: 'lab-app', connection_id: 'homelab', cwd: '/srv/app', git_repo_root: '/srv/app' }),
  makeSessionInfo({ id: 'studio-app', connection_id: 'studio', cwd: '/work/app/src', git_repo_root: '/work/app' }),
  makeSessionInfo({ id: 'studio-infra', connection_id: 'studio', cwd: '/work/infra' })
]

const foreign = (session: { connection_id?: null | string }) =>
  session.connection_id && session.connection_id !== 'local' ? session.connection_id : null

const label = (id: string) => ({ homelab: 'Homelab', studio: 'Studio' })[id] ?? id

const owned = (tree: SidebarProjectTree[]) =>
  Object.fromEntries(tree.map(node => [node.label, [...(node.sessionIds ?? [])].sort()]))

describe('withGatewayProjectRows', () => {
  it('files every row another gateway owns: folder-less to Home, the rest under their folder on that gateway', () => {
    const tree = withGatewayProjectRows([home, localApp], rows, foreign, label)

    expect(owned(tree)).toEqual({
      Home: ['lab-loose', 'local-1'],
      app: [],
      'app · Homelab': ['lab-app'],
      'app · Studio': ['studio-app'],
      infra: ['studio-infra']
    })
    expect(tree.find(node => node.isNoProject)?.sessionCount).toBe(2)
    expect(tree.find(node => node.label === 'infra')?.id).toBe(gatewayProjectId('studio', '/work/infra'))
    // Every row the tree was given lands in exactly one project.
    expect(tree.flatMap(node => node.sessionIds ?? []).sort()).toEqual(rows.map(row => row.id).sort())
  })

  it('reaches the overview previews, and leaves the tree untouched with no foreign rows', () => {
    const tree = withGatewayProjectRows([home], rows, foreign, label)
    const previews = overlayLivePreviews(tree, rows, [], 10)

    expect(Object.values(previews).flat().length).toBe(rows.length)
    expect(withGatewayProjectRows([home], [rows[0]], foreign, label)[0]).toBe(home)
  })

  it('adds a Home bucket when the serving gateway has none', () => {
    const tree = withGatewayProjectRows([], [rows[1]], foreign, label)

    expect(tree).toHaveLength(1)
    expect(tree[0]).toMatchObject({ id: NO_PROJECT_ID, isNoProject: true, sessionCount: 1, sessionIds: ['lab-loose'] })
  })
})
