import { expect, test } from 'vitest'

import { CanonicalDesktopProtocol } from './canonical-protocol'

const workspace = {
  requestId: 'workspace-intent', sourcePath: '/source', cwd: '/source/.worktrees/task',
  projectId: 'fixture-project', branch: 'hermes/task', repoRoot: '/source'
}

test('the canonical create frame preserves a prepared workspace and its retry identity', () => {
  const protocol = new CanonicalDesktopProtocol()

  const input = {
    source: 'desktop', profile: 'coder', cwd: workspace.cwd, cwd_explicit: true,
    request_id: workspace.requestId, coding_workspace: workspace
  }

  const first = protocol.prepare('session.create', input)
  expect(first).toEqual({ ...input, source: 'gui' })
  expect(protocol.prepare('session.create', { ...input })).toEqual(first)

  const projected = protocol.result('session.create', first, {
    session_id: 'canonical-workspace', stored_session_id: 'canonical-workspace',
    revision: 1, execution_generation: 0, messages: [], prompts: [],
    info: { cwd: workspace.cwd, coding_workspace: { ...workspace, artifactsPath: '/artifacts/canonical-workspace' } }
  })

  expect(projected).toMatchObject({ info: { coding_workspace: { ...workspace } } })
})
