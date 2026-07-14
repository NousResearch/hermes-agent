import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { archiveWorkspaceDoc, listWorkspaceDocs, readWorkspaceDoc, writeWorkspaceDoc } from './workspace-docs'

const api = vi.fn(async ({ path }: { path: string }): Promise<unknown> => {
  if (path.startsWith('/api/workspace-docs/list?')) {
    return { documents: [{ docType: 'runbook', path: 'runbook.md', status: 'draft', title: 'A runbook', valid: true }] }
  }

  if (path.startsWith('/api/workspace-docs/read?')) {
    return {
      body: 'Body text',
      content: '---\n---\nBody text',
      frontmatter: {
        applyState: 'unapplied',
        docType: 'runbook',
        status: 'draft',
        tags: [],
        title: 'A runbook'
      },
      path: 'runbook.md'
    }
  }

  throw new Error(`unexpected GET path ${path}`)
})

function stubBridge() {
  vi.stubGlobal('window', { hermesDesktop: { api } })
}

describe('workspace docs client', () => {
  beforeEach(() => {
    stubBridge()
    $connection.set(null)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
    $connection.set(null)
  })

  it('lists documents for a workspace root', async () => {
    await expect(listWorkspaceDocs('/work/project')).resolves.toEqual([
      { docType: 'runbook', path: 'runbook.md', status: 'draft', title: 'A runbook', valid: true }
    ])

    expect(api).toHaveBeenCalledWith({ path: '/api/workspace-docs/list?workspaceRoot=%2Fwork%2Fproject' })
  })

  it('reads a document by relative path', async () => {
    await expect(readWorkspaceDoc('/work/project', 'runbook.md')).resolves.toMatchObject({
      body: 'Body text',
      path: 'runbook.md'
    })

    expect(api).toHaveBeenCalledWith({
      path: '/api/workspace-docs/read?path=runbook.md&workspaceRoot=%2Fwork%2Fproject'
    })
  })

  it('routes reads/lists to the active profile backend', async () => {
    $connection.set({ mode: 'remote', profile: 'remote-docker' } as never)

    await listWorkspaceDocs('/work/project')

    expect(api).toHaveBeenCalledWith({
      path: '/api/workspace-docs/list?workspaceRoot=%2Fwork%2Fproject',
      profile: 'remote-docker'
    })
  })

  it('writes a document with a snake_case frontmatter payload', async () => {
    api.mockResolvedValueOnce({
      frontmatter: { applyState: 'unapplied', docType: 'generic-md', status: 'draft', tags: ['x'], title: 'Note' },
      ok: true,
      path: 'note.md'
    })

    await expect(
      writeWorkspaceDoc(
        '/work/project',
        'note.md',
        { description: 'A note', docType: 'generic-md', status: 'draft', tags: ['x'], title: 'Note' },
        '# Note\n'
      )
    ).resolves.toMatchObject({ ok: true, path: 'note.md' })

    expect(api).toHaveBeenCalledWith({
      body: {
        body: '# Note\n',
        frontmatter: { description: 'A note', doc_type: 'generic-md', status: 'draft', tags: ['x'], title: 'Note' },
        path: 'note.md',
        workspaceRoot: '/work/project'
      },
      method: 'POST',
      path: '/api/workspace-docs',
      profile: undefined
    })
  })

  it('archives a document', async () => {
    api.mockResolvedValueOnce({
      frontmatter: { applyState: 'unapplied', docType: 'generic-md', status: 'archived', tags: [], title: 'Note' },
      ok: true,
      path: 'note.md'
    })

    await expect(archiveWorkspaceDoc('/work/project', 'note.md')).resolves.toMatchObject({
      frontmatter: { status: 'archived' },
      ok: true
    })

    expect(api).toHaveBeenCalledWith({
      body: { path: 'note.md', workspaceRoot: '/work/project' },
      method: 'POST',
      path: '/api/workspace-docs/archive',
      profile: undefined
    })
  })
})
