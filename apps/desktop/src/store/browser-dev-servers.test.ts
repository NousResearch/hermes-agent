import { describe, expect, it } from 'vitest'

import { browserDevServerCandidates, devServerUnavailableMessage, normalizeDevServerUrl } from './browser-dev-servers'

describe('browser dev server candidates', () => {
  it('derives common localhost dev-server URLs for the active workspace', () => {
    expect(browserDevServerCandidates({ cwd: '/repo/app', worktrees: [] }).map(candidate => candidate.url)).toEqual([
      'http://localhost:3000',
      'http://localhost:5173',
      'http://localhost:5174',
      'http://localhost:4173',
      'http://localhost:8000',
      'http://localhost:8080'
    ])
  })

  it('adds labeled candidates for repo worktrees', () => {
    const candidates = browserDevServerCandidates({
      cwd: '',
      worktrees: [
        { branch: 'feature/ui', path: '/repo/feature-ui' },
        { branch: '', path: '/repo/preview' }
      ]
    })

    expect(candidates[0]).toEqual({
      id: 'worktree-feature-ui-3000',
      label: 'Open feature/ui dev server :3000',
      url: 'http://localhost:3000'
    })
    expect(candidates.some(candidate => candidate.label.includes('preview'))).toBe(true)
  })

  it('returns no candidates when there is no workspace context', () => {
    expect(browserDevServerCandidates({ cwd: '', worktrees: [] })).toEqual([])
  })

  it('normalizes localhost dev-server inputs used by BrowserPane and command palette flows', () => {
    expect(normalizeDevServerUrl('localhost:3000')).toBe('http://localhost:3000')
    expect(normalizeDevServerUrl('127.0.0.1:5173')).toBe('http://127.0.0.1:5173')
    expect(normalizeDevServerUrl('[::1]:8080')).toBe('http://[::1]:8080')
    expect(normalizeDevServerUrl('https://localhost:3000/app')).toBe('https://localhost:3000/app')
  })

  it('returns a clear unavailable-port message for failed local dev-server opens', () => {
    expect(devServerUnavailableMessage('http://localhost:5173')).toContain('localhost:5173')
    expect(devServerUnavailableMessage('http://localhost:5173')).toContain('dev server')
  })
})
