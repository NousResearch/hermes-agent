import type { HermesGitWorktree } from '@/global'

export interface BrowserDevServerCandidate {
  id: string
  label: string
  url: string
}

interface BrowserDevServerCandidatesInput {
  cwd?: null | string
  worktrees: Array<Pick<HermesGitWorktree, 'branch' | 'path'>>
}

const COMMON_DEV_PORTS: ReadonlyArray<{ label: string; port: number }> = [
  { label: 'dev server', port: 3000 },
  { label: 'Vite dev server', port: 5173 },
  { label: 'Vite fallback', port: 5174 },
  { label: 'Vite preview', port: 4173 },
  { label: 'Python/dev server', port: 8000 },
  { label: 'HTTP dev server', port: 8080 }
]

export function browserDevServerCandidates({ cwd, worktrees }: BrowserDevServerCandidatesInput): BrowserDevServerCandidate[] {
  const contexts = devServerContexts(cwd, worktrees)

  return contexts.flatMap(context =>
    COMMON_DEV_PORTS.map(({ label, port }) => ({
      id: `${context.id}-${port}`,
      label: `Open ${context.label === 'workspace' ? label : `${context.label} dev server`} :${port}`,
      url: normalizeDevServerUrl(`localhost:${port}`)
    }))
  )
}

export function normalizeDevServerUrl(value: string): string {
  const trimmed = value.trim()

  if (/^(localhost|127(?:\.\d{1,3}){3}|\[[^\]]+\]):\d+(?:\/.*)?$/i.test(trimmed)) {
    return `http://${trimmed}`
  }

  return trimmed
}

export function devServerUnavailableMessage(url: string): string {
  const normalized = normalizeDevServerUrl(url)
  const target = normalized.replace(/^https?:\/\//i, '')

  return `No local dev server responded at ${target}. Start the app server or choose a different localhost port, then reload the BrowserPane.`
}

function devServerContexts(cwd: BrowserDevServerCandidatesInput['cwd'], worktrees: BrowserDevServerCandidatesInput['worktrees']) {
  const contexts: Array<{ id: string; label: string }> = []

  if (cwd?.trim()) {
    contexts.push({ id: 'cwd', label: 'workspace' })
  }

  for (const worktree of worktrees) {
    const label = worktree.branch?.trim() || tailName(worktree.path) || 'worktree'
    const id = `worktree-${slug(label)}`

    if (!contexts.some(context => context.id === id)) {
      contexts.push({ id, label })
    }
  }

  return contexts
}

function tailName(path: string): string {
  return path.split(/[\\/]+/).filter(Boolean).at(-1) ?? ''
}

function slug(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'worktree'
}
