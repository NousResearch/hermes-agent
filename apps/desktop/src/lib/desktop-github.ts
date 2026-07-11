import type {
  HermesGithubPullRequestDetail,
  HermesGithubPullRequestFilter,
  HermesGithubPullRequestList,
  HermesGithubPullRequestRef
} from '@/global'

import { desktopFsProfile, isDesktopFsRemoteMode } from './desktop-fs'

function api<T>(path: string): Promise<T> {
  return window.hermesDesktop.api<T>({ path, profile: desktopFsProfile() })
}

export function githubProfileKey(): string {
  return desktopFsProfile() || 'default'
}

export function listGithubPullRequests(filter: HermesGithubPullRequestFilter): Promise<HermesGithubPullRequestList> {
  if (!isDesktopFsRemoteMode()) {
    const bridge = window.hermesDesktop.github?.pullRequests

    if (!bridge) {
      throw new Error('GitHub Desktop bridge is unavailable')
    }

    return bridge.list(filter)
  }

  const query = new URLSearchParams({ kind: filter.kind, state: filter.state, limit: String(filter.limit ?? 100) })

  return api(`/api/github/pull-requests?${query}`)
}

export function getGithubPullRequestDetail(ref: HermesGithubPullRequestRef): Promise<HermesGithubPullRequestDetail> {
  if (!isDesktopFsRemoteMode()) {
    const bridge = window.hermesDesktop.github?.pullRequests

    if (!bridge) {
      throw new Error('GitHub Desktop bridge is unavailable')
    }

    return bridge.detail(ref)
  }

  const query = new URLSearchParams({ repository: ref.repository, number: String(ref.number) })

  return api(`/api/github/pull-requests/detail?${query}`)
}
