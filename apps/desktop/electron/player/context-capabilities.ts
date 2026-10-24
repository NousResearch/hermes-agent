import { randomBytes } from 'node:crypto'

import {
  PLAYER_CONTEXT_TOKEN_TTL_MS,
  type PlayerAgentScope,
  type PlayerMediaTarget
} from '../../src/lib/player-context-contract'

export interface CapturedPlayerContext {
  ownerId: number
  linkId: string
  scope: PlayerAgentScope
  generation: number
  expiresAtMs: number
  target: PlayerMediaTarget | null
}

export function samePlayerScope(left: PlayerAgentScope, right: PlayerAgentScope): boolean {
  return (
    left.connectionId === right.connectionId && left.profileId === right.profileId && left.sessionId === right.sessionId
  )
}

export function playerControlGenerationIsCurrent(
  context: Pick<CapturedPlayerContext, 'generation'>,
  generation: number
): boolean {
  return context.generation === generation
}

export function playerOwnerMatches(
  requestOwnerId: number,
  ...boundOwnerIds: Array<number | null | undefined>
): boolean {
  return boundOwnerIds.every(ownerId => ownerId == null || ownerId === requestOwnerId)
}

/** Issues opaque, short-lived capabilities bound to a window and explicit chat link. */
export class PlayerContextRegistry {
  private readonly contexts = new Map<string, CapturedPlayerContext>()

  issue(input: Omit<CapturedPlayerContext, 'expiresAtMs'>, now = Date.now()): { token: string; expiresAtMs: number } {
    this.expire(now)
    const token = randomBytes(32).toString('base64url')
    const expiresAtMs = now + PLAYER_CONTEXT_TOKEN_TTL_MS
    this.contexts.set(token, {
      ...input,
      scope: { ...input.scope },
      target: input.target
        ? {
            ...input.target,
            title: input.target.title ? { ...input.target.title } : null
          }
        : null,
      expiresAtMs
    })

    return { token, expiresAtMs }
  }

  resolve(
    token: string,
    ownerId: number,
    linkId: string,
    scope: PlayerAgentScope,
    now = Date.now()
  ): CapturedPlayerContext | null {
    const value = this.contexts.get(token)

    if (!value) {
      return null
    }

    if (now >= value.expiresAtMs) {
      this.contexts.delete(token)

      return null
    }

    if (value.ownerId !== ownerId || value.linkId !== linkId || !samePlayerScope(value.scope, scope)) {
      return null
    }

    return {
      ...value,
      scope: { ...value.scope },
      target: value.target ? { ...value.target, title: value.target.title ? { ...value.target.title } : null } : null
    }
  }

  clear(): void {
    this.contexts.clear()
  }

  private expire(now: number): void {
    for (const [token, context] of this.contexts) {
      if (context.expiresAtMs <= now) {
        this.contexts.delete(token)
      }
    }
  }
}

/** Only a validated canonical YouTube watch URL creates an immutable media target. */
export function youtubeTargetFromUrl(
  source: string,
  rawUrl: string,
  metadata: Pick<PlayerMediaTarget, 'title' | 'capturedTimeSeconds' | 'adStatus'>
): PlayerMediaTarget | null {
  if (source !== 'youtube') {
    return null
  }

  try {
    const url = new URL(rawUrl)

    if (
      url.protocol !== 'https:' ||
      !['youtube.com', 'www.youtube.com', 'm.youtube.com'].includes(url.hostname) ||
      url.pathname !== '/watch'
    ) {
      return null
    }

    const videoId = url.searchParams.get('v')

    if (!videoId || !/^[A-Za-z0-9_-]{11}$/.test(videoId)) {
      return null
    }

    return {
      source: 'youtube',
      videoId,
      canonicalUrl: `https://www.youtube.com/watch?v=${videoId}`,
      title: metadata.title ? { ...metadata.title } : null,
      capturedTimeSeconds: metadata.capturedTimeSeconds,
      adStatus: metadata.adStatus
    }
  } catch {
    return null
  }
}
