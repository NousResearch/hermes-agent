/** Existing SDK request and probe timeout shared by hosted-room routing. */

import { host } from '@hermes/plugin-sdk'

import { botsText } from './i18n'
import type { ProfileRoute } from './types'

export async function requestHostedConnection<T>(
  route: ProfileRoute,
  method: string,
  params: Record<string, unknown> = {}
): Promise<T> {
  if (!route?.connectionId || typeof host.requestProfile !== 'function') {
    throw new Error(botsText().group.hostRouteMissing)
  }

  return host.requestProfile(route, method, params) as Promise<T>
}

export async function withHostedRoomProbeTimeout<T>(task: Promise<T>, timeoutMs = 3000) {
  let timer: null | ReturnType<typeof setTimeout> = null

  try {
    return await Promise.race([
      task,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error('Host check timed out')), timeoutMs)
      })
    ])
  } finally {
    if (timer !== null) {
      clearTimeout(timer)
    }
  }
}
