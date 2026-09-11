/**
 * Max size for local files Desktop loads as data URLs (composer attach, image
 * previews, etc.). Main owns the real cap + its JSON under userData — this
 * atom only mirrors it for Settings → Chat. See electron/main.ts
 * (`hermes:data-url-read-max:*`). The default and the clamp are shared with
 * main via apps/shared so the two ends cannot drift.
 */

import {
  clampDataUrlReadMaxMb,
  DATA_URL_READ_DEFAULT_MAX_MB,
  DATA_URL_READ_MAX_MAX_MB,
  DATA_URL_READ_MIN_MAX_MB
} from '@hermes/shared'
import { atom } from 'nanostores'

import { notifyError } from '@/store/notifications'

export { clampDataUrlReadMaxMb, DATA_URL_READ_DEFAULT_MAX_MB, DATA_URL_READ_MAX_MAX_MB, DATA_URL_READ_MIN_MAX_MB }

export const $dataUrlReadMaxMb = atom<number>(DATA_URL_READ_DEFAULT_MAX_MB)

export async function refreshDataUrlReadMaxMb(): Promise<number> {
  const api = window.hermesDesktop?.dataUrlReadMax

  if (!api) {
    return $dataUrlReadMaxMb.get()
  }

  try {
    const result = await api.get()
    const maxMb = clampDataUrlReadMaxMb(result.maxMb)
    $dataUrlReadMaxMb.set(maxMb)

    return maxMb
  } catch {
    return $dataUrlReadMaxMb.get()
  }
}

export async function setDataUrlReadMaxMb(maxMb: number): Promise<number> {
  const next = clampDataUrlReadMaxMb(maxMb)
  const api = window.hermesDesktop?.dataUrlReadMax

  if (!api) {
    $dataUrlReadMaxMb.set(next)

    return next
  }

  try {
    const result = await api.set(next)
    const applied = clampDataUrlReadMaxMb(result.maxMb)
    $dataUrlReadMaxMb.set(applied)

    return applied
  } catch (error) {
    // Leave the atom at the last known-good value and surface the failure —
    // an optimistic set here would show a cap that was never persisted and
    // silently reverts on restart.
    notifyError(error, 'Could not save the max attachment size')

    return $dataUrlReadMaxMb.get()
  }
}

if (typeof window !== 'undefined' && window.hermesDesktop?.dataUrlReadMax) {
  void refreshDataUrlReadMaxMb()
}
