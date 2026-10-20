import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

import { recordFeatureToggle } from './desktop-metrics'

const HIDE_THREAD_TIMELINE_STORAGE_KEY = 'hermes.desktop.hideThreadTimeline'

/** Desktop-local appearance preference, shared by all threads in this window. */
export const $hideThreadTimeline = atom(storedBoolean(HIDE_THREAD_TIMELINE_STORAGE_KEY, false))

$hideThreadTimeline.subscribe(value => persistBoolean(HIDE_THREAD_TIMELINE_STORAGE_KEY, value))

export function setHideThreadTimeline(value: boolean) {
  recordFeatureToggle('thread_timeline', !$hideThreadTimeline.get(), !value)
  $hideThreadTimeline.set(value)
}
