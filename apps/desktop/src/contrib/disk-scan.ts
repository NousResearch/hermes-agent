import { atom } from 'nanostores'

/** True while the disk door's FIRST scan is in flight. Boot code that must
 *  not mistake a not-yet-registered plugin route for a stale one
 *  (remembered-route restore, routed session resume) waits on this instead of
 *  on timing. Its own module so a hook can wait on it without loading the
 *  runtime loader. */
export const $diskPluginsScanPending = atom(false)
