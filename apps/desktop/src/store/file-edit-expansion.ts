/**
 * Whether a new file-edit tool row mounts with its inline diff expanded.
 *
 * Settings → Appearance owns the lever (#74302): users review edits
 * differently — some want every diff visible as it lands, others prefer a
 * compact transcript they expand on demand. Off (the shipped default) keeps
 * each settled row at its one-line summary; a manual toggle on a specific row
 * still wins over this default via `$toolDisclosureOpen`.
 */

import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const EXPAND_FILE_EDITS_STORAGE_KEY = 'hermes.desktop.fileEdits.expandedByDefault'

/** Desktop-local presentation preference; shared backend config must not be changed by a single window. */
export const $expandFileEditsByDefault = atom(storedBoolean(EXPAND_FILE_EDITS_STORAGE_KEY, false))

$expandFileEditsByDefault.subscribe(value => persistBoolean(EXPAND_FILE_EDITS_STORAGE_KEY, value))

export function setExpandFileEditsByDefault(value: boolean) {
  $expandFileEditsByDefault.set(value)
}
