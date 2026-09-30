/**
 * `desktop.composer.context_suggestions` — whether the composer offers
 * context-file suggestions (#65950): the live `@` file/folder completions
 * above the input (`use-at-completions`) and the per-session `@file:`
 * prefetch behind them (`use-context-suggestions`).
 *
 * Default on, matching hermes_cli/config_defaults.py. Reading or toggling it
 * only changes what the composer suggests — manual `@file:`/`@folder:` refs
 * keep working either way, so it is prompt-cache safe.
 */

import { atom } from 'nanostores'

export const $composerContextSuggestions = atom<boolean>(true)

export function setComposerContextSuggestionsFromConfig(value: unknown): void {
  $composerContextSuggestions.set(value !== false)
}
