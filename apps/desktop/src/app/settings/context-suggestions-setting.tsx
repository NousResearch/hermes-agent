import { saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { $composerContextSuggestions } from '@/store/composer-context-suggestions'
import { notifyError } from '@/store/notifications'

import { setHermesConfigCache, useHermesConfigRecord } from '../hooks/use-config-record'

import { setNested } from './helpers'
import { ToggleRow } from './primitives'

// `desktop.composer.context_suggestions` — the composer's context-file
// suggestions (#65950): the live `@` file/folder rows above the input and the
// per-session `@file:` prefetch behind them. Same config record + sparse-patch
// save as resume_last_session; the renderer store it feeds lives in
// store/composer-context-suggestions.ts.
export function ContextSuggestionsSetting() {
  const { t } = useI18n()
  const a = t.settings.appearance
  const configQuery = useHermesConfigRecord()
  const config = configQuery.data
  const writeScope = configQuery.writeScope

  const checked =
    ((config?.desktop as { composer?: { context_suggestions?: unknown } } | undefined)?.composer
      ?.context_suggestions) !== false

  const update = (on: boolean) => {
    if (!config) {
      return
    }

    const next = setNested(config, 'desktop.composer.context_suggestions', on)
    setHermesConfigCache(next)
    // Apply at once — the composer gates on the renderer store, which otherwise
    // only refreshes on the next gateway config reload.
    $composerContextSuggestions.set(on)
    // Sparse patch: PUT /api/config deep-merges, and echoing the cached
    // snapshot would overwrite keys other surfaces changed since it loaded.
    void saveHermesConfig(setNested({}, 'desktop.composer.context_suggestions', on), writeScope)
      .then(result => {
        if (!result.ok) {
          throw new Error(t.settings.config.autosaveFailed)
        }
      })
      .catch(error => {
        setHermesConfigCache(config)
        $composerContextSuggestions.set(!on)
        notifyError(error, t.settings.config.autosaveFailed)
      })
  }

  return (
    <ToggleRow
      checked={checked}
      description={a.contextSuggestionsDesc}
      disabled={!config}
      label={a.contextSuggestionsTitle}
      onChange={update}
    />
  )
}
