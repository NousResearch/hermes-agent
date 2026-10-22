import { useStore } from '@nanostores/react'

import { useI18n } from '@/i18n'
import { $inlinePreviewWide, setInlinePreviewWide } from '@/store/inline-preview-width'
import { $reactionsEnabled, setReactionsEnabled } from '@/store/reactions-enabled'

import { ToggleRow } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'

const ids = SETTING_IDS.appearance

/**
 * Chat-display rows that live outside the `AppearanceSettings` body.
 *
 * That component is already past its function budget, so each row that shows
 * up there rides on an extraction rather than growing it. Callers gate the
 * whole group with their own `show('chat-display')`.
 */
export function ChatDisplayToggles() {
  const { t } = useI18n()
  const a = t.settings.appearance
  const reactionsEnabled = useStore($reactionsEnabled)
  const inlinePreviewWide = useStore($inlinePreviewWide)

  return (
    <>
      <ToggleRow
        checked={reactionsEnabled}
        description={a.reactionsDesc}
        id={settingElementId(ids.reactions)}
        label={a.reactionsTitle}
        onChange={setReactionsEnabled}
      />
      <ToggleRow
        checked={inlinePreviewWide}
        description={a.inlinePreviewWideDesc}
        id={settingElementId(ids.inlinePreviewWide)}
        label={a.inlinePreviewWideTitle}
        onChange={setInlinePreviewWide}
      />
    </>
  )
}
