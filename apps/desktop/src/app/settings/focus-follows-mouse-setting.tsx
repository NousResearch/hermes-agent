import { useStore } from '@nanostores/react'

import { useI18n } from '@/i18n'
import { $composerPopoutGesturesEnabled, setComposerPopoutGesturesEnabled } from '@/store/composer-popout'
import { $focusFollowsMouse, setFocusFollowsMouse } from '@/store/focus-follows-mouse'

import { ToggleRow } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'

function FocusFollowsMouseSetting() {
  const { t } = useI18n()
  const checked = useStore($focusFollowsMouse)

  return (
    <ToggleRow
      checked={checked}
      description={t.settings.appearance.focusFollowsMouseDesc}
      id={settingElementId(SETTING_IDS.appearance.focusFollowsMouse)}
      label={t.settings.appearance.focusFollowsMouseTitle}
      onChange={setFocusFollowsMouse}
    />
  )
}

/** Window/layout owns both composer placement and cross-pane pointer focus. */
export function ComposerInteractionSettings() {
  const { t } = useI18n()
  const gesturesEnabled = useStore($composerPopoutGesturesEnabled)

  return (
    <>
      <ToggleRow
        checked={gesturesEnabled}
        description={t.settings.appearance.composerPopoutDesc}
        id={settingElementId(SETTING_IDS.appearance.composerPopout)}
        label={t.settings.appearance.composerPopoutTitle}
        onChange={setComposerPopoutGesturesEnabled}
      />
      <FocusFollowsMouseSetting />
    </>
  )
}
