import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import {
  LARGE_PASTE_ATTACHMENT_THRESHOLD,
  LARGE_PASTE_ATTACHMENT_THRESHOLD_MAX,
  LARGE_PASTE_ATTACHMENT_THRESHOLD_MIN,
  normalizeLargePasteAttachmentThreshold
} from '@/app/chat/composer/large-paste'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { isSubmitEnter } from '@/lib/ime'
import { $largePasteAttachmentThreshold, setLargePasteAttachmentThreshold } from '@/store/large-paste-threshold'

import { ListRow } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'

export function LargePasteThresholdSetting() {
  const { t } = useI18n()
  const c = t.settings.config
  const stored = useStore($largePasteAttachmentThreshold)
  const [draft, setDraft] = useState(String(stored))

  useEffect(() => {
    setDraft(String(stored))
  }, [stored])

  const commit = (input: HTMLInputElement) => {
    // Browsers expose an unfinished numeric entry (e.g. "1e") as an empty
    // value with badInput set. It must not be mistaken for a default reset.
    if (input.validity.badInput) {
      setDraft(String(stored))

      return
    }

    const candidate = draft.trim() === '' ? LARGE_PASTE_ATTACHMENT_THRESHOLD : Number(draft)
    const applied = normalizeLargePasteAttachmentThreshold(candidate)

    // Invalid edits leave the committed preference alone; blank resets it.
    if (applied !== candidate) {
      setDraft(String(stored))

      return
    }

    if (applied !== stored) {
      setLargePasteAttachmentThreshold(applied)
      triggerHaptic('selection')
    }

    setDraft(String(applied))
  }

  return (
    <ListRow
      action={
        <Input
          aria-label={c.largePasteThresholdTitle}
          className="w-28"
          inputMode="numeric"
          max={LARGE_PASTE_ATTACHMENT_THRESHOLD_MAX}
          min={LARGE_PASTE_ATTACHMENT_THRESHOLD_MIN}
          onBlur={event => commit(event.currentTarget)}
          onChange={event => setDraft(event.target.value)}
          onKeyDown={event => {
            if (isSubmitEnter(event)) {
              event.currentTarget.blur()
            }
          }}
          step={1}
          type="number"
          value={draft}
        />
      }
      description={c.largePasteThresholdDesc}
      id={settingElementId(SETTING_IDS.chat.largePasteThreshold)}
      title={c.largePasteThresholdTitle}
    />
  )
}
