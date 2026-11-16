import { useStore } from '@nanostores/react'

import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { $embedAllowed, $embedMode, clearEmbedAllowed, type EmbedMode, setEmbedMode } from '@/store/embed-consent'
import { $remoteMediaMode, type RemoteMediaMode, setRemoteMediaMode } from '@/store/remote-media'

import { ListRow, RowFootnoteAction } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'

const ids = SETTING_IDS.appearance

/** Appearance › Chat display rows that decide when chat content may contact
 *  third-party sites: inline embeds, and remote images/media + link titles. */
export function ChatPrivacySettings() {
  const { t } = useI18n()
  const a = t.settings.appearance
  const embedMode = useStore($embedMode)
  const embedAllowed = useStore($embedAllowed)
  const remoteMediaMode = useStore($remoteMediaMode)

  const embedOptions = [
    { id: 'ask', label: a.embedsAsk },
    { id: 'always', label: a.embedsAlways },
    { id: 'off', label: a.embedsOff }
  ] as const satisfies readonly { id: EmbedMode; label: string }[]

  const remoteMediaOptions = [
    { id: 'ask', label: a.remoteMediaAsk },
    { id: 'always', label: a.remoteMediaAlways }
  ] as const satisfies readonly { id: RemoteMediaMode; label: string }[]

  return (
    <>
      <ListRow
        action={
          <SegmentedControl
            onChange={id => {
              triggerHaptic('selection')
              setEmbedMode(id)
            }}
            options={embedOptions}
            value={embedMode}
          />
        }
        below={
          embedAllowed.length > 0 && (
            <RowFootnoteAction onClick={clearEmbedAllowed}>{a.embedsReset(embedAllowed.length)}</RowFootnoteAction>
          )
        }
        description={a.embedsDesc}
        id={settingElementId(ids.embeds)}
        title={a.embedsTitle}
      />
      <ListRow
        action={
          <SegmentedControl
            onChange={id => {
              triggerHaptic('selection')
              setRemoteMediaMode(id)
            }}
            options={remoteMediaOptions}
            value={remoteMediaMode}
          />
        }
        description={a.remoteMediaDesc}
        id={settingElementId(ids.remoteMedia)}
        title={a.remoteMediaTitle}
      />
    </>
  )
}
