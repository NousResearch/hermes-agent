import { useStore } from '@nanostores/react'

import { Checkbox } from '@/components/ui/checkbox'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { EXTRA_LIST_STYLES } from '@/lib/markdown-lists'
import { $composerListStyles, setComposerListStyle } from '@/store/composer-list-styles'

import { ListRow } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'

/** Numbered, bullet and checkbox lists always continue; these opt into the non-Markdown styles. */
export function ComposerListsSetting() {
  const { t } = useI18n()
  const a = t.settings.appearance
  const styles = useStore($composerListStyles)

  return (
    <ListRow
      below={
        <div className="mt-2 flex flex-wrap gap-x-5 gap-y-2 text-sm">
          {EXTRA_LIST_STYLES.map(style => (
            <label className="flex cursor-pointer items-center gap-2" key={style}>
              <Checkbox
                checked={styles.has(style)}
                onCheckedChange={checked => {
                  triggerHaptic('selection')
                  setComposerListStyle(style, checked === true)
                }}
              />
              {a.composerListStyles[style]}
            </label>
          ))}
        </div>
      }
      description={a.composerListsDesc}
      id={settingElementId(SETTING_IDS.appearance.composerLists)}
      title={a.composerListsTitle}
      wide
    />
  )
}
