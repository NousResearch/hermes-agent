import { useStore } from '@nanostores/react'
import { type ReactElement, useEffect } from 'react'

import { Switch } from '@/components/ui/switch'
import { relativeTime } from '@/components/update-status'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { $autoUpdate, loadAutoUpdate, setAutoUpdateEnabled } from '@/store/auto-update'

import { SETTING_IDS, settingElementId } from './settings-manifest'

/**
 * "Install updates automatically" (#123674) — the opt-in switch that lives in
 * the About → Version & updates card, between the status line and Check now.
 * Stacked title / description / switch, the same shape as Sessions'
 * "Auto-archive stale chats" row when it stacks.
 *
 * Device-scoped: the setting is stored by Electron (userData/auto-update.json),
 * not config.yaml, because it governs this app's own launch behaviour.
 */
export function AutoUpdateSetting(): ReactElement | null {
  const { t } = useI18n()
  const u = t.updates
  const auto = useStore($autoUpdate)

  useEffect(() => {
    void loadAutoUpdate()
  }, [])

  if (!auto.loaded || typeof window === 'undefined' || !window.hermesDesktop?.updates?.auto) {
    return null
  }

  const last = auto.lastAttempt
  const titleId = 'auto-update-setting-title'

  return (
    <div className="mt-3 border-t border-border/60 pt-3" id={settingElementId(SETTING_IDS.about.autoUpdate)}>
      <div className="text-[length:var(--conversation-text-font-size)] font-medium text-foreground" id={titleId}>
        {u.autoUpdate.title}
      </div>
      <div className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {auto.supported ? u.autoUpdate.desc : u.autoUpdate.unsupported}
      </div>
      {auto.enabled && last && (
        <div className="mt-1 text-xs text-muted-foreground">
          {u.autoUpdate.last(u.autoUpdate.outcomes[last.outcome] ?? last.outcome, relativeTime(last.at, u))}
        </div>
      )}
      <div className="mt-3">
        <Switch
          aria-labelledby={titleId}
          checked={auto.enabled}
          disabled={!auto.supported || auto.saving}
          onCheckedChange={on => {
            triggerHaptic('selection')
            void setAutoUpdateEnabled(on)
          }}
        />
      </div>
    </div>
  )
}
