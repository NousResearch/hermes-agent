import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { ListRow } from '@/app/settings/primitives'
import { SETTING_IDS, settingElementId } from '@/app/settings/settings-manifest'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import {
  $remoteLivenessTimeoutMs,
  loadRemoteLivenessTimeout,
  saveRemoteLivenessTimeout
} from '@/store/remote-liveness-timeout'

// Bounds imported from main's clamp module so the advertised input range can
// never drift from what main actually clamps to (review note on #121941).
import { REMOTE_LIVENESS_TIMEOUT_BOUNDS } from '../../../electron/remote-liveness-timeout'

/** Settings → Advanced: how long the desktop app waits for a remote backend's
 *  health probe before treating it as unreachable. Device-local (not
 *  profile-scoped): each machine's network/CPU headroom to a remote host is
 *  its own, and changes apply live — no restart. */
export function RemoteLivenessTimeoutSetting() {
  const { t } = useI18n()
  const timeoutMs = useStore($remoteLivenessTimeoutMs)
  const [draft, setDraft] = useState(String(timeoutMs))

  useEffect(() => {
    void loadRemoteLivenessTimeout()
  }, [])

  useEffect(() => {
    setDraft(String(timeoutMs))
  }, [timeoutMs])

  const commit = () => {
    const parsed = Number(draft)

    if (!Number.isFinite(parsed) || parsed === timeoutMs) {
      setDraft(String(timeoutMs))

      return
    }

    void saveRemoteLivenessTimeout(parsed)
      .then(() => undefined)
      .catch(() => setDraft(String($remoteLivenessTimeoutMs.get())))
  }

  return (
    <ListRow
      action={
        <div className="flex items-center gap-2">
          <Input
            aria-label={t.settings.remoteLivenessTimeout.aria}
            className="w-24"
            inputMode="numeric"
            max={REMOTE_LIVENESS_TIMEOUT_BOUNDS.max}
            min={REMOTE_LIVENESS_TIMEOUT_BOUNDS.min}
            onBlur={commit}
            onChange={event => setDraft(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter') {
                event.currentTarget.blur()
              }
            }}
            type="number"
            value={draft}
          />
          <span className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">ms</span>
        </div>
      }
      description="How long to wait for a remote backend to answer a health check before treating it as unreachable. Raise this if a remote host under heavy load causes false 'cannot connect' errors."
      id={settingElementId(SETTING_IDS.advanced.remoteLivenessTimeout)}
      title={t.settings.remoteLivenessTimeout.title}
    />
  )
}
