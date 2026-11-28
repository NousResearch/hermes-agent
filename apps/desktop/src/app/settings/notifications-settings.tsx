import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'
import type { NotificationSettingsTranslations } from '@/i18n/types_notifications'
import { COMPLETION_SOUND_VARIANTS, previewCompletionSound } from '@/lib/completion-sound'
import { triggerHaptic } from '@/lib/haptics'
import { hostHas } from '@/lib/host-capabilities'
import { Bell, Play } from '@/lib/icons'
import { coarseElapsed } from '@/lib/time'
import { cn } from '@/lib/utils'
import {
  $browserNotificationPermission,
  askBrowserNotificationPermission,
  refreshBrowserNotificationPermission
} from '@/store/browser-notification-permission'
import { $completionSoundVariantId, setCompletionSoundVariantId } from '@/store/completion-sound'
import { $gateway } from '@/store/gateway'
import { $hapticsMuted, setHapticsMuted } from '@/store/haptics'
import {
  $nativeNotifyPrefs,
  NATIVE_NOTIFICATION_KINDS,
  sendTestNativeNotification,
  setNativeDockBounce,
  setNativeNotifyEnabled,
  setNativeNotifyKind
} from '@/store/native-notifications'
import { notify } from '@/store/notifications'
import { type PresenceInputAge, windowPresenceReporter } from '@/store/presence'

import { CONTROL_TEXT } from './constants'
import { ListRow, SectionHeading, SettingsContent, ToggleRow } from './primitives'
import { notificationKindSettingId, SETTING_IDS, settingElementId } from './settings-manifest'
import { useSettingDeepLink } from './use-setting-deep-link'

const CAPTION = 'text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)'

function Caption({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cn(CAPTION, className)}>{children}</p>
}

/** Webapp only: the browser's notification permission, asked from a click here
 *  or on Send test notification, never on page load. The tab title flashes
 *  whatever the answer. */
function BrowserPermissionRow() {
  const { t } = useI18n()
  const copy = t.settings.notifications.browserPermission
  const permission = useStore($browserNotificationPermission)

  useEffect(refreshBrowserNotificationPermission, [])

  return (
    <ListRow
      action={
        permission === 'default' ? (
          <Button
            onClick={() => void askBrowserNotificationPermission()}
            size="sm"
            type="button"
            variant="outline"
          >
            {copy.allow}
          </Button>
        ) : undefined
      }
      description={copy.states[permission]}
      id={settingElementId(SETTING_IDS.notifications.browserPermission)}
      title={copy.title}
    />
  )
}

const PRESENCE_READOUT_MS = 5000

/** "4 sec", "2 min": a duration in the locale's short units, for sentences that place it themselves. */
function formatElapsed(seconds: number, locale: string): string {
  const { unit, value } = coarseElapsed(seconds * 1000)

  return new Intl.NumberFormat(locale, { style: 'unit', unit, unitDisplay: 'short' }).format(value)
}

type PresenceReadout = { kind: 'disconnected' } | { kind: 'unsupported' } | ({ kind: 'watching' } & PresenceInputAge)

/** What this window would do with a conversation open: decided by its main connection (the row cannot
 *  know which conversation will be opened) and the evidence the reporter holds now. */
function readPresence(): PresenceReadout {
  const gateway = $gateway.get()

  if (gateway?.connectionState !== 'open') {
    return { kind: 'disconnected' }
  }

  const age = gateway.clientActivity ? windowPresenceReporter()?.inputAge() : undefined

  return age ? { kind: 'watching', ...age } : { kind: 'unsupported' }
}

function presenceDescription(readout: PresenceReadout, copy: NotificationSettingsTranslations['phonePresence'], locale: string) {
  if (readout.kind !== 'watching') {
    return readout.kind === 'disconnected' ? copy.disconnected : copy.unsupported
  }

  // Only page input can be missing: the OS idle clock always has an age.
  if (readout.idleSeconds === null) {
    return copy.watchingWindowNoInput
  }

  return (readout.basis === 'system' ? copy.watchingComputer : copy.watchingWindow)(formatElapsed(readout.idleSeconds, locale))
}

/** Read-only: what this window tells Hermes while a conversation is open in it, with the evidence it
 *  has now. What a phone does with it is the plugin's decision. */
function PhonePresenceRow() {
  const { locale, t } = useI18n()
  const copy = t.settings.notifications.phonePresence
  const [readout, setReadout] = useState<PresenceReadout>(readPresence)

  // Read in a timer, not during render: the evidence moves with the clock, not with any prop or store.
  useEffect(() => {
    const read = () => setReadout(readPresence())
    read()
    const timer = setInterval(read, PRESENCE_READOUT_MS)

    return () => clearInterval(timer)
  }, [])

  return (
    <ListRow
      description={
        <>
          {copy.description} {presenceDescription(readout, copy, locale)}
        </>
      }
      id={settingElementId(SETTING_IDS.notifications.phonePresence)}
      title={copy.title}
    />
  )
}

interface NotificationsSettingsProps {
  subpage?: string
}

export function NotificationsSettings({ subpage }: NotificationsSettingsProps = {}) {
  const { t } = useI18n()
  const prefs = useStore($nativeNotifyPrefs)
  const muted = useStore($hapticsMuted)
  const completionSoundVariantId = useStore($completionSoundVariantId)
  const copy = t.settings.notifications
  const showAlerts = subpage === undefined || subpage === 'alerts'
  const showSounds = subpage === undefined || subpage === 'sounds'

  useSettingDeepLink('notifications', page => subpage === undefined || page === subpage)

  const runTest = async () => {
    triggerHaptic('open')

    if (hostHas('browserNotifications')) {
      await askBrowserNotificationPermission()
    }

    const ok = await sendTestNativeNotification(copy.testTitle, copy.testBody)
    notify({ kind: ok ? 'info' : 'error', message: ok ? copy.testSent : copy.testUnsupported })
  }

  return (
    <SettingsContent>
      {showAlerts && (
        <>
          {subpage === undefined && <SectionHeading icon={Bell} title={copy.title} />}
          <Caption className="mb-2 leading-(--conversation-caption-line-height)">{copy.intro}</Caption>

          <ToggleRow
            checked={prefs.enabled}
            description={copy.enableAllDesc}
            id={settingElementId(SETTING_IDS.notifications.enableAll)}
            label={copy.enableAll}
            onChange={setNativeNotifyEnabled}
          />

          {hostHas('browserNotifications') && <BrowserPermissionRow />}

          {NATIVE_NOTIFICATION_KINDS.map(kind => (
            <ToggleRow
              checked={prefs.enabled && prefs.kinds[kind]}
              description={copy.kinds[kind].description}
              disabled={!prefs.enabled}
              id={settingElementId(notificationKindSettingId(kind))}
              key={kind}
              label={copy.kinds[kind].label}
              onChange={on => setNativeNotifyKind(kind, on)}
            />
          ))}

          <ToggleRow
            checked={prefs.dockBounce}
            description={copy.dockBounceDesc}
            disabled={!prefs.enabled}
            id={settingElementId(SETTING_IDS.notifications.dockBounce)}
            label={copy.dockBounce}
            onChange={setNativeDockBounce}
          />

          <PhonePresenceRow />
        </>
      )}

      {showSounds && (
        <>
          <ToggleRow
            checked={muted}
            description={copy.muteSoundsDesc}
            id={settingElementId(SETTING_IDS.notifications.muteSounds)}
            label={copy.muteSoundsTitle}
            onChange={setHapticsMuted}
          />
          <ListRow
            action={
              <>
                <Select
                  onValueChange={value => {
                    const variantId = Number.parseInt(value, 10)

                    setCompletionSoundVariantId(variantId)
                    previewCompletionSound(variantId)
                    triggerHaptic('selection')
                  }}
                  value={String(completionSoundVariantId)}
                >
                  <SelectTrigger className={cn('min-w-56', CONTROL_TEXT)}>
                    <SelectValue />
                  </SelectTrigger>

                  <SelectContent>
                    {COMPLETION_SOUND_VARIANTS.map(variant => (
                      <SelectItem key={variant.id} value={String(variant.id)}>
                        {variant.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <Button
                  className="gap-1.5"
                  onClick={() => {
                    previewCompletionSound()
                    triggerHaptic('crisp')
                  }}
                  size="sm"
                  type="button"
                  variant="outline"
                >
                  <Play className="size-3.5" />
                  {copy.completionSoundPreview}
                </Button>
              </>
            }
            description={copy.completionSoundDesc}
            id={settingElementId(SETTING_IDS.notifications.completionSound)}
            title={copy.completionSoundTitle}
          />
        </>
      )}

      {showAlerts && (
        <div className="mt-4 flex flex-col gap-2">
          <Button className="self-start" onClick={() => void runTest()} size="sm" type="button" variant="outline">
            <Bell />
            {copy.test}
          </Button>
          <Caption>{copy.focusedHint}</Caption>
        </div>
      )}
    </SettingsContent>
  )
}
