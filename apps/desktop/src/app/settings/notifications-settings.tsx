import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Bell } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $nativeNotifyPrefs,
  NATIVE_NOTIFICATION_KINDS,
  sendTestNativeNotification,
  setNativeNotifyEnabled,
  setNativeNotifyKind
} from '@/store/native-notifications'
import { notify } from '@/store/notifications'

import { ListRow, SectionHeading, SettingsContent } from './primitives'

const CAPTION = 'text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)'

function Caption({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cn(CAPTION, className)}>{children}</p>
}

function ToggleRow(props: {
  checked: boolean
  description: string
  disabled?: boolean
  label: string
  onChange: (on: boolean) => void
}) {
  return (
    <ListRow
      action={
        <Switch
          aria-label={props.label}
          checked={props.checked}
          disabled={props.disabled}
          onCheckedChange={on => {
            triggerHaptic('selection')
            props.onChange(on)
          }}
        />
      }
      description={props.description}
      title={props.label}
    />
  )
}

export function NotificationsSettings() {
  const { t } = useI18n()
  const prefs = useStore($nativeNotifyPrefs)
  const copy = t.settings.notifications

  const runTest = async () => {
    triggerHaptic('open')
    const ok = await sendTestNativeNotification(copy.testTitle, copy.testBody)
    notify({ kind: ok ? 'info' : 'error', message: ok ? copy.testSent : copy.testUnsupported })
  }

  return (
    <SettingsContent>
      <SectionHeading icon={Bell} title={copy.title} />
      <Caption className="mb-2 leading-(--conversation-caption-line-height)">{copy.intro}</Caption>

      <ToggleRow
        checked={prefs.enabled}
        description={copy.enableAllDesc}
        label={copy.enableAll}
        onChange={setNativeNotifyEnabled}
      />

      <div className="my-1 h-px bg-border/30" />

      {NATIVE_NOTIFICATION_KINDS.map(kind => (
        <ToggleRow
          checked={prefs.enabled && prefs.kinds[kind]}
          description={copy.kinds[kind].description}
          disabled={!prefs.enabled}
          key={kind}
          label={copy.kinds[kind].label}
          onChange={on => setNativeNotifyKind(kind, on)}
        />
      ))}

      <div className="mt-4 flex flex-col gap-2">
        <Button className="self-start" onClick={() => void runTest()} size="sm" type="button" variant="outline">
          <Bell />
          {copy.test}
        </Button>
        <Caption>{copy.focusedHint}</Caption>
      </div>
    </SettingsContent>
  )
}
