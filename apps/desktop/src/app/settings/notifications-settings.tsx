import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'
import { useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { useI18n } from '@/i18n'
import {
  CUSTOM_AUDIO_VARIANT_ID,
  COMPLETION_SOUND_VARIANTS,
  getCustomAudioDataUrl,
  previewCompletionSound,
  setCustomAudioDataUrl
} from '@/lib/completion-sound'
import { triggerHaptic } from '@/lib/haptics'
import { Bell, Play } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $completionSoundVariantId, $completionSoundVolume, setCompletionSoundVariantId, setCompletionSoundVolume } from '@/store/completion-sound'
import {
  $nativeNotifyPrefs,
  NATIVE_NOTIFICATION_KINDS,
  sendTestNativeNotification,
  setNativeNotifyEnabled,
  setNativeNotifyKind
} from '@/store/native-notifications'
import { notify } from '@/store/notifications'

import { CONTROL_TEXT } from './constants'
import { ListRow, SectionHeading, SettingsContent, ToggleRow } from './primitives'

const CAPTION = 'text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)'

function Caption({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cn(CAPTION, className)}>{children}</p>
}

function fileNameFromDataUrl(dataUrl: string): string {
  try {
    const parsed = new URL(dataUrl)
    const name = parsed.searchParams.get('name')

    return name || 'custom-audio'
  } catch {
    return 'custom-audio'
  }
}

export function NotificationsSettings() {
  const { t } = useI18n()
  const prefs = useStore($nativeNotifyPrefs)
  const completionSoundVariantId = useStore($completionSoundVariantId)
  const completionSoundVolume = useStore($completionSoundVolume)
  const copy = t.settings.notifications
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [customAudioName, setCustomAudioName] = useState<string | null>(() => {
    const dataUrl = getCustomAudioDataUrl()

    return dataUrl ? fileNameFromDataUrl(dataUrl) : null
  })
  // Draft volume (percent) tracked locally so the thumb follows the pointer
  // while dragging. The store is only written on commit to avoid replaying the
  // sound on every pixel of movement.
  const [draftVolumePercent, setDraftVolumePercent] = useState(() => Math.round(completionSoundVolume * 100))

  const isCustomSelected = completionSoundVariantId === CUSTOM_AUDIO_VARIANT_ID

  const runTest = async () => {
    triggerHaptic('open')
    const ok = await sendTestNativeNotification(copy.testTitle, copy.testBody)
    notify({ kind: ok ? 'info' : 'error', message: ok ? copy.testSent : copy.testUnsupported })
  }

  const handleCustomFile = async (file: File) => {
    // localStorage holds the file as a base64 data URL; cap the payload so a
    // giant file can't silently fail the quota. ~2MB of audio → ~2.7MB base64.
    if (file.size > 2 * 1024 * 1024) {
      notify({ kind: 'error', message: copy.completionSoundCustomTooLarge })

      return
    }

    try {
      const dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()

        reader.onload = () => resolve(String(reader.result))
        reader.onerror = () => reject(reader.error)
        reader.readAsDataURL(file)
      })

      setCustomAudioDataUrl(dataUrl)
      setCustomAudioName(file.name)

      // Preview the freshly loaded file immediately so the user hears it.
      previewCompletionSound(CUSTOM_AUDIO_VARIANT_ID)
      notify({ kind: 'info', message: copy.completionSoundCustomLoaded(file.name) })
    } catch {
      notify({ kind: 'error', message: copy.testUnsupported })
    }
  }

  const removeCustomAudio = () => {
    setCustomAudioDataUrl(null)
    setCustomAudioName(null)
    triggerHaptic('selection')
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

      <ListRow
        action={
          <div className="flex flex-wrap items-center justify-end gap-2">
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
                <SelectItem value={String(CUSTOM_AUDIO_VARIANT_ID)}>{copy.completionSoundCustom}</SelectItem>
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
          </div>
        }
        description={copy.completionSoundDesc}
        title={copy.completionSoundTitle}
      />

      <ListRow
        action={
          <div className="flex w-full min-w-56 items-center gap-3 @2xl:w-72">
            <Slider
              aria-label={copy.completionSoundVolumeLabel}
              max={400}
              min={0}
              onValueChange={([value]) => setDraftVolumePercent(value)}
              onValueCommit={([value]) => {
                setCompletionSoundVolume(value / 100)
                setDraftVolumePercent(value)
                previewCompletionSound()
                triggerHaptic('selection')
              }}
              step={5}
              value={[draftVolumePercent]}
            />
            <span className="w-10 shrink-0 text-right tabular-nums text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
              {draftVolumePercent}%
            </span>
          </div>
        }
        description={copy.completionSoundVolumeDesc(draftVolumePercent)}
        title={copy.completionSoundVolumeLabel}
      />

      {isCustomSelected && (
        <ListRow
          action={
            <div className="flex flex-wrap items-center justify-end gap-2">
              <input
                accept="audio/*,.mp3,.wav,.ogg,.m4a,.flac"
                className="hidden"
                onChange={event => {
                  const file = event.target.files?.[0]

                  if (file) {
                    void handleCustomFile(file)
                  }

                  event.target.value = ''
                }}
                ref={fileInputRef}
                type="file"
              />
              <Button
                onClick={() => fileInputRef.current?.click()}
                size="sm"
                type="button"
                variant="outline"
              >
                {copy.completionSoundCustomChoose}
              </Button>
              {customAudioName && (
                <Button
                  onClick={removeCustomAudio}
                  size="sm"
                  type="button"
                  variant="ghost"
                >
                  {copy.completionSoundCustomRemove}
                </Button>
              )}
            </div>
          }
          below={
            customAudioName ? (
              <div className="mt-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                {copy.completionSoundCustomLoaded(customAudioName)}
              </div>
            ) : null
          }
          description={copy.completionSoundCustomDesc}
          title={copy.completionSoundCustom}
        />
      )}

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
