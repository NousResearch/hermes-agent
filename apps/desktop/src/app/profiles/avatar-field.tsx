import { useCallback } from 'react'

import { ProfileAvatar } from '@/components/profile-avatar'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { downscaleAvatar } from '@/lib/avatar-image'

// Open the OS file picker, read the chosen image, and downscale it to a small
// square data URL ready to PUT as a profile avatar. Returns null when the user
// cancels; throws (with a user-facing message) when reading/decoding fails so
// callers can surface it via their existing error path.
export function useAvatarPicker() {
  const { t } = useI18n()

  return useCallback(async (): Promise<null | string> => {
    const paths = await window.hermesDesktop?.selectPaths({
      title: t.profiles.pickPictureTitle,
      multiple: false,
      filters: [{ name: t.composer.images, extensions: ['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'] }]
    })

    const path = paths?.[0]

    if (!path) {
      return null
    }

    const dataUrl = await window.hermesDesktop.readFileDataUrl(path)

    return downscaleAvatar(dataUrl)
  }, [t])
}

// Form row for picking a profile picture in the create flow. Shows a square
// preview (the picked image, or the colored-initial fallback for the typed
// name) beside choose/remove buttons. Follows the dialog's label/hint pattern.
export function AvatarField({
  busy,
  name,
  onChange,
  onError,
  value
}: {
  busy?: boolean
  name: string
  onChange: (value: null | string) => void
  onError: (message: string) => void
  value: null | string
}) {
  const { t } = useI18n()
  const p = t.profiles
  const pickAvatar = useAvatarPicker()

  const handlePick = useCallback(async () => {
    try {
      const next = await pickAvatar()

      if (next) {
        onChange(next)
      }
    } catch (err) {
      onError(err instanceof Error ? err.message : p.failedSavePicture)
    }
  }, [onChange, onError, p, pickAvatar])

  return (
    <div className="grid gap-1.5">
      {/* Not a <label>: there's no form control to bind — the buttons carry
          their own accessible names. */}
      <span className="text-xs font-medium">
        {p.pictureLabel} <span className="font-normal text-muted-foreground">- {p.pictureOptional}</span>
      </span>
      <div className="flex items-center gap-3">
        {value ? (
          <span className="inline-grid size-14 shrink-0 place-items-center overflow-hidden rounded-lg">
            <img alt="" className="size-full object-cover" draggable={false} src={value} />
          </span>
        ) : (
          <ProfileAvatar className="size-14 rounded-lg text-xl" name={name || '?'} />
        )}
        <div className="flex items-center gap-1.5">
          <Button disabled={busy} onClick={() => void handlePick()} size="sm" type="button" variant="outline">
            {value ? p.changePicture : p.choosePicture}
          </Button>
          {value && (
            <Button disabled={busy} onClick={() => onChange(null)} size="sm" type="button" variant="ghost">
              {p.removePicture}
            </Button>
          )}
        </div>
      </div>
      <p className="text-[0.66rem] leading-4 text-muted-foreground">{p.pictureHint}</p>
    </div>
  )
}
