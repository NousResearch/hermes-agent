/**
 * Edit Profile: appearance, title, description, and the advanced profile
 * config disclosure for an existing bot.
 */

import {
  Button,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DisclosureCaret,
  host,
  Input,
  queryClient,
  Textarea,
  useI18n,
  useValue
} from '@hermes/plugin-sdk'
import { useState } from 'react'

import { avatarColor, botAppearance, BotFace } from './avatar'
import { AvatarPicker } from './avatar-picker'
import { $botMeta, botSelectionKey, ROSTER_KEY, saveBotMeta } from './data'
import { labeled } from './dialog-parts'
import { useBots } from './i18n'
import { displayName } from './labels'
import { AdvancedProfileConfig, applyAdvancedConfig, emptyAdvancedState } from './profile-config'
import { botRosterMeta, requestForBot } from './routing'
import type { AvatarAppearance, RosterRow } from './types'

// ── edit profile dialog ──────────────────────────────────────────────────────

/** AvatarAppearance, minus the `image` guarantee — the dialog's no-bot fallback
 *  literal doesn't supply one. See the TODO in EditProfileDialog. */
interface EditProfileAppearance extends Omit<AvatarAppearance, 'image'> {
  image?: null | string
}
interface EditProfileDialogProps {
  bot: null | RosterRow
  onClose: () => void
  open: boolean
}
export function EditProfileDialog({ bot, open, onClose }: EditProfileDialogProps) {
  const { t } = useI18n()
  const b = useBots()
  const metaAll = useValue($botMeta)
  const meta = bot ? botRosterMeta(bot, metaAll) : null
  // TODO(bot-mode-types): the no-bot fallback omits `image`, which the state
  // seeding below reads — `appearance.image` is undefined on that branch, so
  // this is NOT an AvatarAppearance. Harmless today only because the component
  // returns null a few lines down when `bot` is null, so the seeded value is
  // thrown away before anything renders.
  const appearance: EditProfileAppearance = bot
    ? botAppearance(bot.name, meta)
    : {
        shape: 'circle',
        color: null
      }
  const [shape, setShape] = useState(appearance.shape)
  const [color, setColor] = useState<null | string>(appearance.color)
  const [image, setImage] = useState<null | string>(appearance.image ?? null)
  const [title, setTitle] = useState(meta?.title || '')
  const [description, setDescription] = useState(bot?.description || '')
  const [busy, setBusy] = useState(false)
  const [advanced, setAdvanced] = useState(false)
  const [adv, setAdv] = useState(emptyAdvancedState())

  // Re-seed local state each time a different bot opens the dialog.
  const [seedKey, setSeedKey] = useState<null | string>(null)
  const currentKey = bot ? `${botSelectionKey(bot)}:${open}` : null
  if (currentKey !== seedKey) {
    setSeedKey(currentKey)
    if (bot && open) {
      setShape(appearance.shape)
      setColor(appearance.color)
      setImage(appearance.image ?? null)
      setTitle(meta?.title || '')
      setDescription(bot.description || '')
      setBusy(false)
      setAdvanced(false)
      setAdv(emptyAdvancedState())
    }
  }
  if (!bot) {
    return null
  }
  const submit = async () => {
    if (busy) {
      return
    }
    setBusy(true)
    let advancedFailed = false
    const persistence = await saveBotMeta(bot, {
      shape,
      color: color ?? undefined,
      image,
      imageKind: image ? 'photo' : 'shape',
      title: title.trim(),
      custom: true
    })
    // Only an explicit remote failure is an error — 'unsupported' is the
    // documented older-gateway fallback (local wins, silently), and toasting
    // it would flag every save on every legacy setup forever.
    const lookFailed = persistence.serverOutcome === 'failed'
    if (lookFailed) {
      host.notify({
        kind: 'error',
        message: b.avatar.savedLocally
      })
    }
    if (persistence.serverOutcome === 'persisted') {
      queryClient.invalidateQueries({
        queryKey: ROSTER_KEY
      })
    }
    const desc = description.trim()
    if (desc !== (bot.description || '').trim()) {
      try {
        await requestForBot(bot, 'cli.exec', {
          argv: ['profile', 'describe', bot.name, '--text', desc]
        })
        queryClient.invalidateQueries({
          queryKey: ROSTER_KEY
        })
      } catch (err) {
        host.notifyError(err, b.avatar.savedLocallyDescriptionFailed)
      }
    }
    if (adv.loaded && (adv.dirtyModel || adv.dirtySoul || adv.dirtySkills || adv.dirtyToolsets || adv.dirtyMcp)) {
      try {
        const res = await applyAdvancedConfig(bot, adv)
        const failed = Object.entries(res?.applied || {}).filter(([, ok]) => !ok)
        if (failed.length) {
          advancedFailed = true
          host.notify({
            kind: 'error',
            message: `Some sections failed: ${failed.map(([k]) => k).join(', ')}`
          })
        }
      } catch (err) {
        advancedFailed = true
        host.notifyError(err, b.bot.advancedFailed)
      }
    }
    if (!advancedFailed && !lookFailed) {
      host.notify({
        kind: 'success',
        message: `${displayName(bot, {
          title
        })} updated`
      })
    }
    setBusy(false)
    onClose()
  }
  return (
    <Dialog open={open} onOpenChange={value => !value && !busy && onClose()}>
      <DialogContent
        className={advanced ? 'max-w-3xl' : 'max-w-sm'} // Same resizable-window treatment as the create dialog.
        style={
          advanced
            ? {
                resize: 'both',
                overflow: 'auto',
                minWidth: 420,
                minHeight: 360,
                maxWidth: '95vw',
                maxHeight: '90vh'
              }
            : undefined
        }
      >
        <DialogHeader>
          <DialogTitle>{b.bot.editTitle}</DialogTitle>
          <DialogDescription>{`Appearance and role for ${displayName(bot, null)} (${bot.name}).`}</DialogDescription>
        </DialogHeader>
        <div className="grid gap-4">
          <div className="flex justify-center py-1">
            <BotFace shape={shape} color={avatarColor(color, bot.name)} image={image} size={64} name={bot.name} />
          </div>
          <AvatarPicker
            shape={shape}
            color={color}
            image={image}
            onShape={setShape}
            onColor={setColor}
            onImage={setImage}
            generateSeed={{
              name: bot.name,
              title,
              description
            }}
          />
          {labeled(
            'Title',
            <Input
              placeholder={displayName(bot, null)}
              value={title}
              onChange={event => setTitle(event.target.value)}
            />
          )}
          {labeled(
            'Description',
            <Textarea
              className="min-h-16"
              placeholder={b.bot.helpPromptPlaceholder}
              value={description}
              onChange={event => setDescription(event.target.value)}
            />
          )}
          <Button
            className="flex items-center gap-1 text-xs font-medium text-(--ui-text-tertiary) hover:text-(--ui-text-secondary)"
            onClick={() => setAdvanced(v => !v)}
            size="inline"
            variant="text"
          >
            <DisclosureCaret open={advanced} />
            {b.bot.advancedHint}
          </Button>
          {advanced ? (
            <div className="rounded-md border border-(--ui-stroke-secondary) p-3">
              <AdvancedProfileConfig bot={bot} state={adv} setState={setAdv} />
            </div>
          ) : null}
        </div>
        <DialogFooter>
          <Button variant="ghost" disabled={busy} onClick={onClose}>
            {t.common.cancel}
          </Button>
          <Button disabled={busy} onClick={submit}>
            {busy ? t.common.saving : t.common.save}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
