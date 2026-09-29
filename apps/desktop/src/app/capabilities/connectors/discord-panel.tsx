import { useCallback, useEffect, useState } from 'react'

import {
  deleteDiscordDesk,
  getDiscordDesk,
  listDeskPlatforms,
  listDiscordChannels,
  saveDeskPlatform,
  saveDiscordDesk,
  type DiscordDeskStatus,
  type DiscordGuildChannels
} from '@/api/discord-desk'
import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { ConnectorLogo } from '@/components/ui/connector-logo'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import type { ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { notify, readableError } from '@/store/notifications'
import type { MessagingEnvVarInfo, MessagingPlatformInfo } from '@/types/hermes'

import { discordSnowflake, discordTokenAccepted } from './discord-form'

function editableFields(platform: MessagingPlatformInfo): MessagingEnvVarInfo[] {
  const visible = platform.env_vars.filter(field => !field.advanced && field.required)

  return visible.length > 0 ? visible : platform.env_vars.filter(field => field.required)
}

export function DiscordPanel({ profile }: { profile: ProfileScope }) {
  const { t } = useI18n()
  const copy = t.connectorsPage.discord
  const [status, setStatus] = useState<DiscordDeskStatus | null>(null)
  const [platforms, setPlatforms] = useState<MessagingPlatformInfo[]>([])
  const [pickerOpen, setPickerOpen] = useState(false)
  const [editorOpen, setEditorOpen] = useState(false)
  const [editing, setEditing] = useState<MessagingPlatformInfo | null>(null)
  const [pendingDelete, setPendingDelete] = useState<MessagingPlatformInfo | 'discord' | null>(null)
  const [token, setToken] = useState('')
  const [userId, setUserId] = useState('')
  const [channelId, setChannelId] = useState('')
  const [guilds, setGuilds] = useState<DiscordGuildChannels[]>([])
  const [values, setValues] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState<null | 'channels' | 'save'>(null)
  const [error, setError] = useState<null | string>(null)

  const load = useCallback(async () => {
    try {
      const next = await getDiscordDesk(profile)

      setStatus(next)
      setUserId(current => current || next.user_id || '')
      setChannelId(current => current || next.channel_id || '')
      setError(null)
    } catch (err) {
      setError(readableError(err, copy.loadFailed).message)
    }

    try {
      const catalog = await listDeskPlatforms(profile)

      setPlatforms(catalog.platforms)
    } catch {
      setPlatforms([])
    }
  }, [copy.loadFailed, profile])

  useEffect(() => {
    void load()
  }, [load])

  const tokenReady = discordTokenAccepted(token) || (token.trim() === '' && status?.token_set === true)
  const userReady = discordSnowflake(userId) !== null
  const channelReady = discordSnowflake(channelId) !== null
  const channelName =
    guilds.flatMap(guild => guild.channels).find(channel => channel.id === channelId)?.name ||
    status?.channel_name ||
    ''
  const others = platforms.filter(platform => platform.id !== 'discord' && platform.configured)
  const choices = platforms.filter(platform => platform.id !== 'discord' && editableFields(platform).length > 0)
  const fields = editing ? editableFields(editing) : []
  const fieldsReady = fields.every(field => field.is_set || (values[field.key] || '').trim() !== '')

  function openDiscordEditor() {
    setPickerOpen(false)
    setEditing(null)
    setToken('')
    setError(null)
    setEditorOpen(true)
  }

  function openPlatformEditor(platform: MessagingPlatformInfo) {
    setPickerOpen(false)
    setEditing(platform)
    setValues({})
    setError(null)
    setEditorOpen(true)
  }

  async function showChannels() {
    setBusy('channels')
    setError(null)

    try {
      const result = await listDiscordChannels(token.trim(), profile)

      setGuilds(result.guilds)
      const first = result.guilds[0]?.channels[0]

      setChannelId(current => current || first?.id || '')
    } catch (err) {
      setError(readableError(err, copy.loadFailed).message)
    } finally {
      setBusy(null)
    }
  }

  async function saveDiscord() {
    const user = discordSnowflake(userId)
    const channel = discordSnowflake(channelId)

    if (!user || !channel) {
      return
    }

    setBusy('save')
    setError(null)

    try {
      const result = await saveDiscordDesk(
        { channel_id: channel, channel_name: channelName || channel, token: token.trim(), user_id: user },
        profile
      )

      setToken('')
      setEditorOpen(false)
      notify({
        kind: 'success',
        message:
          result.restart_started === false && result.restart_error
            ? result.restart_error
            : copy.channelLine(result.channel_name),
        title: copy.saved
      })
      await load()
    } catch (err) {
      setError(readableError(err, copy.loadFailed).message)
    } finally {
      setBusy(null)
    }
  }

  async function savePlatform() {
    if (!editing) {
      return
    }

    const env = Object.fromEntries(
      fields
        .map(field => [field.key, (values[field.key] || '').trim()] as const)
        .filter(([, value]) => value !== '')
    )

    setBusy('save')
    setError(null)

    try {
      const result = await saveDeskPlatform(editing.id, { enabled: true, env }, profile)

      setEditorOpen(false)
      notify({
        kind: 'success',
        message: result.hot_served ? t.messaging.connectingLive : t.messaging.restartToReconnect,
        title: copy.savedName(editing.name)
      })
      await load()
    } catch (err) {
      setError(readableError(err, copy.loadFailed).message)
    } finally {
      setBusy(null)
    }
  }

  return (
    <section className="shrink-0 rounded-md border border-(--ui-stroke-tertiary) p-3" data-slot="discord-panel">
      <div className="flex items-center justify-end gap-3">
        <Button onClick={() => setPickerOpen(true)} size="xs" type="button" variant="outline">
          {copy.addAction}
        </Button>
      </div>

      <div className="mt-3 grid gap-2">
        {status?.configured ? (
          <ConnectedRow
            detail={status.channel_name ? copy.channelLine(status.channel_name) : undefined}
            name={copy.title}
            onDelete={() => setPendingDelete('discord')}
            onEdit={openDiscordEditor}
            slug="discord"
            status={copy.connectedName(copy.title)}
          />
        ) : null}
        {others.map(platform => (
          <ConnectedRow
            key={platform.id}
            name={platform.name}
            onDelete={() => setPendingDelete(platform)}
            onEdit={() => openPlatformEditor(platform)}
            slug={platform.id}
            status={copy.connectedName(platform.name)}
          />
        ))}
        {status && !status.configured && others.length === 0 ? (
          <p className="text-xs text-(--ui-text-tertiary)">{copy.none}</p>
        ) : null}
      </div>

      {error && !editorOpen ? <p className="mt-2 text-[0.7rem] text-(--ui-red)">{error}</p> : null}

      <Dialog onOpenChange={setPickerOpen} open={pickerOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{copy.addTitle}</DialogTitle>
            <DialogDescription>{copy.addBody}</DialogDescription>
          </DialogHeader>
          <div className="grid max-h-80 gap-2 overflow-y-auto">
            <PickerRow
              added={status?.configured === true}
              description={status?.configured ? copy.connectedName(copy.title) : copy.body}
              name={copy.title}
              onOpen={openDiscordEditor}
              slug="discord"
            />
            {choices.map(platform => (
              <PickerRow
                added={platform.configured}
                description={platform.configured ? copy.connectedName(platform.name) : platform.description}
                key={platform.id}
                name={platform.name}
                onOpen={() => openPlatformEditor(platform)}
                slug={platform.id}
              />
            ))}
          </div>
        </DialogContent>
      </Dialog>

      <Dialog
        onOpenChange={next => {
          setEditorOpen(next)
          if (!next) {
            setEditing(null)
          }
        }}
        open={editorOpen}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{editing ? editing.name : copy.title}</DialogTitle>
            <DialogDescription>{editing ? editing.description : copy.body}</DialogDescription>
          </DialogHeader>
          {editing ? (
            <div className="grid gap-3">
              {fields.map(field => (
                <label className="grid gap-1 text-xs text-(--ui-text-secondary)" key={field.key}>
                  {field.prompt}
                  <Input
                    autoComplete="off"
                    onChange={event => setValues(current => ({ ...current, [field.key]: event.currentTarget.value }))}
                    placeholder={field.is_set ? copy.keepSaved : undefined}
                    size="sm"
                    type={field.is_password ? 'password' : 'text'}
                    value={values[field.key] || ''}
                  />
                </label>
              ))}
              {error ? <p className="text-[0.7rem] text-(--ui-red)">{error}</p> : null}
            </div>
          ) : (
            <div className="grid gap-3">
              <label className="grid gap-1 text-xs text-(--ui-text-secondary)">
                {copy.token}
                <Input
                  autoComplete="off"
                  onChange={event => setToken(event.currentTarget.value)}
                  placeholder={status?.token_set ? copy.tokenKeep : undefined}
                  size="sm"
                  type="password"
                  value={token}
                />
              </label>
              <label className="grid gap-1 text-xs text-(--ui-text-secondary)">
                {copy.userId}
                <Input onChange={event => setUserId(event.currentTarget.value)} size="sm" value={userId} />
                <span className="text-[0.65rem] text-(--ui-text-tertiary)">{copy.userIdHint}</span>
              </label>
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  disabled={!tokenReady || busy !== null}
                  onClick={() => void showChannels()}
                  size="xs"
                  type="button"
                >
                  {busy === 'channels' ? copy.showingChannels : copy.showChannels}
                </Button>
                {guilds.length > 0 ? (
                  <select
                    className="h-7 min-w-48 rounded-md border border-input bg-transparent px-2 text-xs"
                    onChange={event => setChannelId(event.currentTarget.value)}
                    value={channelId}
                  >
                    {guilds.flatMap(guild =>
                      guild.channels.map(channel => (
                        <option key={channel.id} value={channel.id}>
                          {guild.guild} / #{channel.name}
                        </option>
                      ))
                    )}
                  </select>
                ) : (
                  <span className="text-[0.65rem] text-(--ui-text-tertiary)">{copy.channelEmpty}</span>
                )}
              </div>
              {error ? <p className="text-[0.7rem] text-(--ui-red)">{error}</p> : null}
            </div>
          )}
          <DialogFooter>
            <Button
              disabled={
                busy !== null || (editing ? !fieldsReady : !tokenReady || !userReady || !channelReady)
              }
              onClick={() => void (editing ? savePlatform() : saveDiscord())}
              size="sm"
              type="button"
            >
              {busy === 'save' ? copy.saving : copy.save}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        confirmLabel={copy.remove}
        description={
          pendingDelete && pendingDelete !== 'discord' ? copy.removeBodyFor(pendingDelete.name) : copy.removeBody
        }
        destructive
        onClose={() => setPendingDelete(null)}
        onConfirm={async () => {
          if (pendingDelete === 'discord') {
            const result = await deleteDiscordDesk(profile)

            notify({
              kind: 'success',
              message: result.restart_started === false && result.restart_error ? result.restart_error : copy.removed,
              title: copy.removed
            })
          } else if (pendingDelete) {
            await saveDeskPlatform(
              pendingDelete.id,
              {
                clear_env: pendingDelete.env_vars.filter(field => field.is_set).map(field => field.key),
                enabled: false
              },
              profile
            )
            notify({ kind: 'success', message: copy.removedName(pendingDelete.name), title: copy.removedName(pendingDelete.name) })
          }

          setEditorOpen(false)
          setToken('')
          await load()
        }}
        open={pendingDelete !== null}
        title={
          pendingDelete && pendingDelete !== 'discord' ? copy.removeTitleFor(pendingDelete.name) : copy.removeTitle
        }
      />
    </section>
  )
}

function ConnectedRow({
  detail,
  name,
  onDelete,
  onEdit,
  slug,
  status
}: {
  detail?: string
  name: string
  onDelete: () => void
  onEdit: () => void
  slug: string
  status: string
}) {
  const { t } = useI18n()
  const copy = t.connectorsPage.discord

  return (
    <div className="flex items-center gap-3 rounded-lg border border-(--ui-stroke-quaternary) bg-(--ui-bg-elevated) p-3">
      <ConnectorLogo className="size-9 shrink-0 rounded-[9px]" connector={{ name: slug, title: name }} />
      <div className="grid min-w-0 flex-1 gap-0.5">
        <span className="truncate text-[0.8125rem] font-semibold text-(--ui-text-primary)">{name}</span>
        <span className="inline-flex items-center gap-1 text-[0.6875rem] text-(--ui-text-secondary)">
          <span className="size-1.5 rounded-full bg-(--ui-green)" />
          {status}
        </span>
        {detail ? <span className="truncate text-[0.7rem] text-(--ui-text-tertiary)">{detail}</span> : null}
      </div>
      <Button onClick={onEdit} size="xs" type="button" variant="outline">
        {copy.edit}
      </Button>
      <Button className="text-destructive" onClick={onDelete} size="xs" type="button" variant="outline">
        {copy.remove}
      </Button>
    </div>
  )
}

function PickerRow({
  added,
  description,
  name,
  onOpen,
  slug
}: {
  added: boolean
  description: string
  name: string
  onOpen: () => void
  slug: string
}) {
  const { t } = useI18n()
  const copy = t.connectorsPage.discord

  return (
    <div className="flex items-center gap-3 rounded-lg border border-(--ui-stroke-quaternary) p-3">
      <ConnectorLogo className="size-8 shrink-0 rounded-[8px]" connector={{ name: slug, title: name }} />
      <div className="grid min-w-0 flex-1">
        <span className="text-sm font-medium text-(--ui-text-primary)">{name}</span>
        <span className="line-clamp-2 text-[0.7rem] text-(--ui-text-tertiary)">{description}</span>
      </div>
      <Button onClick={onOpen} size="xs" type="button" variant="outline">
        {added ? copy.edit : copy.pickerAdd}
      </Button>
    </div>
  )
}
