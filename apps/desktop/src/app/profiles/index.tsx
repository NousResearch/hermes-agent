import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import {
  createProfile,
  deleteProfile,
  getProfiles,
  getProfileSetupCommand,
  getProfileSoul,
  type ProfileInfo,
  renameProfile,
  updateProfileSoul
} from '@/hermes'
import { useTranslation } from '@/i18n'
import { AlertTriangle, Pencil, Save, Terminal, Trash2, Users } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { OverlayMain, OverlaySidebar, OverlaySplitLayout } from '../overlays/overlay-split-layout'
import { OverlayView } from '../overlays/overlay-view'

const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

const PROFILE_NAME_HINT_KEY = 'profiles.nameHint'

function isValidProfileName(name: string): boolean {
  return PROFILE_NAME_RE.test(name.trim())
}

interface ProfilesViewProps {
  onClose: () => void
}

export function ProfilesView({ onClose }: ProfilesViewProps) {
  const t = useTranslation()
  const [profiles, setProfiles] = useState<null | ProfileInfo[]>(null)
  const [selectedName, setSelectedName] = useState<null | string>(null)
  const [createOpen, setCreateOpen] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<null | ProfileInfo>(null)
  const [deleting, setDeleting] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const { profiles: list } = await getProfiles()
      setProfiles(list)
      setSelectedName(current => {
        if (current && list.some(p => p.name === current)) {
          return current
        }

        return list.find(p => p.is_default)?.name ?? list[0]?.name ?? null
      })
    } catch (err) {
      notifyError(err, t('profiles.notifications.loadFailed'))
    }
  }, [t])

  useRefreshHotkey(refresh)

  useEffect(() => {
    void refresh()
  }, [refresh])

  const selected = useMemo(() => {
    if (!profiles) {
      return null
    }

    return profiles.find(p => p.name === selectedName) ?? profiles[0] ?? null
  }, [profiles, selectedName])

  const handleCreate = useCallback(
    async (name: string, cloneFromDefault: boolean) => {
      const trimmed = name.trim()

      if (!isValidProfileName(trimmed)) {
        throw new Error(t('profiles.invalidName'))
      }

      await createProfile({ name: trimmed, clone_from_default: cloneFromDefault })
      notify({ kind: 'success', title: t('profiles.notifications.created'), message: trimmed })
      setSelectedName(trimmed)
      await refresh()
    },
    [refresh, t]
  )

  const handleRename = useCallback(
    async (from: string, to: string): Promise<void> => {
      const target = to.trim()

      if (target === from) {
        return
      }

      if (!isValidProfileName(target)) {
        throw new Error(t('profiles.invalidName'))
      }

      await renameProfile(from, target)
      notify({ kind: 'success', title: t('profiles.notifications.renamed'), message: `${from} → ${target}` })
      setSelectedName(target)
      await refresh()
    },
    [refresh, t]
  )

  const handleConfirmDelete = useCallback(async () => {
    if (!pendingDelete) {
      return
    }

    setDeleting(true)

    try {
      await deleteProfile(pendingDelete.name)
      notify({ kind: 'success', title: t('profiles.notifications.deleted'), message: pendingDelete.name })
      setPendingDelete(null)
      setSelectedName(null)
      await refresh()
    } catch (err) {
      notifyError(err, t('profiles.notifications.deleteFailed'))
    } finally {
      setDeleting(false)
    }
  }, [pendingDelete, refresh, t])

  return (
    <OverlayView closeLabel={t('profiles.close')} onClose={onClose}>
      {!profiles ? (
        <PageLoader label={t('profiles.loading')} />
      ) : (
        <OverlaySplitLayout>
          <OverlaySidebar>
            <div className="mb-1 flex items-center justify-between gap-2 pl-1.5 pr-0.5">
              <span className="text-[0.7rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary)">
                {t('profiles.title')}
              </span>
              <Button
                aria-label={t('profiles.newProfile')}
                className="text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground"
                onClick={() => setCreateOpen(true)}
                size="icon-xs"
                variant="ghost"
              >
                <Codicon name="add" size="0.875rem" />
              </Button>
            </div>
            {profiles.map(profile => (
              <ProfileRow
                active={selected?.name === profile.name}
                key={profile.name}
                onSelect={() => setSelectedName(profile.name)}
                profile={profile}
              />
            ))}
            {profiles.length === 0 && (
              <p className="px-1.5 py-3 text-xs text-muted-foreground">{t('profiles.empty')}</p>
            )}
          </OverlaySidebar>

          <OverlayMain className="px-0">
            {selected ? (
              <ProfileDetail
                key={selected.name}
                onDelete={() => setPendingDelete(selected)}
                onRename={newName => handleRename(selected.name, newName)}
                profile={selected}
              />
            ) : (
              <div className="grid h-full place-items-center px-6 py-12 text-center text-sm text-muted-foreground">
                <div>
                  <Users className="mx-auto size-6 text-muted-foreground/60" />
                  <p className="mt-3">{t('profiles.selectPrompt')}</p>
                </div>
              </div>
            )}
          </OverlayMain>
        </OverlaySplitLayout>
      )}

      <CreateProfileDialog
        onClose={() => setCreateOpen(false)}
        onCreate={async (name, cloneFromDefault) => handleCreate(name, cloneFromDefault)}
        open={createOpen}
      />

      <Dialog onOpenChange={open => !open && !deleting && setPendingDelete(null)} open={pendingDelete !== null}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{t('profiles.deleteDialog.title')}</DialogTitle>
            <DialogDescription>
              {pendingDelete ? (
                <>
                  {t('profiles.deleteDialog.before')}{' '}
                  <span className="font-medium text-foreground">{pendingDelete.name}</span>{' '}
                  {t('profiles.deleteDialog.middle')} <span className="font-mono text-xs">{pendingDelete.path}</span>{' '}
                  {t('profiles.deleteDialog.after')}
                </>
              ) : null}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button disabled={deleting} onClick={() => setPendingDelete(null)} variant="outline">
              {t('common.cancel')}
            </Button>
            <Button disabled={deleting} onClick={() => void handleConfirmDelete()} variant="destructive">
              {deleting ? t('profiles.deleting') : t('profiles.delete')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </OverlayView>
  )
}

function ProfileRow({ active, onSelect, profile }: { active: boolean; onSelect: () => void; profile: ProfileInfo }) {
  const t = useTranslation()

  return (
    <button
      className={cn(
        'flex w-full flex-col items-start gap-0.5 rounded-md px-2 py-1.5 text-left transition-colors',
        active
          ? 'bg-(--ui-row-active-background) text-foreground'
          : 'text-(--ui-text-secondary) hover:bg-(--ui-row-hover-background) hover:text-foreground'
      )}
      onClick={onSelect}
      type="button"
    >
      <span className="flex w-full items-center justify-between gap-2">
        <span className="truncate text-sm font-medium">{profile.name}</span>
        {profile.is_default && <span className="text-[0.6rem] text-primary">{t('profiles.default')}</span>}
      </span>
      <span className="text-[0.66rem] text-muted-foreground">
        {t('profiles.skillCount', { count: profile.skill_count })}
        {profile.has_env ? ' · env' : ''}
      </span>
    </button>
  )
}

function ProfileDetail({
  onDelete,
  onRename,
  profile
}: {
  onDelete: () => void
  onRename: (newName: string) => Promise<void>
  profile: ProfileInfo
}) {
  const t = useTranslation()
  const [renameOpen, setRenameOpen] = useState(false)
  const [copying, setCopying] = useState(false)

  const handleCopySetup = useCallback(async () => {
    setCopying(true)

    try {
      const { command } = await getProfileSetupCommand(profile.name)
      await navigator.clipboard.writeText(command)
      notify({ kind: 'success', title: t('profiles.notifications.setupCopied'), message: command })
    } catch (err) {
      notifyError(err, t('profiles.notifications.setupCopyFailed'))
    } finally {
      setCopying(false)
    }
  }, [profile.name, t])

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-2xl space-y-6 px-6 py-6">
          <header className="space-y-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-xl font-semibold tracking-tight">{profile.name}</h3>
                  {profile.is_default && <Badge>{t('profiles.default')}</Badge>}
                  {profile.has_env && <Badge variant="muted">.env</Badge>}
                </div>
                <p className="mt-1 font-mono text-[0.7rem] text-muted-foreground" title={profile.path}>
                  {profile.path}
                </p>
              </div>
              <div className="flex shrink-0 items-center gap-3">
                {!profile.is_default && (
                  <Button onClick={() => setRenameOpen(true)} size="sm" variant="text">
                    <Pencil />
                    {t('profiles.rename')}
                  </Button>
                )}
                <Button disabled={copying} onClick={() => void handleCopySetup()} size="sm" variant="text">
                  <Terminal />
                  {copying ? t('profiles.copying') : t('profiles.copySetup')}
                </Button>
                {!profile.is_default && (
                  <Button
                    className="hover:text-destructive hover:no-underline"
                    onClick={onDelete}
                    size="sm"
                    variant="text"
                  >
                    <Trash2 />
                    {t('profiles.delete')}
                  </Button>
                )}
              </div>
            </div>

            <dl className="grid gap-2 text-xs sm:grid-cols-2">
              <DetailRow label={t('profiles.model')}>
                {profile.model ? (
                  <>
                    <span className="font-mono">{profile.model}</span>
                    {profile.provider && <span className="text-muted-foreground"> · {profile.provider}</span>}
                  </>
                ) : (
                  <span className="text-muted-foreground">{t('profiles.notSet')}</span>
                )}
              </DetailRow>
              <DetailRow label={t('profiles.skills')}>{profile.skill_count}</DetailRow>
            </dl>
          </header>

          <SoulEditor profileName={profile.name} />
        </div>
      </div>

      <RenameProfileDialog
        currentName={profile.name}
        onClose={() => setRenameOpen(false)}
        onRename={async newName => {
          await onRename(newName)
          setRenameOpen(false)
        }}
        open={renameOpen}
      />
    </div>
  )
}

function DetailRow({ children, label }: { children: React.ReactNode; label: string }) {
  return (
    <div className="flex flex-wrap items-baseline gap-2">
      <dt className="text-[0.65rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">{label}</dt>
      <dd className="text-xs text-foreground">{children}</dd>
    </div>
  )
}

function SoulEditor({ profileName }: { profileName: string }) {
  const t = useTranslation()
  const [content, setContent] = useState('')
  const [original, setOriginal] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)
  const requestRef = useRef<string>(profileName)

  useEffect(() => {
    requestRef.current = profileName
    setLoading(true)
    setError(null)
    setContent('')
    setOriginal('')

    void (async () => {
      try {
        const soul = await getProfileSoul(profileName)

        if (requestRef.current === profileName) {
          setContent(soul.content)
          setOriginal(soul.content)
        }
      } catch (err) {
        if (requestRef.current === profileName) {
          setError(err instanceof Error ? err.message : t('profiles.soul.loadFailed'))
        }
      } finally {
        if (requestRef.current === profileName) {
          setLoading(false)
        }
      }
    })()
  }, [profileName, t])

  const dirty = content !== original
  const isEmpty = !content.trim()

  async function handleSave() {
    setSaving(true)
    setError(null)

    try {
      await updateProfileSoul(profileName, content)
      setOriginal(content)
      notify({ kind: 'success', title: t('profiles.soul.saved'), message: profileName })
    } catch (err) {
      setError(err instanceof Error ? err.message : t('profiles.soul.saveFailed'))
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h4 className="text-[0.7rem] font-semibold uppercase tracking-[0.14em] text-muted-foreground">SOUL.md</h4>
          <p className="text-xs text-muted-foreground">{t('profiles.soul.description')}</p>
        </div>
        {dirty && <span className="text-[0.65rem] text-muted-foreground">{t('common.unsavedChanges')}</span>}
      </div>

      {loading ? (
        <PageLoader className="min-h-44" label={t('profiles.soul.loading')} />
      ) : (
        <Textarea
          className="min-h-72 font-mono text-xs leading-5"
          onChange={event => setContent(event.target.value)}
          placeholder={isEmpty ? t('profiles.soul.emptyPlaceholder') : undefined}
          value={content}
        />
      )}

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="flex justify-end">
        <Button disabled={!dirty || saving || loading} onClick={() => void handleSave()} size="sm">
          <Save />
          {saving ? t('common.saving') : t('profiles.soul.save')}
        </Button>
      </div>
    </section>
  )
}

function CreateProfileDialog({
  onClose,
  onCreate,
  open
}: {
  onClose: () => void
  onCreate: (name: string, cloneFromDefault: boolean) => Promise<void>
  open: boolean
}) {
  const t = useTranslation()
  const [name, setName] = useState('')
  const [cloneFromDefault, setCloneFromDefault] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)

  useEffect(() => {
    if (!open) {
      return
    }

    setName('')
    setCloneFromDefault(true)
    setError(null)
    setSaving(false)
  }, [open])

  const trimmed = name.trim()
  const invalid = trimmed !== '' && !isValidProfileName(trimmed)

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()

    if (!trimmed || invalid) {
      setError(invalid ? t('profiles.invalidName') : t('profiles.nameRequired'))

      return
    }

    setSaving(true)
    setError(null)

    try {
      await onCreate(trimmed, cloneFromDefault)
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : t('profiles.notifications.createFailed'))
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog onOpenChange={value => !value && !saving && onClose()} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{t('profiles.newProfile')}</DialogTitle>
          <DialogDescription>{t('profiles.createDialog.description')}</DialogDescription>
        </DialogHeader>

        <form className="grid gap-4" onSubmit={handleSubmit}>
          <div className="grid gap-1.5">
            <label className="text-xs font-medium" htmlFor="new-profile-name">
              {t('profiles.name')}
            </label>
            <Input
              aria-invalid={invalid}
              autoFocus
              id="new-profile-name"
              onChange={event => setName(event.target.value)}
              placeholder="my-profile"
              value={name}
            />
            <p className={cn('text-[0.66rem] leading-4', invalid ? 'text-destructive' : 'text-muted-foreground')}>
              {t(PROFILE_NAME_HINT_KEY)}
            </p>
          </div>

          <label className="flex cursor-pointer items-center gap-2 rounded-md border border-border/40 bg-background/50 px-3 py-2 text-sm">
            <input
              checked={cloneFromDefault}
              className="size-4 accent-primary"
              onChange={event => setCloneFromDefault(event.target.checked)}
              type="checkbox"
            />
            <span>
              <span className="font-medium">{t('profiles.createDialog.cloneFromDefault')}</span>
              <span className="ml-2 text-xs text-muted-foreground">{t('profiles.createDialog.cloneDescription')}</span>
            </span>
          </label>

          {error && (
            <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <DialogFooter>
            <Button disabled={saving} onClick={onClose} type="button" variant="outline">
              {t('common.cancel')}
            </Button>
            <Button disabled={saving || !trimmed || invalid} type="submit">
              {saving ? t('profiles.creating') : t('profiles.createProfile')}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function RenameProfileDialog({
  currentName,
  onClose,
  onRename,
  open
}: {
  currentName: string
  onClose: () => void
  onRename: (newName: string) => Promise<void>
  open: boolean
}) {
  const t = useTranslation()
  const [name, setName] = useState(currentName)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)

  useEffect(() => {
    if (!open) {
      return
    }

    setName(currentName)
    setError(null)
    setSaving(false)
  }, [currentName, open])

  const trimmed = name.trim()
  const unchanged = trimmed === currentName
  const invalid = trimmed !== '' && !unchanged && !isValidProfileName(trimmed)

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()

    if (unchanged) {
      onClose()

      return
    }

    if (!trimmed || invalid) {
      setError(invalid ? t('profiles.invalidName') : t('profiles.nameRequired'))

      return
    }

    setSaving(true)
    setError(null)

    try {
      await onRename(trimmed)
    } catch (err) {
      setError(err instanceof Error ? err.message : t('profiles.notifications.renameFailed'))
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog onOpenChange={value => !value && !saving && onClose()} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{t('profiles.renameProfile')}</DialogTitle>
          <DialogDescription>
            {t('profiles.renameDialog.before')} <span className="font-mono">~/.local/bin</span>.
          </DialogDescription>
        </DialogHeader>

        <form className="grid gap-3" onSubmit={handleSubmit}>
          <div className="grid gap-1.5">
            <label className="text-xs font-medium" htmlFor="rename-profile-name">
              {t('profiles.newName')}
            </label>
            <Input
              aria-invalid={invalid}
              autoFocus
              id="rename-profile-name"
              onChange={event => setName(event.target.value)}
              value={name}
            />
            <p className={cn('text-[0.66rem] leading-4', invalid ? 'text-destructive' : 'text-muted-foreground')}>
              {t(PROFILE_NAME_HINT_KEY)}
            </p>
          </div>

          {error && (
            <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <DialogFooter>
            <Button disabled={saving} onClick={onClose} type="button" variant="outline">
              {t('common.cancel')}
            </Button>
            <Button disabled={saving || invalid || unchanged} type="submit">
              {saving ? t('profiles.renaming') : t('profiles.rename')}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
