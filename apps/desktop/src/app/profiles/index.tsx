import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { CodeEditor } from '@/components/chat/code-editor'
import { PageLoader } from '@/components/page-loader'
import { ProfileAvatar } from '@/components/profile-avatar'
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
import { SanitizedInput } from '@/components/ui/sanitized-input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  createProfile,
  deleteProfile,
  deleteProfileAvatar,
  getProfileSoul,
  type ProfileInfo,
  renameProfile,
  updateProfileAvatar,
  updateProfileSoul
} from '@/hermes'
import { useI18n } from '@/i18n'
import { AlertTriangle, Pencil, Save, X } from '@/lib/icons'
import { slug } from '@/lib/sanitize'
import { normalize } from '@/lib/text'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import {
  $profileAvatars,
  normalizeProfileKey,
  refreshProfiles,
  removeProfileLocal,
  renameProfileLocal,
  setProfileAvatarLocal
} from '@/store/profile'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import {
  Panel,
  PanelAddButton,
  PanelBody,
  PanelDetail,
  PanelEmpty,
  PanelHeader,
  PanelList,
  PanelListRow,
  PanelMeta,
  PanelPill,
  PanelRowMenu,
  PanelSectionLabel
} from '../overlays/panel'

import { useAvatarPicker } from './avatar-field'

const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

function isValidProfileName(name: string): boolean {
  return PROFILE_NAME_RE.test(name.trim())
}

interface ProfilesViewProps {
  onClose: () => void
}

export function ProfilesView({ onClose }: ProfilesViewProps) {
  const { t } = useI18n()
  const p = t.profiles
  const [profiles, setProfiles] = useState<null | ProfileInfo[]>(null)
  const [selectedName, setSelectedName] = useState<null | string>(null)
  const [query, setQuery] = useState('')
  const [createOpen, setCreateOpen] = useState(false)
  const [pendingRename, setPendingRename] = useState<null | ProfileInfo>(null)
  const [pendingDelete, setPendingDelete] = useState<null | ProfileInfo>(null)
  const [deleting, setDeleting] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const list = await refreshProfiles()
      setProfiles(list)
      setSelectedName(current => {
        if (current && list.some(p => p.name === current)) {
          return current
        }

        return list.find(p => p.is_default)?.name ?? list[0]?.name ?? null
      })
    } catch (err) {
      notifyError(err, p.failedLoad)
    }
  }, [p])

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

  const visibleProfiles = useMemo(() => {
    const q = normalize(query)

    if (!profiles || !q) {
      return profiles ?? []
    }

    return profiles.filter(
      profile => profile.name.toLowerCase().includes(q) || (profile.model ?? '').toLowerCase().includes(q)
    )
  }, [profiles, query])

  const handleCreate = useCallback(
    async (name: string, cloneFrom: null | string) => {
      const trimmed = name.trim()

      if (!isValidProfileName(trimmed)) {
        throw new Error(p.nameHint)
      }

      await createProfile({ name: trimmed, clone_from: cloneFrom })
      notify({ kind: 'success', title: p.created, message: trimmed })
      setSelectedName(trimmed)
      await refresh()
    },
    [p, refresh]
  )

  const handleRename = useCallback(
    async (from: string, to: string): Promise<void> => {
      const target = to.trim()

      if (target === from) {
        return
      }

      if (!isValidProfileName(target)) {
        throw new Error(p.nameHint)
      }

      await renameProfile(from, target)
      // Optimistically re-key the cached list, cosmetics (color/order/avatar)
      // and gateway routing to the new name so the rail repaints instantly and
      // no reconnect loop is left dialing the old, now-torn-down backend.
      renameProfileLocal(from, target)
      notify({ kind: 'success', title: p.renamed, message: `${from} → ${target}` })
      setSelectedName(target)
      await refresh()
    },
    [p, refresh]
  )

  const handleConfirmDelete = useCallback(async () => {
    if (!pendingDelete) {
      return
    }

    setDeleting(true)

    try {
      await deleteProfile(pendingDelete.name)
      // Optimistically drop the row/avatar and retarget any routing at the
      // deleted profile to default, so the rail square disappears at once and
      // its reconnect backoff stops respawning a backend for a gone directory.
      removeProfileLocal(pendingDelete.name)
      notify({ kind: 'success', title: p.deleted, message: pendingDelete.name })
      setPendingDelete(null)
      setSelectedName(null)
      await refresh()
    } catch (err) {
      notifyError(err, p.failedDelete)
    } finally {
      setDeleting(false)
    }
  }, [p, pendingDelete, refresh])

  return (
    <Panel closeLabel={p.close} onClose={onClose}>
      {!profiles ? (
        <PageLoader label={p.loading} />
      ) : profiles.length === 0 ? (
        <PanelEmpty
          action={
            <Button onClick={() => setCreateOpen(true)} size="sm">
              {p.newProfile}
            </Button>
          }
          description={p.createDesc}
          icon="organization"
          title={p.noProfiles}
        />
      ) : (
        <>
          <PanelHeader subtitle={p.count(profiles.length)} title={p.title} />
          <PanelBody>
            <PanelList
              onSearchChange={setQuery}
              searchLabel={p.search}
              searchPlaceholder={p.search}
              searchValue={query}
            >
              {visibleProfiles.map(profile => (
                <ProfileRow
                  active={selected?.name === profile.name}
                  key={profile.name}
                  menu={
                    <PanelRowMenu
                      items={
                        profile.is_default
                          ? []
                          : [
                              { icon: 'edit', label: p.renameMenu, onSelect: () => setPendingRename(profile) },
                              {
                                icon: 'trash',
                                label: t.common.delete,
                                onSelect: () => setPendingDelete(profile),
                                tone: 'danger'
                              }
                            ]
                      }
                    />
                  }
                  onSelect={() => setSelectedName(profile.name)}
                  profile={profile}
                />
              ))}
              <PanelAddButton label={p.newProfile} onClick={() => setCreateOpen(true)} />
            </PanelList>

            {selected ? (
              <ProfileDetail key={selected.name} profile={selected} />
            ) : (
              <PanelEmpty description={p.selectPrompt} icon="account" />
            )}
          </PanelBody>
        </>
      )}

      <RenameProfileDialog
        currentName={pendingRename?.name ?? ''}
        onClose={() => setPendingRename(null)}
        onRename={async newName => {
          if (pendingRename) {
            await handleRename(pendingRename.name, newName)
            setPendingRename(null)
          }
        }}
        open={pendingRename !== null}
      />

      <CreateProfileDialog
        onClose={() => setCreateOpen(false)}
        onCreate={async (name, cloneFrom) => handleCreate(name, cloneFrom)}
        open={createOpen}
        profiles={profiles ?? []}
      />

      <Dialog onOpenChange={open => !open && !deleting && setPendingDelete(null)} open={pendingDelete !== null}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{p.deleteTitle}</DialogTitle>
            <DialogDescription>
              {pendingDelete ? (
                <>
                  {p.deleteDescPrefix}
                  <span className="font-medium text-foreground">{pendingDelete.name}</span>
                  {p.deleteDescMid}
                  <span className="font-mono text-xs">{pendingDelete.path}</span>
                  {p.deleteDescSuffix}
                </>
              ) : null}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button disabled={deleting} onClick={() => setPendingDelete(null)} variant="outline">
              {t.common.cancel}
            </Button>
            <Button disabled={deleting} onClick={() => void handleConfirmDelete()} variant="destructive">
              {deleting ? p.deleting : t.common.delete}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Panel>
  )
}

function ProfileRow({
  active,
  menu,
  onSelect,
  profile
}: {
  active: boolean
  menu?: React.ReactNode
  onSelect: () => void
  profile: ProfileInfo
}) {
  return (
    <PanelListRow
      active={active}
      lead={<ProfileGlyph isDefault={profile.is_default} name={profile.name} />}
      menu={menu}
      onSelect={onSelect}
      rowKey={profile.name}
      title={profile.name}
    />
  )
}

// Leading glyph for a profile row, mirroring the sidebar rail: the default
// profile keeps the `home` icon; named profiles show their picture (or the
// colored-initial fallback ProfileAvatar draws when no picture is set).
function ProfileGlyph({ isDefault, name }: { isDefault: boolean; name: string }) {
  if (isDefault) {
    return <Codicon className="shrink-0 text-muted-foreground/70" name="home" size="0.9rem" />
  }

  return <ProfileAvatar className="size-4 rounded-[3px] text-[0.5rem]" name={name} />
}

// The profile picture in the detail header, doubling as its editor: click to
// pick a new image (hover reveals a pencil overlay), and a corner button removes
// the current one. Mirrors the SOUL.md save pattern — optimistic local update
// plus a success toast — so the rail and lists refresh instantly.
function ProfileDetailAvatar({ name }: { name: string }) {
  const { t } = useI18n()
  const p = t.profiles
  const pickAvatar = useAvatarPicker()
  const avatars = useStore($profileAvatars)
  const hasPicture = Boolean(avatars[normalizeProfileKey(name)])
  const [busy, setBusy] = useState(false)

  const handlePick = useCallback(async () => {
    setBusy(true)

    try {
      const next = await pickAvatar()

      if (next) {
        await updateProfileAvatar(name, next)
        setProfileAvatarLocal(name, next)
        notify({ kind: 'success', title: p.pictureSaved, message: name })
      }
    } catch (err) {
      notifyError(err, p.failedSavePicture)
    } finally {
      setBusy(false)
    }
  }, [name, p, pickAvatar])

  const handleRemove = useCallback(async () => {
    setBusy(true)

    try {
      await deleteProfileAvatar(name)
      setProfileAvatarLocal(name, null)
    } catch (err) {
      notifyError(err, p.failedRemovePicture)
    } finally {
      setBusy(false)
    }
  }, [name, p])

  return (
    <div className="group/avatar relative shrink-0">
      <button
        aria-label={hasPicture ? p.changePicture : p.choosePicture}
        className="block rounded-lg outline-none ring-offset-2 ring-offset-background focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-60"
        disabled={busy}
        onClick={() => void handlePick()}
        type="button"
      >
        <ProfileAvatar className="size-12 rounded-lg text-lg" name={name} />
        <span className="absolute inset-0 grid place-items-center rounded-lg bg-black/45 text-white opacity-0 transition-opacity group-hover/avatar:opacity-100">
          <Pencil className="size-4" />
        </span>
      </button>
      {hasPicture && (
        <button
          aria-label={p.removePicture}
          className="absolute -right-1.5 -top-1.5 grid size-5 place-items-center rounded-full border border-border bg-background text-muted-foreground opacity-0 transition hover:text-destructive group-hover/avatar:opacity-100 disabled:opacity-60"
          disabled={busy}
          onClick={() => void handleRemove()}
          type="button"
        >
          <X className="size-3" />
        </button>
      )}
    </div>
  )
}

function ProfileDetail({ profile }: { profile: ProfileInfo }) {
  const { t } = useI18n()
  const p = t.profiles

  return (
    <PanelDetail>
      <header className="space-y-3">
        <div className="flex items-start gap-3">
          <ProfileDetailAvatar name={profile.name} />
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-[0.95rem] font-semibold tracking-tight text-foreground">{profile.name}</h3>
              {profile.is_default && <PanelPill tone="good">{p.defaultBadge}</PanelPill>}
              {profile.has_env && <PanelPill tone="muted">.env</PanelPill>}
            </div>
            <p className="mt-1 truncate font-mono text-[0.66rem] text-muted-foreground/55" title={profile.path}>
              {profile.path}
            </p>
          </div>
        </div>

        <PanelMeta
          rows={[
            {
              label: p.modelLabel,
              value: profile.model ? (
                <span className="font-mono">
                  {profile.model}
                  {profile.provider ? <span className="text-muted-foreground/55"> · {profile.provider}</span> : null}
                </span>
              ) : (
                <span className="text-muted-foreground/55">{p.notSet}</span>
              )
            },
            { label: p.skillsLabel, value: profile.skill_count }
          ]}
        />
      </header>

      <SoulEditor profileName={profile.name} />
    </PanelDetail>
  )
}

function SoulEditor({ profileName }: { profileName: string }) {
  const { t } = useI18n()
  const p = t.profiles
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
          setError(err instanceof Error ? err.message : p.failedLoadSoul)
        }
      } finally {
        if (requestRef.current === profileName) {
          setLoading(false)
        }
      }
    })()
  }, [p, profileName])

  const dirty = content !== original

  async function handleSave() {
    setSaving(true)
    setError(null)

    try {
      await updateProfileSoul(profileName, content)
      setOriginal(content)
      notify({ kind: 'success', title: p.soulSaved, message: profileName })
    } catch (err) {
      setError(err instanceof Error ? err.message : p.failedSaveSoul)
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <PanelSectionLabel className="text-[0.7rem] tracking-[0.14em]">SOUL.md</PanelSectionLabel>
          <p className="text-xs text-muted-foreground">{p.soulDesc}</p>
        </div>
        {dirty && <span className="text-[0.65rem] text-muted-foreground">{p.unsavedChanges}</span>}
      </div>

      {loading ? (
        <PageLoader className="min-h-44" label={p.loadingSoul} />
      ) : (
        <div className="min-h-48">
          <CodeEditor
            filePath="SOUL.md"
            framed
            initialValue={content}
            key={profileName}
            onChange={setContent}
            onSave={() => void handleSave()}
          />
        </div>
      )}

      {error && (
        <div className="flex items-start gap-2 rounded bg-destructive/10 px-3 py-2 text-xs text-destructive">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="flex justify-end">
        <Button disabled={!dirty || saving || loading} onClick={() => void handleSave()} size="sm">
          <Save />
          {saving ? p.saving : p.saveSoul}
        </Button>
      </div>
    </section>
  )
}

function CreateProfileDialog({
  onClose,
  onCreate,
  open,
  profiles
}: {
  onClose: () => void
  onCreate: (name: string, cloneFrom: null | string) => Promise<void>
  open: boolean
  profiles: ProfileInfo[]
}) {
  const { t } = useI18n()
  const p = t.profiles
  const [name, setName] = useState('')
  const [cloneFrom, setCloneFrom] = useState<null | string>('default')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)

  useEffect(() => {
    if (!open) {
      return
    }

    setName('')
    setCloneFrom('default')
    setError(null)
    setSaving(false)
  }, [open])

  const trimmed = name.trim()
  const invalid = trimmed !== '' && !isValidProfileName(trimmed)

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()

    if (!trimmed || invalid) {
      setError(invalid ? p.invalidName(p.nameHint) : p.nameRequired)

      return
    }

    setSaving(true)
    setError(null)

    try {
      await onCreate(trimmed, cloneFrom)
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : p.failedCreate)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog onOpenChange={value => !value && !saving && onClose()} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{p.newProfile}</DialogTitle>
          <DialogDescription>{p.createDesc}</DialogDescription>
        </DialogHeader>

        <form className="grid gap-4" onSubmit={handleSubmit}>
          <div className="grid gap-1.5">
            <label className="text-xs font-medium" htmlFor="new-profile-name">
              {p.nameLabel}
            </label>
            <SanitizedInput
              aria-invalid={invalid}
              autoFocus
              id="new-profile-name"
              onValueChange={setName}
              placeholder="my-profile"
              sanitize={slug}
              value={name}
            />
            <p className={cn('text-[0.66rem] leading-4', invalid ? 'text-destructive' : 'text-muted-foreground')}>
              {p.nameHint}
            </p>
          </div>

          <div className="grid gap-1.5">
            <label className="text-xs font-medium" htmlFor="new-profile-clone-from">
              {p.cloneFrom}
            </label>
            <Select
              onValueChange={value => setCloneFrom(value === '__none__' ? null : value)}
              value={cloneFrom ?? '__none__'}
            >
              <SelectTrigger className="h-9 rounded-md" id="new-profile-clone-from">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__none__">{p.cloneFromNone}</SelectItem>
                {profiles.map(profile => (
                  <SelectItem key={profile.name} value={profile.name}>
                    {profile.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">{p.cloneFromDesc}</p>
          </div>

          {error && (
            <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <DialogFooter>
            <Button disabled={saving} onClick={onClose} type="button" variant="outline">
              {t.common.cancel}
            </Button>
            <Button disabled={saving || !trimmed || invalid} type="submit">
              {saving ? p.creating : p.createAction}
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
  const { t } = useI18n()
  const p = t.profiles
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
      setError(invalid ? p.invalidName(p.nameHint) : p.nameRequired)

      return
    }

    setSaving(true)
    setError(null)

    try {
      await onRename(trimmed)
    } catch (err) {
      setError(err instanceof Error ? err.message : p.failedRename)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog onOpenChange={value => !value && !saving && onClose()} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{p.renameTitle}</DialogTitle>
          <DialogDescription>
            {p.renameDescPrefix}
            <span className="font-mono">~/.local/bin</span>
            {p.renameDescSuffix}
          </DialogDescription>
        </DialogHeader>

        <form className="grid gap-3" onSubmit={handleSubmit}>
          <div className="grid gap-1.5">
            <label className="text-xs font-medium" htmlFor="rename-profile-name">
              {p.newNameLabel}
            </label>
            <SanitizedInput
              aria-invalid={invalid}
              autoFocus
              id="rename-profile-name"
              onValueChange={setName}
              sanitize={slug}
              value={name}
            />
            <p className={cn('text-[0.66rem] leading-4', invalid ? 'text-destructive' : 'text-muted-foreground')}>
              {p.nameHint}
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
              {t.common.cancel}
            </Button>
            <Button disabled={saving || invalid || unchanged} type="submit">
              {saving ? p.renaming : p.rename}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
