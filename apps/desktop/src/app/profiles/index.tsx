import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { ActionStatus } from '@/components/ui/action-status'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Textarea } from '@/components/ui/textarea'
import { Tip } from '@/components/ui/tooltip'
import { createProfile, getProfiles, getProfileSoul, type ProfileInfo, updateProfileSoul } from '@/hermes'
import { AlertTriangle, Save, Users } from '@/lib/icons'
import { profileColor } from '@/lib/profile-color'
import { cn } from '@/lib/utils'
import { $activeProfile, switchProfile } from '@/store/profile'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { OverlayMain, OverlaySidebar, OverlaySplitLayout } from '../overlays/overlay-split-layout'
import { OverlayView } from '../overlays/overlay-view'

import { CreateProfileDialog } from './create-profile-dialog'
import { DeleteProfileDialog } from './delete-profile-dialog'
import { RenameProfileDialog } from './rename-profile-dialog'

// Pick a free "<source>-copy" name for a duplicated profile, appending a numeric
// suffix when the base is taken. Source is truncated to leave room for the
// suffix and to stay within the 64-char profile-name limit.
function uniqueCloneName(source: string, existing: Set<string>): string {
  const base = `${source}-copy`.slice(0, 58)

  if (!existing.has(base)) {
    return base
  }

  for (let i = 2; i < 1000; i++) {
    const candidate = `${base}-${i}`

    if (!existing.has(candidate)) {
      return candidate
    }
  }

  return `${base}-${Date.now()}`
}

// Three-state affordance shared by every save/create/rename/delete button:
// spinner while pending, a check on success, then back to the idle icon+label.
interface ProfilesViewProps {
  onClose: () => void
}

export function ProfilesView({ onClose }: ProfilesViewProps) {
  const [profiles, setProfiles] = useState<null | ProfileInfo[]>(null)
  const [selectedName, setSelectedName] = useState<null | string>(null)
  const [createOpen, setCreateOpen] = useState(false)
  const [pendingRename, setPendingRename] = useState<null | ProfileInfo>(null)
  const [pendingDelete, setPendingDelete] = useState<null | ProfileInfo>(null)
  const [loadError, setLoadError] = useState<null | string>(null)

  const refresh = useCallback(async () => {
    try {
      const { profiles: list } = await getProfiles()
      setProfiles(list)
      setLoadError(null)
      setSelectedName(current => {
        if (current && list.some(p => p.name === current)) {
          return current
        }

        return list.find(p => p.is_default)?.name ?? list[0]?.name ?? null
      })
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : 'Failed to load profiles')
      setProfiles(prev => prev ?? [])
    }
  }, [])

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

  const handleClone = useCallback(
    async (source: ProfileInfo) => {
      const existing = new Set((profiles ?? []).map(p => p.name))
      const target = uniqueCloneName(source.name, existing)

      try {
        await createProfile({ name: target, clone_from: source.name })
        setSelectedName(target)
        await refresh()
      } catch (err) {
        setLoadError(err instanceof Error ? err.message : `Failed to duplicate ${source.name}`)
      }
    },
    [profiles, refresh]
  )

  const handleMakeDefault = useCallback(async (profile: ProfileInfo) => {
    try {
      // Relaunches the backend under this profile's HERMES_HOME and reloads the
      // window, so control normally doesn't return here.
      await switchProfile(profile.name)
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : `Failed to switch to ${profile.name}`)
    }
  }, [])

  return (
    <OverlayView closeLabel="Close profiles" onClose={onClose}>
      {!profiles ? (
        <PageLoader label="Loading profiles..." />
      ) : (
        <OverlaySplitLayout>
          <OverlaySidebar>
            <div className="mb-1 flex items-center justify-between gap-2 pl-1.5 pr-0.5">
              <span className="text-[0.7rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary)">
                Profiles
              </span>
              <Button
                aria-label="New profile"
                className="text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground"
                onClick={() => setCreateOpen(true)}
                size="icon-xs"
                variant="ghost"
              >
                <Codicon name="add" size="0.875rem" />
              </Button>
            </div>
            {loadError && (
              <div className="mb-1 flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-2 py-1.5 text-[0.66rem] text-destructive">
                <AlertTriangle className="mt-0.5 size-3 shrink-0" />
                <span>{loadError}</span>
              </div>
            )}
            {profiles.map(profile => (
              <ProfileRow
                active={selected?.name === profile.name}
                key={profile.name}
                onClone={() => void handleClone(profile)}
                onDelete={() => setPendingDelete(profile)}
                onMakeDefault={() => void handleMakeDefault(profile)}
                onRename={() => setPendingRename(profile)}
                onSelect={() => setSelectedName(profile.name)}
                profile={profile}
              />
            ))}
            {profiles.length === 0 && <p className="px-1.5 py-3 text-xs text-muted-foreground">No profiles yet.</p>}
          </OverlaySidebar>

          <OverlayMain className="px-0">
            {selected ? (
              <ProfileDetail key={selected.name} profile={selected} />
            ) : (
              <div className="grid h-full place-items-center px-6 py-12 text-center text-sm text-muted-foreground">
                <div>
                  <Users className="mx-auto size-6 text-muted-foreground/60" />
                  <p className="mt-3">Select a profile to view its details.</p>
                </div>
              </div>
            )}
          </OverlayMain>
        </OverlaySplitLayout>
      )}

      <CreateProfileDialog
        onClose={() => setCreateOpen(false)}
        onCreated={async name => {
          setSelectedName(name)
          await refresh()
        }}
        open={createOpen}
      />

      <RenameProfileDialog
        currentName={pendingRename?.name ?? ''}
        onClose={() => setPendingRename(null)}
        onRenamed={async name => {
          setSelectedName(name)
          await refresh()
        }}
        open={pendingRename !== null}
      />

      <DeleteProfileDialog
        onClose={() => setPendingDelete(null)}
        onDeleted={async () => {
          setSelectedName(null)
          await refresh()
        }}
        open={pendingDelete !== null}
        profile={pendingDelete}
      />
    </OverlayView>
  )
}

function ProfileRow({
  active,
  onClone,
  onDelete,
  onMakeDefault,
  onRename,
  onSelect,
  profile
}: {
  active: boolean
  onClone: () => void
  onDelete: () => void
  onMakeDefault: () => void
  onRename: () => void
  onSelect: () => void
  profile: ProfileInfo
}) {
  const running = useStore($activeProfile)
  const isRunning = profile.name === running

  return (
    <div
      className={cn(
        'group relative flex items-center rounded-md border transition-colors',
        active
          ? 'border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary)'
          : 'border-transparent hover:bg-(--chrome-action-hover)'
      )}
    >
      <button
        className={cn(
          'flex min-w-0 flex-1 flex-col items-start gap-0.5 rounded-md px-2 py-1.5 text-left text-[length:var(--conversation-text-font-size)] transition-colors',
          active ? 'text-foreground' : 'text-(--ui-text-secondary) group-hover:text-foreground'
        )}
        onClick={onSelect}
        type="button"
      >
        <span className="flex w-full items-center gap-1.5 pr-6">
          {profile.is_default ? null : (
            <span
              aria-hidden="true"
              className="size-2 shrink-0 rounded-full"
              style={{ backgroundColor: profileColor(profile.name) ?? 'var(--ui-text-quaternary)' }}
            />
          )}
          <span className="truncate text-sm font-medium">{profile.name}</span>
          {isRunning && (
            <Tip label="Current default profile">
              <Codicon className="shrink-0 text-(--ui-accent)" name="pass-filled" size="0.75rem" />
            </Tip>
          )}
        </span>
        <span className="text-[0.66rem] text-muted-foreground">
          {isRunning ? 'default · ' : ''}
          {profile.skill_count} {profile.skill_count === 1 ? 'skill' : 'skills'}
        </span>
      </button>

      <ProfileActionsMenu
        isRunning={isRunning}
        onClone={onClone}
        onDelete={onDelete}
        onMakeDefault={onMakeDefault}
        onRename={onRename}
        profile={profile}
      >
        <Button
          aria-label={`Actions for ${profile.name}`}
          className="absolute right-1 top-1 size-6 bg-transparent text-(--ui-text-tertiary) opacity-0 transition-opacity hover:bg-(--ui-control-active-background) hover:text-foreground focus-visible:opacity-100 group-hover:opacity-100 data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground data-[state=open]:opacity-100"
          size="icon-xs"
          title="Profile actions"
          variant="ghost"
        >
          <Codicon name="ellipsis" size="0.875rem" />
        </Button>
      </ProfileActionsMenu>
    </div>
  )
}

function ProfileActionsMenu({
  children,
  isRunning,
  onClone,
  onDelete,
  onMakeDefault,
  onRename,
  profile
}: {
  children: React.ReactNode
  isRunning: boolean
  onClone: () => void
  onDelete: () => void
  onMakeDefault: () => void
  onRename: () => void
  profile: ProfileInfo
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>{children}</DropdownMenuTrigger>
      <DropdownMenuContent align="end" aria-label={`Actions for ${profile.name}`} className="w-44" sideOffset={6}>
        <DropdownMenuItem disabled={isRunning} onSelect={onMakeDefault}>
          <Codicon name="pass" size="0.875rem" />
          <span>{isRunning ? 'Current default' : 'Make default'}</span>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        {!profile.is_default && (
          <DropdownMenuItem onSelect={onRename}>
            <Codicon name="edit" size="0.875rem" />
            <span>Rename</span>
          </DropdownMenuItem>
        )}
        <DropdownMenuItem onSelect={onClone}>
          <Codicon name="copy" size="0.875rem" />
          <span>Duplicate</span>
        </DropdownMenuItem>
        {!profile.is_default && (
          <DropdownMenuItem
            className="text-destructive focus:text-destructive"
            onSelect={onDelete}
            variant="destructive"
          >
            <Codicon name="trash" size="0.875rem" />
            <span>Delete</span>
          </DropdownMenuItem>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function ProfileDetail({ profile }: { profile: ProfileInfo }) {
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-2xl space-y-6 px-6 py-6">
          <header className="space-y-3">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="text-xl font-semibold tracking-tight">{profile.name}</h3>
                {profile.is_default && <Badge>Default</Badge>}
              </div>
              <Tip label={profile.path}>
                <p className="mt-1 font-mono text-[0.7rem] text-muted-foreground">{profile.path}</p>
              </Tip>
            </div>

            <dl className="grid gap-2 text-xs sm:grid-cols-2">
              <DetailRow label="Model">
                {profile.model ? (
                  <>
                    <span className="font-mono">{profile.model}</span>
                    {profile.provider && <span className="text-muted-foreground"> · {profile.provider}</span>}
                  </>
                ) : (
                  <span className="text-muted-foreground">Not set</span>
                )}
              </DetailRow>
              <DetailRow label="Skills">{profile.skill_count}</DetailRow>
            </dl>
          </header>

          <SoulEditor profileName={profile.name} />
        </div>
      </div>
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
  const [content, setContent] = useState('')
  const [original, setOriginal] = useState('')
  const [loading, setLoading] = useState(true)
  const [status, setStatus] = useState<'idle' | 'saved' | 'saving'>('idle')
  const [error, setError] = useState<null | string>(null)
  const requestRef = useRef<string>(profileName)
  const savedTimerRef = useRef<null | number>(null)

  useEffect(() => {
    requestRef.current = profileName
    setLoading(true)
    setError(null)
    setStatus('idle')
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
          setError(err instanceof Error ? err.message : 'Failed to load SOUL.md')
        }
      } finally {
        if (requestRef.current === profileName) {
          setLoading(false)
        }
      }
    })()
  }, [profileName])

  useEffect(
    () => () => {
      if (savedTimerRef.current !== null) {
        window.clearTimeout(savedTimerRef.current)
      }
    },
    []
  )

  const dirty = content !== original
  const isEmpty = !content.trim()
  const saving = status === 'saving'

  async function handleSave() {
    setStatus('saving')
    setError(null)

    if (savedTimerRef.current !== null) {
      window.clearTimeout(savedTimerRef.current)
    }

    try {
      await updateProfileSoul(profileName, content)
      setOriginal(content)
      setStatus('saved')
      savedTimerRef.current = window.setTimeout(() => {
        setStatus(current => (current === 'saved' ? 'idle' : current))
      }, 2200)
    } catch (err) {
      setStatus('idle')
      setError(err instanceof Error ? err.message : 'Failed to save SOUL.md')
    }
  }

  return (
    <section className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h4 className="text-[0.7rem] font-semibold uppercase tracking-[0.14em] text-muted-foreground">SOUL.md</h4>
          <p className="text-xs text-muted-foreground">
            The system prompt and persona instructions baked into this profile.
          </p>
        </div>
        {dirty && <span className="text-[0.65rem] text-muted-foreground">Unsaved changes</span>}
      </div>

      {loading ? (
        <PageLoader className="min-h-44" label="Loading SOUL.md" />
      ) : (
        <Textarea
          className="min-h-72 font-mono text-xs leading-5"
          onChange={event => setContent(event.target.value)}
          placeholder={isEmpty ? 'Empty SOUL.md — start writing the persona...' : undefined}
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
        <Button disabled={loading || saving || !dirty} onClick={() => void handleSave()} size="sm">
          <ActionStatus
            busy="Saving…"
            done="Saved"
            idle="Save SOUL.md"
            idleIcon={<Save />}
            state={saving ? 'saving' : status === 'saved' && !dirty ? 'done' : 'idle'}
          />
        </Button>
      </div>
    </section>
  )
}

