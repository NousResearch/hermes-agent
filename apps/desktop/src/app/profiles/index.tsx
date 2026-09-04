import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { ProfileScope } from '@/api/client'
import { CodeEditor } from '@/components/chat/code-editor'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import type { DesktopRegistryConnection } from '@/global'
import { getProfilesForScope, getProfileSoul, type ProfileInfo, updateProfileSoul } from '@/hermes'
import { useI18n } from '@/i18n'
import { sortConnectionsForDisplay } from '@/lib/connection-display'
import { displayPath } from '@/lib/display-path'
import { AlertTriangle, Save } from '@/lib/icons'
import { resolveProfileColor } from '@/lib/profile-color'
import { normalize } from '@/lib/text'
import { $activeConnectionId, $connectionsRegistry, $hasMultipleConnections } from '@/store/connections'
import { notify, notifyError } from '@/store/notifications'
import { $profileColors, profileLabel, refreshProfiles } from '@/store/profile'

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
  type PanelMenuItem,
  PanelMeta,
  PanelPill,
  PanelSectionLabel
} from '../overlays/panel'

import { CreateProfileDialog } from './create-profile-dialog'
import { DeleteProfileDialog } from './delete-profile-dialog'
import { RenameProfileDialog } from './rename-profile-dialog'

interface ProfilesViewProps {
  onClose: () => void
}

interface ProfileEntry {
  connectionId: string
  connectionLabel: string
  profile: ProfileInfo
}

const entryKey = (entry: ProfileEntry) => `${entry.connectionId || 'ambient'}::${entry.profile.name}`

export function ProfilesView({ onClose }: ProfilesViewProps) {
  const { t } = useI18n()
  const p = t.profiles
  const registry = useStore($connectionsRegistry)
  const multipleConnections = useStore($hasMultipleConnections)
  const activeConnectionId = useStore($activeConnectionId)

  const sources = useMemo(
    () => (multipleConnections ? sortConnectionsForDisplay(registry?.connections ?? []) : []),
    [multipleConnections, registry]
  )

  const [entries, setEntries] = useState<null | ProfileEntry[]>(null)
  const [failedConnections, setFailedConnections] = useState<Set<string>>(new Set())
  const [selectedKey, setSelectedKey] = useState<null | string>(null)
  const [query, setQuery] = useState('')
  const [createOn, setCreateOn] = useState<null | string>(null)
  const [pendingRename, setPendingRename] = useState<null | ProfileEntry>(null)
  const [pendingDelete, setPendingDelete] = useState<null | ProfileEntry>(null)
  const refreshGeneration = useRef(0)

  const refresh = useCallback(async () => {
    const generation = ++refreshGeneration.current

    if (!multipleConnections) {
      try {
        const profiles = await refreshProfiles()

        if (generation !== refreshGeneration.current) {
          return
        }

        const next = profiles.map(profile => ({ connectionId: '', connectionLabel: '', profile }))
        setEntries(next)
        setFailedConnections(new Set())
        setSelectedKey(current =>
          current && next.some(entry => entryKey(entry) === current)
            ? current
            : next[0]
              ? entryKey(next.find(entry => entry.profile.is_default) ?? next[0])
              : null
        )
      } catch (error) {
        if (generation === refreshGeneration.current) {
          notifyError(error, p.failedLoad)
        }
      }

      return
    }

    const results = await Promise.all(
      sources.map(async source => {
        try {
          const profiles = (await getProfilesForScope({ connectionId: source.id })).profiles

          return {
            entries: profiles.map(profile => ({
              connectionId: source.id,
              connectionLabel: source.label,
              profile
            })),
            failed: false,
            source
          }
        } catch {
          return { entries: [] as ProfileEntry[], failed: true, source }
        }
      })
    )

    const next = results.flatMap(result => result.entries)
    const failed = new Set(results.filter(result => result.failed).map(result => result.source.id))

    if (generation !== refreshGeneration.current) {
      return
    }

    setEntries(next)
    setFailedConnections(failed)
    setSelectedKey(current => {
      if (current && next.some(entry => entryKey(entry) === current)) {
        return current
      }

      const preferred =
        next.find(entry => entry.connectionId === activeConnectionId && entry.profile.is_default) ??
        next.find(entry => entry.profile.is_default) ??
        next[0]

      return preferred ? entryKey(preferred) : null
    })
  }, [activeConnectionId, multipleConnections, p.failedLoad, sources])

  useRefreshHotkey(refresh)

  useEffect(() => {
    void refresh()
  }, [refresh])

  const selected = useMemo(
    () => entries?.find(entry => entryKey(entry) === selectedKey) ?? entries?.[0] ?? null,
    [entries, selectedKey]
  )

  const visibleEntries = useMemo(() => {
    const q = normalize(query)

    if (!entries || !q) {
      return entries ?? []
    }

    return entries.filter(
      entry =>
        entry.profile.name.toLowerCase().includes(q) ||
        (entry.profile.model ?? '').toLowerCase().includes(q) ||
        entry.connectionLabel.toLowerCase().includes(q)
    )
  }, [entries, query])

  const scopeFor = (connectionId: string, profile?: string): ProfileScope =>
    multipleConnections ? { connectionId, profile } : undefined

  const selectAndRefresh = useCallback(
    async (connectionId: string, name: string) => {
      setSelectedKey(`${connectionId || 'ambient'}::${name}`)
      await refresh()
    },
    [refresh]
  )

  const menuFor = (entry: ProfileEntry): PanelMenuItem[] =>
    entry.profile.is_default
      ? [{ icon: 'edit', label: p.renameMenu, onSelect: () => setPendingRename(entry) }]
      : [
          { icon: 'edit', label: p.renameMenu, onSelect: () => setPendingRename(entry) },
          { icon: 'trash', label: t.common.delete, onSelect: () => setPendingDelete(entry), tone: 'danger' }
        ]

  const renderRows = (source: null | DesktopRegistryConnection) => {
    const connectionId = source?.id ?? ''
    const rows = visibleEntries.filter(entry => entry.connectionId === connectionId)

    return (
      <Fragment key={connectionId || 'ambient'}>
        {source ? <PanelSectionLabel>{source.label}</PanelSectionLabel> : null}
        {rows.map(entry => (
          <ProfileRow
            active={selected ? entryKey(selected) === entryKey(entry) : false}
            key={entryKey(entry)}
            menuItems={menuFor(entry)}
            onSelect={() => setSelectedKey(entryKey(entry))}
            profile={entry.profile}
          />
        ))}
        {source && failedConnections.has(source.id) ? (
          <p className="px-3 py-2 text-xs text-muted-foreground">{p.fleet.gatewayUnreachable(source.label)}</p>
        ) : (
          <PanelAddButton label={p.newProfile} onClick={() => setCreateOn(connectionId)} />
        )}
      </Fragment>
    )
  }

  return (
    <Panel closeLabel={p.close} onClose={onClose}>
      {!entries ? (
        <PageLoader label={p.loading} />
      ) : entries.length === 0 && !multipleConnections ? (
        <PanelEmpty
          action={
            <Button onClick={() => setCreateOn('')} size="sm">
              {p.newProfile}
            </Button>
          }
          description={p.createDesc}
          icon="organization"
          title={p.noProfiles}
        />
      ) : (
        <>
          <PanelHeader subtitle={p.count(entries.length)} title={p.title} />
          <PanelBody>
            <PanelList
              onSearchChange={setQuery}
              searchLabel={p.search}
              searchPlaceholder={p.search}
              searchValue={query}
            >
              {multipleConnections ? sources.map(source => renderRows(source)) : renderRows(null)}
            </PanelList>

            {selected ? (
              <ProfileDetail
                connectionId={multipleConnections ? selected.connectionId : null}
                connectionLabel={multipleConnections ? selected.connectionLabel : ''}
                key={entryKey(selected)}
                profile={selected.profile}
              />
            ) : (
              <PanelEmpty description={p.selectPrompt} icon="account" />
            )}
          </PanelBody>
        </>
      )}

      <RenameProfileDialog
        currentName={pendingRename?.profile.name ?? ''}
        isDefault={pendingRename?.profile.is_default ?? false}
        onClose={() => setPendingRename(null)}
        onRenamed={name => selectAndRefresh(pendingRename?.connectionId ?? '', name)}
        open={pendingRename !== null}
        scope={pendingRename ? scopeFor(pendingRename.connectionId, pendingRename.profile.name) : undefined}
      />

      <CreateProfileDialog
        onClose={() => setCreateOn(null)}
        onCreated={name => selectAndRefresh(createOn ?? '', name)}
        open={createOn !== null}
        profiles={(entries ?? []).filter(entry => entry.connectionId === (createOn ?? '')).map(entry => entry.profile)}
        scope={createOn !== null ? scopeFor(createOn) : undefined}
      />

      <DeleteProfileDialog
        gatewayLabel={multipleConnections ? pendingDelete?.connectionLabel : undefined}
        onClose={() => setPendingDelete(null)}
        onDeleted={async () => {
          setSelectedKey(null)
          await refresh()
        }}
        open={pendingDelete !== null}
        profile={pendingDelete?.profile ?? null}
        scope={pendingDelete ? scopeFor(pendingDelete.connectionId, pendingDelete.profile.name) : undefined}
      />
    </Panel>
  )
}

function ProfileRow({
  active,
  menuItems,
  onSelect,
  profile
}: {
  active: boolean
  menuItems: PanelMenuItem[]
  onSelect: () => void
  profile: ProfileInfo
}) {
  const colors = useStore($profileColors)

  return (
    <PanelListRow
      active={active}
      lead={
        <ProfileGlyph
          aria-hidden="true"
          color={resolveProfileColor(profile.name, colors)}
          isDefault={profile.is_default}
          name={profile.name}
        />
      }
      menuItems={menuItems}
      menuLabel={profileLabel(profile)}
      onSelect={onSelect}
      rowKey={profile.name}
      title={profileLabel(profile)}
    />
  )
}

function ProfileDetail({
  connectionId,
  connectionLabel,
  profile
}: {
  connectionId: null | string
  connectionLabel: string
  profile: ProfileInfo
}) {
  const { t } = useI18n()
  const p = t.profiles

  return (
    <PanelDetail>
      <header className="space-y-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-[0.95rem] font-semibold tracking-tight text-foreground">{profileLabel(profile)}</h3>
            {profile.is_default && <PanelPill tone="good">{p.defaultBadge}</PanelPill>}
            {profile.has_env && <PanelPill tone="muted">.env</PanelPill>}
            {connectionLabel ? <PanelPill tone="muted">{connectionLabel}</PanelPill> : null}
          </div>
          <p
            className="mt-1 truncate font-mono text-[0.66rem] text-muted-foreground/55"
            title={displayPath(profile.path)}
          >
            {displayPath(profile.path)}
          </p>
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

      <SoulEditor connectionId={connectionId} profileName={profile.name} />
    </PanelDetail>
  )
}

function SoulEditor({ connectionId, profileName }: { connectionId: null | string; profileName: string }) {
  const { t } = useI18n()
  const p = t.profiles
  const [content, setContent] = useState('')
  const [original, setOriginal] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)
  const requestRef = useRef(`${connectionId ?? 'ambient'}::${profileName}`)
  const requestKey = `${connectionId ?? 'ambient'}::${profileName}`

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    requestRef.current = requestKey
    setLoading(true)
    setError(null)
    setContent('')
    setOriginal('')

    void (async () => {
      try {
        const soul = connectionId
          ? await getProfileSoul(profileName, { connectionId, profile: profileName })
          : await getProfileSoul(profileName)

        if (requestRef.current === requestKey) {
          setContent(soul.content)
          setOriginal(soul.content)
        }
      } catch (err) {
        if (requestRef.current === requestKey) {
          setError(err instanceof Error ? err.message : p.failedLoadSoul)
        }
      } finally {
        if (requestRef.current === requestKey) {
          setLoading(false)
        }
      }
    })()
  }, [connectionId, p.failedLoadSoul, profileName, requestKey])

  const dirty = content !== original

  async function handleSave() {
    setSaving(true)
    setError(null)

    try {
      if (connectionId) {
        await updateProfileSoul(profileName, content, { connectionId, profile: profileName })
      } else {
        await updateProfileSoul(profileName, content)
      }

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
            key={requestKey}
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
