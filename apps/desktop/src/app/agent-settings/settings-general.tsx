import { useQuery } from '@tanstack/react-query'
import { useLayoutEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { CodeEditor } from '@/components/chat/code-editor'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { getProfiles, getProfileSoul, updateProfileSoul } from '@/hermes'
import { useI18n } from '@/i18n'
import { refreshFleetRoster } from '@/store/fleet-roster'

import { CreateProfileDialog } from '../profiles/create-profile-dialog'
import { DeleteProfileDialog } from '../profiles/delete-profile-dialog'
import { RenameProfileDialog } from '../profiles/rename-profile-dialog'
import { ListRow } from '../settings/primitives'
import { settingsScopeLabel } from '../settings/profile-scope'

import { agentSettingsKey, type AgentSettingsOwner } from './data'
import { useAgentSettingsEditorCopy } from './settings-copy'

export function AgentGeneralSettings({ owner }: { owner: AgentSettingsOwner }) {
  const { t } = useI18n()
  const c = useAgentSettingsEditorCopy()
  const navigate = useNavigate()
  const location = useLocation()
  const [dialog, setDialog] = useState<'rename' | 'duplicate' | 'delete' | null>(null)
  const [deleted, setDeleted] = useState(false)
  const mounted = useRef(true)
  useLayoutEffect(() => {
    mounted.current = true

    return () => {
      mounted.current = false
    }
  }, [])
  const profiles = useQuery({ queryKey: [...agentSettingsKey(owner), 'profiles'], queryFn: () => getProfiles(owner) })
  const profile = profiles.data?.profiles.find(item => item.name === owner.profile)

  if (profiles.isPending) {
    return <PageLoader />
  }

  if (profiles.isError) {
    return <Button onClick={() => void profiles.refetch()}>{t.common.retry}</Button>
  }

  if (deleted || !profile) {
    return <p role="status">{deleted ? t.profiles.deleted : t.errors.genericFailure}</p>
  }

  return (
    <>
      <div className="agent-settings-card">
        <ListRow
          action={
            <Button onClick={() => setDialog('rename')} variant="secondary">
              {t.profiles.rename}
            </Button>
          }
          description={settingsScopeLabel(profile)}
          title={t.profiles.nameLabel}
        />
        <p className="agent-settings-path">{profile.path}</p>
      </div>
      <div className="agent-settings-actions">
        <Button onClick={() => setDialog('duplicate')} variant="secondary">
          {c.duplicate}
        </Button>
        {!profile.is_default && owner.profile !== 'default' && (
          <Button onClick={() => setDialog('delete')} variant="destructiveOutline">
            {t.common.delete}
          </Button>
        )}
      </div>
      <RenameProfileDialog
        currentName={owner.profile}
        isDefault={profile.is_default}
        onClose={() => setDialog(null)}
        onRenamed={async name => {
          await refreshFleetRoster({ force: true })

          if (!mounted.current) {
            return
          }

          if (profile.is_default) {
            void profiles.refetch()

            return
          }

          const params = new URLSearchParams(location.search)
          params.set('agentProfile', name)
          navigate({ pathname: location.pathname, search: `?${params}` }, { replace: true })
        }}
        open={dialog === 'rename'}
        scope={owner}
      />
      <CreateProfileDialog
        initialCloneFrom={owner.profile}
        onClose={() => setDialog(null)}
        onCreated={() => {
          void profiles.refetch()
          void refreshFleetRoster({ force: true })
        }}
        open={dialog === 'duplicate'}
        profiles={profiles.data?.profiles}
        scope={owner}
      />
      <DeleteProfileDialog
        onClose={() => setDialog(null)}
        onDeleted={() => {
          if (mounted.current) {
            setDeleted(true)
          }

          void refreshFleetRoster({ force: true })
        }}
        open={dialog === 'delete'}
        profile={profile}
        scope={owner}
      />
    </>
  )
}

export function AgentInstructionsSettings({ owner }: { owner: AgentSettingsOwner }) {
  const { t } = useI18n()
  const [draft, setDraft] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const soul = useQuery({
    queryKey: [...agentSettingsKey(owner), 'soul'],
    queryFn: () => getProfileSoul(owner.profile, owner)
  })

  async function save() {
    if (saving || draft === null) {
      return
    }

    setSaving(true)
    setError('')

    try {
      await updateProfileSoul(owner.profile, draft, owner)
      await soul.refetch()
      setDraft(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : t.profiles.failedSaveSoul)
    } finally {
      setSaving(false)
    }
  }

  if (soul.isPending) {
    return <PageLoader />
  }

  if (soul.isError) {
    return <Button onClick={() => void soul.refetch()}>{t.common.retry}</Button>
  }

  return (
    <>
      {soul.data.exists === false && <p>{t.profiles.soulMissing}</p>}
      <div className="agent-settings-instructions">
        <CodeEditor
          disabled={saving}
          filePath="SOUL.md"
          framed
          initialValue={soul.data.content}
          onChange={setDraft}
          onSave={() => void save()}
        />
      </div>
      {error && <p role="alert">{error}</p>}
      <Button disabled={saving || draft === null} loading={saving} onClick={() => void save()}>
        {t.profiles.saveSoul}
      </Button>
    </>
  )
}
