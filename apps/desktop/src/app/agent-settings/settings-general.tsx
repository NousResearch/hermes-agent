import { useQuery } from '@tanstack/react-query'
import { createContext, type Dispatch, type SetStateAction, useContext, useLayoutEffect, useRef, useState } from 'react'

import { usePortalRedesign } from '@/app/redesign/use-portal-redesign'
import { CodeEditor } from '@/components/chat/code-editor'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import { ErrorState } from '@/components/ui/error-state'
import { getProfiles, getProfileSoul, updateProfileSoul } from '@/hermes'
import { useI18n } from '@/i18n'
import { refreshFleetRoster } from '@/store/fleet-roster'

import { CreateProfileDialog } from '../profiles/create-profile-dialog'
import { DeleteProfileDialog } from '../profiles/delete-profile-dialog'
import { RenameProfileDialog } from '../profiles/rename-profile-dialog'
import { ListRow } from '../settings/primitives'
import { settingsScopeLabel } from '../settings/profile-scope'

import { agentSettingsKey, type AgentSettingsOwner } from './data'
import { useOverviewActionsCopy } from './overview-actions-copy'
import { useAgentSettingsEditorCopy } from './settings-copy'

interface AgentInstructionsDraft {
  text: string | null
  revision: number
  saving: boolean
  error: string
}

export const EMPTY_AGENT_INSTRUCTIONS_DRAFT: AgentInstructionsDraft = {
  text: null,
  revision: 0,
  saving: false,
  error: ''
}

export const AgentInstructionsDraftContext = createContext<
  [AgentInstructionsDraft, Dispatch<SetStateAction<AgentInstructionsDraft>>] | null
>(null)

export function AgentGeneralSettings({
  owner,
  onProfileRenamed
}: {
  owner: AgentSettingsOwner
  onProfileRenamed: (name: string) => Promise<void>
}) {
  const { t } = useI18n()
  const c = useAgentSettingsEditorCopy()
  const actions = useOverviewActionsCopy()
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
    return (
      <ErrorState description={profiles.error.message} title={t.profiles.failedLoad}>
        <Button onClick={() => void profiles.refetch()} variant="secondary">
          {t.common.retry}
        </Button>
      </ErrorState>
    )
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
        {profile.path && (
          <div className="agent-settings-path">
            <span>{profile.path}</span>
            <CopyButton buttonVariant="secondary" label={actions.copyPath} text={profile.path} />
          </div>
        )}
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

          if (profile.is_default) {
            if (mounted.current) {
              void profiles.refetch()
            }

            return
          }

          await onProfileRenamed(name)
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
  const { locale, t } = useI18n()
  const c = useAgentSettingsEditorCopy()
  const redesign = usePortalRedesign()
  const retainedDraft = useContext(AgentInstructionsDraftContext)
  const localDraft = useState(EMPTY_AGENT_INSTRUCTIONS_DRAFT)
  const [draft, setDraft] = retainedDraft ?? localDraft

  const soul = useQuery({
    queryKey: [...agentSettingsKey(owner), 'soul'],
    queryFn: () => getProfileSoul(owner.profile, owner)
  })

  async function save() {
    if (draft.saving || draft.text === null) {
      return
    }

    const submittedRevision = draft.revision
    setDraft(current => ({ ...current, saving: true, error: '' }))

    try {
      await updateProfileSoul(owner.profile, draft.text, owner)
      await soul.refetch()
      setDraft(current => (current.revision === submittedRevision ? { ...current, text: null } : current))
    } catch (err) {
      setDraft(current => ({ ...current, error: err instanceof Error ? err.message : t.profiles.failedSaveSoul }))
    } finally {
      setDraft(current => ({ ...current, saving: false }))
    }
  }

  if (soul.isPending) {
    return <PageLoader />
  }

  if (soul.isError) {
    return (
      <ErrorState description={soul.error.message} title={t.profiles.failedLoadSoul}>
        <Button onClick={() => void soul.refetch()} variant="secondary">
          {t.common.retry}
        </Button>
      </ErrorState>
    )
  }

  return (
    <>
      {soul.data.exists === false && <p>{t.profiles.soulMissing}</p>}
      <section className="agent-settings-instructions-section">
        {redesign && (
          <header>
            <h3>SOUL.md</h3>
            <p>{c.soulDescription}</p>
          </header>
        )}
        <div className="agent-settings-instructions">
          <CodeEditor
            className="rounded-none border-0"
            disabled={draft.saving}
            filePath="SOUL.md"
            framed
            initialValue={draft.text ?? soul.data.content}
            onChange={text => setDraft(current => ({ ...current, text, revision: current.revision + 1 }))}
            onSave={() => void save()}
          />
          <div className="agent-settings-instructions-footer">
            {redesign && (
              <span className="agent-settings-instructions-meta">
                SOUL.md · {c.characters}: {(draft.text ?? soul.data.content).length.toLocaleString(locale)}
              </span>
            )}
            <Button
              aria-label={t.profiles.saveSoul}
              className="h-9"
              disabled={draft.saving || draft.text === null}
              loading={draft.saving}
              onClick={() => void save()}
            >
              {redesign ? t.common.save : t.profiles.saveSoul}
            </Button>
          </div>
        </div>
      </section>
      {draft.error && <p role="alert">{draft.error}</p>}
    </>
  )
}
