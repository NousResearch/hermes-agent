import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import {
  createContext,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
  useContext,
  useLayoutEffect,
  useRef,
  useState
} from 'react'
import { useLocation, useNavigate } from 'react-router'

import { CodeEditor } from '@/components/chat/code-editor'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import { ErrorState } from '@/components/ui/error-state'
import { getProfiles, getProfileSoul, updateProfileSoul } from '@/hermes'
import { useI18n } from '@/i18n'
import { PROFILE_SWATCHES, resolveProfileColor } from '@/lib/profile-color'
import { refreshFleetRoster } from '@/store/fleet-roster'
import { $profileColors, setProfileColor } from '@/store/profile'
import { usePortalShell } from '@/store/shell-variant'

import { CreateProfileDialog } from '../profiles/create-profile-dialog'
import { DeleteProfileDialog } from '../profiles/delete-profile-dialog'
import { RenameProfileDialog } from '../profiles/rename-profile-dialog'
import { ListRow, ToggleRow } from '../settings/primitives'
import { settingsScopeLabel } from '../settings/profile-scope'

import { agentSettingsKey, type AgentSettingsOwner } from './data'
import { useDefaultAgent } from './default-agent'
import { useOverviewActionsCopy } from './overview-actions-copy'
import { useAgentSettingsEditorCopy } from './settings-copy'
import { agentSettingsSearch } from './settings-routes'

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
  const g = c.generalPage
  const actions = useOverviewActionsCopy()
  const navigate = useNavigate()
  const location = useLocation()

  const openKeys = () =>
    navigate({ pathname: location.pathname, search: agentSettingsSearch(location.search, 'providers') })

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
  const defaultAgent = useDefaultAgent(owner, profile)

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
      <AgentSettingsGroup title={g.identity}>
        <ListRow
          action={
            <>
              <span className="agent-settings-value">{settingsScopeLabel(profile)}</span>
              <Button onClick={() => setDialog('rename')} variant="secondary">
                {t.profiles.rename}
              </Button>
            </>
          }
          description={g.nameDescription}
          title={t.profiles.nameLabel}
        />
        <ListRow
          action={<AgentColorSwatches label={g.colorSwatch} profile={owner.profile} />}
          description={g.colorDescription}
          title={g.color}
        />
      </AgentSettingsGroup>
      {profile.path && (
        <AgentSettingsGroup title={g.profile}>
          <ListRow
            action={<CopyButton buttonVariant="secondary" label={actions.copyPath} text={profile.path} />}
            description={g.folderDescription}
            hint={profile.path}
            title={g.folder}
          />
          <ListRow
            action={
              <Button onClick={openKeys} variant="secondary">
                {g.editKeys}
              </Button>
            }
            description={g.envKeysDescription}
            title={g.envKeys}
          />
          <ToggleRow
            checked={defaultAgent.isDefault}
            description={defaultAgent.canSetDefault ? g.defaultAgentDescription : actions.defaultUnavailable}
            disabled={defaultAgent.isDefault || !defaultAgent.canSetDefault || defaultAgent.saving}
            label={g.defaultAgent}
            onChange={on => on && void defaultAgent.makeDefault()}
          />
        </AgentSettingsGroup>
      )}
      <AgentSettingsGroup title={g.manage}>
        <ListRow
          action={
            <Button onClick={() => setDialog('duplicate')} variant="secondary">
              {c.duplicate}
            </Button>
          }
          description={g.duplicateDescription}
          title={c.duplicate}
        />
        {!profile.is_default && owner.profile !== 'default' && (
          <ListRow
            action={
              <Button onClick={() => setDialog('delete')} variant="destructiveOutline">
                {g.deleteTitle}
              </Button>
            }
            description={g.deleteDescription}
            title={g.deleteTitle}
          />
        )}
      </AgentSettingsGroup>
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
  const redesign = usePortalShell()
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

interface AgentSettingsGroupProps {
  title: string
  children: ReactNode
}

/** A titled card of settings rows (895:9666): 15 Medium heading, 12px to an r4 card. */
function AgentSettingsGroup({ title, children }: AgentSettingsGroupProps) {
  return (
    <section className="agent-settings-group">
      <h3>{title}</h3>
      <div className="agent-settings-card">{children}</div>
    </section>
  )
}

// Six of the rail's twelve evenly spaced hues: enough to tell agents apart in one row (895:9675).
const AGENT_SWATCHES = PROFILE_SWATCHES.filter((_, index) => index % 2 === 0)

interface AgentColorSwatchesProps {
  profile: string
  label: string
}

/** The same per-agent color the rail and agent list read; a local, cosmetic preference. */
function AgentColorSwatches({ profile, label }: AgentColorSwatchesProps) {
  const colors = useStore($profileColors)
  const current = resolveProfileColor(profile, colors)

  return (
    <div aria-label={label} className="agent-color-swatches" role="group">
      {AGENT_SWATCHES.map((swatch, index) => (
        <button
          aria-label={`${label} ${index + 1}`}
          aria-pressed={swatch === current}
          key={swatch}
          onClick={() => setProfileColor(profile, swatch)}
          style={{ backgroundColor: swatch }}
          type="button"
        />
      ))}
    </div>
  )
}
