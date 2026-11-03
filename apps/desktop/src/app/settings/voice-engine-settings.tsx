import { useStore } from '@nanostores/react'
import { useEffect, useMemo } from 'react'

import { setEnvVar } from '@/api/config'
import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { Button } from '@/components/ui/button'
import { $pluginRecords } from '@/contrib/plugins-store'
import { type ResolvedLiveVoiceEngine, useLiveVoiceEngines } from '@/contrib/voice-engines'
import { useI18n } from '@/i18n'
import { openExternalLink } from '@/lib/external-link'
import { isExtensionVoiceChatMode } from '@/lib/voice-live'
import {
  $agentPluginBusy,
  $agentPlugins,
  $agentPluginsProfile,
  $agentPluginsStatus,
  loadAgentPlugins,
  saveAgentPluginSettings
} from '@/store/agent-plugins'
import { confirm } from '@/store/confirm'
import { notify, notifyError } from '@/store/notifications'
import {
  $voiceSignIn,
  cancelVoiceSignIn,
  refreshVoiceSignIn,
  signInKey,
  signOutVoiceProvider,
  startVoiceSignIn,
  voiceSignInState
} from '@/store/voice-engine-sign-in'

import { agentPackageName, desktopPackageName } from '../capabilities/plugins/plugin-packages'

import { ENUM_OPTIONS } from './constants'
import { PluginSettingsForm } from './plugin-settings-form'
import { ListRow } from './primitives'

/** `voice.voice_chat_mode` choices: the built-ins plus every registered live
 *  voice engine, labelled with the engine's own name (never its raw id). A
 *  saved engine whose plugin is gone keeps a row so the select still shows it. */
export function useVoiceChatModeChoices(current: unknown): { labels: Record<string, string>; options: string[] } {
  const { t } = useI18n()
  const engines = useLiveVoiceEngines()

  return useMemo(() => {
    const options = [...(ENUM_OPTIONS['voice.voice_chat_mode'] ?? []), ...engines.map(engine => engine.id)]
    const labels = Object.fromEntries(engines.map(engine => [engine.id, engine.label]))

    if (isExtensionVoiceChatMode(current) && !labels[current]) {
      options.push(current)
      labels[current] = t.composer.voiceEngineMissing(current)
    }

    return { labels, options }
  }, [current, engines, t])
}

/** The provider sign-in row of an engine that declared `signIn`. */
function VoiceEngineSignInRow({ profile, provider }: { profile: string | undefined; provider: string }) {
  const { t } = useI18n()
  const c = t.composer
  const key = signInKey(provider, profile)
  const state = useStore($voiceSignIn)[key] ?? voiceSignInState(provider, profile)
  const { flow, loggedIn, name } = state

  useEffect(() => {
    void refreshVoiceSignIn(provider, profile)
  }, [profile, provider])

  const signOut = async () => {
    const ok = await confirm({
      confirmLabel: c.voiceEngineSignOutAction,
      description: c.voiceEngineSignOutBody(name),
      destructive: true,
      title: c.voiceEngineSignOutTitle(name)
    })

    if (ok) {
      await signOutVoiceProvider(provider, profile).catch(error => notifyError(error, c.voiceEngineSignOutFailed))
    }
  }

  const description =
    flow.status === 'pending'
      ? c.voiceEngineSignInCode(flow.userCode)
      : flow.status === 'error'
        ? c.voiceEngineSignInFailed(flow.message)
        : loggedIn
          ? c.voiceEngineSignedIn(name)
          : c.voiceEngineNotSignedIn(name)

  const action =
    flow.status === 'pending' || flow.status === 'starting' ? (
      <Button onClick={() => void cancelVoiceSignIn(provider, profile)} size="sm" variant="outline">
        {t.common.cancel}
      </Button>
    ) : loggedIn ? (
      state.disconnectable ? (
        <Button onClick={() => void signOut()} size="sm" variant="outline">
          {c.voiceEngineSignOutAction}
        </Button>
      ) : null
    ) : (
      <Button onClick={() => void startVoiceSignIn(provider, profile, openExternalLink)} size="sm">
        {c.voiceEngineSignInAction}
      </Button>
    )

  return (
    <ListRow
      action={action}
      data-tour={`voice-engine-sign-in-${provider}`}
      description={description}
      title={c.voiceEngineAccount(name)}
    />
  )
}

/** The engine's own `config_schema` keys it asked to show here, edited with the
 *  same form (and the same `plugins.manage settings` save) as Settings ▸ Plugins. */
function VoiceEnginePluginSettings({
  engine,
  profile
}: {
  engine: ResolvedLiveVoiceEngine
  profile: string | undefined
}) {
  const { t } = useI18n()
  const s = t.skills.plugins.settingsForm
  const { requestGateway } = useGatewayRequest()
  const owner = profile ?? null
  const rows = useStore($agentPlugins)
  const loadedFor = useStore($agentPluginsProfile)
  const status = useStore($agentPluginsStatus)
  const busyKey = useStore($agentPluginBusy)
  const records = useStore($pluginRecords)
  const owned = loadedFor === owner

  useEffect(() => {
    // A failed load is not retried here: Settings ▸ Plugins owns that recovery.
    if (!owned && (status === 'idle' || status === 'ready')) {
      void loadAgentPlugins(requestGateway, owner)
    }
  }, [owned, owner, requestGateway, status])

  // The desktop half's package folder is the join key to its agent half.
  const record = records[engine.pluginId]
  const pkg = (record && desktopPackageName(record)) || engine.pluginId
  const row = owned ? rows.find(candidate => agentPackageName(candidate) === pkg) : undefined
  const wanted = new Set(engine.settingsKeys)
  const fields = (row?.settings_schema ?? []).filter(field => wanted.has(field.key))

  if (!row || fields.length === 0) {
    return null
  }

  return (
    <PluginSettingsForm
      disabled={busyKey === row.key}
      fields={fields}
      idPrefix={`voice-engine-${engine.id.replace(':', '-')}`}
      onSave={async changes => {
        const ok = await saveAgentPluginSettings(requestGateway, {
          failMessage: s.saveFailed(engine.label),
          key: row.key ?? row.name,
          profile: owner,
          secrets: changes.secrets,
          values: changes.values,
          writeSecret: (env, value) => setEnvVar(env, value, owner ?? undefined)
        })

        if (ok) {
          notify({ kind: 'success', message: s.saved(engine.label) })
        }

        return ok
      }}
      title={engine.label}
    />
  )
}

/** Under Settings ▸ Voice ▸ Voice Chat Mode, for a selected extension engine:
 *  its provider sign-in and its inline settings. Nothing for built-in modes. */
export function VoiceEngineSettings({ mode, profile }: { mode: unknown; profile: string | undefined }) {
  const engine = useLiveVoiceEngines().find(candidate => candidate.id === mode)

  if (!engine) {
    return null
  }

  return (
    <div className="grid gap-1 pl-4" data-voice-engine-settings={engine.id}>
      {engine.signIn ? <VoiceEngineSignInRow profile={profile} provider={engine.signIn.provider} /> : null}
      {engine.settingsKeys.length > 0 ? <VoiceEnginePluginSettings engine={engine} profile={profile} /> : null}
    </div>
  )
}
