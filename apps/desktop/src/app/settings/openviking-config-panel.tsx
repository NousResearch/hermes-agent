import { type ReactNode, useCallback, useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { getOpenVikingSetup, saveOpenVikingSetup, startOpenVikingLocal, validateOpenVikingSetup } from '@/hermes'
import { ExternalLink } from '@/lib/external-link'
import { AlertCircle, CheckCircle2, Loader2, Play, RefreshCw, Save, Settings2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import type { OpenVikingConnectionValues, OpenVikingHealth, OpenVikingProfile, OpenVikingSetup } from '@/types/hermes'

import { CONTROL_TEXT } from './constants'
import { LoadingState, Pill } from './primitives'

type SourceMode = 'custom' | 'profile' | 'service'
type WizardStep = 'details' | 'source' | 'validate'
type ValidationState = { message: string; tone: 'error' | 'success' }
type RoleMismatch = 'root-key-selected-as-user' | 'user-key-selected-as-root'

interface FormValues {
  account: string
  actor_peer_id: string
  api_key: string
  api_key_type: 'none' | 'root' | 'user'
  url: string
  user: string
}

const WIZARD_STEPS: Array<{ id: WizardStep; label: string }> = [
  { id: 'source', label: 'Source' },
  { id: 'details', label: 'Details' },
  { id: 'validate', label: 'Review' }
]

const OPENVIKING_SERVICE_API_KEY_URL =
  'https://console.volcengine.com/vikingdb/openviking/region:openviking+cn-beijing'

const CREDENTIAL_OPTIONS: Array<{ label: string; value: FormValues['api_key_type'] }> = [
  { label: 'No API key', value: 'none' },
  { label: 'User API key', value: 'user' },
  { label: 'Root API key', value: 'root' }
]

function normalizeComparableUrl(url: string) {
  return url.trim().replace(/\/+$/, '')
}

function connectionKindForUrl(url: string, serviceUrl: string) {
  return normalizeComparableUrl(url) === normalizeComparableUrl(serviceUrl) ? 'OpenViking Service' : 'Custom'
}

function modeFromSetup(setup: OpenVikingSetup): SourceMode {
  if (setup.active.source === 'ovcli') {
    return 'profile'
  }

  return connectionKindForUrl(setup.active.url, setup.defaults.service_url) === 'OpenViking Service' ? 'service' : 'custom'
}

function apiKeyTypeFromConnection(
  values: Pick<OpenVikingConnectionValues, 'account' | 'api_key_set' | 'api_key_type' | 'user'>
): FormValues['api_key_type'] {
  if (values.api_key_type) {
    return values.api_key_type
  }

  if (!values.api_key_set) {
    return 'none'
  }

  return values.account || values.user ? 'root' : 'user'
}

function seedValues(setup: OpenVikingSetup, mode: SourceMode): FormValues {
  const active = setup.active
  const customUrl = active.url && active.url !== setup.defaults.service_url ? active.url : setup.defaults.url

  return {
    account: active.account ?? '',
    actor_peer_id: active.actor_peer_id || setup.defaults.actor_peer_id,
    api_key: '',
    api_key_type: mode === 'service' ? 'user' : apiKeyTypeFromConnection(active),
    url: mode === 'service' ? setup.defaults.service_url : customUrl,
    user: active.user ?? ''
  }
}

function profileValues(profile: OpenVikingProfile, fallbackAgentId: string): FormValues {
  return {
    account: profile.account,
    actor_peer_id: profile.actor_peer_id || fallbackAgentId,
    api_key: '',
    api_key_type: apiKeyTypeFromConnection(profile),
    url: profile.url,
    user: profile.user
  }
}

function isLocalUrl(url: string): boolean {
  return /^http:\/\/(127\.0\.0\.1|localhost|\[::1\])(?::\d+)?(?:\/|$)/i.test(url.trim())
}

function defaultProfileName(mode: SourceMode): string {
  return mode === 'service' ? 'openviking_service' : 'openviking_custom'
}

function sourceTitle(mode: SourceMode): string {
  if (mode === 'profile') {
    return 'Existing Profiles'
  }

  return mode === 'service' ? 'OpenViking Service' : 'Custom Server'
}

function buildConnectionValues(values: FormValues, mode: SourceMode, setup: OpenVikingSetup): OpenVikingConnectionValues {
  const apiKeyType = mode === 'service' ? 'user' : values.api_key_type

  const payload: OpenVikingConnectionValues = {
    actor_peer_id: values.actor_peer_id.trim() || setup.defaults.actor_peer_id,
    api_key_type: apiKeyType,
    url: mode === 'service' ? setup.defaults.service_url : values.url.trim()
  }

  if (apiKeyType === 'none') {
    payload.api_key = ''
  } else if (values.api_key.trim()) {
    payload.api_key = values.api_key.trim()

    if (apiKeyType === 'root') {
      payload.root_api_key = values.api_key.trim()
    }
  }

  if (apiKeyType === 'root') {
    payload.account = values.account.trim()
    payload.user = values.user.trim()
  }

  return payload
}

function requiresRootIdentity(values: FormValues | null, mode: SourceMode): boolean {
  return mode === 'custom' && values?.api_key_type === 'root'
}

function messageFromError(err: unknown): string {
  if (err instanceof Error) {
    const jsonMatch = err.message.match(/\{.*\}$/)

    if (jsonMatch) {
      try {
        const parsed = JSON.parse(jsonMatch[0]) as { detail?: unknown }

        if (typeof parsed.detail === 'string') {
          return parsed.detail
        }
      } catch {
        return err.message
      }
    }

    return err.message
  }

  return typeof err === 'string' ? err : ''
}

function validationFailureMessage(message: string): string {
  const raw = message.trim()
  const lower = raw.toLowerCase()

  if (!raw) {
    return 'OpenViking could not be validated. Check the URL and credentials, then try again.'
  }

  if (
    lower.includes('profile file was not found') ||
    lower.includes('profile no longer exists') ||
    lower.includes('profile could not be loaded')
  ) {
    return 'The selected OpenViking profile no longer exists. Refresh profiles or choose another profile.'
  }

  if (
    lower.includes('connection refused') ||
    lower.includes('connection reset') ||
    lower.includes('timed out') ||
    lower.includes('timeout') ||
    lower.includes('not reachable')
  ) {
    return 'OpenViking could not be reached. Check that the server URL is correct and the server is running.'
  }

  if (lower.includes('require') && lower.includes('api key')) {
    return 'An API key is required for this OpenViking server. Enter an API key and try again.'
  }

  if (
    lower.includes('authentication') ||
    lower.includes('unauthorized') ||
    lower.includes('forbidden') ||
    lower.includes('401') ||
    lower.includes('403')
  ) {
    return 'OpenViking rejected the credentials. Check the API key and account/user details, then try again.'
  }

  if (lower.includes('reported unhealthy')) {
    return 'OpenViking is reachable but reports an unhealthy status. Check the server health and try again.'
  }

  return raw
    .replace(/^OpenViking validation failed:\s*/i, '')
    .replace(/^OpenViking authentication validation failed:\s*/i, 'Authentication failed: ')
    .replace(/^OpenViking root API key validation failed:\s*/i, 'Root API key validation failed: ')
}

function Field({
  children,
  hint,
  id,
  label
}: {
  children: ReactNode
  hint?: string
  id: string
  label: string
}) {
  return (
    <div className="grid gap-1.5">
      <label className="text-xs font-medium text-muted-foreground" htmlFor={id}>
        {label}
      </label>
      {children}
      {hint ? <div className="text-xs text-muted-foreground">{hint}</div> : null}
    </div>
  )
}

function SetupChoice({
  active,
  children,
  description,
  onClick,
  title
}: {
  active: boolean
  children: ReactNode
  description: string
  onClick: () => void
  title: string
}) {
  return (
    <button
      aria-label={title}
      aria-pressed={active}
      className={cn(
        'grid min-h-20 gap-1 rounded-md border border-border bg-background px-3 py-2 text-left outline-none transition-colors hover:bg-(--chrome-action-hover) focus-visible:ring-[0.1875rem] focus-visible:ring-ring/50',
        active && 'border-primary/70 bg-primary/8'
      )}
      onClick={onClick}
      type="button"
    >
      <span className="flex items-center justify-between gap-3 text-xs font-semibold text-foreground">
        {title}
        {active ? <CheckCircle2 className="size-4 shrink-0 text-primary" /> : null}
      </span>
      <span className="text-xs leading-5 text-muted-foreground">{description}</span>
      {children}
    </button>
  )
}

function Notice({ state }: { state: ValidationState }) {
  const Icon = state.tone === 'success' ? CheckCircle2 : AlertCircle

  return (
    <div
      className={cn(
        'flex items-start gap-2 rounded-md border px-3 py-2 text-xs leading-5',
        state.tone === 'success'
          ? 'border-primary/30 bg-primary/8 text-foreground'
          : 'border-destructive/40 bg-destructive/10 text-destructive'
      )}
      role={state.tone === 'error' ? 'alert' : 'status'}
    >
      <Icon className="mt-0.5 size-3.5 shrink-0" />
      <span>{state.message}</span>
    </div>
  )
}

function RoleMismatchPrompt({
  mismatch,
  onReenterRootKey,
  onReenterUserKey,
  onUseAsRootKey,
  onUseAsUserKey
}: {
  mismatch: RoleMismatch
  onReenterRootKey: () => void
  onReenterUserKey: () => void
  onUseAsRootKey: () => void
  onUseAsUserKey: () => void
}) {
  const rootSelectedAsUser = mismatch === 'root-key-selected-as-user'
  const title = rootSelectedAsUser ? 'This key has root access.' : 'This key is a user API key.'

  const message = rootSelectedAsUser
    ? 'Switch to Root API key and provide account/user, or enter a user API key.'
    : 'Switch to User API key, or enter a root API key.'

  return (
    <div
      className="grid gap-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2.5 text-xs leading-5 text-foreground"
      role="alert"
    >
      <div className="flex items-start gap-2">
        <AlertCircle className="mt-0.5 size-3.5 shrink-0 text-amber-600 dark:text-amber-300" />
        <div className="grid gap-1">
          <div className="font-medium">{title}</div>
          <div className="text-muted-foreground">{message}</div>
        </div>
      </div>
      <div className="flex flex-wrap justify-end gap-2">
        {rootSelectedAsUser ? (
          <>
            <Button onClick={onReenterUserKey} size="sm" type="button" variant="ghost">
              Enter User API key
            </Button>
            <Button onClick={onUseAsRootKey} size="sm" type="button" variant="secondary">
              Use as Root API key
            </Button>
          </>
        ) : (
          <>
            <Button onClick={onReenterRootKey} size="sm" type="button" variant="ghost">
              Enter Root API key
            </Button>
            <Button onClick={onUseAsUserKey} size="sm" type="button" variant="secondary">
              Use as User API key
            </Button>
          </>
        )}
      </div>
    </div>
  )
}

function healthTone(status: OpenVikingHealth['status']) {
  if (status === 'healthy') {
    return {
      dot: 'bg-emerald-500',
      shell: 'text-emerald-700 dark:text-emerald-300'
    }
  }

  if (status === 'unhealthy') {
    return {
      dot: 'bg-amber-500 shadow-[0_0_0_3px_rgba(245,158,11,0.18)]',
      shell: 'border-amber-500/25 bg-amber-500/10 text-amber-700 dark:text-amber-300'
    }
  }

  return {
    dot: 'bg-rose-500 shadow-[0_0_0_3px_rgba(244,63,94,0.16)]',
    shell: 'border-rose-500/25 bg-rose-500/10 text-rose-700 dark:text-rose-300'
  }
}

function healthSummary(status: OpenVikingHealth['status']) {
  if (status === 'healthy') {
    return 'Ready to use'
  }

  if (status === 'unhealthy') {
    return 'Server reached, review details'
  }

  return 'Cannot reach server'
}

function HealthStatus({ health }: { health: OpenVikingHealth }) {
  const [detailsOpen, setDetailsOpen] = useState(false)
  const tone = healthTone(health.status)
  const summary = healthSummary(health.status)
  const canShowDetails = health.status !== 'healthy' && Boolean(health.message)
  const DetailIcon = health.status === 'healthy' ? CheckCircle2 : AlertCircle
  const showSummary = health.status !== 'healthy'

  return (
    <div className="min-w-0 sm:text-right">
      <div className="text-[0.6875rem] font-medium uppercase text-muted-foreground">Status</div>
      <div className="mt-1 flex min-w-0 flex-col gap-1 sm:items-end">
        {canShowDetails ? (
          <button
            aria-label="View OpenViking status details"
            className={cn(
              'inline-flex w-fit items-center gap-2 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors hover:bg-(--chrome-action-hover) focus-visible:ring-[0.1875rem] focus-visible:ring-ring/50',
              tone.shell
            )}
            onClick={() => setDetailsOpen(true)}
            title={health.message}
            type="button"
          >
            <span className={cn('size-2 rounded-full', tone.dot)} />
            {health.label}
          </button>
        ) : (
          <div
            className={cn(
              'inline-flex w-fit items-center gap-1.5 text-xs font-medium',
              tone.shell
            )}
            role="status"
            title={health.message}
          >
            <span className={cn('size-1.5 rounded-full', tone.dot)} />
            {health.label}
          </div>
        )}
        {showSummary ? <div className="text-xs text-muted-foreground">{summary}</div> : null}
      </div>
      <Dialog onOpenChange={setDetailsOpen} open={detailsOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader className="pr-28 sm:pr-40">
            <DialogTitle icon={DetailIcon}>OpenViking status details</DialogTitle>
            <DialogDescription>{summary}</DialogDescription>
          </DialogHeader>
          <div className="rounded-md border border-border bg-background px-3 py-2 font-mono text-xs leading-5 break-words text-foreground">
            {health.message}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export function OpenVikingConfigPanel() {
  const [setup, setSetup] = useState<OpenVikingSetup | null>(null)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [mode, setMode] = useState<SourceMode>('service')
  const [step, setStep] = useState<WizardStep>('source')
  const [values, setValues] = useState<FormValues | null>(null)
  const [profilePath, setProfilePath] = useState('')
  const [profileName, setProfileName] = useState(defaultProfileName('service'))
  const [saving, setSaving] = useState(false)
  const [validating, setValidating] = useState(false)
  const [starting, setStarting] = useState(false)
  const [refreshingProfiles, setRefreshingProfiles] = useState(false)
  const [validation, setValidation] = useState<ValidationState | null>(null)
  const [roleMismatch, setRoleMismatch] = useState<RoleMismatch | null>(null)

  const applySetupSnapshot = useCallback(
    (next: OpenVikingSetup, preferred?: { mode?: SourceMode; profilePath?: string }) => {
      const nextMode = preferred?.mode ?? modeFromSetup(next)
      const activeProfile = next.profiles.find(profile => profile.is_active) ?? next.profiles[0]

      const preferredProfile = preferred?.profilePath
        ? next.profiles.find(profile => profile.path === preferred.profilePath)
        : null

      const profile = preferredProfile ?? activeProfile

      setSetup(next)
      setMode(nextMode)
      setProfilePath(profile?.path ?? '')
      setProfileName(defaultProfileName(nextMode))
      setValues(nextMode === 'profile' && profile ? profileValues(profile, next.defaults.actor_peer_id) : seedValues(next, nextMode))
      setValidation(null)
      setRoleMismatch(null)
    },
    []
  )

  const refresh = useCallback(async (preferred?: { mode?: SourceMode; profilePath?: string }) => {
    try {
      const next = await getOpenVikingSetup()
      applySetupSnapshot(next, preferred)
    } catch (err) {
      notifyError(err, 'OpenViking settings failed to load')
      setSetup(null)
      setValues(null)
    }
  }, [applySetupSnapshot])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const activeProfile = useMemo(
    () => setup?.profiles.find(profile => profile.is_active) ?? null,
    [setup?.profiles]
  )

  const selectedProfile = useMemo(
    () => setup?.profiles.find(profile => profile.path === profilePath) ?? null,
    [profilePath, setup?.profiles]
  )

  const currentStepIndex = WIZARD_STEPS.findIndex(item => item.id === step)
  const currentStepNumber = currentStepIndex >= 0 ? currentStepIndex + 1 : 1
  const currentStepLabel = WIZARD_STEPS[currentStepNumber - 1]?.label ?? WIZARD_STEPS[0].label
  const canStartLocal = mode === 'custom' && values ? isLocalUrl(values.url) : false
  const rootIdentityRequired = requiresRootIdentity(values, mode)
  const hasRootIdentity = Boolean(values?.account.trim()) && Boolean(values?.user.trim())

  const canContinueDetails =
    mode === 'profile'
      ? Boolean(profilePath)
      : Boolean(profileName.trim()) &&
        Boolean(values?.url.trim()) &&
        Boolean(values?.actor_peer_id.trim()) &&
        (!rootIdentityRequired || hasRootIdentity)

  const resetValidation = useCallback(() => {
    setValidation(null)
    setRoleMismatch(null)
  }, [])

  const openWizard = useCallback(() => {
    setStep('source')
    setValidation(null)
    setRoleMismatch(null)
    setDialogOpen(true)
    void refresh()
  }, [refresh])

  const refreshProfiles = useCallback(async () => {
    setRefreshingProfiles(true)

    try {
      await refresh({ mode: 'profile', profilePath })
    } finally {
      setRefreshingProfiles(false)
    }
  }, [profilePath, refresh])

  const chooseMode = useCallback(
    async (nextMode: SourceMode) => {
      if (!setup) {
        return
      }

      setMode(nextMode)
      setProfileName(defaultProfileName(nextMode))
      resetValidation()

      if (nextMode === 'profile') {
        setRefreshingProfiles(true)

        try {
          await refresh({ mode: 'profile', profilePath })
        } finally {
          setRefreshingProfiles(false)
        }

        return
      }

      setValues(seedValues(setup, nextMode))
    },
    [profilePath, refresh, resetValidation, setup]
  )

  const updateValue = useCallback(
    <K extends keyof FormValues>(key: K, value: FormValues[K]) => {
      resetValidation()
      setValues(current => (current ? { ...current, [key]: value } : current))
    },
    [resetValidation]
  )

  const chooseProfile = useCallback(
    (path: string) => {
      if (!setup) {
        return
      }

      const profile = setup.profiles.find(item => item.path === path)
      setProfilePath(path)
      resetValidation()

      if (profile) {
        setValues(profileValues(profile, setup.defaults.actor_peer_id))
      }
    },
    [resetValidation, setup]
  )

  const validateCurrentSetup = useCallback(async () => {
    if (!setup || !values) {
      return false
    }

    setValidating(true)
    setValidation(null)
    setRoleMismatch(null)

    try {
      const result =
        mode === 'profile'
          ? await validateOpenVikingSetup({ profile_path: profilePath, values: {} })
          : await validateOpenVikingSetup({
              require_api_key: mode === 'service' || !isLocalUrl(values.url),
              values: buildConnectionValues(values, mode, setup)
            })

      if (!result.ok) {
        setValidation({ message: validationFailureMessage(result.message), tone: 'error' })

        return false
      }

      if (mode === 'custom' && result.ok) {
        if (values.api_key_type === 'user' && result.role === 'root') {
          setRoleMismatch('root-key-selected-as-user')

          return false
        }

        if (values.api_key_type === 'root' && result.role === 'user') {
          setRoleMismatch('user-key-selected-as-root')

          return false
        }
      }

      setValidation(null)

      return true
    } catch (err) {
      setValidation({ message: validationFailureMessage(messageFromError(err)), tone: 'error' })

      return false
    } finally {
      setValidating(false)
    }
  }, [mode, profilePath, setup, values])

  const useAsRootKey = useCallback(() => {
    setValues(current => (current ? { ...current, api_key_type: 'root', account: '', user: '' } : current))
    setValidation(null)
    setRoleMismatch(null)
    setStep('details')
  }, [])

  const useAsUserKey = useCallback(() => {
    setValues(current => (current ? { ...current, api_key_type: 'user', account: '', user: '' } : current))
    setValidation(null)
    setRoleMismatch(null)
    setStep('details')
  }, [])

  const reenterRootKey = useCallback(() => {
    setValues(current => (current ? { ...current, api_key: '', api_key_type: 'root' } : current))
    setValidation(null)
    setRoleMismatch(null)
    setStep('details')
  }, [])

  const reenterUserKey = useCallback(() => {
    setValues(current => (current ? { ...current, api_key: '', api_key_type: 'user', account: '', user: '' } : current))
    setValidation(null)
    setRoleMismatch(null)
    setStep('details')
  }, [])

  const save = useCallback(async () => {
    if (!setup || !values) {
      return
    }

    const validated = await validateCurrentSetup()

    if (!validated) {
      return
    }

    setSaving(true)

    try {
      if (mode === 'profile') {
        await saveOpenVikingSetup({ profile_path: profilePath, save_mode: 'profile', values: {} })
      } else {
        await saveOpenVikingSetup({
          profile_name: profileName.trim(),
          save_mode: 'profile',
          values: buildConnectionValues(values, mode, setup)
        })
      }

      notify({ kind: 'success', title: 'OpenViking saved', message: 'Memory provider configuration updated.' })
      setDialogOpen(false)
      await refresh()
    } catch (err) {
      notifyError(err, 'Failed to save OpenViking settings')
    } finally {
      setSaving(false)
    }
  }, [mode, profileName, profilePath, refresh, setup, validateCurrentSetup, values])

  const startLocal = useCallback(async () => {
    if (!values) {
      return
    }

    setStarting(true)
    setValidation(null)
    setRoleMismatch(null)

    try {
      const result = await startOpenVikingLocal(values.url)
      setValidation({ message: result.message, tone: result.ok ? 'success' : 'error' })
    } catch (err) {
      notifyError(err, 'Could not start local OpenViking')
    } finally {
      setStarting(false)
    }
  }, [values])

  if (!setup || !values) {
    return <LoadingState label="Loading OpenViking settings..." />
  }

  const activeConnectionKind = connectionKindForUrl(setup.active.url, setup.defaults.service_url)

  const activeTitle = activeProfile
    ? `${activeProfile.display_name} (${activeConnectionKind})`
    : sourceTitle(modeFromSetup(setup))

  const activeLabel = activeProfile ? 'Active profile' : setup.active.source === 'ovcli' ? 'Linked profile' : 'Active setup'

  return (
    <section className="py-3">
      <div className="grid gap-4 rounded-xl bg-background/60 p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
              OpenViking settings
            </div>
            <div className="mt-1 text-xs text-muted-foreground">
              Configure Hermes memory with OpenViking profiles, service, or a self-hosted server.
            </div>
          </div>
          <Button onClick={openWizard} size="sm" type="button" variant="secondary">
            <Settings2 className="size-3.5" />
            Configure
          </Button>
        </div>

        <div className="grid gap-3 rounded-md border border-border bg-background px-3 py-2.5 sm:grid-cols-[minmax(0,1.1fr)_minmax(0,1.5fr)_minmax(12rem,auto)] sm:items-center">
          <div className="min-w-0">
            <div className="text-[0.6875rem] font-medium uppercase text-muted-foreground">{activeLabel}</div>
            <div className="mt-0.5 truncate text-xs font-medium text-foreground">{activeTitle}</div>
          </div>
          <div className="min-w-0">
            <div className="text-[0.6875rem] font-medium uppercase text-muted-foreground">OpenViking URL</div>
            <div className="mt-0.5 truncate font-mono text-xs text-foreground">{setup.active.url}</div>
          </div>
          <HealthStatus health={setup.health} />
        </div>
        {setup.legacy_env_present.length > 0 ? (
          <div
            className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-100"
            title={setup.legacy_env_present.join(', ')}
          >
            <div className="font-medium">Legacy OpenViking env vars ignored</div>
            <div className="mt-0.5 text-amber-800/80 dark:text-amber-100/80">
              Save this setup to clear old variable names.
            </div>
          </div>
        ) : null}
      </div>

      <Dialog onOpenChange={setDialogOpen} open={dialogOpen}>
        <DialogContent className="max-w-2xl" onInteractOutside={event => event.preventDefault()}>
          <div className="absolute right-5 top-11 text-xs font-medium text-muted-foreground">
            Step {currentStepNumber} of {WIZARD_STEPS.length}: {currentStepLabel}
          </div>
          <DialogHeader>
            <DialogTitle icon={Settings2}>Configure OpenViking</DialogTitle>
            <DialogDescription>Use a profile, the OpenViking service, or a custom server.</DialogDescription>
          </DialogHeader>

          <div className="grid gap-4">
            {step === 'source' ? (
              <div className="grid gap-3">
                <div className="text-sm font-medium text-foreground">Choose setup</div>
                <div className="grid gap-2 sm:grid-cols-3">
                  <SetupChoice
                    active={mode === 'service'}
                    description="Managed endpoint"
                    onClick={() => void chooseMode('service')}
                    title="OpenViking Service"
                  >
                    <span className="truncate font-mono text-[0.6875rem] text-muted-foreground">
                      {setup.defaults.service_url}
                    </span>
                  </SetupChoice>
                  <SetupChoice
                    active={mode === 'profile'}
                    description={
                      refreshingProfiles
                        ? 'Refreshing profiles'
                        : setup.profiles.length
                          ? 'Use an ovcli profile'
                          : 'No profiles found'
                    }
                    onClick={() => void chooseMode('profile')}
                    title="Existing Profiles"
                  >
                    <span className="truncate text-[0.6875rem] text-muted-foreground">
                      {selectedProfile?.display_name ?? setup.profiles[0]?.display_name ?? 'Create one first'}
                    </span>
                  </SetupChoice>
                  <SetupChoice
                    active={mode === 'custom'}
                    description="Local, VPS, or self-hosted"
                    onClick={() => void chooseMode('custom')}
                    title="Custom Server"
                  >
                    <span className="truncate font-mono text-[0.6875rem] text-muted-foreground">
                      {mode === 'custom' ? values.url : setup.defaults.url}
                    </span>
                  </SetupChoice>
                </div>
              </div>
            ) : null}

            {step === 'details' ? (
              <div className="grid gap-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="text-sm font-medium text-foreground">{sourceTitle(mode)}</div>
                  {mode === 'service' ? (
                    <ExternalLink
                      className="text-xs font-medium text-primary decoration-primary/30 hover:decoration-primary/70"
                      href={OPENVIKING_SERVICE_API_KEY_URL}
                      showExternalIcon={false}
                    >
                      Get OpenViking API key
                    </ExternalLink>
                  ) : null}
                  {mode === 'profile' ? (
                    <Button
                      disabled={refreshingProfiles}
                      onClick={() => void refreshProfiles()}
                      size="sm"
                      type="button"
                      variant="secondary"
                    >
                      {refreshingProfiles ? (
                        <Loader2 className="size-3.5 animate-spin" />
                      ) : (
                        <RefreshCw className="size-3.5" />
                      )}
                      Refresh profiles
                    </Button>
                  ) : null}
                </div>
                {mode === 'profile' ? (
                  setup.profiles.length > 0 ? (
                    <Field
                      hint={selectedProfile?.description}
                      id="openviking-profile"
                      label="OpenViking profile"
                    >
                      <Select onValueChange={chooseProfile} value={profilePath}>
                        <SelectTrigger className={CONTROL_TEXT} id="openviking-profile">
                          <SelectValue placeholder="Choose profile" />
                        </SelectTrigger>
                        <SelectContent>
                          {setup.profiles.map(profile => (
                            <SelectItem key={profile.path} value={profile.path}>
                              {profile.display_name}
                              {profile.is_active ? ' (Active)' : ''}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </Field>
                  ) : (
                    <div className="rounded-md border border-border bg-background px-3 py-2 text-xs text-muted-foreground">
                      No OpenViking profiles were found.
                    </div>
                  )
                ) : (
                  <div className="grid gap-4">
                    <Field id="openviking-profile-name" label="Profile name">
                      <Input
                        className="font-mono"
                        id="openviking-profile-name"
                        onChange={event => {
                          resetValidation()
                          setProfileName(event.target.value)
                        }}
                        value={profileName}
                      />
                    </Field>

                    <Field id="openviking-url" label="OpenViking URL">
                      <Input
                        className="font-mono"
                        id="openviking-url"
                        onChange={event => updateValue('url', event.target.value)}
                        readOnly={mode === 'service'}
                        value={mode === 'service' ? setup.defaults.service_url : values.url}
                      />
                    </Field>

                    {mode === 'custom' ? (
                      <Field id="openviking-credential" label="Credential">
                        <Select
                          onValueChange={value => updateValue('api_key_type', value as FormValues['api_key_type'])}
                          value={values.api_key_type}
                        >
                          <SelectTrigger className={CONTROL_TEXT} id="openviking-credential">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {CREDENTIAL_OPTIONS.map(option => (
                              <SelectItem key={option.value} value={option.value}>
                                {option.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </Field>
                    ) : null}

                    {(mode === 'service' || values.api_key_type !== 'none') && (
                      <Field id="openviking-api-key" label="OpenViking API key">
                        <Input
                          className="font-mono"
                          id="openviking-api-key"
                          onChange={event => updateValue('api_key', event.target.value)}
                          type="password"
                          value={values.api_key}
                        />
                      </Field>
                    )}

                    {values.api_key_type === 'root' ? (
                      <div className="grid gap-3 sm:grid-cols-2">
                        <Field id="openviking-account" label="Account">
                          <Input
                            className="font-mono"
                            id="openviking-account"
                            onChange={event => updateValue('account', event.target.value)}
                            value={values.account}
                          />
                        </Field>
                        <Field id="openviking-user" label="User">
                          <Input
                            className="font-mono"
                            id="openviking-user"
                            onChange={event => updateValue('user', event.target.value)}
                            value={values.user}
                          />
                        </Field>
                      </div>
                    ) : null}

                    <Field
                      hint="Identifies this Hermes agent inside OpenViking."
                      id="openviking-agent-id"
                      label="Agent ID"
                    >
                      <Input
                        className="font-mono"
                        id="openviking-agent-id"
                        onChange={event => updateValue('actor_peer_id', event.target.value)}
                        value={values.actor_peer_id}
                      />
                    </Field>

                    {canStartLocal ? (
                      <div className="flex justify-start">
                        <Button disabled={starting} onClick={() => void startLocal()} size="sm" type="button" variant="secondary">
                          {starting ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />}
                          Start local server
                        </Button>
                      </div>
                    ) : null}
                    {validation ? <Notice state={validation} /> : null}
                  </div>
                )}
              </div>
            ) : null}

            {step === 'validate' ? (
              <div className="grid gap-4">
                <div className="grid gap-3 rounded-md border border-border bg-background px-3 py-2.5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="text-sm font-medium text-foreground">{sourceTitle(mode)}</div>
                    {mode !== 'profile' ? <Pill>{profileName.trim()}</Pill> : null}
                  </div>
                  <div className="grid gap-2 text-xs sm:grid-cols-2">
                    <div className="min-w-0">
                      <div className="text-muted-foreground">OpenViking URL</div>
                      <div className="truncate font-mono text-foreground">{values.url}</div>
                    </div>
                    <div className="min-w-0">
                      <div className="text-muted-foreground">Agent ID</div>
                      <div className="truncate font-mono text-foreground">{values.actor_peer_id}</div>
                    </div>
                  </div>
                  {mode === 'profile' && selectedProfile ? (
                    <div className="flex min-w-0 items-center gap-1 text-xs text-muted-foreground">
                      <span className="shrink-0">Profile Path:</span>
                      <span className="truncate font-mono">{selectedProfile.path}</span>
                    </div>
                  ) : null}
                </div>

                {validation ? <Notice state={validation} /> : null}
                {roleMismatch ? (
                  <RoleMismatchPrompt
                    mismatch={roleMismatch}
                    onReenterRootKey={reenterRootKey}
                    onReenterUserKey={reenterUserKey}
                    onUseAsRootKey={useAsRootKey}
                    onUseAsUserKey={useAsUserKey}
                  />
                ) : null}

                {canStartLocal ? (
                  <div className="flex justify-start">
                    <Button disabled={starting} onClick={() => void startLocal()} size="sm" type="button" variant="secondary">
                      {starting ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />}
                      Start local server
                    </Button>
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>

          <DialogFooter>
            {step !== 'source' ? (
              <Button
                disabled={saving || validating || starting}
                onClick={() => setStep(step === 'validate' ? 'details' : 'source')}
                size="sm"
                type="button"
                variant="ghost"
              >
                Back
              </Button>
            ) : null}
            {step === 'source' ? (
              <Button
                disabled={mode === 'profile' && (refreshingProfiles || setup.profiles.length === 0)}
                onClick={() => setStep('details')}
                size="sm"
                type="button"
              >
                Next
              </Button>
            ) : null}
            {step === 'details' ? (
              <Button disabled={!canContinueDetails} onClick={() => setStep('validate')} size="sm" type="button">
                Continue
              </Button>
            ) : null}
            {step === 'validate' ? (
              <Button
                disabled={saving || validating || (mode === 'profile' && !profilePath)}
                onClick={() => void save()}
                size="sm"
                type="button"
              >
                {saving || validating ? <Loader2 className="size-3.5 animate-spin" /> : <Save className="size-3.5" />}
                {validating ? 'Checking...' : saving ? 'Saving...' : 'Save profile'}
              </Button>
            ) : null}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  )
}
