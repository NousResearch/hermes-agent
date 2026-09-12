import { useQuery } from '@tanstack/react-query'
import { useCallback, useState } from 'react'

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { getBrowserRealProfile, type ProfileScope, profileScopeKey, saveHermesConfigRecord } from '@/hermes'
import { useI18n } from '@/i18n'
import { AlertTriangle, Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import type { HermesConfigRecord, RealProfileBrowser, RealProfileCandidates } from '@/types/hermes'

import { hermesConfigCacheWriter, useHermesConfigRecord } from '../hooks/use-config-record'

import { CONTROL_TEXT } from './constants'
import { ListRow, ToggleRow } from './primitives'

interface BrowserRealProfilePanelProps {
  /** Capabilities profile-scope override — the toggle reads/writes THIS
   *  profile's config.yaml instead of the app-wide active one. */
  profile?: ProfileScope
}

/** Sentinel for the "follow the system / last used" rows. Empty string is what
 *  the config keys actually store, but Radix Select reserves '' for "no value",
 *  so the option value is a marker translated at the write boundary. */
const AUTO = '__auto__'

function browserSection(record: Record<string, unknown> | undefined): Record<string, unknown> {
  const browser = record?.browser

  return browser && typeof browser === 'object' && !Array.isArray(browser)
    ? (browser as Record<string, unknown>)
    : {}
}

function readStringSetting(record: Record<string, unknown> | undefined, key: string): string {
  const value = browserSection(record)[key]

  return typeof value === 'string' ? value.trim() : ''
}

/** Shared with the Browser pane's first-open consent prompt, so both surfaces
 *  agree on what "on" means for `browser.use_real_profile`. */
export function readUseRealProfile(record: Record<string, unknown> | undefined): boolean {
  return Boolean(browserSection(record).use_real_profile)
}

/** Cache key for the discovery fetch. Scoped like the config record it sits
 *  next to: two profiles (or two gateways' same-named profiles) resolve to
 *  different browsers, so they must never share a cache row. */
export const realProfileCandidatesKey = (profile?: ProfileScope) =>
  ['browser-real-profile', profileScopeKey(profile)] as const

/** Human name of the identity a launch would use right now, e.g. "Brave · Personal".
 *  Falls back to the raw directory when the browser reports no display name. */
export function describeResolved(
  data: RealProfileCandidates | undefined,
  copy: { browsingAs: (browser: string, profile: string) => string }
): null | string {
  if (!data?.resolved_browser || !data.resolved_profile) {
    return null
  }

  const browser = data.browsers.find(row => row.key === data.resolved_browser)
  const profile = browser?.profiles.find(row => row.directory === data.resolved_profile)

  return copy.browsingAs(browser?.label ?? data.resolved_browser, profile?.name ?? data.resolved_profile)
}

/**
 * Real-profile browsing controls, rendered at the top of the Capabilities →
 * Tools → Browser detail pane: the `browser.use_real_profile` consent toggle
 * plus — once it is on — pickers for WHICH browser and WHICH profile inside it
 * the agent borrows logins from (`browser.real_profile_browser` /
 * `browser.real_profile_pin`).
 *
 * Why the pickers matter: unpinned, the snapshot follows the OS default browser
 * and its last-used profile, so on a machine with work and personal profiles
 * "whichever you touched last" silently decides the agent's identity. Pinning is
 * what lets one Hermes profile browse as Brave/Work while another browses as
 * Chrome/Personal — the settings live in each profile's own config.yaml, and
 * this panel is profile-scoped, so both are configurable from one window.
 *
 * All three settings write config.yaml through the same deep-merging
 * PUT /api/config every other settings surface uses (applies to new sessions).
 * The candidate list is READ from the gateway that would launch the browser, so
 * a desktop attached to a remote gateway lists that machine's browsers — the
 * only correct answer, since that is where the snapshot is taken.
 */
export function BrowserRealProfilePanel({ profile }: BrowserRealProfilePanelProps) {
  const { t } = useI18n()
  const copy = t.settings.toolsets.browserRealProfile
  const pickerCopy = copy.picker
  const { data: config } = useHermesConfigRecord(profile)
  const setConfig = hermesConfigCacheWriter(profile)
  const [busy, setBusy] = useState(false)

  const enabled = readUseRealProfile(config)

  const {
    data: candidates,
    error: candidatesError,
    isLoading: candidatesLoading,
    refetch: refetchCandidates
  } = useQuery({
    queryKey: realProfileCandidatesKey(profile),
    queryFn: () => getBrowserRealProfile(profile ?? undefined),
    // Only probe the filesystem for browsers once the user has consented —
    // before that the panel is a single toggle and the answer is unused.
    enabled,
    staleTime: 30_000
  })

  /** Deep-merge `browser.*` keys and persist, rolling the cache back on failure. */
  const writeBrowserSettings = useCallback(
    async (patch: Record<string, unknown>, toast?: { title: string; message: string }) => {
      if (!config) {
        return
      }

      const next: HermesConfigRecord = { ...config, browser: { ...browserSection(config), ...patch } }

      setBusy(true)
      setConfig(next)

      try {
        await saveHermesConfigRecord(next, profile)

        if (toast) {
          notify({ kind: 'info', title: toast.title, message: toast.message })
        }

        // The resolved identity is computed server-side from these very keys,
        // so re-read it rather than guessing what the backend now resolves to.
        void refetchCandidates()
      } catch (err) {
        setConfig(config)
        notifyError(err, copy.failedSave)
      } finally {
        setBusy(false)
      }
    },
    [config, copy.failedSave, profile, refetchCandidates, setConfig]
  )

  const toggle = useCallback(
    (on: boolean) =>
      writeBrowserSettings(
        { use_real_profile: on },
        {
          title: on ? copy.enabledTitle : copy.disabledTitle,
          message: on ? copy.enabledMessage : copy.disabledMessage
        }
      ),
    [copy, writeBrowserSettings]
  )

  const selectedBrowser = readStringSetting(config, 'real_profile_browser')
  const selectedPin = readStringSetting(config, 'real_profile_pin')

  // Profiles of the browser the picker currently targets: the explicit pick if
  // there is one, else whatever the backend resolved (system default).
  const activeBrowser: RealProfileBrowser | undefined = candidates?.browsers.find(
    row => row.key === (selectedBrowser || candidates.resolved_browser)
  )

  const resolvedLabel = describeResolved(candidates, pickerCopy)

  // A pin the backend rejected (renamed/removed profile) must stay visible in
  // the trigger instead of silently reading as "Last used" — the config still
  // says it, and the error row explains why it is not being honored.
  const pinIsMissing = Boolean(
    selectedPin && activeBrowser && !activeBrowser.profiles.some(row => row.directory === selectedPin)
  )

  return (
    <div className="grid gap-1">
      <ToggleRow
        checked={enabled}
        description={copy.description}
        disabled={busy || !config}
        label={copy.label}
        onChange={on => void toggle(on)}
      />

      {enabled && candidatesLoading && (
        <div className="flex items-center gap-2 px-1 pb-2 text-xs text-muted-foreground">
          <Loader2 className="size-3.5 animate-spin" />
          {pickerCopy.loading}
        </div>
      )}

      {enabled && Boolean(candidatesError) && (
        <p className="px-1 pb-2 text-[0.68rem] text-amber-600 dark:text-amber-300">{pickerCopy.failedLoad}</p>
      )}

      {enabled && candidates && !candidates.supported && (
        <p className="px-1 pb-2 text-[0.68rem] text-muted-foreground">
          {pickerCopy.unsupportedPlatform(candidates.platform)}
        </p>
      )}

      {enabled && candidates?.supported && (
        <>
          <ListRow
            action={
              <Select
                disabled={busy}
                onValueChange={value =>
                  void writeBrowserSettings(
                    // Switching browsers clears the profile pin: a pin names a
                    // directory inside ONE browser's user-data dir, so carrying
                    // it across would fail closed against the new browser.
                    { real_profile_browser: value === AUTO ? '' : value, real_profile_pin: '' }
                  )
                }
                value={selectedBrowser || AUTO}
              >
                <SelectTrigger className={cn('min-w-56', CONTROL_TEXT)}>
                  <SelectValue />
                </SelectTrigger>

                <SelectContent>
                  <SelectItem value={AUTO}>
                    {candidates.detected_default
                      ? pickerCopy.systemDefaultNamed(
                          candidates.browsers.find(row => row.key === candidates.detected_default)?.label ??
                            candidates.detected_default
                        )
                      : pickerCopy.systemDefault}
                  </SelectItem>

                  {candidates.browsers.map(row => (
                    // A browser can't be driven unless BOTH halves exist: a
                    // profile to snapshot and a binary to run the copy on. Show
                    // the row disabled with the reason — the absence is
                    // information, not something to hide.
                    <SelectItem disabled={!row.has_profile || !row.installed} key={row.key} value={row.key}>
                      {row.label}
                      {!row.installed && ` — ${pickerCopy.notInstalled}`}
                      {row.installed && !row.has_profile && ` — ${pickerCopy.noProfile}`}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            }
            description={pickerCopy.browserDescription}
            title={pickerCopy.browserLabel}
          />

          <ListRow
            action={
              <Select
                disabled={busy || !activeBrowser?.profiles.length}
                onValueChange={value =>
                  void writeBrowserSettings(
                    { real_profile_pin: value === AUTO ? '' : value },
                    resolvedLabel
                      ? undefined
                      : { title: pickerCopy.savedTitle, message: pickerCopy.savedMessage(value) }
                  )
                }
                value={selectedPin || AUTO}
              >
                <SelectTrigger className={cn('min-w-56', CONTROL_TEXT)}>
                  <SelectValue />
                </SelectTrigger>

                <SelectContent>
                  <SelectItem value={AUTO}>
                    {(() => {
                      const lastUsed = activeBrowser?.profiles.find(row => row.last_used)

                      return lastUsed ? pickerCopy.lastUsedNamed(lastUsed.name) : pickerCopy.lastUsed
                    })()}
                  </SelectItem>

                  {activeBrowser?.profiles.map(row => (
                    <SelectItem key={row.directory} value={row.directory}>
                      {row.name}
                      {row.name !== row.directory && ` (${row.directory})`}
                    </SelectItem>
                  ))}

                  {/* Keep a stale pin selectable so the trigger shows the truth. */}
                  {pinIsMissing && <SelectItem value={selectedPin}>{selectedPin}</SelectItem>}
                </SelectContent>
              </Select>
            }
            description={pickerCopy.profileDescription}
            title={pickerCopy.profileLabel}
          />

          {resolvedLabel && <p className="px-1 pb-1 text-[0.68rem] text-muted-foreground">{resolvedLabel}</p>}

          {candidates.error && (
            <p className="flex items-start gap-1 px-1 pb-2 text-[0.68rem] text-amber-600 dark:text-amber-300">
              <AlertTriangle className="mt-0.5 size-3 shrink-0" />
              {candidates.error}
            </p>
          )}
        </>
      )}
    </div>
  )
}
