import { atom } from 'nanostores'

import { translateNow } from '@/i18n'
import { notify } from '@/store/notifications'

// ── Per-profile remote overrides (connection.json `profiles.<name>`) ────────
// The Electron main already routes a profile with a remote entry to its own
// pooled backend (profileRemoteOverride → resolveProfileBackendRoute case 2);
// until now the only way to WRITE that entry was hand-editing connection.json
// (#91349). This store is the renderer-side cache of which profiles carry an
// override, so the rail can badge them and the dialog can be opened from
// anywhere (context menu, auth-failure toast) without threading callbacks.
//
// Deliberately no import from store/profile: profile.ts imports THIS module
// for the auth-failure toast, so the dependency must stay one-way.

export interface ProfileRemoteOverride {
  /** hostname[:port] parsed from the override URL — the display label. */
  host: string
  url: string
}

/** profile key → its remote override, refreshed from Electron on demand. */
export const $profileRemoteOverrides = atom<Record<string, ProfileRemoteOverride>>({})

// The profile whose "Connect to a remote host" dialog is open, or null. An
// atom (not component state) so the rail's context menu AND the re-enter-token
// toast action can both open the same dialog.
export const $remoteOverrideDialogProfile = atom<null | string>(null)

export function openRemoteOverrideDialog(profile: string): void {
  $remoteOverrideDialogProfile.set(profile)
}

export function closeRemoteOverrideDialog(): void {
  $remoteOverrideDialogProfile.set(null)
}

/** hostname[:port] for a gateway URL, or '' when it does not parse. */
export function remoteHostLabel(url: string): string {
  try {
    const parsed = new URL(String(url || ''))

    if (!parsed.hostname) {
      return ''
    }

    return parsed.port && parsed.port !== '80' && parsed.port !== '443'
      ? `${parsed.hostname}:${parsed.port}`
      : parsed.hostname
  } catch {
    return ''
  }
}

/**
 * Re-pull each named profile's connection scope from Electron and publish the
 * ones that resolve to a remote/cloud override. Best-effort per profile: a
 * single failed read keeps that profile unbadged rather than failing the lot.
 */
export async function refreshProfileRemoteOverrides(names: string[]): Promise<void> {
  const getConnectionConfig = window.hermesDesktop?.getConnectionConfig

  if (!getConnectionConfig) {
    return
  }

  const next: Record<string, ProfileRemoteOverride> = {}

  await Promise.all(
    names.map(async name => {
      const key = String(name || '').trim()

      if (!key) {
        return
      }

      try {
        const config = await getConnectionConfig(key)

        if ((config.mode === 'remote' || config.mode === 'cloud') && config.remoteUrl) {
          next[key] = { host: remoteHostLabel(config.remoteUrl) || config.remoteUrl, url: config.remoteUrl }
        }
      } catch {
        // Backend bridge hiccup — leave this profile unbadged until the next refresh.
      }
    })
  )

  $profileRemoteOverrides.set(next)
}

// The token-rotation failure shape: the remote host answered but refused the
// credentials. Mirrors the auth-vocabulary used by suggestion-providers/repair
// — connectivity failures (timeout, refused, DNS) stay generic so a down host
// doesn't misread as a revoked token.
const AUTH_FAILURE_RE =
  /\b(401|403|unauthorized|forbidden)\b|invalid[_ ]?token|token .*(expired|invalid|rejected)|authenticat\w+ (failed|required|expired)/i

/**
 * When switching to a profile with a remote override fails because the host
 * rejected the saved token (rotated/revoked), surface a re-enter-token toast
 * whose action reopens the override dialog — never a silently dead profile.
 * Returns true when the toast was shown (callers keep their generic handling
 * for everything else).
 */
export function notifyRemoteOverrideAuthFailure(profile: string, error: unknown): boolean {
  const key = String(profile || '').trim()
  const override = key ? $profileRemoteOverrides.get()[key] : undefined

  if (!override) {
    return false
  }

  const message = error instanceof Error ? error.message : String(error ?? '')

  if (!AUTH_FAILURE_RE.test(message)) {
    return false
  }

  notify({
    kind: 'error',
    title: translateNow('profiles.remoteOverride.authFailedTitle'),
    message: translateNow('profiles.remoteOverride.authFailedMessage', key, override.host),
    action: {
      label: translateNow('profiles.remoteOverride.updateToken'),
      onClick: () => openRemoteOverrideDialog(key)
    }
  })

  return true
}
