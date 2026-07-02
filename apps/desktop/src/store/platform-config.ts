/**
 * Platform client-config sync — renderer apply side.
 *
 * The main process caches the platform-served versioned client config at
 * userData/apex-client-config.json (fetched fail-soft at boot and after a
 * successful managed sign-in — see electron/apex-client-config.cjs and the
 * main.cjs wiring). Once the gateway is open, this module:
 *
 *   1. reads the cached state over IPC (hermesDesktop.clientConfig.get — a
 *      local disk read, no network),
 *   2. when `version > appliedVersion` and payload.config_yaml is a non-empty
 *      object, applies each dotted key through the SAME global-config write
 *      path the settings pages use (getHermesConfigRecord → setNested →
 *      saveHermesConfig against the runtime's /api/config dashboard API),
 *   3. records the version back via hermesDesktop.clientConfig.markApplied —
 *      but ONLY when every key applied, so a partial failure retries on the
 *      next gateway-open / boot.
 *
 * Strictly best-effort: a missing bridge (web build / older main process), an
 * unreachable backend, or a malformed payload all no-op. Contract v1 payload:
 * { config_yaml: { "<dotted.key>": <scalar>, … } }; unknown payload fields and
 * non-scalar values are ignored (forward compat), never treated as failures.
 */

import { setNested } from '@/app/settings/helpers'
import type { DesktopClientConfigState } from '@/global'
import { getHermesConfigRecord, saveHermesConfig } from '@/hermes'
import type { HermesConfigRecord } from '@/types/hermes'

/** Scalar values contract v1 allows as dotted-key config values. */
export type PlatformConfigScalar = boolean | null | number | string

export type PlatformConfigEntry = [key: string, value: PlatformConfigScalar]

export interface PlatformConfigApplyResult {
  status: 'applied' | 'failed' | 'no-bridge' | 'nothing-to-apply' | 'partial' | 'up-to-date'
  version: number
  appliedKeys: string[]
  failedKeys: string[]
}

/**
 * Mirror of electron/apex-client-config.cjs::shouldApply (the renderer cannot
 * require that cjs module; keep the two in lockstep). Apply only when the
 * fetched version is a positive integer strictly greater than what was already
 * applied — a same-version re-read and a version regression both no-op.
 */
export function shouldApplyClientConfig(version: unknown, appliedVersion: unknown): boolean {
  const fetched = normalizeVersion(version)

  return fetched > 0 && fetched > normalizeVersion(appliedVersion)
}

function normalizeVersion(value: unknown): number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0 ? value : 0
}

function isPlatformConfigScalar(value: unknown): value is PlatformConfigScalar {
  return (
    value === null || typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean'
  )
}

/**
 * Extract the applicable dotted-key entries from a cached payload. Contract v1
 * carries them under `config_yaml`. Forward-compat rules: a missing/malformed
 * config_yaml yields no entries, and non-scalar values (some future contract's
 * nested shapes) are DROPPED rather than failed — they must never wedge the
 * apply loop into retrying forever.
 */
export function configYamlEntries(payload: Record<string, unknown> | null | undefined): PlatformConfigEntry[] {
  if (!payload || typeof payload !== 'object') {
    return []
  }

  const configYaml = (payload as { config_yaml?: unknown }).config_yaml

  if (!configYaml || typeof configYaml !== 'object' || Array.isArray(configYaml)) {
    return []
  }

  const entries: PlatformConfigEntry[] = []

  for (const [key, value] of Object.entries(configYaml)) {
    if (key.trim() && isPlatformConfigScalar(value)) {
      entries.push([key, value])
    }
  }

  return entries
}

// One apply pass at a time: gatewayState can flap (reconnects) and each 'open'
// transition calls applyPlatformConfig again; the persisted appliedVersion
// makes repeat passes no-op, and this guard covers the in-flight window.
let applyInFlight = false

/**
 * Apply the cached platform config to the runtime's global config, if a newer
 * version is pending. Never throws. Call on gateway-open — it self-guards
 * against concurrent and repeat invocations.
 */
export async function applyPlatformConfig(): Promise<PlatformConfigApplyResult> {
  if (applyInFlight) {
    return { status: 'up-to-date', version: 0, appliedKeys: [], failedKeys: [] }
  }

  applyInFlight = true

  try {
    return await applyPlatformConfigOnce()
  } catch {
    // Belt-and-suspenders: nothing inside should throw, but platform sync must
    // never take the boot flow down with it.
    return { status: 'failed', version: 0, appliedKeys: [], failedKeys: [] }
  } finally {
    applyInFlight = false
  }
}

async function applyPlatformConfigOnce(): Promise<PlatformConfigApplyResult> {
  const noResult = (status: PlatformConfigApplyResult['status'], version = 0): PlatformConfigApplyResult => ({
    status,
    version,
    appliedKeys: [],
    failedKeys: []
  })

  const bridge = window.hermesDesktop?.clientConfig

  if (!bridge?.get) {
    return noResult('no-bridge')
  }

  let state: DesktopClientConfigState

  try {
    state = await bridge.get()
  } catch {
    return noResult('failed')
  }

  if (!state || !shouldApplyClientConfig(state.version, state.appliedVersion)) {
    return noResult('up-to-date', state?.version ?? 0)
  }

  const entries = configYamlEntries(state.payload)

  if (!entries.length) {
    // Nothing this client knows how to apply (empty config_yaml or a payload
    // of future-only fields). Vacuously complete — record the version so we
    // don't rescan it on every boot.
    await markApplied(bridge, state.version)

    return noResult('nothing-to-apply', state.version)
  }

  let config: HermesConfigRecord

  try {
    config = await getHermesConfigRecord()
  } catch (error) {
    console.warn('[platform-config] could not read runtime config; will retry next boot', error)

    return noResult('failed', state.version)
  }

  // Fold every key into one record and save once — the exact write path the
  // settings page uses (setNested + saveHermesConfig), batched. A single bad
  // key (e.g. an unsafe path part) is logged and skipped; the rest still land.
  const appliedKeys: string[] = []
  const failedKeys: string[] = []
  let next = config

  for (const [key, value] of entries) {
    try {
      next = setNested(next, key, value)
      appliedKeys.push(key)
    } catch (error) {
      failedKeys.push(key)
      console.warn(`[platform-config] skipped config key "${key}":`, error)
    }
  }

  if (appliedKeys.length) {
    try {
      await saveHermesConfig(next)
    } catch (error) {
      console.warn('[platform-config] saving runtime config failed; will retry next boot', error)

      return { status: 'failed', version: state.version, appliedKeys: [], failedKeys: entries.map(([k]) => k) }
    }

    for (const key of appliedKeys) {
      console.info(`[platform-config] applied ${key} (v${state.version})`)
    }
  }

  if (failedKeys.length) {
    console.warn(
      `[platform-config] v${state.version} partially applied (${failedKeys.length}/${entries.length} keys failed); will retry next boot`
    )

    return { status: 'partial', version: state.version, appliedKeys, failedKeys }
  }

  await markApplied(bridge, state.version)

  return { status: 'applied', version: state.version, appliedKeys, failedKeys }
}

async function markApplied(
  bridge: NonNullable<Window['hermesDesktop']['clientConfig']>,
  version: number
): Promise<void> {
  if (!bridge.markApplied) {
    return
  }

  try {
    await bridge.markApplied(version)
  } catch (error) {
    // Non-fatal: the worst case is a redundant (idempotent) re-apply next boot.
    console.warn('[platform-config] markApplied failed:', error)
  }
}
