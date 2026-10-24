import * as electronApi from 'electron'

const READY_TIMEOUT_MS = 60_000
const COMPONENT_ID_RE = /^[A-Za-z0-9._+:-]{1,120}$/

interface RuntimeComponents {
  WIDEVINE_CDM_ID?: string
  whenReady?: (ids: string[]) => Promise<unknown>
  status?: () => unknown
}

const componentsApi = (electronApi as unknown as { components?: RuntimeComponents }).components

function hasReadyResult(value: unknown, id: string): boolean {
  return (
    Array.isArray(value) &&
    value.some(row => {
      if (!row || typeof row !== 'object') {
        return false
      }

      const entry = row as { id?: unknown; status?: unknown }

      return entry.id === id && typeof entry.status === 'string' && COMPONENT_ID_RE.test(entry.status)
    })
  )
}

function hasInstalledStatus(value: unknown, id: string): boolean {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }

  const row = (value as Record<string, unknown>)[id]

  return Boolean(
    row &&
    typeof row === 'object' &&
    typeof (row as { status?: unknown }).status === 'string' &&
    COMPONENT_ID_RE.test((row as { status: string }).status)
  )
}

function safeError(error: unknown): string {
  const code = error && typeof error === 'object' ? (error as { code?: unknown }).code : undefined

  return typeof code === 'string' && /^[A-Za-z0-9._-]{1,48}$/.test(code) ? code : 'component-readiness-failed'
}

/** Read the runtime's actual CDM component API; Electron's version string is not evidence. */
export async function resolveWidevineReadiness(
  api: RuntimeComponents | undefined = componentsApi,
  timeoutMs = READY_TIMEOUT_MS,
  platform: NodeJS.Platform = process.platform
): Promise<{ state: 'ready' | 'restart-required' | 'unsupported'; detail?: string }> {
  const id = api?.WIDEVINE_CDM_ID

  if (!id || !api?.whenReady || !api.status) {
    return { state: 'unsupported', detail: 'Widevine component API unavailable in this Electron runtime' }
  }

  let timer: ReturnType<typeof setTimeout> | undefined

  try {
    const results = await Promise.race([
      api.whenReady([id]),
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(
          () => reject(Object.assign(new Error('timeout'), { code: 'component-readiness-timeout' })),
          timeoutMs
        )
      })
    ])

    if (!hasReadyResult(results, id)) {
      return { state: 'unsupported', detail: 'Widevine readiness result did not include the requested component' }
    }

    const status = api.status()

    if (!hasInstalledStatus(status, id)) {
      return { state: 'unsupported', detail: 'Widevine component status is unavailable' }
    }

    const statusRow = (status as Record<string, { status: string }>)[id]

    if (platform === 'linux' && statusRow.status === 'updated') {
      return {
        state: 'restart-required',
        detail: 'Widevine was installed or updated; restart Hermes before protected playback'
      }
    }

    return { state: 'ready' }
  } catch (error) {
    return { state: 'unsupported', detail: safeError(error) }
  } finally {
    if (timer) {
      clearTimeout(timer)
    }
  }
}
