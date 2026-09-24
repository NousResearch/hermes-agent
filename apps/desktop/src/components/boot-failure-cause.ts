/**
 * Plain-language cause for a LOCAL backend boot failure. The main process
 * reports what it saw ("exited before it became ready (1)", "Timed out
 * connecting to Hermes backend after 45000ms", an EACCES, a Python traceback
 * tail…). None of that tells a non-developer what went wrong, so the overlay
 * leads with one classified sentence and keeps the raw text behind
 * "Show recent logs". Pure, table-driven, unit-tested without React.
 *
 * SSH and remote-reauth failures have their own classifiers
 * (`sshFailureMessage`, `isRemoteReauthFailure`) and are not handled here.
 */

export type LocalBootCause = 'diskFull' | 'exitedEarly' | 'installMissing' | 'permission' | 'portInUse' | 'timedOut'

// Order matters: the more specific causes (disk, permission, port, missing
// install) are checked before the generic exit/timeout shapes that usually
// accompany them in the same output.
const CAUSE_PATTERNS: readonly [LocalBootCause, RegExp][] = [
  ['diskFull', /no space left on device|database or disk is full|\bENOSPC\b|disk full/i],
  ['permission', /permission denied|\bEACCES\b|\bEPERM\b|read-only file system|\bEROFS\b|operation not permitted/i],
  ['portInUse', /address already in use|\bEADDRINUSE\b|port .* (?:is )?(?:already )?in use/i],
  [
    'installMissing',
    /installation is missing|is missing or incomplete|venv missing|no module named|modulenotfounderror/i
  ],
  ['timedOut', /timed out|timeout/i],
  ['exitedEarly', /exited before|exited \(|process exited|exited with|traceback \(most recent call last\)/i]
]

export function classifyLocalBootFailure(error: string | null | undefined): LocalBootCause | null {
  const text = String(error || '')

  if (!text) {
    return null
  }

  return CAUSE_PATTERNS.find(([, pattern]) => pattern.test(text))?.[0] ?? null
}

export interface LocalBootFailureCopy {
  /** The one classified sentence shown in the red box. */
  headline: string
  /** Raw error text worth keeping for the collapsed details — null when the
   *  raw text is short enough that the headline already carries it. */
  rawDetail: string | null
}

/** Copy for the overlay's red box: a classified cause when one is known,
 *  otherwise the raw error's FIRST line (never a traceback dump). */
export function localBootFailureCopy(
  error: string | null | undefined,
  causes: Record<LocalBootCause, string>
): LocalBootFailureCopy {
  const raw = String(error || '').trim()
  const cause = classifyLocalBootFailure(raw)

  if (cause) {
    return { headline: causes[cause], rawDetail: raw || null }
  }

  const firstLine = raw.split('\n')[0]?.trim() ?? ''

  return { headline: firstLine, rawDetail: firstLine === raw ? null : raw }
}
