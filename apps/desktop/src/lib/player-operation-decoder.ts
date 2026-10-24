import type { PlayerAgentOperation, PlayerControlAction, PlayerDownloadQuality } from './player-context-contract'

type Wire = Record<string, unknown>
type Decoder = (wire: Wire) => PlayerAgentOperation | null
const isRecord = (value: unknown): value is Wire => typeof value === 'object' && value !== null && !Array.isArray(value)
const exactKeys = (wire: Wire, allowed: string[]) => Object.keys(wire).every(key => allowed.includes(key))

const token = (value: unknown): value is string =>
  typeof value === 'string' && value.length >= 24 && value.length <= 256

const boundedNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 86_400

function control(wire: Wire): PlayerAgentOperation | null {
  const actions: PlayerControlAction[] = [
    'play',
    'pause',
    'mute',
    'unmute',
    'seek',
    'pin',
    'unpin',
    'fullscreen',
    'exit-fullscreen'
  ]

  if (
    !exactKeys(wire, ['type', 'context_token', 'navigation_generation', 'action', 'position_seconds']) ||
    !token(wire.context_token)
  )
    {return null}

  if (!Number.isSafeInteger(wire.navigation_generation) || Number(wire.navigation_generation) < 0) {return null}

  if (typeof wire.action !== 'string' || !actions.includes(wire.action as PlayerControlAction)) {return null}

  if (wire.action === 'seek' ? !boundedNumber(wire.position_seconds) : wire.position_seconds !== undefined) {return null}

  return {
    type: 'control',
    contextToken: wire.context_token,
    navigationGeneration: Number(wire.navigation_generation),
    action: wire.action as PlayerControlAction,
    ...(wire.position_seconds === undefined ? {} : { positionSeconds: wire.position_seconds as number })
  }
}

function validRange(value: unknown): value is Wire {
  if (!isRecord(value) || !Object.keys(value).length || !exactKeys(value, ['start_seconds', 'end_seconds']))
    {return false}

  if (value.start_seconds !== undefined && !boundedNumber(value.start_seconds)) {return false}

  if (value.end_seconds !== undefined && !boundedNumber(value.end_seconds)) {return false}

  return (
    value.start_seconds === undefined ||
    value.end_seconds === undefined ||
    Number(value.end_seconds) >= Number(value.start_seconds)
  )
}

function transcript(wire: Wire): PlayerAgentOperation | null {
  if (!exactKeys(wire, ['type', 'context_token', 'language', 'range']) || !token(wire.context_token)) {return null}

  if (
    wire.language !== undefined &&
    (typeof wire.language !== 'string' ||
      wire.language.length > 35 ||
      !/^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$/.test(wire.language))
  )
    {return null}

  if (wire.range !== undefined && !validRange(wire.range)) {return null}

  return {
    type: 'transcript.start',
    contextToken: wire.context_token,
    ...(typeof wire.language === 'string' ? { language: wire.language } : {}),
    ...(isRecord(wire.range)
      ? {
          range: {
            ...(typeof wire.range.start_seconds === 'number' ? { startSeconds: wire.range.start_seconds } : {}),
            ...(typeof wire.range.end_seconds === 'number' ? { endSeconds: wire.range.end_seconds } : {})
          }
        }
      : {})
  }
}

function download(wire: Wire): PlayerAgentOperation | null {
  if (!exactKeys(wire, ['type', 'context_token', 'quality']) || !token(wire.context_token)) {return null}

  if (
    wire.quality !== undefined &&
    (typeof wire.quality !== 'string' || !['best', '720p', '1080p', 'audio'].includes(wire.quality))
  )
    {return null}

  return {
    type: 'download.start',
    contextToken: wire.context_token,
    ...(typeof wire.quality === 'string' ? { quality: wire.quality as PlayerDownloadQuality } : {})
  }
}

function job(wire: Wire): PlayerAgentOperation | null {
  if (
    !exactKeys(wire, ['type', 'job_id']) ||
    typeof wire.job_id !== 'string' ||
    !/^[A-Za-z0-9_-]{1,128}$/.test(wire.job_id)
  )
    {return null}

  return { type: wire.type as 'job.status' | 'job.cancel', jobId: wire.job_id }
}

const decoders: Record<string, Decoder> = {
  context: wire => (exactKeys(wire, ['type']) ? { type: 'context' } : null),
  'jobs.list': wire => (exactKeys(wire, ['type']) ? { type: 'jobs.list' } : null),
  control,
  'transcript.start': transcript,
  'download.start': download,
  'job.status': job,
  'job.cancel': job
}

/** Decode only the closed gateway wire schema, with no route, URL or filesystem authority. */
export function decodePlayerAgentOperation(value: unknown): PlayerAgentOperation | null {
  if (!isRecord(value) || typeof value.type !== 'string' || !Object.hasOwn(decoders, value.type)) {return null}

  return decoders[value.type](value)
}
