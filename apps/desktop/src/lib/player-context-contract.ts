import type { PlayerSource } from './player-services'

/**
 * Session scope originates in the gateway's server request, never model
 * arguments. Electron additionally binds every link and token to one owner
 * BrowserWindow before accepting a request.
 */
export interface PlayerAgentScope {
  connectionId: string
  profileId: string
  sessionId: string
}

export const PLAYER_CONTEXT_TOKEN_TTL_MS = 5 * 60 * 1000

export type PlayerAdStatus = 'ad' | 'content' | 'unknown'

/** Text read from a provider page is untrusted input, not an instruction. */
export interface PlayerUntrustedPageText {
  text: string
  trust: 'untrusted-page'
}

export interface PlayerContextMetadata {
  identity: 'youtube-watch' | 'unknown'
  videoId: string | null
  /** Reconstructed from the validated YouTube watch ID; never copied from a page URL. */
  canonicalUrl: string | null
  title: PlayerUntrustedPageText | null
  channel: PlayerUntrustedPageText | null
  positionSeconds: number | null
  durationSeconds: number | null
  paused: boolean | null
  muted: boolean | null
  adStatus: PlayerAdStatus
}

export interface PlayerContextSnapshot {
  source: PlayerSource
  /** Exact validated service origin; auth hosts and raw URLs are never exposed. */
  safeOrigin: string | null
  navigationGeneration: number
  contextToken: string
  capturedAt: string
  expiresAt: string
  metadata: PlayerContextMetadata
}

export type PlayerControlAction =
  'play' | 'pause' | 'mute' | 'unmute' | 'seek' | 'pin' | 'unpin' | 'fullscreen' | 'exit-fullscreen'

export type PlayerMediaControlAction = 'play' | 'pause' | 'mute' | 'unmute' | 'seek'

export interface PlayerMediaRange {
  startSeconds?: number
  endSeconds?: number
}

export type PlayerDownloadQuality = 'best' | '720p' | '1080p' | 'audio'

export type PlayerAgentOperation =
  | { type: 'context' }
  | {
      type: 'control'
      contextToken: string
      navigationGeneration: number
      action: PlayerControlAction
      positionSeconds?: number
    }
  | { type: 'transcript.start'; contextToken: string; language?: string; range?: PlayerMediaRange }
  | { type: 'download.start'; contextToken: string; quality?: PlayerDownloadQuality }
  | { type: 'jobs.list' }
  | { type: 'job.status'; jobId: string }
  | { type: 'job.cancel'; jobId: string }

export interface PlayerAgentRequest {
  scope: PlayerAgentScope
  operation: PlayerAgentOperation
}

/** Sent only by an explicitly activated, trusted Hermes renderer. */
export interface PlayerAwarenessLinkState {
  linked: boolean
  /** Native-generated opaque link identifier; never accepted as an input. */
  linkId: string | null
  scope: PlayerAgentScope | null
}

export interface PlayerMediaTarget {
  source: 'youtube'
  videoId: string
  canonicalUrl: string
  title: PlayerUntrustedPageText | null
  capturedTimeSeconds: number | null
  adStatus: PlayerAdStatus
}

export type PlayerMediaJobKind = 'transcript' | 'download'
export type PlayerMediaJobState =
  'queued' | 'running' | 'cancelling' | 'completed' | 'failed' | 'cancelled' | 'interrupted'

export interface PlayerTranscriptSegment {
  startSeconds: number
  endSeconds: number | null
  text: PlayerUntrustedPageText
}

export interface PlayerMediaJob {
  id: string
  kind: PlayerMediaJobKind
  state: PlayerMediaJobState
  progress: number | null
  createdAt: string
  updatedAt: string
  source: PlayerMediaTarget
  error: { code: string; message: string } | null
  transcript: {
    language: string
    trackKind: 'manual' | 'automatic' | 'unknown'
    retrievedAt: string
    segments: PlayerTranscriptSegment[]
  } | null
  artifact: {
    fileName: string
    sizeBytes: number
    format: string
  } | null
}

export interface PlayerMediaJobCapabilities {
  available: boolean
  reason: string | null
}

export interface PlayerMediaJobStartOptions {
  kind: PlayerMediaJobKind
  language?: string
  range?: PlayerMediaRange
  quality?: PlayerDownloadQuality
}

export interface PlayerAgentResponse {
  operation: PlayerAgentOperation['type']
  context?: PlayerContextSnapshot
  control?: {
    applied: boolean
    action: PlayerControlAction
    positionSeconds: number | null
    paused: boolean | null
    muted: boolean | null
    pinned?: boolean
    fullscreen?: boolean
  }
  job?: PlayerMediaJob
  jobs?: PlayerMediaJob[]
  capabilities?: PlayerMediaJobCapabilities
  error?: { code: string; message: string }
}
