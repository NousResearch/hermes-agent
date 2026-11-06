/** Narrow renderer contract for Hermes' native service player surface. */

import type {
  PlayerAgentRequest,
  PlayerAgentResponse,
  PlayerAgentScope,
  PlayerAwarenessLinkState,
  PlayerMediaControlAction,
  PlayerMediaJobCapabilities
} from './player-context-contract'
import type { PlayerSource } from './player-services'
export type {
  PlayerAgentRequest,
  PlayerAgentResponse,
  PlayerAgentScope,
  PlayerAwarenessLinkState,
  PlayerContextMetadata,
  PlayerContextSnapshot,
  PlayerMediaJob,
  PlayerMediaJobCapabilities,
  PlayerMediaJobKind,
  PlayerMediaJobStartOptions,
  PlayerMediaRange,
  PlayerMediaTarget,
  PlayerUntrustedPageText
} from './player-context-contract'
export type { PlayerSource } from './player-services'

export type PlayerCommand =
  | { type: 'open'; source: PlayerSource }
  | { type: 'status' }
  | { type: 'dock' }
  | { type: 'popout' }
  | { type: 'home' }
  | { type: 'back' }
  | { type: 'forward' }
  | { type: 'media-control'; action: PlayerMediaControlAction; positionSeconds?: number }
  | { type: 'pin'; pinned: boolean }
  | { type: 'fullscreen'; fullscreen: boolean }
  | { type: 'close' }
  /** Host adapter teardown; not exposed as a plugin controller command. */
  | { type: 'release' }
  | {
      type: 'bounds'
      bounds: { x: number; y: number; width: number; height: number }
      visible: boolean
      revision: number
    }

export interface PlayerSnapshot {
  status: 'closed' | 'loading' | 'ready' | 'error'
  source: PlayerSource | null
  presentation: 'docked' | 'floating'
  pinned: boolean
  pinStatus: 'requested' | 'observed' | 'unsupported' | 'unknown'
  pinMode: 'topmost' | 'all-workspaces' | 'unsupported'
  fullscreen: boolean
  currentTime: number | null
  duration: number | null
  paused: boolean | null
  muted: boolean | null
  canGoBack: boolean
  canGoForward: boolean
  awarenessLinked: boolean
  safeOrigin: string | null
  guestId: number | null
  runtime: { state: 'not-required' | 'starting' | 'ready' | 'restart-required' | 'unsupported'; detail?: string }
  error: string | null
}

export type PlayerListener = (snapshot: PlayerSnapshot) => void

export interface PlayerBridge {
  request(command: PlayerCommand): Promise<PlayerSnapshot>
  onState(listener: PlayerListener): () => void
  linkAwareness(scope: PlayerAgentScope): Promise<PlayerAwarenessLinkState>
  unlinkAwareness(): Promise<PlayerAwarenessLinkState>
  agentRequest(request: PlayerAgentRequest): Promise<PlayerAgentResponse>
  setupMediaHelpers(): Promise<PlayerMediaJobCapabilities>
  revealMediaJob(scope: PlayerAgentScope, jobId: string): Promise<boolean>
  clearLocalData(): Promise<PlayerSnapshot>
}
