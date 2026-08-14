// Production renderer diagnostics: a ring-buffered hitch capture that ships in
// the packaged app and costs nothing until main arms it. See `capture.ts` for
// the cost/sanitization contract and `arming-bridge.ts` for the IPC surface.

export { type DiagnosticsArmPayload, initDiagnosticsArming, initDiagnosticsSnapshots } from './arming-bridge'
export {
  activeCaptureId,
  armDiagnostics,
  type DiagnosticsCaptureSnapshot,
  type DiagnosticsEvent,
  disarmDiagnostics,
  type GatewayEventAppliedEvent,
  isDiagnosticsArmed,
  type LongFrameEvent,
  type MemorySampleEvent,
  readDiagnosticsCapture,
  recordGatewayEventApplied,
  recordStreamDeltaApplied,
  type StreamDeltaAppliedEvent
} from './capture'
export {
  createLongFrameObserver,
  type LongFrameSample,
  type LongFrameScript,
  supportsLongAnimationFrames,
  toLongFrameSample
} from './long-frames'
