// What Hermes is saying and when the listener hears it, for plugins that follow speech.
// Times are `performance.now()` ms with the output device's delay already added.
import { atom } from 'nanostores'

import { getElementAnalyser } from '@/lib/element-analyser'
import { $voice } from '@/store/voice-playback'

export interface SpokenPiece {
  endMs: number
  /** `sentence` when the voice reports no finer timing. */
  kind: 'fragment' | 'sentence' | 'word'
  startMs: number
  text: string
}

export interface SpokenSegment {
  endMs: number
  pieces: readonly SpokenPiece[]
  startMs: number
  text: string
}

export interface VoiceTimeline {
  segments: readonly SpokenSegment[]
}

export const $voiceTimeline = atom<VoiceTimeline>({ segments: [] })

// Ended segments stay readable briefly so a plugin sampling a little behind still finds them.
const KEEP_ENDED_MS = 5_000

/** Add a segment heard from `startMs`. `units` are timed relative to it; without any, the whole
 *  text is one sentence piece that `extendVoiceSentence` grows as its audio is scheduled. */
export function publishVoiceSegment(text: string, units: readonly SpokenPiece[], startMs: number) {
  const timed = units.length ? units : [{ endMs: 0, kind: 'sentence' as const, startMs: 0, text }]
  const pieces = timed.map(unit => ({ ...unit, endMs: startMs + unit.endMs, startMs: startMs + unit.startMs }))
  const kept = $voiceTimeline.get().segments.filter(segment => segment.endMs > performance.now() - KEEP_ENDED_MS)

  $voiceTimeline.set({ segments: [...kept, { endMs: Math.max(...pieces.map(p => p.endMs)), pieces, startMs, text }] })
}

/** The newest segment's audio is now scheduled until `endMs`. Timed segments already know their end. */
export function extendVoiceSentence(endMs: number) {
  const segments = $voiceTimeline.get().segments
  const last = segments.at(-1)

  if (last?.pieces[0]?.kind === 'sentence') {
    const pieces = [{ ...last.pieces[0], endMs }]

    $voiceTimeline.set({ segments: [...segments.slice(0, -1), { ...last, endMs, pieces }] })
  }
}

// Speech cut short: clip what was being heard and drop what was still to come.
$voice.listen(({ endReason }) => {
  if (endReason !== 'stopped') {
    return
  }

  const now = performance.now()

  const clip = <T extends { endMs: number; startMs: number }>(items: readonly T[]) =>
    items.filter(item => item.startMs < now).map(item => ({ ...item, endMs: Math.min(item.endMs, now) }))

  const segments = clip($voiceTimeline.get().segments).map(segment => ({ ...segment, pieces: clip(segment.pieces) }))

  $voiceTimeline.set({ segments })
})

let levelSource: AnalyserNode | HTMLAudioElement | null = null
let outputLatencyMs = 0

/** Tell the level meter what is playing now. */
export function setVoiceLevelSource(source: AnalyserNode | HTMLAudioElement) {
  levelSource = source
}

/** RMS loudness 0..1 of the voice now playing; 0 when nothing is. */
export function readVoiceLevel(): number {
  if (!levelSource || $voice.get().status !== 'speaking') {
    return 0
  }

  // An <audio> element is routed through an analyser only once something asks for its level.
  const analyser = 'fftSize' in levelSource ? levelSource : getElementAnalyser(levelSource)?.analyser

  if (!analyser) {
    return 0
  }

  const samples = new Float32Array(analyser.fftSize)
  analyser.getFloatTimeDomainData(samples)

  // Scaled so that loud speech reaches ~1.
  return Math.min(1, Math.sqrt(samples.reduce((sum, sample) => sum + sample * sample, 0) / samples.length) * 4)
}

export const getVoiceOutputLatencyMs = () => outputLatencyMs

/** The context's output delay in ms, remembered for plugins. */
export function measureOutputLatencyMs(context: AudioContext): number {
  outputLatencyMs = ((context.baseLatency || 0) + (context.outputLatency || 0)) * 1000

  return outputLatencyMs
}

/** Page time at which the listener hears context time `contextTime`. */
export function listenerMsForContextTime(context: AudioContext, contextTime: number): number {
  const latencyMs = measureOutputLatencyMs(context)
  const stamp = context.getOutputTimestamp?.()

  // The output timestamp already includes device delay (Bluetooth too); it is empty before output starts.
  return stamp?.performanceTime && stamp.contextTime !== undefined
    ? stamp.performanceTime + (contextTime - stamp.contextTime) * 1000
    : performance.now() + (contextTime - context.currentTime) * 1000 + latencyMs
}
