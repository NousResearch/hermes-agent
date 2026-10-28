// Bundled as a separate worklet asset; these globals belong to AudioWorkletGlobalScope.
declare abstract class AudioWorkletProcessor {
  readonly port: MessagePort
}
declare const currentFrame: number
declare const sampleRate: number
declare function registerProcessor(name: string, processor: typeof AudioWorkletProcessor): void

const WINDOW_SIZE = 256
const REPORT_HZ = 60

class VoiceLevelProcessor extends AudioWorkletProcessor {
  private readonly samples = new Float32Array(WINDOW_SIZE)
  private cursor = 0
  private received = 0
  private nextReport = sampleRate / REPORT_HZ

  private report(audioTime: number): void {
    let sum = 0

    for (const sample of this.samples) {
      // Preserve AnalyserNode.getByteTimeDomainData's scale and existing VAD thresholds.
      const centered = Math.max(0, Math.min(255, Math.floor(128 * (sample + 1)))) - 128
      sum += centered * centered
    }

    this.port.postMessage({ level: Math.min(1, Math.sqrt(sum / WINDOW_SIZE) / 42), audioTime })
  }

  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const input = inputs[0]?.[0]
    const frameCount = outputs[0][0].length

    for (let index = 0; index < frameCount; index++) {
      this.samples[this.cursor] = input?.[index] ?? 0
      this.cursor = (this.cursor + 1) % WINDOW_SIZE
      this.received++

      if (this.received >= this.nextReport) {
        this.report((currentFrame + index + 1) / sampleRate)
        this.nextReport += sampleRate / REPORT_HZ
      }
    }

    // The output stays silent; the microphone is never routed to the speakers.
    return true
  }
}

registerProcessor('hermes-voice-level', VoiceLevelProcessor)

export {}
