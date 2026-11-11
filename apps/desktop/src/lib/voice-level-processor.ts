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

  private report(frame: number): void {
    let sum = 0

    for (const sample of this.samples) {
      // Preserve AnalyserNode.getByteTimeDomainData's scale and existing VAD thresholds.
      const centered = Math.max(0, Math.min(255, Math.floor(128 * (sample + 1)))) - 128
      sum += centered * centered
    }

    this.port.postMessage({
      level: Math.min(1, Math.sqrt(sum / WINDOW_SIZE) / 42),
      audioTime: frame / sampleRate,
      // Include frame rounding so freshness uses the next actual report, not a latency allowance.
      nextReportTime: (frame + Math.ceil(this.nextReport) - this.received) / sampleRate
    })
  }

  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const input = inputs[0]?.[0]
    const frameCount = outputs[0][0].length

    for (let index = 0; index < frameCount; index++) {
      this.samples[this.cursor] = input?.[index] ?? 0
      this.cursor = (this.cursor + 1) % WINDOW_SIZE
      this.received++

      if (this.received >= this.nextReport) {
        this.nextReport += sampleRate / REPORT_HZ
        this.report(currentFrame + index + 1)
      }
    }

    // The output stays silent; the microphone is never routed to the speakers.
    return true
  }
}

registerProcessor('hermes-voice-level', VoiceLevelProcessor)

export {}
