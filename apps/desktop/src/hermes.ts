export function transcribeAudio(dataUrl: string, mimeType?: string): Promise<AudioTranscriptionResponse> {
  return window.hermesDesktop.api<AudioTranscriptionResponse>({
    path: '/api/audio/transcribe',
    method: 'POST',
    timeoutMs: 120_000,
    body: {
      data_url: dataUrl,
      mime_type: mimeType
    }
  })
}