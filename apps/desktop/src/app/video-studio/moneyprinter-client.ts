export type VideoAspect = '9:16' | '16:9' | '1:1'
export type VideoSource = 'pexels' | 'pixabay' | 'coverr' | 'local'
export type SubtitlePosition = 'bottom' | 'top' | 'center' | 'custom'
export type VideoConcatMode = 'random' | 'sequential'
export type VideoTransitionMode = '' | 'FadeIn' | 'FadeOut' | 'Shuffle' | 'SlideIn' | 'SlideOut'

export interface MoneyPrinterLocalMaterial {
  file: string
  kind: 'image' | 'video'
  name: string
  size: number
  updatedAt?: number
}

export interface MoneyPrinterLocalMaterialUploadInput {
  contentBase64?: string
  filename: string
  sourcePath?: string
}

export interface MoneyPrinterAudioAsset {
  file: string
  kind: 'audio'
  name: string
  size: number
  updatedAt?: number
}

export interface MoneyPrinterFontAsset {
  file: string
  name: string
  size: number
  updatedAt?: number
}

export interface MoneyPrinterAssets {
  bgms: MoneyPrinterAudioAsset[]
  customAudio: MoneyPrinterAudioAsset[]
  fonts: MoneyPrinterFontAsset[]
  voices: string[]
}

export interface VideoGenerationForm {
  bgmFile: string
  bgmType: string
  bgmVolume: number
  customAudioFile: string
  customPosition: number
  customSystemPrompt: string
  fontName: string
  fontSize: number
  localMaterials: string[]
  matchMaterialsToScript: boolean
  paragraphNumber: number
  roundedSubtitleBackground: boolean
  searchTermsAmount: number
  strokeColor: string
  strokeWidth: number
  subtitleEnabled: boolean
  subtitlePosition: SubtitlePosition
  textBackgroundColor: string
  textForeColor: string
  videoAspect: VideoAspect
  videoClipDuration: number
  videoConcatMode: VideoConcatMode
  videoCount: number
  videoLanguage: string
  videoScript: string
  videoScriptPrompt: string
  videoSource: VideoSource
  videoSubject: string
  videoTerms: string
  videoTransitionMode: VideoTransitionMode
  voiceName: string
  voiceRate: number
  voiceVolume: number
}

export interface CreateVideoPayload {
  bgm_file?: string
  bgm_type: string
  bgm_volume: number
  custom_audio_file?: string
  custom_position: number
  custom_system_prompt: string
  font_name: string
  font_size: number
  match_materials_to_script: boolean
  paragraph_number: number
  rounded_subtitle_background: boolean
  stroke_color: string
  stroke_width: number
  subtitle_enabled: boolean
  subtitle_position: SubtitlePosition
  text_background_color?: boolean | string
  text_fore_color: string
  video_aspect: VideoAspect
  video_clip_duration: number
  video_concat_mode: VideoConcatMode
  video_count: number
  video_language: string
  video_materials?: { duration: number; provider: 'local'; url: string }[]
  video_script: string
  video_script_prompt: string
  video_source: VideoSource
  video_subject: string
  video_terms?: string[]
  video_transition_mode?: Exclude<VideoTransitionMode, ''>
  voice_name: string
  voice_rate: number
  voice_volume: number
}

export interface MoneyPrinterError {
  code: string
  details?: Record<string, unknown>
  message: string
}

export interface MoneyPrinterResponse<T> {
  data: T | null
  error: MoneyPrinterError | null
  ok: boolean
}

export interface MoneyPrinterHealth {
  apiBaseUrl?: string
  config?: MoneyPrinterConfigSummary
  ffmpeg?: boolean
  imagemagick?: boolean
  installed: boolean
  message?: string
  serviceRunning: boolean
  storageWritable?: boolean
  upstreamCommit?: string
}

export interface MoneyPrinterConfigSummary {
  apiKeyConfigured?: boolean
  baseUrl?: string
  configExists: boolean
  llmProvider: string
  materialProviders: {
    coverr: boolean
    pexels: boolean
    pixabay: boolean
  }
  modelConfigured: boolean
  modelName?: string
}

export interface MoneyPrinterConfigInput {
  apiKey: string
  baseUrl: string
  coverrApiKey: string
  llmProvider: string
  modelName: string
  pexelsApiKey: string
  pixabayApiKey: string
}

export type MoneyPrinterTaskState = 'complete' | 'failed' | 'pending' | 'processing' | 'queued' | 'unknown'

export interface MoneyPrinterVideoOutput {
  downloadUrl?: string
  file?: string
  name: string
  streamUrl?: string
}

const DEFAULT_MONEYPRINTER_API_BASE_URL = 'http://127.0.0.1:8080'
const MONEYPRINTER_CAPABILITY_MEDIA_RE = /^\/api\/capabilities\/moneyprinter\/(download|stream)\/(.+)$/
const MONEYPRINTER_PUBLIC_VIDEO_RE = /^(?:combined|final)-\d+\.mp4$/i

export interface MoneyPrinterTask {
  audioFile?: string
  error?: string
  id: string
  progress: number
  script?: string
  state: MoneyPrinterTaskState
  subject?: string
  subtitlePath?: string
  terms?: string[] | string
  videos: MoneyPrinterVideoOutput[]
}

export interface CreateVideoResult {
  task: MoneyPrinterTask
}

export interface GenerateScriptPayload {
  custom_system_prompt: string
  paragraph_number: number
  video_language: string
  video_script_prompt: string
  video_subject: string
}

export interface GenerateTermsPayload {
  amount: number
  match_materials_to_script: boolean
  video_script: string
  video_subject: string
}

export interface GenerateScriptResult {
  script?: string
  video_script?: string
}

export interface GenerateTermsResult {
  terms?: string[] | string
  video_terms?: string[] | string
}

export const defaultVideoGenerationForm: VideoGenerationForm = {
  bgmFile: '',
  bgmType: 'random',
  bgmVolume: 0.2,
  customAudioFile: '',
  customPosition: 70,
  customSystemPrompt: '',
  fontName: 'STHeitiMedium.ttc',
  fontSize: 60,
  localMaterials: [],
  matchMaterialsToScript: false,
  paragraphNumber: 1,
  roundedSubtitleBackground: false,
  searchTermsAmount: 5,
  strokeColor: '#000000',
  strokeWidth: 1.5,
  subtitleEnabled: true,
  subtitlePosition: 'bottom',
  textBackgroundColor: '',
  textForeColor: '#FFFFFF',
  videoAspect: '9:16',
  videoClipDuration: 5,
  videoConcatMode: 'random',
  videoCount: 1,
  videoLanguage: '',
  videoScript: '',
  videoScriptPrompt: '',
  videoSource: 'pexels',
  videoSubject: '',
  videoTerms: '',
  videoTransitionMode: '',
  voiceName: 'zh-CN-XiaoxiaoNeural-Female',
  voiceRate: 1,
  voiceVolume: 1
}

export const defaultMoneyPrinterConfigForm: MoneyPrinterConfigInput = {
  apiKey: '',
  baseUrl: '',
  coverrApiKey: '',
  llmProvider: 'openai',
  modelName: 'gpt-4o-mini',
  pexelsApiKey: '',
  pixabayApiKey: ''
}

export function videoStudioApiPath(path: string): string {
  const suffix = path.startsWith('/') ? path : `/${path}`

  return `/api/capabilities/moneyprinter${suffix}`
}

function normalizeApiBaseUrl(apiBaseUrl?: string): string {
  return (apiBaseUrl || DEFAULT_MONEYPRINTER_API_BASE_URL).replace(/\/+$/, '')
}

function decodePathSegment(segment: string): string {
  try {
    return decodeURIComponent(segment)
  } catch {
    return segment
  }
}

function encodeMediaPath(mediaPath: string): string {
  return mediaPath
    .split('/')
    .filter(Boolean)
    .map(segment => encodeURIComponent(decodePathSegment(segment)))
    .join('/')
}

function mediaPathFromUrl(url: string): string {
  if (url.startsWith('/')) {
    return url
  }

  try {
    return new URL(url).pathname
  } catch {
    return ''
  }
}

export function resolveMoneyPrinterMediaUrl(url?: string, apiBaseUrl?: string): string | undefined {
  const value = String(url || '').trim()

  if (!value) {
    return undefined
  }

  const match = mediaPathFromUrl(value).match(MONEYPRINTER_CAPABILITY_MEDIA_RE)

  if (!match) {
    return value
  }

  const [, kind, mediaPath] = match

  return `${normalizeApiBaseUrl(apiBaseUrl)}/api/v1/${kind}/${encodeMediaPath(mediaPath)}`
}

export function isMoneyPrinterPreviewVideo(video: MoneyPrinterVideoOutput): boolean {
  const name = video.name || video.file?.split('/').pop() || ''

  return MONEYPRINTER_PUBLIC_VIDEO_RE.test(name)
}

export function configFormFromSummary(summary: MoneyPrinterConfigSummary): MoneyPrinterConfigInput {
  return {
    ...defaultMoneyPrinterConfigForm,
    baseUrl: summary.baseUrl || '',
    llmProvider: summary.llmProvider || defaultMoneyPrinterConfigForm.llmProvider,
    modelName: summary.modelName || defaultMoneyPrinterConfigForm.modelName
  }
}

export function splitVideoTerms(value: string): string[] {
  return value
    .split(/[\n,，]/)
    .map(term => term.trim())
    .filter(Boolean)
}

function normalizeTextBackgroundColor(value: string): boolean | string | undefined {
  const trimmed = value.trim()

  if (!trimmed) {
    return undefined
  }

  if (trimmed.toLowerCase() === 'true') {
    return true
  }

  if (trimmed.toLowerCase() === 'false') {
    return false
  }

  return trimmed
}

export function scriptTextFromResult(result: GenerateScriptResult): string {
  return String(result.video_script || result.script || '').trim()
}

export function termsTextFromResult(result: GenerateTermsResult): string {
  const terms = result.video_terms ?? result.terms ?? []

  if (Array.isArray(terms)) {
    return terms.map(term => String(term).trim()).filter(Boolean).join('\n')
  }

  return String(terms || '').trim()
}

export function toGenerateScriptPayload(form: VideoGenerationForm): GenerateScriptPayload {
  return {
    custom_system_prompt: form.customSystemPrompt,
    paragraph_number: form.paragraphNumber,
    video_language: form.videoLanguage,
    video_script_prompt: form.videoScriptPrompt,
    video_subject: form.videoSubject.trim()
  }
}

export function toGenerateTermsPayload(form: VideoGenerationForm): GenerateTermsPayload {
  return {
    amount: form.searchTermsAmount,
    match_materials_to_script: form.matchMaterialsToScript,
    video_script: form.videoScript,
    video_subject: form.videoSubject.trim()
  }
}

export function toCreateVideoPayload(form: VideoGenerationForm): CreateVideoPayload {
  const payload: CreateVideoPayload = {
    bgm_type: form.bgmType,
    bgm_volume: form.bgmVolume,
    custom_position: form.customPosition,
    custom_system_prompt: form.customSystemPrompt,
    font_name: form.fontName,
    font_size: form.fontSize,
    match_materials_to_script: form.matchMaterialsToScript,
    paragraph_number: form.paragraphNumber,
    rounded_subtitle_background: form.roundedSubtitleBackground,
    stroke_color: form.strokeColor,
    stroke_width: form.strokeWidth,
    subtitle_enabled: form.subtitleEnabled,
    subtitle_position: form.subtitlePosition,
    text_fore_color: form.textForeColor,
    video_aspect: form.videoAspect,
    video_clip_duration: form.videoClipDuration,
    video_concat_mode: form.videoConcatMode,
    video_count: form.videoCount,
    video_language: form.videoLanguage,
    video_script: form.videoScript,
    video_script_prompt: form.videoScriptPrompt,
    video_source: form.videoSource,
    video_subject: form.videoSubject.trim(),
    voice_name: form.voiceName,
    voice_rate: form.voiceRate,
    voice_volume: form.voiceVolume
  }

  const videoTerms = splitVideoTerms(form.videoTerms)

  if (videoTerms.length > 0) {
    payload.video_terms = videoTerms
  }

  if (form.bgmFile.trim()) {
    payload.bgm_file = form.bgmFile.trim()
  }

  if (form.customAudioFile.trim()) {
    payload.custom_audio_file = form.customAudioFile.trim()
  }

  const textBackgroundColor = normalizeTextBackgroundColor(form.textBackgroundColor)

  if (textBackgroundColor !== undefined) {
    payload.text_background_color = textBackgroundColor
  }

  if (form.videoTransitionMode) {
    payload.video_transition_mode = form.videoTransitionMode
  }

  if (form.videoSource === 'local') {
    payload.video_materials = form.localMaterials
      .map(file => file.trim())
      .filter(Boolean)
      .map(file => ({ duration: 0, provider: 'local' as const, url: file }))
  }

  return payload
}

function responseError(message: string, code = 'MONEYPRINTER_DESKTOP_API_UNAVAILABLE'): MoneyPrinterError {
  return { code, message }
}

async function apiRequest<T>(path: string, init?: { body?: unknown; method?: string }): Promise<MoneyPrinterResponse<T>> {
  const bridge = typeof window === 'undefined' ? null : window.hermesDesktop

  if (!bridge?.api) {
    return {
      data: null,
      error: responseError('Hermes Desktop API bridge is not available in this renderer context.'),
      ok: false
    }
  }

  try {
    return await bridge.api<MoneyPrinterResponse<T>>({
      body: init?.body,
      method: init?.method || 'GET',
      path: videoStudioApiPath(path)
    })
  } catch (err) {
    return {
      data: null,
      error: responseError(err instanceof Error ? err.message : String(err), 'MONEYPRINTER_REQUEST_FAILED'),
      ok: false
    }
  }
}

export const moneyprinterClient = {
  createVideo(form: VideoGenerationForm): Promise<MoneyPrinterResponse<CreateVideoResult>> {
    return apiRequest<CreateVideoResult>('/videos', {
      body: toCreateVideoPayload(form),
      method: 'POST'
    })
  },

  createAudio(form: VideoGenerationForm): Promise<MoneyPrinterResponse<CreateVideoResult>> {
    return apiRequest<CreateVideoResult>('/audio', {
      body: toCreateVideoPayload(form),
      method: 'POST'
    })
  },

  createSubtitle(form: VideoGenerationForm): Promise<MoneyPrinterResponse<CreateVideoResult>> {
    return apiRequest<CreateVideoResult>('/subtitle', {
      body: toCreateVideoPayload(form),
      method: 'POST'
    })
  },

  deleteTask(taskId: string): Promise<MoneyPrinterResponse<{ taskId: string }>> {
    return apiRequest<{ taskId: string }>(`/tasks/${encodeURIComponent(taskId)}`, { method: 'DELETE' })
  },

  getHealth(): Promise<MoneyPrinterResponse<MoneyPrinterHealth>> {
    return apiRequest<MoneyPrinterHealth>('/health')
  },

  generateScript(form: VideoGenerationForm): Promise<MoneyPrinterResponse<GenerateScriptResult>> {
    return apiRequest<GenerateScriptResult>('/scripts', {
      body: toGenerateScriptPayload(form),
      method: 'POST'
    })
  },

  generateTerms(form: VideoGenerationForm): Promise<MoneyPrinterResponse<GenerateTermsResult>> {
    return apiRequest<GenerateTermsResult>('/terms', {
      body: toGenerateTermsPayload(form),
      method: 'POST'
    })
  },

  getConfig(): Promise<MoneyPrinterResponse<MoneyPrinterConfigSummary>> {
    return apiRequest<MoneyPrinterConfigSummary>('/config')
  },

  getTask(taskId: string): Promise<MoneyPrinterResponse<MoneyPrinterTask>> {
    return apiRequest<MoneyPrinterTask>(`/tasks/${encodeURIComponent(taskId)}`)
  },

  listTasks(): Promise<MoneyPrinterResponse<{ tasks: MoneyPrinterTask[] }>> {
    return apiRequest<{ tasks: MoneyPrinterTask[] }>('/tasks')
  },

  listLocalMaterials(): Promise<MoneyPrinterResponse<{ materials: MoneyPrinterLocalMaterial[] }>> {
    return apiRequest<{ materials: MoneyPrinterLocalMaterial[] }>('/materials')
  },

  listAssets(): Promise<MoneyPrinterResponse<MoneyPrinterAssets>> {
    return apiRequest<MoneyPrinterAssets>('/assets')
  },

  saveConfig(config: MoneyPrinterConfigInput): Promise<MoneyPrinterResponse<MoneyPrinterConfigSummary>> {
    return apiRequest<MoneyPrinterConfigSummary>('/config', {
      body: config,
      method: 'POST'
    })
  },

  startService(): Promise<MoneyPrinterResponse<MoneyPrinterHealth>> {
    return apiRequest<MoneyPrinterHealth>('/service/start', { method: 'POST' })
  },

  uploadLocalMaterial(
    input: MoneyPrinterLocalMaterialUploadInput
  ): Promise<MoneyPrinterResponse<{ material: MoneyPrinterLocalMaterial }>> {
    return apiRequest<{ material: MoneyPrinterLocalMaterial }>('/materials', {
      body: input,
      method: 'POST'
    })
  },

  uploadBgm(input: MoneyPrinterLocalMaterialUploadInput): Promise<MoneyPrinterResponse<{ bgm: MoneyPrinterAudioAsset }>> {
    return apiRequest<{ bgm: MoneyPrinterAudioAsset }>('/bgms', {
      body: input,
      method: 'POST'
    })
  },

  uploadCustomAudio(input: MoneyPrinterLocalMaterialUploadInput): Promise<MoneyPrinterResponse<{ audio: MoneyPrinterAudioAsset }>> {
    return apiRequest<{ audio: MoneyPrinterAudioAsset }>('/custom-audio', {
      body: input,
      method: 'POST'
    })
  }
}
