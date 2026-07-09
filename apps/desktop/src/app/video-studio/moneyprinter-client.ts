export type VideoAspect = '9:16' | '16:9' | '1:1'
export type VideoSource = 'pexels' | 'pixabay' | 'coverr' | 'local'
export type SubtitlePosition = 'bottom' | 'top' | 'center' | 'custom'

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

export interface VideoGenerationForm {
  bgmType: string
  bgmVolume: number
  fontName: string
  fontSize: number
  localMaterials: string[]
  matchMaterialsToScript: boolean
  strokeColor: string
  strokeWidth: number
  subtitleEnabled: boolean
  subtitlePosition: SubtitlePosition
  textForeColor: string
  videoAspect: VideoAspect
  videoClipDuration: number
  videoCount: number
  videoLanguage: string
  videoScript: string
  videoSource: VideoSource
  videoSubject: string
  voiceName: string
  voiceRate: number
  voiceVolume: number
}

export interface CreateVideoPayload {
  bgm_type: string
  bgm_volume: number
  font_name: string
  font_size: number
  match_materials_to_script: boolean
  stroke_color: string
  stroke_width: number
  subtitle_enabled: boolean
  subtitle_position: SubtitlePosition
  text_fore_color: string
  video_aspect: VideoAspect
  video_clip_duration: number
  video_count: number
  video_language: string
  video_materials?: { duration: number; provider: 'local'; url: string }[]
  video_script: string
  video_source: VideoSource
  video_subject: string
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
  configExists: boolean
  llmProvider: string
  materialProviders: {
    coverr: boolean
    pexels: boolean
    pixabay: boolean
  }
  modelConfigured: boolean
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

export const defaultVideoGenerationForm: VideoGenerationForm = {
  bgmType: 'random',
  bgmVolume: 0.2,
  fontName: 'STHeitiMedium.ttc',
  fontSize: 60,
  localMaterials: [],
  matchMaterialsToScript: false,
  strokeColor: '#000000',
  strokeWidth: 1.5,
  subtitleEnabled: true,
  subtitlePosition: 'bottom',
  textForeColor: '#FFFFFF',
  videoAspect: '9:16',
  videoClipDuration: 5,
  videoCount: 1,
  videoLanguage: '',
  videoScript: '',
  videoSource: 'pexels',
  videoSubject: '',
  voiceName: 'zh-CN-XiaoxiaoNeural-Female',
  voiceRate: 1,
  voiceVolume: 1
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

export function toCreateVideoPayload(form: VideoGenerationForm): CreateVideoPayload {
  const payload: CreateVideoPayload = {
    bgm_type: form.bgmType,
    bgm_volume: form.bgmVolume,
    font_name: form.fontName,
    font_size: form.fontSize,
    match_materials_to_script: form.matchMaterialsToScript,
    stroke_color: form.strokeColor,
    stroke_width: form.strokeWidth,
    subtitle_enabled: form.subtitleEnabled,
    subtitle_position: form.subtitlePosition,
    text_fore_color: form.textForeColor,
    video_aspect: form.videoAspect,
    video_clip_duration: form.videoClipDuration,
    video_count: form.videoCount,
    video_language: form.videoLanguage,
    video_script: form.videoScript,
    video_source: form.videoSource,
    video_subject: form.videoSubject.trim(),
    voice_name: form.voiceName,
    voice_rate: form.voiceRate,
    voice_volume: form.voiceVolume
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

  deleteTask(taskId: string): Promise<MoneyPrinterResponse<{ taskId: string }>> {
    return apiRequest<{ taskId: string }>(`/tasks/${encodeURIComponent(taskId)}`, { method: 'DELETE' })
  },

  getHealth(): Promise<MoneyPrinterResponse<MoneyPrinterHealth>> {
    return apiRequest<MoneyPrinterHealth>('/health')
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
  }
}
