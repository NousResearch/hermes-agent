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

export interface VideoLibraryAsset {
  created_at: number
  duration_seconds?: number | null
  fps?: number | null
  height?: number | null
  id: string
  managed_path: string
  original_name: string
  size_bytes: number
  source_path: string
  status: string
  updated_at: number
  width?: number | null
}

export interface VideoLibraryTag {
  confidence: number
  id: string
  name: string
  source: string
}

export interface VideoLibraryClip {
  asset_id: string
  clip_index: number
  created_at: number
  description: string
  duration_seconds: number
  end_seconds: number
  file_path: string
  id: string
  keyframe_path?: string
  start_seconds: number
  status: string
  tags: VideoLibraryTag[]
  updated_at: number
}

export interface VideoLibraryAnalysisResult {
  asset: VideoLibraryAsset
  clips: VideoLibraryClip[]
  job: {
    error: string
    id: string
    progress: number
    state: string
  }
}

export interface VideoLibraryTimelineResult {
  id: string
  path: string
  timeline: Record<string, unknown>
}

export interface MoneyPrinterAudioAsset {
  downloadUrl?: string
  file: string
  kind: 'audio'
  name: string
  size: number
  streamUrl?: string
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
  ffmpegPath?: string
  imagemagick?: boolean
  installed: boolean
  message?: string
  missingDependencies?: string[]
  runtimePython?: string
  runtimeReady?: boolean
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
  minimax?: {
    apiKeyConfigured?: boolean
    baseUrl?: string
    musicModel?: string
    t2aModel?: string
    voiceCloneModel?: string
  }
  modelConfigured: boolean
  modelName?: string
}

export interface MoneyPrinterConfigInput {
  apiKey: string
  baseUrl: string
  coverrApiKey: string
  llmProvider: string
  minimaxApiKey: string
  minimaxBaseUrl: string
  minimaxMusicModel: string
  minimaxT2aModel: string
  minimaxVoiceCloneModel: string
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

export interface MiniMaxCloneVoiceInput {
  activate: boolean
  cloneAudio: MoneyPrinterLocalMaterialUploadInput
  model: string
  promptAudio?: MoneyPrinterLocalMaterialUploadInput
  promptText: string
  trialText: string
  voiceId: string
}

export interface MiniMaxCloneVoicePayload {
  activate: boolean
  clone_audio: MoneyPrinterLocalMaterialUploadInput
  model: string
  prompt_audio?: MoneyPrinterLocalMaterialUploadInput
  prompt_text: string
  trial_text: string
  voice_id: string
}

export interface MiniMaxVoiceRecord {
  category: 'local_preview' | 'system' | 'voice_cloning' | 'voice_generation'
  id: string
  name: string
  providerConfirmed: boolean
}

export interface MiniMaxTtsInput {
  model: string
  text: string
  voiceId: string
}

export interface MiniMaxTtsPayload {
  model: string
  save_as_custom_audio: boolean
  speed: number
  text: string
  voice_id: string
  volume: number
}

export interface MiniMaxAudioResult {
  activated?: boolean
  audio?: MoneyPrinterAudioAsset
  file?: string
  previewError?: string
  trialAudio?: MoneyPrinterAudioAsset
  trialAudioFile?: string
  voice_id?: string
  voiceNameForVideo?: string
}

export interface MiniMaxMusicInput {
  isInstrumental: boolean
  lyrics: string
  lyricsOptimizer: boolean
  model: string
  prompt: string
  saveAsBgm: boolean
}

export interface MiniMaxMusicPayload {
  is_instrumental: boolean
  lyrics: string
  lyrics_optimizer: boolean
  model: string
  prompt: string
  save_as_bgm: boolean
}

export interface MiniMaxLyricsInput {
  lyrics: string
  mode: 'edit' | 'write_full_song'
  prompt: string
  title: string
}

export interface MiniMaxLyricsResult {
  lyrics?: string
  song_title?: string
  style_tags?: string[]
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
  minimaxApiKey: '',
  minimaxBaseUrl: 'https://api.minimax.io/v1',
  minimaxMusicModel: 'music-2.6-free',
  minimaxT2aModel: 'speech-2.8-hd',
  minimaxVoiceCloneModel: 'speech-2.8-hd',
  modelName: 'gpt-4o-mini',
  pexelsApiKey: '',
  pixabayApiKey: ''
}

export function videoStudioApiPath(path: string): string {
  const suffix = path.startsWith('/') ? path : `/${path}`

  return `/api/capabilities/moneyprinter${suffix}`
}

export function videoLibraryApiPath(path: string): string {
  const suffix = path.startsWith('/') ? path : `/${path}`

  return `/api/capabilities/video-library${suffix}`
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

export function resolveMoneyPrinterMediaUrl(url?: string, _apiBaseUrl?: string): string | undefined {
  const value = String(url || '').trim()

  if (!value) {
    return undefined
  }

  const mediaPath = mediaPathFromUrl(value)
  const match = mediaPath.match(MONEYPRINTER_CAPABILITY_MEDIA_RE)

  if (!match) {
    return value
  }

  return `hermes-media://gateway/${encodeURIComponent(mediaPath)}`
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
    minimaxBaseUrl: summary.minimax?.baseUrl || defaultMoneyPrinterConfigForm.minimaxBaseUrl,
    minimaxMusicModel: summary.minimax?.musicModel || defaultMoneyPrinterConfigForm.minimaxMusicModel,
    minimaxT2aModel: summary.minimax?.t2aModel || defaultMoneyPrinterConfigForm.minimaxT2aModel,
    minimaxVoiceCloneModel:
      summary.minimax?.voiceCloneModel || defaultMoneyPrinterConfigForm.minimaxVoiceCloneModel,
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
    return terms
      .map(term => String(term).trim())
      .filter(Boolean)
      .join('\n')
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

export function toMiniMaxCloneVoicePayload(input: MiniMaxCloneVoiceInput): MiniMaxCloneVoicePayload {
  return {
    activate: input.activate,
    clone_audio: input.cloneAudio,
    model: input.model,
    ...(input.promptAudio ? { prompt_audio: input.promptAudio } : {}),
    prompt_text: input.promptText,
    trial_text: input.trialText,
    voice_id: input.voiceId
  }
}

export function toMiniMaxMusicPayload(input: MiniMaxMusicInput): MiniMaxMusicPayload {
  return {
    is_instrumental: input.isInstrumental,
    lyrics: input.lyrics,
    lyrics_optimizer: input.lyricsOptimizer,
    model: input.model,
    prompt: input.prompt,
    save_as_bgm: input.saveAsBgm
  }
}

export function toMiniMaxTtsPayload(input: MiniMaxTtsInput): MiniMaxTtsPayload {
  return {
    model: input.model,
    save_as_custom_audio: true,
    speed: 1,
    text: input.text,
    voice_id: input.voiceId,
    volume: 1
  }
}

function responseError(message: string, code = 'MONEYPRINTER_DESKTOP_API_UNAVAILABLE'): MoneyPrinterError {
  return { code, message }
}

async function desktopApiRequest<T>(
  path: string,
  init?: { body?: unknown; method?: string; timeoutMs?: number },
  codes: { request: string; unavailable: string } = {
    request: 'MONEYPRINTER_REQUEST_FAILED',
    unavailable: 'MONEYPRINTER_DESKTOP_API_UNAVAILABLE'
  }
): Promise<MoneyPrinterResponse<T>> {
  const bridge = typeof window === 'undefined' ? null : window.hermesDesktop

  if (!bridge?.api) {
    return {
      data: null,
      error: responseError('Hermes Desktop API bridge is not available in this renderer context.', codes.unavailable),
      ok: false
    }
  }

  try {
    return await bridge.api<MoneyPrinterResponse<T>>({
      body: init?.body,
      method: init?.method || 'GET',
      path,
      timeoutMs: init?.timeoutMs
    })
  } catch (err) {
    return {
      data: null,
      error: responseError(err instanceof Error ? err.message : String(err), codes.request),
      ok: false
    }
  }
}

async function apiRequest<T>(
  path: string,
  init?: { body?: unknown; method?: string; timeoutMs?: number }
): Promise<MoneyPrinterResponse<T>> {
  return desktopApiRequest<T>(videoStudioApiPath(path), init)
}

async function videoLibraryRequest<T>(
  path: string,
  init?: { body?: unknown; method?: string }
): Promise<MoneyPrinterResponse<T>> {
  return desktopApiRequest<T>(videoLibraryApiPath(path), init, {
    request: 'VIDEO_LIBRARY_REQUEST_FAILED',
    unavailable: 'VIDEO_LIBRARY_DESKTOP_API_UNAVAILABLE'
  })
}

export const videoLibraryClient = {
  analyzeAsset(assetId: string): Promise<MoneyPrinterResponse<VideoLibraryAnalysisResult>> {
    return videoLibraryRequest<VideoLibraryAnalysisResult>(`/assets/${encodeURIComponent(assetId)}/analyze`, {
      body: {},
      method: 'POST'
    })
  },

  createTimeline(clipIds: string[], aspect: VideoAspect): Promise<MoneyPrinterResponse<VideoLibraryTimelineResult>> {
    return videoLibraryRequest<VideoLibraryTimelineResult>('/timelines', {
      body: { aspect, clipIds },
      method: 'POST'
    })
  },

  importAsset(sourcePath: string): Promise<MoneyPrinterResponse<{ asset: VideoLibraryAsset }>> {
    return videoLibraryRequest<{ asset: VideoLibraryAsset }>('/assets', {
      body: { sourcePath },
      method: 'POST'
    })
  },

  listAssets(): Promise<MoneyPrinterResponse<{ assets: VideoLibraryAsset[]; total: number }>> {
    return videoLibraryRequest<{ assets: VideoLibraryAsset[]; total: number }>('/assets')
  },

  listClips(assetId?: string): Promise<MoneyPrinterResponse<{ clips: VideoLibraryClip[]; total: number }>> {
    const query = assetId ? `?asset_id=${encodeURIComponent(assetId)}` : ''

    return videoLibraryRequest<{ clips: VideoLibraryClip[]; total: number }>(`/clips${query}`)
  },

  replaceClipTags(
    clipId: string,
    tags: string[]
  ): Promise<MoneyPrinterResponse<{ clipId: string; tags: VideoLibraryTag[] }>> {
    return videoLibraryRequest<{ clipId: string; tags: VideoLibraryTag[] }>(
      `/clips/${encodeURIComponent(clipId)}/tags`,
      {
        body: { tags },
        method: 'POST'
      }
    )
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

  cloneMiniMaxVoice(input: MiniMaxCloneVoiceInput): Promise<MoneyPrinterResponse<MiniMaxAudioResult>> {
    return apiRequest<MiniMaxAudioResult>('/minimax/voices/clone', {
      body: toMiniMaxCloneVoicePayload(input),
      method: 'POST',
      timeoutMs: 180_000
    })
  },

  generateMiniMaxTts(input: MiniMaxTtsInput): Promise<MoneyPrinterResponse<MiniMaxAudioResult>> {
    return apiRequest<MiniMaxAudioResult>('/minimax/tts', {
      body: toMiniMaxTtsPayload(input),
      method: 'POST',
      timeoutMs: 180_000
    })
  },

  generateMiniMaxMusic(input: MiniMaxMusicInput): Promise<MoneyPrinterResponse<Record<string, unknown>>> {
    return apiRequest<Record<string, unknown>>('/minimax/music', {
      body: toMiniMaxMusicPayload(input),
      method: 'POST',
      timeoutMs: 240_000
    })
  },

  generateMiniMaxLyrics(input: MiniMaxLyricsInput): Promise<MoneyPrinterResponse<MiniMaxLyricsResult>> {
    return apiRequest<MiniMaxLyricsResult>('/minimax/lyrics', {
      body: input,
      method: 'POST',
      timeoutMs: 90_000
    })
  },

  listMiniMaxVoices(): Promise<MoneyPrinterResponse<{ voices: MiniMaxVoiceRecord[] }>> {
    return apiRequest<{ voices: MiniMaxVoiceRecord[] }>('/minimax/voices', { timeoutMs: 90_000 })
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

  uploadBgm(
    input: MoneyPrinterLocalMaterialUploadInput
  ): Promise<MoneyPrinterResponse<{ bgm: MoneyPrinterAudioAsset }>> {
    return apiRequest<{ bgm: MoneyPrinterAudioAsset }>('/bgms', {
      body: input,
      method: 'POST'
    })
  },

  uploadCustomAudio(
    input: MoneyPrinterLocalMaterialUploadInput
  ): Promise<MoneyPrinterResponse<{ audio: MoneyPrinterAudioAsset }>> {
    return apiRequest<{ audio: MoneyPrinterAudioAsset }>('/custom-audio', {
      body: input,
      method: 'POST'
    })
  }
}
