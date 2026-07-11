import type * as React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { selectDesktopPaths } from '@/lib/desktop-fs'
import { readKey, writeKey } from '@/lib/storage'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

import {
  createMiniMaxCloneVoiceId,
  miniMaxVoiceErrorMessage,
  miniMaxVoiceName,
  validateMiniMaxCloneInput
} from './minimax-voice-workflows'
import {
  cacheFilenameForSelection,
  confirmedTimelineFormPatch,
  timelineVideoSelections
} from './material-cache'
import {
  configFormFromSummary,
  defaultMoneyPrinterConfigForm,
  defaultVideoGenerationForm,
  isMoneyPrinterPreviewVideo,
  type MiniMaxAudioResult,
  type MiniMaxVoiceRecord,
  type MoneyPrinterAssets,
  moneyprinterClient,
  type MoneyPrinterConfigInput,
  type MoneyPrinterConfigSummary,
  type MoneyPrinterHealth,
  type MoneyPrinterTask,
  resolveMoneyPrinterMediaUrl,
  scriptTextFromResult,
  termsTextFromResult,
  type VideoGenerationForm,
  videoLibraryClient,
} from './moneyprinter-client'
import { UnifiedMaterialLibraryPanel } from './unified-material-library-panel'
import { useNamedVideoLibrary } from './use-named-video-library'

interface VideoStudioViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

const PIPELINE_STEPS = ['主题', '文案', '关键词', '语音', '素材', '字幕', '合成', '预览']

const VIDEO_STUDIO_DRAFT_KEY = 'hermes-video-studio-moneyprinter-draft-v1'

const EMPTY_ASSETS: MoneyPrinterAssets = {
  bgms: [],
  customAudio: [],
  fonts: [],
  voices: []
}

function fieldLabelClass() {
  return 'text-xs font-medium text-(--ui-text-secondary)'
}

function numberValue(value: string, fallback: number): number {
  const parsed = Number(value)

  return Number.isFinite(parsed) ? parsed : fallback
}

function formatBytes(size: number): string {
  if (!Number.isFinite(size) || size <= 0) {
    return '0 B'
  }

  const units = ['B', 'KB', 'MB', 'GB']
  let value = size
  let unit = 0

  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }

  return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`
}

function savedStateLabel(configured?: boolean): string {
  return configured ? '已保存' : '未配置'
}

function storedVideoGenerationForm(): VideoGenerationForm {
  const raw = readKey(VIDEO_STUDIO_DRAFT_KEY)

  if (!raw) {
    return defaultVideoGenerationForm
  }

  try {
    const parsed = JSON.parse(raw) as Partial<VideoGenerationForm>

    if (!parsed || typeof parsed !== 'object') {
      return defaultVideoGenerationForm
    }

    return {
      ...defaultVideoGenerationForm,
      ...parsed,
      localMaterials: Array.isArray(parsed.localMaterials)
        ? parsed.localMaterials.filter((item): item is string => typeof item === 'string')
        : defaultVideoGenerationForm.localMaterials
    }
  } catch {
    return defaultVideoGenerationForm
  }
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error || new Error('Unable to read local material file'))
    reader.onload = () => resolve(String(reader.result || ''))
    reader.readAsDataURL(file)
  })
}

export function VideoStudioView({
  className,
  setStatusbarItemGroup: _setStatusbarItemGroup,
  ...props
}: VideoStudioViewProps) {
  const [form, setForm] = useState<VideoGenerationForm>(() => storedVideoGenerationForm())
  const [configForm, setConfigForm] = useState<MoneyPrinterConfigInput>(defaultMoneyPrinterConfigForm)
  const [configSummary, setConfigSummary] = useState<MoneyPrinterConfigSummary | null>(null)
  const [health, setHealth] = useState<MoneyPrinterHealth | null>(null)
  const [tasks, setTasks] = useState<MoneyPrinterTask[]>([])
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [configBusy, setConfigBusy] = useState(false)
  const [scriptBusy, setScriptBusy] = useState(false)
  const [termsBusy, setTermsBusy] = useState(false)
  const [audioBusy, setAudioBusy] = useState(false)
  const [subtitleBusy, setSubtitleBusy] = useState(false)
  const [assets, setAssets] = useState<MoneyPrinterAssets>(EMPTY_ASSETS)
  const [assetsBusy, setAssetsBusy] = useState(false)
  const [bgmUploadBusy, setBgmUploadBusy] = useState(false)
  const [customAudioUploadBusy, setCustomAudioUploadBusy] = useState(false)
  const [serviceBusy, setServiceBusy] = useState(false)
  const [miniMaxVoices, setMiniMaxVoices] = useState<MiniMaxVoiceRecord[]>([])
  const [miniMaxExistingVoiceId, setMiniMaxExistingVoiceId] = useState('Korean_GentleBoss')

  const [miniMaxExistingPreviewText, setMiniMaxExistingPreviewText] = useState(
    '안녕하세요. Hermes 음성 미리듣기입니다.'
  )

  const [miniMaxExistingPreview, setMiniMaxExistingPreview] = useState<MiniMaxAudioResult | null>(null)
  const [miniMaxExistingBusy, setMiniMaxExistingBusy] = useState(false)
  const [miniMaxCloneBusy, setMiniMaxCloneBusy] = useState(false)
  const [miniMaxCloneFile, setMiniMaxCloneFile] = useState<File | null>(null)
  const [miniMaxPromptFile, setMiniMaxPromptFile] = useState<File | null>(null)
  const [miniMaxVoiceId, setMiniMaxVoiceId] = useState(() => createMiniMaxCloneVoiceId())
  const [miniMaxPromptText, setMiniMaxPromptText] = useState('')
  const [miniMaxTrialText, setMiniMaxTrialText] = useState('这是 MiniMax 复刻音色试听。')
  const [miniMaxClonePreview, setMiniMaxClonePreview] = useState<MiniMaxAudioResult | null>(null)
  const [miniMaxActivateBusy, setMiniMaxActivateBusy] = useState(false)
  const [miniMaxMusicBusy, setMiniMaxMusicBusy] = useState(false)
  const [miniMaxLyricsBusy, setMiniMaxLyricsBusy] = useState(false)
  const [miniMaxMusicPrompt, setMiniMaxMusicPrompt] = useState('适合短视频的现代感背景音乐')
  const [miniMaxLyrics, setMiniMaxLyrics] = useState('')
  const namedLibrary = useNamedVideoLibrary({
    client: videoLibraryClient,
    script: form.videoScript,
    terms: form.videoTerms
  })

  const selectedTask = useMemo(
    () => tasks.find(task => task.id === selectedTaskId) ?? tasks[0] ?? null,
    [selectedTaskId, tasks]
  )

  const selectedTaskVideos = useMemo(
    () => selectedTask?.videos.filter(isMoneyPrinterPreviewVideo) ?? [],
    [selectedTask]
  )

  useEffect(() => {
    writeKey(VIDEO_STUDIO_DRAFT_KEY, JSON.stringify(form))
  }, [form])

  const updateForm = useCallback(<K extends keyof VideoGenerationForm>(key: K, value: VideoGenerationForm[K]) => {
    setForm(current => ({ ...current, [key]: value }))
  }, [])

  const updateConfig = useCallback(
    <K extends keyof MoneyPrinterConfigInput>(key: K, value: MoneyPrinterConfigInput[K]) => {
      setConfigForm(current => ({ ...current, [key]: value }))
    },
    []
  )

  const applyConfigSummary = useCallback((summary: MoneyPrinterConfigSummary, clearSecrets = false) => {
    const visibleConfig = configFormFromSummary(summary)

    setConfigSummary(summary)
    setConfigForm(current => ({
      ...visibleConfig,
      apiKey: clearSecrets ? '' : current.apiKey,
      coverrApiKey: clearSecrets ? '' : current.coverrApiKey,
      minimaxApiKey: clearSecrets ? '' : current.minimaxApiKey,
      pexelsApiKey: clearSecrets ? '' : current.pexelsApiKey,
      pixabayApiKey: clearSecrets ? '' : current.pixabayApiKey
    }))
  }, [])

  const loadConfig = useCallback(async () => {
    const response = await moneyprinterClient.getConfig()

    if (!response.ok || !response.data) {
      notifyError(response.error?.message, 'Unable to load MoneyPrinter config')

      return
    }

    applyConfigSummary(response.data, true)
  }, [applyConfigSummary])

  const refreshTasks = useCallback(async () => {
    const response = await moneyprinterClient.listTasks()

    if (!response.ok || !response.data) {
      notifyError(response.error?.message, 'Unable to load MoneyPrinter tasks')

      return
    }

    setTasks(response.data.tasks)
    setSelectedTaskId(current => current || response.data?.tasks[0]?.id || null)
  }, [])

  const refreshAssets = useCallback(async () => {
    setAssetsBusy(true)

    try {
      const [response, voicesResponse] = await Promise.all([
        moneyprinterClient.listAssets(),
        moneyprinterClient.listMiniMaxVoices()
      ])

      if (!response.ok || !response.data) {
        notifyError(response.error?.message, 'Unable to load MoneyPrinter assets')
      } else {
        setAssets(response.data)
      }

      if (!voicesResponse.ok || !voicesResponse.data) {
        notifyError(voicesResponse.error?.message, 'Unable to load MiniMax voices')
      } else {
        setMiniMaxVoices(voicesResponse.data.voices)
      }
    } finally {
      setAssetsBusy(false)
    }
  }, [])

  useEffect(() => {
    void loadConfig()
    void refreshTasks()
    void refreshAssets()
  }, [loadConfig, refreshAssets, refreshTasks])

  const selectMaterialLibrary = useCallback(
    (libraryId: string) => {
      namedLibrary.selectLibrary(libraryId)
      setForm(current => ({
        ...current,
        localMaterials: [],
        matchMaterialsToScript: true,
        videoConcatMode: 'sequential',
        videoSource: 'local'
      }))
    },
    [namedLibrary]
  )

  const addNamedLibraryFiles = useCallback(async () => {
    const paths = await selectDesktopPaths({
      directories: false,
      filters: [{ extensions: ['mp4', 'mov', 'mkv', 'avi', 'flv', 'webm'], name: 'Video' }],
      multiple: true,
      title: '添加素材到当前资产库'
    })
    if (paths.length === 0) return
    try {
      await namedLibrary.importFiles(paths)
    } catch {
      // The controller owns the user-visible per-operation error.
    }
  }, [namedLibrary])

  const addNamedLibraryDirectory = useCallback(async () => {
    const [directory] = await selectDesktopPaths({
      directories: true,
      multiple: false,
      title: '选择当前资产库的素材目录'
    })
    if (!directory) return
    try {
      await namedLibrary.addSourceRoot(directory)
    } catch {
      // The controller owns the user-visible per-operation error.
    }
  }, [namedLibrary])

  const createNamedLibraryTimeline = useCallback(async () => {
    try {
      const result = await namedLibrary.createTimeline(form.videoAspect)
      const selections = timelineVideoSelections(result.timeline)
      if (selections.length === 0) throw new Error('素材时间线缺少可验证的来源信息')
      const uploadedFiles: string[] = []

      for (const selection of selections) {
        const filename = cacheFilenameForSelection(selection)
        const response = await moneyprinterClient.uploadLocalMaterial({
          filename,
          sourcePath: selection.file
        })
        if (!response.ok || !response.data) {
          throw new Error(response.error?.message || `无法加入镜头 ${filename}`)
        }
        const material = response.data.material
        uploadedFiles.push(material.file)
      }

      setForm(current => ({
        ...current,
        ...confirmedTimelineFormPatch(uploadedFiles)
      }))
      notify({
        kind: 'success',
        title: '具名资产库时间线已创建',
        message: `${uploadedFiles.length} 个人工确认镜头已加入本地混剪。`
      })
    } catch (reason) {
      notifyError(reason instanceof Error ? reason.message : String(reason), '无法创建素材时间线')
    }
  }, [form.videoAspect, namedLibrary])

  const uploadBgmFiles = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget
    const files = Array.from(input.files || [])

    if (files.length === 0) {
      return
    }

    setBgmUploadBusy(true)

    try {
      const uploadedFiles: string[] = []

      for (const file of files) {
        let sourcePath = ''

        try {
          sourcePath = window.hermesDesktop?.getPathForFile?.(file) || ''
        } catch {
          sourcePath = ''
        }

        const response = await moneyprinterClient.uploadBgm({
          ...(sourcePath ? { sourcePath } : { contentBase64: await fileToDataUrl(file) }),
          filename: file.name
        })

        if (!response.ok || !response.data) {
          notifyError(response.error?.message, `Unable to upload ${file.name}`)

          continue
        }

        const bgm = response.data.bgm
        uploadedFiles.push(bgm.file)
        setAssets(current => ({ ...current, bgms: [bgm, ...current.bgms.filter(item => item.file !== bgm.file)] }))
        setForm(current => ({ ...current, bgmFile: bgm.file, bgmType: 'custom' }))
      }

      if (uploadedFiles.length > 0) {
        notify({ kind: 'success', title: 'BGM uploaded', message: uploadedFiles.join(', ') })
      }
    } finally {
      setBgmUploadBusy(false)
      input.value = ''
    }
  }, [])

  const uploadCustomAudioFiles = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget
    const files = Array.from(input.files || [])

    if (files.length === 0) {
      return
    }

    setCustomAudioUploadBusy(true)

    try {
      const uploadedFiles: string[] = []

      for (const file of files) {
        let sourcePath = ''

        try {
          sourcePath = window.hermesDesktop?.getPathForFile?.(file) || ''
        } catch {
          sourcePath = ''
        }

        const response = await moneyprinterClient.uploadCustomAudio({
          ...(sourcePath ? { sourcePath } : { contentBase64: await fileToDataUrl(file) }),
          filename: file.name
        })

        if (!response.ok || !response.data) {
          notifyError(response.error?.message, `Unable to upload ${file.name}`)

          continue
        }

        const audio = response.data.audio
        uploadedFiles.push(audio.file)
        setAssets(current => ({
          ...current,
          customAudio: [audio, ...current.customAudio.filter(item => item.file !== audio.file)]
        }))
        setForm(current => ({ ...current, customAudioFile: audio.file }))
      }

      if (uploadedFiles.length > 0) {
        notify({ kind: 'success', title: 'Custom audio uploaded', message: uploadedFiles.join(', ') })
      }
    } finally {
      setCustomAudioUploadBusy(false)
      input.value = ''
    }
  }, [])

  const previewExistingMiniMaxVoice = useCallback(async () => {
    const voiceId = miniMaxExistingVoiceId.trim()
    const text = miniMaxExistingPreviewText.trim()

    if (!voiceId || !text) {
      notifyError('请填写已有 Voice ID 和试听文本', 'MiniMax existing voice preview is incomplete')

      return
    }

    setMiniMaxExistingBusy(true)

    try {
      const response = await moneyprinterClient.generateMiniMaxTts({
        model: configForm.minimaxT2aModel,
        text,
        voiceId
      })

      if (!response.ok || !response.data) {
        notifyError(response.error?.message, 'MiniMax existing voice preview failed')

        return
      }

      setMiniMaxExistingPreview(response.data)
      notify({ kind: 'success', title: '已有音色试听已生成', message: voiceId })
    } finally {
      setMiniMaxExistingBusy(false)
    }
  }, [configForm.minimaxT2aModel, miniMaxExistingPreviewText, miniMaxExistingVoiceId])

  const cloneMiniMaxVoice = useCallback(async () => {
    const voiceId = miniMaxVoiceId.trim()

    const validationError = validateMiniMaxCloneInput({
      cloneFile: Boolean(miniMaxCloneFile),
      promptFile: Boolean(miniMaxPromptFile),
      promptText: miniMaxPromptText,
      voiceId
    })

    if (validationError || !miniMaxCloneFile) {
      notifyError(validationError, 'MiniMax voice clone is incomplete')

      return
    }

    const toUploadInput = async (file: File) => {
      let sourcePath = ''

      try {
        sourcePath = window.hermesDesktop?.getPathForFile?.(file) || ''
      } catch {
        sourcePath = ''
      }

      return {
        ...(sourcePath ? { sourcePath } : { contentBase64: await fileToDataUrl(file) }),
        filename: file.name
      }
    }

    setMiniMaxCloneBusy(true)

    try {
      const response = await moneyprinterClient.cloneMiniMaxVoice({
        activate: false,
        cloneAudio: await toUploadInput(miniMaxCloneFile),
        model: configForm.minimaxVoiceCloneModel,
        ...(miniMaxPromptFile ? { promptAudio: await toUploadInput(miniMaxPromptFile) } : {}),
        promptText: miniMaxPromptFile ? miniMaxPromptText.trim() : '',
        trialText: miniMaxTrialText.trim(),
        voiceId
      })

      if (!response.ok || !response.data) {
        notifyError(
          miniMaxVoiceErrorMessage(response.error?.message),
          'MiniMax voice clone failed'
        )

        return
      }

      setMiniMaxClonePreview(response.data)
      await refreshAssets()
      notify({
        kind: response.data.previewError ? 'warning' : 'success',
        title: response.data.previewError ? '音色已创建，但试听不可用' : '克隆试听已生成（未激活）',
        message: response.data.previewError || voiceId
      })
    } finally {
      setMiniMaxCloneBusy(false)
    }
  }, [configForm.minimaxVoiceCloneModel, miniMaxCloneFile, miniMaxPromptFile, miniMaxPromptText, miniMaxTrialText, miniMaxVoiceId, refreshAssets])

  const activateClonedMiniMaxVoice = useCallback(async () => {
    const voiceId = miniMaxVoiceId.trim()
    const text = miniMaxTrialText.trim()

    if (!miniMaxClonePreview || !voiceId || !text) {
      notifyError('请先生成克隆试听', 'MiniMax clone activation is incomplete')

      return
    }

    if (
      !window.confirm(
        '首次正式使用克隆音色预计收取一次性 ¥9.9 复刻费，并另计少量 TTS 字符费。是否继续？'
      )
    ) {
      return
    }

    setMiniMaxActivateBusy(true)

    try {
      const response = await moneyprinterClient.generateMiniMaxTts({
        model: configForm.minimaxT2aModel,
        text,
        voiceId
      })

      if (!response.ok || !response.data) {
        notifyError(response.error?.message, 'MiniMax clone activation failed')

        return
      }

      setMiniMaxClonePreview(current => ({ ...current, activated: true, audio: response.data?.audio }))
      updateForm('voiceName', miniMaxVoiceName(voiceId))
      await refreshAssets()
      notify({ kind: 'success', title: '克隆音色已正式激活并选用', message: miniMaxVoiceName(voiceId) })
    } finally {
      setMiniMaxActivateBusy(false)
    }
  }, [configForm.minimaxT2aModel, miniMaxClonePreview, miniMaxTrialText, miniMaxVoiceId, refreshAssets, updateForm])

  const generateMiniMaxLyrics = useCallback(async () => {
    setMiniMaxLyricsBusy(true)

    try {
      const response = await moneyprinterClient.generateMiniMaxLyrics({
        lyrics: miniMaxLyrics,
        mode: miniMaxLyrics.trim() ? 'edit' : 'write_full_song',
        prompt: miniMaxMusicPrompt.trim(),
        title: form.videoSubject.trim()
      })

      if (!response.ok || !response.data) {
        notifyError(response.error?.message, 'MiniMax lyrics generation failed')

        return
      }

      setMiniMaxLyrics(response.data.lyrics || '')
      notify({ kind: 'success', title: 'MiniMax 歌词已生成', message: '可继续编辑歌词或直接生成音乐。' })
    } finally {
      setMiniMaxLyricsBusy(false)
    }
  }, [form.videoSubject, miniMaxLyrics, miniMaxMusicPrompt])

  const generateMiniMaxMusic = useCallback(async () => {
    setMiniMaxMusicBusy(true)

    try {
      const response = await moneyprinterClient.generateMiniMaxMusic({
        isInstrumental: !miniMaxLyrics.trim(),
        lyrics: miniMaxLyrics.trim(),
        lyricsOptimizer: true,
        model: configForm.minimaxMusicModel,
        prompt: miniMaxMusicPrompt.trim(),
        saveAsBgm: true
      })

      if (!response.ok || !response.data) {
        notifyError(response.error?.message, 'MiniMax music generation failed')

        return
      }

      const bgm = response.data.bgm as { file?: string } | undefined

      if (bgm?.file) {
        updateForm('bgmFile', bgm.file)
        updateForm('bgmType', 'custom')
      }

      await refreshAssets()
      notify({ kind: 'success', title: 'MiniMax 音乐已保存为 BGM', message: bgm?.file || '' })
    } finally {
      setMiniMaxMusicBusy(false)
    }
  }, [configForm.minimaxMusicModel, miniMaxLyrics, miniMaxMusicPrompt, refreshAssets, updateForm])

  const checkHealth = useCallback(async () => {
    setServiceBusy(true)

    try {
      const response = await moneyprinterClient.getHealth()

      if (!response.ok || !response.data) {
        notifyError(response.error?.message, 'MoneyPrinter service health check failed')

        return
      }

      setHealth(response.data)

      if (response.data.config) {
        applyConfigSummary(response.data.config, true)
      }

      notify({ kind: 'success', title: 'Video Studio health checked', message: response.data.message || '' })
    } finally {
      setServiceBusy(false)
    }
  }, [applyConfigSummary])

  const startService = useCallback(async () => {
    setServiceBusy(true)

    try {
      const response = await moneyprinterClient.startService()

      if (!response.ok || !response.data) {
        notifyError(response.error?.message, 'Unable to start MoneyPrinter service')

        return
      }

      setHealth(response.data)

      if (response.data.config) {
        applyConfigSummary(response.data.config, true)
      }

      notify({ kind: 'success', title: 'MoneyPrinter service started', message: response.data.message || '' })
    } finally {
      setServiceBusy(false)
    }
  }, [applyConfigSummary])

  const saveConfig = useCallback(async () => {
    setConfigBusy(true)

    try {
      const response = await moneyprinterClient.saveConfig(configForm)

      if (!response.ok || !response.data) {
        notifyError(response.error?.message, 'Unable to save MoneyPrinter config')

        return
      }

      const savedConfig = response.data

      if (!savedConfig) {
        notifyError(null, 'Unable to save MoneyPrinter config')

        return
      }

      setHealth(current => (current ? { ...current, config: savedConfig } : current))
      applyConfigSummary(savedConfig, true)
      notify({
        kind: 'success',
        message: 'Secrets were written to ignored external/MoneyPrinterTurbo/config.toml.',
        title: 'MoneyPrinter config saved'
      })
    } finally {
      setConfigBusy(false)
    }
  }, [applyConfigSummary, configForm])

  const generateScript = useCallback(async () => {
    if (!form.videoSubject.trim()) {
      notifyError(null, '请输入视频主题后再生成文案')

      return
    }

    setScriptBusy(true)

    try {
      const response = await moneyprinterClient.generateScript(form)

      if (!response.ok || !response.data) {
        notifyError(response.error?.message, 'Unable to generate MoneyPrinter script')

        return
      }

      const script = scriptTextFromResult(response.data)

      if (!script) {
        notifyError(null, 'MoneyPrinter returned an empty script')

        return
      }

      updateForm('videoScript', script)
      notify({ kind: 'success', title: '文案已生成', message: '已回填到视频文案，可继续人工修改。' })
    } finally {
      setScriptBusy(false)
    }
  }, [form, updateForm])

  const generateTerms = useCallback(async () => {
    if (!form.videoSubject.trim()) {
      notifyError(null, '请输入视频主题后再生成关键词')

      return
    }

    setTermsBusy(true)

    try {
      const response = await moneyprinterClient.generateTerms(form)

      if (!response.ok || !response.data) {
        notifyError(response.error?.message, 'Unable to generate MoneyPrinter terms')

        return
      }

      const terms = termsTextFromResult(response.data)

      if (!terms) {
        notifyError(null, 'MoneyPrinter returned empty search terms')

        return
      }

      updateForm('videoTerms', terms)
      notify({ kind: 'success', title: '关键词已生成', message: '已回填到关键词列表，可逐行编辑。' })
    } finally {
      setTermsBusy(false)
    }
  }, [form, updateForm])

  const createAudio = useCallback(async () => {
    if (!form.videoScript.trim()) {
      notifyError(null, '请先填写或生成视频文案，再单独生成音频')

      return
    }

    setAudioBusy(true)

    try {
      const response = await moneyprinterClient.createAudio(form)

      if (!response.ok || !response.data) {
        notifyError(response.error?.message, 'Unable to create audio task')

        return
      }

      const createdTask = response.data.task

      setTasks(current => [createdTask, ...current.filter(task => task.id !== createdTask.id)])
      setSelectedTaskId(createdTask.id)
      notify({ kind: 'success', title: 'Audio task created', message: createdTask.id })
    } finally {
      setAudioBusy(false)
    }
  }, [form])

  const createSubtitle = useCallback(async () => {
    if (!form.videoScript.trim()) {
      notifyError(null, '请先填写或生成视频文案，再单独生成字幕')

      return
    }

    setSubtitleBusy(true)

    try {
      const response = await moneyprinterClient.createSubtitle(form)

      if (!response.ok || !response.data) {
        notifyError(response.error?.message, 'Unable to create subtitle task')

        return
      }

      const createdTask = response.data.task

      setTasks(current => [createdTask, ...current.filter(task => task.id !== createdTask.id)])
      setSelectedTaskId(createdTask.id)
      notify({ kind: 'success', title: 'Subtitle task created', message: createdTask.id })
    } finally {
      setSubtitleBusy(false)
    }
  }, [form])

  const createVideo = useCallback(async () => {
    if (!form.videoSubject.trim()) {
      notifyError(null, '请输入视频主题后再生成视频')

      return
    }

    if (!namedLibrary.selectedLibraryId) {
      notifyError(null, '请先选择一个素材库')

      return
    }

    if (form.localMaterials.length === 0) {
      notifyError(null, '请先匹配并人工确认镜头，再创建素材时间线')

      return
    }

    setBusy(true)

    try {
      const response = await moneyprinterClient.createVideo(form)

      if (!response.ok || !response.data) {
        notifyError(response.error?.message, 'Unable to create video task')

        return
      }

      const createdTask = response.data.task

      setTasks(current => [createdTask, ...current.filter(task => task.id !== createdTask.id)])
      setSelectedTaskId(createdTask.id)
      notify({ kind: 'success', title: 'Video task created', message: createdTask.id })
    } finally {
      setBusy(false)
    }
  }, [form, namedLibrary.selectedLibraryId])

  return (
    <section
      {...props}
      className={cn('flex h-full min-h-0 flex-col overflow-hidden bg-(--ui-editor-surface-background)', className)}
    >
      <header className="border-b border-(--ui-stroke-secondary) px-6 py-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-lg font-semibold text-(--ui-text-primary)">
              <Codicon name="device-camera-video" />
              Video Studio
            </div>
            <p className="mt-1 max-w-3xl text-xs leading-5 text-(--ui-text-secondary)">
              Hermes 原生视频生成工作台。对齐原版 MoneyPrinterTurbo 的精细流程：主题 → 文案草稿 → 人工改稿 → 关键词 →
              素材/语音/字幕参数 → 合成 → 预览；页面调用独立 capability API，不进入 Hermes Agent Core。
            </p>
          </div>
          <div className="flex gap-2">
            <Button disabled={serviceBusy} onClick={checkHealth} variant="secondary">
              Health Check
            </Button>
            <Button disabled={serviceBusy} onClick={startService} variant="outline">
              Start Service
            </Button>
          </div>
        </div>
        <div className="mt-4 grid grid-cols-4 gap-2 md:grid-cols-8">
          {PIPELINE_STEPS.map((step, index) => (
            <div
              className="rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) px-3 py-2 text-xs text-(--ui-text-secondary)"
              key={step}
            >
              <span className="mr-2 text-(--ui-text-tertiary)">{index + 1}</span>
              {step}
            </div>
          ))}
        </div>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-0 overflow-y-auto xl:h-full xl:grid-cols-[minmax(22rem,28rem)_minmax(18rem,1fr)_minmax(20rem,32rem)] xl:overflow-hidden">
        <form
          className="min-h-0 overscroll-contain border-b border-(--ui-stroke-secondary) p-5 xl:h-full xl:overflow-y-auto xl:border-r xl:border-b-0"
          onSubmit={event => event.preventDefault()}
        >
          <div className="space-y-5">
            <datalist id="moneyprinter-voices">
              {assets.voices.map(voice => (
                <option key={voice} value={voice} />
              ))}
            </datalist>
            <datalist id="minimax-provider-voices">
              {miniMaxVoices.map(voice => (
                <option key={`${voice.category}:${voice.id}`} value={voice.id}>
                  {voice.name} · {voice.category}
                </option>
              ))}
            </datalist>
            <datalist id="moneyprinter-fonts">
              {assets.fonts.map(font => (
                <option key={font.file} value={font.file} />
              ))}
            </datalist>
            <div className="rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3">
              <div className="mb-3 flex items-center justify-between gap-2">
                <div>
                  <div className="text-xs font-semibold text-(--ui-text-primary)">配置 API Keys</div>
                  <div className="mt-1 text-[0.6875rem] text-(--ui-text-tertiary)">
                    页面会从 config.toml 回填 provider/model/baseUrl；密钥只显示保存状态，不回显明文。
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button disabled={configBusy} onClick={loadConfig} size="xs" type="button" variant="outline">
                    Reload
                  </Button>
                  <Button disabled={configBusy} onClick={saveConfig} size="xs" type="button" variant="secondary">
                    {configBusy ? 'Saving…' : 'Save'}
                  </Button>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>LLM Provider</span>
                  <select
                    className="h-8 w-full rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) px-2 text-xs"
                    onChange={event => updateConfig('llmProvider', event.target.value)}
                    value={configForm.llmProvider}
                  >
                    <option value="openai">OpenAI compatible</option>
                    <option value="deepseek">DeepSeek</option>
                    <option value="gemini">Gemini</option>
                    <option value="qwen">Qwen</option>
                    <option value="grok">Grok / xAI</option>
                  </select>
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>Model</span>
                  <Input
                    onChange={event => updateConfig('modelName', event.target.value)}
                    value={configForm.modelName}
                  />
                </label>
              </div>

              <label className="mt-3 block space-y-2">
                <span className={fieldLabelClass()}>
                  Provider API Key · {savedStateLabel(configSummary?.apiKeyConfigured)}
                </span>
                <Input
                  onChange={event => updateConfig('apiKey', event.target.value)}
                  placeholder={
                    configSummary?.apiKeyConfigured ? '留空则沿用已保存密钥' : 'Paste in UI only; never send it in chat'
                  }
                  type="password"
                  value={configForm.apiKey}
                />
              </label>

              <label className="mt-3 block space-y-2">
                <span className={fieldLabelClass()}>Base URL，可选</span>
                <Input onChange={event => updateConfig('baseUrl', event.target.value)} value={configForm.baseUrl} />
              </label>

              <div className="mt-3 grid grid-cols-2 gap-3">
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>
                    MiniMax API Key · {savedStateLabel(configSummary?.minimax?.apiKeyConfigured)}
                  </span>
                  <Input
                    onChange={event => updateConfig('minimaxApiKey', event.target.value)}
                    placeholder={
                      configSummary?.minimax?.apiKeyConfigured ? '留空则沿用已保存密钥' : 'MiniMax voice and music key'
                    }
                    type="password"
                    value={configForm.minimaxApiKey}
                  />
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>MiniMax API Base URL</span>
                  <Input
                    onChange={event => updateConfig('minimaxBaseUrl', event.target.value)}
                    value={configForm.minimaxBaseUrl}
                  />
                </label>
              </div>

              <div className="mt-3 grid grid-cols-3 gap-3">
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>MiniMax TTS 模型</span>
                  <Input
                    list="minimax-speech-models"
                    onChange={event => updateConfig('minimaxT2aModel', event.target.value)}
                    value={configForm.minimaxT2aModel}
                  />
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>MiniMax 音色复刻模型</span>
                  <Input
                    list="minimax-speech-models"
                    onChange={event => updateConfig('minimaxVoiceCloneModel', event.target.value)}
                    value={configForm.minimaxVoiceCloneModel}
                  />
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>MiniMax 音乐模型</span>
                  <Input
                    list="minimax-music-models"
                    onChange={event => updateConfig('minimaxMusicModel', event.target.value)}
                    value={configForm.minimaxMusicModel}
                  />
                </label>
              </div>
              <datalist id="minimax-speech-models">
                <option value="speech-2.8-hd" />
                <option value="speech-2.8-turbo" />
                <option value="speech-2.6-hd" />
                <option value="speech-2.6-turbo" />
              </datalist>
              <datalist id="minimax-music-models">
                <option value="music-2.6-free" />
                <option value="music-2.6" />
              </datalist>

              <div className="mt-3 grid grid-cols-3 gap-3">
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>
                    Pexels Key · {savedStateLabel(configSummary?.materialProviders.pexels)}
                  </span>
                  <Input
                    onChange={event => updateConfig('pexelsApiKey', event.target.value)}
                    type="password"
                    value={configForm.pexelsApiKey}
                  />
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>
                    Pixabay Key · {savedStateLabel(configSummary?.materialProviders.pixabay)}
                  </span>
                  <Input
                    onChange={event => updateConfig('pixabayApiKey', event.target.value)}
                    type="password"
                    value={configForm.pixabayApiKey}
                  />
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>
                    Coverr Key · {savedStateLabel(configSummary?.materialProviders.coverr)}
                  </span>
                  <Input
                    onChange={event => updateConfig('coverrApiKey', event.target.value)}
                    type="password"
                    value={configForm.coverrApiKey}
                  />
                </label>
              </div>
            </div>

            <div className="rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3">
              <div className="mb-3 flex items-center justify-between gap-2">
                <div>
                  <div className="text-xs font-semibold text-(--ui-text-primary)">1. 主题 → 文案</div>
                  <div className="mt-1 text-[0.6875rem] text-(--ui-text-tertiary)">
                    先生成草稿，再在下方 textarea 手工改到满意；后续合成使用这里的最终文案。
                  </div>
                </div>
                <Button disabled={scriptBusy} onClick={generateScript} size="xs" type="button" variant="secondary">
                  {scriptBusy ? 'Generating…' : '生成文案'}
                </Button>
              </div>

              <label className="block space-y-2">
                <span className={fieldLabelClass()}>视频主题</span>
                <Input
                  onChange={event => updateForm('videoSubject', event.target.value)}
                  placeholder="例如：30 秒介绍 Hermes Agent 的桌面端工作流"
                  value={form.videoSubject}
                />
              </label>

              <div className="mt-3 grid grid-cols-2 gap-3">
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>语言</span>
                  <Input
                    onChange={event => updateForm('videoLanguage', event.target.value)}
                    placeholder="留空自动检测 / zh-CN / en-US"
                    value={form.videoLanguage}
                  />
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>段落数</span>
                  <Input
                    max={10}
                    min={1}
                    onChange={event => updateForm('paragraphNumber', numberValue(event.target.value, 1))}
                    type="number"
                    value={form.paragraphNumber}
                  />
                </label>
              </div>

              <label className="mt-3 block space-y-2">
                <span className={fieldLabelClass()}>文案生成要求（可选）</span>
                <Textarea
                  className="min-h-20"
                  onChange={event => updateForm('videoScriptPrompt', event.target.value)}
                  placeholder="例如：开头 3 秒必须抛痛点；语气像抖音口播；结尾引导关注。"
                  value={form.videoScriptPrompt}
                />
              </label>

              <label className="mt-3 block space-y-2">
                <span className={fieldLabelClass()}>System Prompt（可选）</span>
                <Textarea
                  className="min-h-20"
                  onChange={event => updateForm('customSystemPrompt', event.target.value)}
                  placeholder="高级用户可覆盖 MoneyPrinterTurbo 文案生成 system prompt。"
                  value={form.customSystemPrompt}
                />
              </label>

              <label className="mt-3 block space-y-2">
                <span className={fieldLabelClass()}>视频文案（可编辑）</span>
                <Textarea
                  className="min-h-40"
                  onChange={event => updateForm('videoScript', event.target.value)}
                  placeholder="生成后会回填到这里；也可以直接人工输入最终脚本。"
                  value={form.videoScript}
                />
              </label>
            </div>

            <div className="rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3">
              <div className="mb-3 flex items-center justify-between gap-2">
                <div>
                  <div className="text-xs font-semibold text-(--ui-text-primary)">2. 文案 → 素材关键词</div>
                  <div className="mt-1 text-[0.6875rem] text-(--ui-text-tertiary)">
                    关键词逐行保存；不满意可以自己删改，最终会作为 video_terms 传给 MoneyPrinterTurbo。
                  </div>
                </div>
                <Button disabled={termsBusy} onClick={generateTerms} size="xs" type="button" variant="secondary">
                  {termsBusy ? 'Generating…' : '生成关键词'}
                </Button>
              </div>

              <label className="block space-y-2">
                <span className={fieldLabelClass()}>关键词数量</span>
                <Input
                  min={1}
                  onChange={event => updateForm('searchTermsAmount', numberValue(event.target.value, 5))}
                  type="number"
                  value={form.searchTermsAmount}
                />
              </label>

              <label className="mt-3 block space-y-2">
                <span className={fieldLabelClass()}>素材关键词 / video_terms</span>
                <Textarea
                  className="min-h-28"
                  onChange={event => updateForm('videoTerms', event.target.value)}
                  placeholder="每行一个关键词，例如：AI desktop workflow"
                  value={form.videoTerms}
                />
              </label>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <label className="block space-y-2">
                <span className={fieldLabelClass()}>画面比例</span>
                <select
                  className="h-8 w-full rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) px-2 text-xs"
                  onChange={event =>
                    updateForm('videoAspect', event.target.value as VideoGenerationForm['videoAspect'])
                  }
                  value={form.videoAspect}
                >
                  <option value="9:16">9:16 竖屏</option>
                  <option value="16:9">16:9 横屏</option>
                  <option value="1:1">1:1 方屏</option>
                </select>
              </label>
              <label className="block space-y-2">
                <span className={fieldLabelClass()}>视频数量</span>
                <Input
                  min={1}
                  onChange={event => updateForm('videoCount', numberValue(event.target.value, 1))}
                  type="number"
                  value={form.videoCount}
                />
              </label>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <label className="block space-y-2">
                <span className={fieldLabelClass()}>片段最长秒数</span>
                <Input
                  min={1}
                  onChange={event => updateForm('videoClipDuration', numberValue(event.target.value, 5))}
                  type="number"
                  value={form.videoClipDuration}
                />
              </label>
              <label className="block space-y-2">
                <span className={fieldLabelClass()}>转场</span>
                <select
                  className="h-8 w-full rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) px-2 text-xs"
                  onChange={event =>
                    updateForm('videoTransitionMode', event.target.value as VideoGenerationForm['videoTransitionMode'])
                  }
                  value={form.videoTransitionMode}
                >
                  <option value="">none</option>
                  <option value="Shuffle">Shuffle</option>
                  <option value="FadeIn">FadeIn</option>
                  <option value="FadeOut">FadeOut</option>
                  <option value="SlideIn">SlideIn</option>
                  <option value="SlideOut">SlideOut</option>
                </select>
              </label>
            </div>

            <UnifiedMaterialLibraryPanel
              error={namedLibrary.error}
              libraries={namedLibrary.libraries}
              loadingLibraries={namedLibrary.loadingLibraries}
              loadingLibrary={namedLibrary.loadingLibrary}
              managementBusy={namedLibrary.managementBusy}
              matches={namedLibrary.matches}
              matchingAll={namedLibrary.matchingAll}
              matchingSegmentId={namedLibrary.matchingSegmentId}
              migrationResult={namedLibrary.migrationResult}
              onAddFiles={addNamedLibraryFiles}
              onConfirmClip={namedLibrary.confirmClip}
              onConfirmScan={namedLibrary.confirmScan}
              onCreateTimeline={createNamedLibraryTimeline}
              onMatchAll={namedLibrary.matchAll}
              onMatchSegment={namedLibrary.matchSegment}
              onMigrateLegacy={namedLibrary.migrateLegacyLibrary}
              onPreviewScan={namedLibrary.previewScan}
              onRefresh={namedLibrary.refreshSelectedLibrary}
              onSelectDirectory={addNamedLibraryDirectory}
              onSelectLibrary={selectMaterialLibrary}
              scanBusy={namedLibrary.scanBusy}
              scanPreview={namedLibrary.scanPreview}
              segments={namedLibrary.segments}
              selectedLibraryId={namedLibrary.selectedLibraryId}
              status={namedLibrary.status}
              timelineBusy={namedLibrary.timelineBusy}
            />

            <div className="rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3">
              <div className="mb-3 flex items-center justify-between gap-2">
                <div>
                  <div className="text-xs font-semibold text-(--ui-text-primary)">3. 语音与音频</div>
                  <div className="mt-1 text-[0.6875rem] text-(--ui-text-tertiary)">
                    可选 TTS voice；若指定自定义音频，将跳过 TTS，字幕可用 Whisper 转写。
                  </div>
                </div>
                <Button disabled={assetsBusy} onClick={refreshAssets} size="xs" type="button" variant="secondary">
                  {assetsBusy ? 'Loading…' : 'Refresh assets'}
                </Button>
              </div>

              <label className="block space-y-2">
                <span className={fieldLabelClass()}>语音 / TTS provider voice</span>
                <Input
                  list="moneyprinter-voices"
                  onChange={event => updateForm('voiceName', event.target.value)}
                  value={form.voiceName}
                />
              </label>

              <div className="grid grid-cols-2 gap-3">
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>语速</span>
                  <Input
                    max={3}
                    min={0.2}
                    onChange={event => updateForm('voiceRate', numberValue(event.target.value, 1))}
                    step={0.1}
                    type="number"
                    value={form.voiceRate}
                  />
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>音量</span>
                  <Input
                    max={3}
                    min={0}
                    onChange={event => updateForm('voiceVolume', numberValue(event.target.value, 1))}
                    step={0.1}
                    type="number"
                    value={form.voiceVolume}
                  />
                </label>
              </div>

              <div className="mt-3 space-y-3 border-t border-(--ui-stroke-secondary) pt-3">
                <div>
                  <div className="text-xs font-semibold text-(--ui-text-primary)">使用已有 MiniMax 音色 ID</div>
                  <div className="mt-1 text-[0.6875rem] text-(--ui-text-tertiary)">
                    系统音色或当前账户已有音色直接生成 TTS，不上传复刻音频，也不会产生新的音色复刻费。
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <label className="block space-y-2">
                    <span className={fieldLabelClass()}>Voice ID</span>
                    <Input
                      list="minimax-provider-voices"
                      onChange={event => {
                        setMiniMaxExistingVoiceId(event.target.value)
                        setMiniMaxExistingPreview(null)
                      }}
                      value={miniMaxExistingVoiceId}
                    />
                  </label>
                  <label className="block space-y-2">
                    <span className={fieldLabelClass()}>试听文本</span>
                    <Input
                      onChange={event => setMiniMaxExistingPreviewText(event.target.value)}
                      value={miniMaxExistingPreviewText}
                    />
                  </label>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button
                    disabled={miniMaxExistingBusy || !miniMaxExistingVoiceId.trim() || !miniMaxExistingPreviewText.trim()}
                    onClick={previewExistingMiniMaxVoice}
                    type="button"
                    variant="outline"
                  >
                    {miniMaxExistingBusy ? 'Generating…' : '生成已有音色试听'}
                  </Button>
                  <Button
                    disabled={!miniMaxExistingPreview}
                    onClick={() => updateForm('voiceName', miniMaxVoiceName(miniMaxExistingVoiceId))}
                    type="button"
                    variant="secondary"
                  >
                    选择用于视频
                  </Button>
                </div>
                {miniMaxExistingPreview?.audio?.streamUrl ? (
                  <audio
                    className="w-full"
                    controls
                    src={resolveMoneyPrinterMediaUrl(miniMaxExistingPreview.audio.streamUrl, health?.apiBaseUrl)}
                  />
                ) : null}
              </div>

              <div className="mt-3 space-y-3 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) p-3">
                <div>
                  <div className="text-xs font-semibold text-(--ui-text-primary)">上传自己的声音进行复刻</div>
                  <div className="mt-1 text-[0.6875rem] text-(--ui-text-tertiary)">
                    克隆试听不会激活音色；只有点击“正式激活并选择”并确认后，才可能收取一次性约 ¥9.9 复刻费。
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <label className="block space-y-2">
                    <span className={fieldLabelClass()}>新的克隆 Voice ID</span>
                    <Input
                      onChange={event => {
                        setMiniMaxVoiceId(event.target.value)
                        setMiniMaxClonePreview(null)
                      }}
                      value={miniMaxVoiceId}
                    />
                  </label>
                  <label className="block space-y-2">
                    <span className={fieldLabelClass()}>复刻音频（10 秒–5 分钟，≤20 MB）</span>
                    <Input
                      accept=".m4a,.mp3,.wav,audio/*"
                      onChange={event => {
                        setMiniMaxCloneFile(event.target.files?.[0] || null)
                        setMiniMaxClonePreview(null)
                      }}
                      type="file"
                    />
                  </label>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <label className="block space-y-2">
                    <span className={fieldLabelClass()}>参考音频（可选）</span>
                    <Input
                      accept=".m4a,.mp3,.wav,audio/*"
                      onChange={event => setMiniMaxPromptFile(event.target.files?.[0] || null)}
                      type="file"
                    />
                  </label>
                  <label className="block space-y-2">
                    <span className={fieldLabelClass()}>参考音频文本</span>
                    <Input
                      disabled={!miniMaxPromptFile}
                      onChange={event => setMiniMaxPromptText(event.target.value)}
                      value={miniMaxPromptText}
                    />
                  </label>
                </div>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>克隆试听文本（不会激活音色）</span>
                  <Textarea onChange={event => setMiniMaxTrialText(event.target.value)} rows={2} value={miniMaxTrialText} />
                </label>
                <div className="flex flex-wrap gap-2">
                  <Button
                    disabled={miniMaxCloneBusy || !miniMaxCloneFile || !miniMaxTrialText.trim()}
                    onClick={cloneMiniMaxVoice}
                    type="button"
                    variant="outline"
                  >
                    {miniMaxCloneBusy ? 'Cloning…' : '创建克隆试听（不激活）'}
                  </Button>
                  <Button
                    disabled={!miniMaxClonePreview?.trialAudio || miniMaxActivateBusy || miniMaxClonePreview.activated}
                    onClick={activateClonedMiniMaxVoice}
                    type="button"
                    variant="secondary"
                  >
                    {miniMaxActivateBusy
                      ? 'Activating…'
                      : miniMaxClonePreview?.activated
                        ? '已激活并选用'
                        : '正式激活并选择（约 ¥9.9）'}
                  </Button>
                </div>
                {miniMaxClonePreview?.trialAudio?.streamUrl ? (
                  <audio
                    className="w-full"
                    controls
                    src={resolveMoneyPrinterMediaUrl(miniMaxClonePreview.trialAudio.streamUrl, health?.apiBaseUrl)}
                  />
                ) : null}
                {miniMaxClonePreview?.previewError ? (
                  <div className="text-[0.6875rem] text-(--ui-text-danger)">{miniMaxClonePreview.previewError}</div>
                ) : null}
              </div>
            </div>

            <div className="rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3">
              <div className="mb-3 text-xs font-semibold text-(--ui-text-primary)">字幕、BGM 与合成细节</div>
              <div className="grid grid-cols-2 gap-3">
                <label className="flex items-center gap-2 text-xs text-(--ui-text-secondary)">
                  <input
                    checked={form.subtitleEnabled}
                    onChange={event => updateForm('subtitleEnabled', event.target.checked)}
                    type="checkbox"
                  />
                  启用字幕
                </label>
                <label className="flex items-center gap-2 text-xs text-(--ui-text-secondary)">
                  <input
                    checked={form.roundedSubtitleBackground}
                    onChange={event => updateForm('roundedSubtitleBackground', event.target.checked)}
                    type="checkbox"
                  />
                  圆角字幕背景
                </label>
              </div>

              <div className="mt-3 grid grid-cols-2 gap-3">
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>字幕位置</span>
                  <select
                    className="h-8 w-full rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) px-2 text-xs"
                    onChange={event =>
                      updateForm('subtitlePosition', event.target.value as VideoGenerationForm['subtitlePosition'])
                    }
                    value={form.subtitlePosition}
                  >
                    <option value="bottom">bottom</option>
                    <option value="top">top</option>
                    <option value="center">center</option>
                    <option value="custom">custom</option>
                  </select>
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>自定义位置 %</span>
                  <Input
                    max={100}
                    min={0}
                    onChange={event => updateForm('customPosition', numberValue(event.target.value, 70))}
                    type="number"
                    value={form.customPosition}
                  />
                </label>
              </div>

              <div className="mt-3 grid grid-cols-2 gap-3">
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>字体</span>
                  <Input
                    list="moneyprinter-fonts"
                    onChange={event => updateForm('fontName', event.target.value)}
                    value={form.fontName}
                  />
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>字号</span>
                  <Input
                    min={1}
                    onChange={event => updateForm('fontSize', numberValue(event.target.value, 60))}
                    type="number"
                    value={form.fontSize}
                  />
                </label>
              </div>

              <div className="mt-3 grid grid-cols-2 gap-3">
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>文字色</span>
                  <Input
                    onChange={event => updateForm('textForeColor', event.target.value)}
                    value={form.textForeColor}
                  />
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>背景色 / false / true</span>
                  <Input
                    onChange={event => updateForm('textBackgroundColor', event.target.value)}
                    placeholder="留空使用默认；可填 #000000 或 false"
                    value={form.textBackgroundColor}
                  />
                </label>
              </div>

              <div className="mt-3 grid grid-cols-2 gap-3">
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>描边色</span>
                  <Input onChange={event => updateForm('strokeColor', event.target.value)} value={form.strokeColor} />
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>描边宽度</span>
                  <Input
                    min={0}
                    onChange={event => updateForm('strokeWidth', numberValue(event.target.value, 1.5))}
                    step={0.1}
                    type="number"
                    value={form.strokeWidth}
                  />
                </label>
              </div>

              <div className="mt-3 grid grid-cols-2 gap-3">
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>BGM 类型</span>
                  <select
                    className="h-8 w-full rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) px-2 text-xs"
                    onChange={event => updateForm('bgmType', event.target.value)}
                    value={form.bgmType}
                  >
                    <option value="random">random 随机</option>
                    <option value="custom">custom 指定文件</option>
                    <option value="">none 关闭</option>
                  </select>
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>BGM 音量</span>
                  <Input
                    max={1}
                    min={0}
                    onChange={event => updateForm('bgmVolume', numberValue(event.target.value, 0.2))}
                    step={0.05}
                    type="number"
                    value={form.bgmVolume}
                  />
                </label>
              </div>

              <div className="mt-3 grid grid-cols-[minmax(0,1fr)_auto] gap-3">
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>指定 BGM 文件</span>
                  <select
                    className="h-8 w-full rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) px-2 text-xs"
                    disabled={form.bgmType === ''}
                    onChange={event => updateForm('bgmFile', event.target.value)}
                    value={form.bgmFile}
                  >
                    <option value="">
                      {form.bgmType === 'custom' ? '选择 resource/songs/*.mp3' : '留空使用 random'}
                    </option>
                    {assets.bgms.map(bgm => (
                      <option key={bgm.file} value={bgm.file}>
                        {bgm.name} · {formatBytes(bgm.size)}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="mt-5 inline-flex h-8 cursor-pointer items-center rounded border border-(--ui-stroke-secondary) px-2 text-xs text-(--ui-text-primary) hover:bg-(--chrome-action-hover)">
                  {bgmUploadBusy ? 'Uploading…' : 'Upload MP3'}
                  <input
                    accept=".mp3,audio/mpeg"
                    className="hidden"
                    disabled={bgmUploadBusy}
                    multiple
                    onChange={uploadBgmFiles}
                    type="file"
                  />
                </label>
              </div>

              <div className="mt-3 space-y-3 border-t border-(--ui-stroke-secondary) pt-3">
                <div className="text-xs font-semibold text-(--ui-text-primary)">MiniMax 音乐生成</div>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>音乐风格 / 场景提示词</span>
                  <Input onChange={event => setMiniMaxMusicPrompt(event.target.value)} value={miniMaxMusicPrompt} />
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>歌词（留空生成纯音乐）</span>
                  <Textarea
                    onChange={event => setMiniMaxLyrics(event.target.value)}
                    placeholder="可先点击生成歌词；填写歌词后将生成带人声音乐"
                    rows={4}
                    value={miniMaxLyrics}
                  />
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <Button disabled={miniMaxLyricsBusy || !miniMaxMusicPrompt.trim()} onClick={generateMiniMaxLyrics} type="button" variant="outline">
                    {miniMaxLyricsBusy ? 'Generating…' : '生成 / 润色歌词'}
                  </Button>
                  <Button disabled={miniMaxMusicBusy || !miniMaxMusicPrompt.trim()} onClick={generateMiniMaxMusic} type="button" variant="outline">
                    {miniMaxMusicBusy ? 'Generating…' : '生成并选为 BGM'}
                  </Button>
                </div>
              </div>

              <div className="mt-3 grid grid-cols-[minmax(0,1fr)_auto] gap-3">
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>自定义音频（可选）</span>
                  <select
                    className="h-8 w-full rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) px-2 text-xs"
                    onChange={event => updateForm('customAudioFile', event.target.value)}
                    value={form.customAudioFile}
                  >
                    <option value="">使用 TTS 生成配音</option>
                    {assets.customAudio.map(audio => (
                      <option key={audio.file} value={audio.file}>
                        {audio.name} · {formatBytes(audio.size)}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="mt-5 inline-flex h-8 cursor-pointer items-center rounded border border-(--ui-stroke-secondary) px-2 text-xs text-(--ui-text-primary) hover:bg-(--chrome-action-hover)">
                  {customAudioUploadBusy ? 'Uploading…' : 'Upload audio'}
                  <input
                    accept=".aac,.flac,.m4a,.mp3,.ogg,.wav,audio/*"
                    className="hidden"
                    disabled={customAudioUploadBusy}
                    multiple
                    onChange={uploadCustomAudioFiles}
                    type="file"
                  />
                </label>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2">
              <Button disabled={audioBusy} onClick={createAudio} type="button" variant="outline">
                {audioBusy ? 'Creating…' : 'Generate Audio'}
              </Button>
              <Button disabled={subtitleBusy} onClick={createSubtitle} type="button" variant="outline">
                {subtitleBusy ? 'Creating…' : 'Generate Subtitle'}
              </Button>
              <Button disabled={busy} onClick={createVideo} type="button">
                {busy ? 'Creating…' : 'Generate Video'}
              </Button>
            </div>
          </div>
        </form>

        <section className="min-h-72 overscroll-contain border-b border-(--ui-stroke-secondary) p-5 xl:h-full xl:min-h-0 xl:overflow-y-auto xl:border-r xl:border-b-0">
          <div className="mb-3 flex items-center justify-between gap-2">
            <h2 className="text-sm font-semibold text-(--ui-text-primary)">任务列表</h2>
            <Button onClick={refreshTasks} size="xs" variant="secondary">
              Refresh
            </Button>
          </div>
          <div className="space-y-2">
            {tasks.length === 0 ? (
              <div className="rounded-lg border border-dashed border-(--ui-stroke-secondary) p-6 text-center text-xs text-(--ui-text-tertiary)">
                暂无任务。生成视频后，这里会显示 task id、状态、进度和输出。
              </div>
            ) : (
              tasks.map(task => (
                <button
                  className={cn(
                    'block w-full rounded-lg border p-3 text-left text-xs transition-colors',
                    selectedTask?.id === task.id
                      ? 'border-primary bg-primary/10 text-(--ui-text-primary)'
                      : 'border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) text-(--ui-text-secondary) hover:bg-(--chrome-action-hover)'
                  )}
                  key={task.id}
                  onClick={() => setSelectedTaskId(task.id)}
                  type="button"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono">{task.id}</span>
                    <span>{task.state}</span>
                  </div>
                  <div className="mt-2 h-1.5 overflow-hidden rounded bg-(--ui-bg-tertiary)">
                    <div
                      className="h-full bg-primary"
                      style={{ width: `${Math.max(0, Math.min(100, task.progress))}%` }}
                    />
                  </div>
                  {task.subject && <div className="mt-2 line-clamp-2">{task.subject}</div>}
                </button>
              ))
            )}
          </div>
        </section>

        <aside className="min-h-72 overscroll-contain p-5 xl:h-full xl:min-h-0 xl:overflow-y-auto">
          <h2 className="mb-3 text-sm font-semibold text-(--ui-text-primary)">预览</h2>
          {health && (
            <div className="mb-3 rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3 text-xs text-(--ui-text-secondary)">
              <div>Service: {health.serviceRunning ? 'running' : 'stopped'}</div>
              <div>Installed: {health.installed ? 'yes' : 'no'}</div>
              <div>Runtime: {health.runtimeReady ? 'ready' : 'not ready'}</div>
              {health.runtimePython && <div className="truncate">Python: {health.runtimePython}</div>}
              {health.missingDependencies && health.missingDependencies.length > 0 && (
                <div className="text-destructive">Missing: {health.missingDependencies.join(', ')}</div>
              )}
              {health.ffmpegPath && <div className="truncate">FFmpeg: {health.ffmpegPath}</div>}
              {health.upstreamCommit && <div>Upstream: {health.upstreamCommit}</div>}
            </div>
          )}

          {!selectedTask ? (
            <div className="rounded-lg border border-dashed border-(--ui-stroke-secondary) p-6 text-center text-xs text-(--ui-text-tertiary)">
              选择一个任务后预览脚本、字幕和生成的视频输出。
            </div>
          ) : (
            <div className="space-y-3">
              <div className="rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3 text-xs">
                <div className="font-mono text-(--ui-text-primary)">{selectedTask.id}</div>
                <div className="mt-1 text-(--ui-text-secondary)">状态：{selectedTask.state}</div>
                {selectedTask.error && <div className="mt-1 text-destructive">{selectedTask.error}</div>}
              </div>

              {selectedTask.script && (
                <pre className="max-h-48 overflow-auto rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3 text-xs whitespace-pre-wrap text-(--ui-text-secondary)">
                  {selectedTask.script}
                </pre>
              )}

              {selectedTaskVideos.length === 0 ? (
                <div className="rounded-lg border border-dashed border-(--ui-stroke-secondary) p-6 text-center text-xs text-(--ui-text-tertiary)">
                  视频合成完成后，这里会出现预览播放器和下载入口。
                </div>
              ) : (
                selectedTaskVideos.map(video => {
                  const streamUrl = resolveMoneyPrinterMediaUrl(video.streamUrl, health?.apiBaseUrl)
                  const downloadUrl = resolveMoneyPrinterMediaUrl(video.downloadUrl, health?.apiBaseUrl)

                  return (
                    <div
                      className="rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3"
                      key={video.name}
                    >
                      <div className="mb-2 text-xs font-medium text-(--ui-text-primary)">{video.name}</div>
                      {streamUrl ? <video className="w-full rounded bg-black" controls src={streamUrl} /> : null}
                      {downloadUrl ? (
                        <Button asChild className="mt-3" size="xs" variant="secondary">
                          <a href={downloadUrl}>Download</a>
                        </Button>
                      ) : null}
                    </div>
                  )
                })
              )}
            </div>
          )}
        </aside>
      </div>
    </section>
  )
}
