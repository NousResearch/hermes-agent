import type * as React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { readKey, writeKey } from '@/lib/storage'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

import {
  configFormFromSummary,
  defaultMoneyPrinterConfigForm,
  defaultVideoGenerationForm,
  isMoneyPrinterPreviewVideo,
  type MoneyPrinterAssets,
  moneyprinterClient,
  type MoneyPrinterConfigInput,
  type MoneyPrinterConfigSummary,
  type MoneyPrinterHealth,
  type MoneyPrinterLocalMaterial,
  type MoneyPrinterTask,
  resolveMoneyPrinterMediaUrl,
  scriptTextFromResult,
  termsTextFromResult,
  type VideoGenerationForm,
  type VideoSource
} from './moneyprinter-client'

interface VideoStudioViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

const SOURCE_OPTIONS: { label: string; value: VideoSource }[] = [
  { label: 'Pexels', value: 'pexels' },
  { label: 'Pixabay', value: 'pixabay' },
  { label: 'Coverr', value: 'coverr' },
  { label: 'Local materials', value: 'local' }
]

const PIPELINE_STEPS = [
  '主题',
  '文案',
  '关键词',
  '语音',
  '素材',
  '字幕',
  '合成',
  '预览'
]

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

export function VideoStudioView({ className, setStatusbarItemGroup: _setStatusbarItemGroup, ...props }: VideoStudioViewProps) {
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
  const [localMaterials, setLocalMaterials] = useState<MoneyPrinterLocalMaterial[]>([])
  const [localMaterialsBusy, setLocalMaterialsBusy] = useState(false)
  const [assets, setAssets] = useState<MoneyPrinterAssets>(EMPTY_ASSETS)
  const [assetsBusy, setAssetsBusy] = useState(false)
  const [uploadBusy, setUploadBusy] = useState(false)
  const [bgmUploadBusy, setBgmUploadBusy] = useState(false)
  const [customAudioUploadBusy, setCustomAudioUploadBusy] = useState(false)
  const [serviceBusy, setServiceBusy] = useState(false)

  const selectedTask = useMemo(
    () => tasks.find(task => task.id === selectedTaskId) ?? tasks[0] ?? null,
    [selectedTaskId, tasks]
  )

  const selectedTaskVideos = useMemo(() => selectedTask?.videos.filter(isMoneyPrinterPreviewVideo) ?? [], [selectedTask])

  useEffect(() => {
    writeKey(VIDEO_STUDIO_DRAFT_KEY, JSON.stringify(form))
  }, [form])

  const updateForm = useCallback(<K extends keyof VideoGenerationForm>(key: K, value: VideoGenerationForm[K]) => {
    setForm(current => ({ ...current, [key]: value }))
  }, [])

  const updateConfig = useCallback(<K extends keyof MoneyPrinterConfigInput>(key: K, value: MoneyPrinterConfigInput[K]) => {
    setConfigForm(current => ({ ...current, [key]: value }))
  }, [])

  const applyConfigSummary = useCallback((summary: MoneyPrinterConfigSummary, clearSecrets = false) => {
    const visibleConfig = configFormFromSummary(summary)

    setConfigSummary(summary)
    setConfigForm(current => ({
      ...visibleConfig,
      apiKey: clearSecrets ? '' : current.apiKey,
      coverrApiKey: clearSecrets ? '' : current.coverrApiKey,
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

  const refreshLocalMaterials = useCallback(async () => {
    setLocalMaterialsBusy(true)

    try {
      const response = await moneyprinterClient.listLocalMaterials()

      if (!response.ok || !response.data) {
        notifyError(response.error?.message, 'Unable to load local materials')

        return
      }

      setLocalMaterials(response.data.materials)
    } finally {
      setLocalMaterialsBusy(false)
    }
  }, [])

  const refreshAssets = useCallback(async () => {
    setAssetsBusy(true)

    try {
      const response = await moneyprinterClient.listAssets()

      if (!response.ok || !response.data) {
        notifyError(response.error?.message, 'Unable to load MoneyPrinter assets')

        return
      }

      setAssets(response.data)
    } finally {
      setAssetsBusy(false)
    }
  }, [])

  useEffect(() => {
    void loadConfig()
    void refreshTasks()
    void refreshLocalMaterials()
    void refreshAssets()
  }, [loadConfig, refreshAssets, refreshLocalMaterials, refreshTasks])

  const toggleLocalMaterial = useCallback((file: string, checked: boolean) => {
    setForm(current => {
      const selected = new Set(current.localMaterials)

      if (checked) {
        selected.add(file)
      } else {
        selected.delete(file)
      }

      return { ...current, localMaterials: Array.from(selected) }
    })
  }, [])

  const uploadLocalMaterials = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget
    const files = Array.from(input.files || [])

    if (files.length === 0) {
      return
    }

    setUploadBusy(true)

    try {
      const uploadedFiles: string[] = []

      for (const file of files) {
        let sourcePath = ''

        try {
          sourcePath = window.hermesDesktop?.getPathForFile?.(file) || ''
        } catch {
          sourcePath = ''
        }

        const response = await moneyprinterClient.uploadLocalMaterial({
          ...(sourcePath ? { sourcePath } : { contentBase64: await fileToDataUrl(file) }),
          filename: file.name
        })

        if (!response.ok || !response.data) {
          notifyError(response.error?.message, `Unable to upload ${file.name}`)

          continue
        }

        const material = response.data.material
        uploadedFiles.push(material.file)
        setLocalMaterials(current => [material, ...current.filter(item => item.file !== material.file)])
      }

      if (uploadedFiles.length > 0) {
        setForm(current => ({
          ...current,
          localMaterials: Array.from(new Set([...current.localMaterials, ...uploadedFiles])),
          videoSource: 'local'
        }))
        notify({ kind: 'success', title: 'Local materials uploaded', message: uploadedFiles.join(', ') })
      }
    } finally {
      setUploadBusy(false)
      input.value = ''
    }
  }, [])

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

    if (form.videoSource === 'local' && form.localMaterials.length === 0) {
      notifyError(null, '请选择或上传至少一个本地素材后再生成视频')

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
  }, [form])

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
              Hermes 原生视频生成工作台。对齐原版 MoneyPrinterTurbo 的精细流程：主题 → 文案草稿 → 人工改稿 → 关键词 → 素材/语音/字幕参数 → 合成 → 预览；页面调用独立 capability API，不进入 Hermes Agent Core。
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
        <div className="mt-4 grid gap-2 sm:grid-cols-8">
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

      <div className="grid h-full min-h-0 flex-1 gap-0 overflow-hidden lg:grid-cols-[minmax(22rem,28rem)_minmax(18rem,1fr)_minmax(20rem,32rem)]">
        <form
          className="h-full min-h-0 overflow-y-auto overscroll-contain border-r border-(--ui-stroke-secondary) p-5"
          onSubmit={event => event.preventDefault()}
        >
          <div className="space-y-5">
            <datalist id="moneyprinter-voices">
              {assets.voices.map(voice => (
                <option key={voice} value={voice} />
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
                  <Input onChange={event => updateConfig('modelName', event.target.value)} value={configForm.modelName} />
                </label>
              </div>

              <label className="mt-3 block space-y-2">
                <span className={fieldLabelClass()}>Provider API Key · {savedStateLabel(configSummary?.apiKeyConfigured)}</span>
                <Input
                  onChange={event => updateConfig('apiKey', event.target.value)}
                  placeholder={configSummary?.apiKeyConfigured ? '留空则沿用已保存密钥' : 'Paste in UI only; never send it in chat'}
                  type="password"
                  value={configForm.apiKey}
                />
              </label>

              <label className="mt-3 block space-y-2">
                <span className={fieldLabelClass()}>Base URL，可选</span>
                <Input onChange={event => updateConfig('baseUrl', event.target.value)} value={configForm.baseUrl} />
              </label>

              <div className="mt-3 grid grid-cols-3 gap-3">
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>Pexels Key · {savedStateLabel(configSummary?.materialProviders.pexels)}</span>
                  <Input onChange={event => updateConfig('pexelsApiKey', event.target.value)} type="password" value={configForm.pexelsApiKey} />
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>Pixabay Key · {savedStateLabel(configSummary?.materialProviders.pixabay)}</span>
                  <Input onChange={event => updateConfig('pixabayApiKey', event.target.value)} type="password" value={configForm.pixabayApiKey} />
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>Coverr Key · {savedStateLabel(configSummary?.materialProviders.coverr)}</span>
                  <Input onChange={event => updateConfig('coverrApiKey', event.target.value)} type="password" value={configForm.coverrApiKey} />
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

              <div className="grid grid-cols-2 gap-3">
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>关键词数量</span>
                  <Input
                    min={1}
                    onChange={event => updateForm('searchTermsAmount', numberValue(event.target.value, 5))}
                    type="number"
                    value={form.searchTermsAmount}
                  />
                </label>
                <label className="flex items-end gap-2 text-xs text-(--ui-text-secondary)">
                  <input
                    checked={form.matchMaterialsToScript}
                    onChange={event => updateForm('matchMaterialsToScript', event.target.checked)}
                    type="checkbox"
                  />
                  按文案顺序匹配素材
                </label>
              </div>

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
                  onChange={event => updateForm('videoAspect', event.target.value as VideoGenerationForm['videoAspect'])}
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
                <span className={fieldLabelClass()}>素材来源</span>
                <select
                  className="h-8 w-full rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) px-2 text-xs"
                  onChange={event => {
                    const source = event.target.value as VideoSource

                    updateForm('videoSource', source)

                    if (source === 'local') {
                      void refreshLocalMaterials()
                    }
                  }}
                  value={form.videoSource}
                >
                  {SOURCE_OPTIONS.map(option => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block space-y-2">
                <span className={fieldLabelClass()}>片段秒数</span>
                <Input
                  min={1}
                  onChange={event => updateForm('videoClipDuration', numberValue(event.target.value, 5))}
                  type="number"
                  value={form.videoClipDuration}
                />
              </label>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <label className="block space-y-2">
                <span className={fieldLabelClass()}>拼接模式</span>
                <select
                  className="h-8 w-full rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) px-2 text-xs"
                  onChange={event => updateForm('videoConcatMode', event.target.value as VideoGenerationForm['videoConcatMode'])}
                  value={form.videoConcatMode}
                >
                  <option value="random">random 随机素材</option>
                  <option value="sequential">sequential 顺序素材</option>
                </select>
              </label>
              <label className="block space-y-2">
                <span className={fieldLabelClass()}>转场</span>
                <select
                  className="h-8 w-full rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) px-2 text-xs"
                  onChange={event => updateForm('videoTransitionMode', event.target.value as VideoGenerationForm['videoTransitionMode'])}
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

            {form.videoSource === 'local' && (
              <div className="rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3">
                <div className="mb-3 flex items-center justify-between gap-2">
                  <div>
                    <div className="text-xs font-semibold text-(--ui-text-primary)">本地素材</div>
                    <div className="mt-1 text-[0.6875rem] text-(--ui-text-tertiary)">
                      上传后会保存到 MoneyPrinterTurbo 的 storage/local_videos 白名单目录。
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button disabled={localMaterialsBusy} onClick={refreshLocalMaterials} size="xs" type="button" variant="secondary">
                      {localMaterialsBusy ? 'Loading…' : 'Refresh'}
                    </Button>
                    <label className="inline-flex h-7 cursor-pointer items-center rounded border border-(--ui-stroke-secondary) px-2 text-xs text-(--ui-text-primary) hover:bg-(--chrome-action-hover)">
                      {uploadBusy ? 'Uploading…' : 'Upload'}
                      <input
                        accept=".mp4,.mov,.avi,.flv,.mkv,.jpg,.jpeg,.png,video/*,image/*"
                        className="hidden"
                        disabled={uploadBusy}
                        multiple
                        onChange={uploadLocalMaterials}
                        type="file"
                      />
                    </label>
                  </div>
                </div>

                {localMaterials.length === 0 ? (
                  <div className="rounded border border-dashed border-(--ui-stroke-secondary) p-3 text-xs text-(--ui-text-tertiary)">
                    还没有本地素材。上传 mp4/mov/mkv 或 jpg/png 后，勾选素材即可用于生成。
                  </div>
                ) : (
                  <div className="max-h-44 space-y-2 overflow-auto pr-1">
                    {localMaterials.map(material => (
                      <label
                        className="flex items-center gap-2 rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) px-2 py-2 text-xs text-(--ui-text-secondary)"
                        key={material.file}
                      >
                        <input
                          checked={form.localMaterials.includes(material.file)}
                          onChange={event => toggleLocalMaterial(material.file, event.target.checked)}
                          type="checkbox"
                        />
                        <span className="min-w-0 flex-1 truncate text-(--ui-text-primary)">{material.name}</span>
                        <span className="shrink-0 text-(--ui-text-tertiary)">{material.kind}</span>
                        <span className="shrink-0 text-(--ui-text-tertiary)">{formatBytes(material.size)}</span>
                      </label>
                    ))}
                  </div>
                )}
              </div>
            )}

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
                <Input list="moneyprinter-voices" onChange={event => updateForm('voiceName', event.target.value)} value={form.voiceName} />
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
                    onChange={event => updateForm('subtitlePosition', event.target.value as VideoGenerationForm['subtitlePosition'])}
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
                  <Input list="moneyprinter-fonts" onChange={event => updateForm('fontName', event.target.value)} value={form.fontName} />
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
                  <Input onChange={event => updateForm('textForeColor', event.target.value)} value={form.textForeColor} />
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
                    <option value="">{form.bgmType === 'custom' ? '选择 resource/songs/*.mp3' : '留空使用 random'}</option>
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

        <section className="h-full min-h-0 overflow-y-auto overscroll-contain border-r border-(--ui-stroke-secondary) p-5">
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
                    <div className="h-full bg-primary" style={{ width: `${Math.max(0, Math.min(100, task.progress))}%` }} />
                  </div>
                  {task.subject && <div className="mt-2 line-clamp-2">{task.subject}</div>}
                </button>
              ))
            )}
          </div>
        </section>

        <aside className="h-full min-h-0 overflow-y-auto overscroll-contain p-5">
          <h2 className="mb-3 text-sm font-semibold text-(--ui-text-primary)">预览</h2>
          {health && (
            <div className="mb-3 rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3 text-xs text-(--ui-text-secondary)">
              <div>Service: {health.serviceRunning ? 'running' : 'stopped'}</div>
              <div>Installed: {health.installed ? 'yes' : 'no'}</div>
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
                    <div className="rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3" key={video.name}>
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
