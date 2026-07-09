import type * as React from 'react'
import { useCallback, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

import {
  defaultVideoGenerationForm,
  isMoneyPrinterPreviewVideo,
  moneyprinterClient,
  type MoneyPrinterConfigInput,
  type MoneyPrinterHealth,
  type MoneyPrinterLocalMaterial,
  type MoneyPrinterTask,
  resolveMoneyPrinterMediaUrl,
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
  '语音',
  '素材',
  '字幕',
  '合成',
  '预览'
]

const defaultConfigForm: MoneyPrinterConfigInput = {
  apiKey: '',
  baseUrl: '',
  coverrApiKey: '',
  llmProvider: 'openai',
  modelName: 'gpt-4o-mini',
  pexelsApiKey: '',
  pixabayApiKey: ''
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

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error || new Error('Unable to read local material file'))
    reader.onload = () => resolve(String(reader.result || ''))
    reader.readAsDataURL(file)
  })
}

export function VideoStudioView({ className, setStatusbarItemGroup: _setStatusbarItemGroup, ...props }: VideoStudioViewProps) {
  const [form, setForm] = useState<VideoGenerationForm>(defaultVideoGenerationForm)
  const [configForm, setConfigForm] = useState<MoneyPrinterConfigInput>(defaultConfigForm)
  const [health, setHealth] = useState<MoneyPrinterHealth | null>(null)
  const [tasks, setTasks] = useState<MoneyPrinterTask[]>([])
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [configBusy, setConfigBusy] = useState(false)
  const [localMaterials, setLocalMaterials] = useState<MoneyPrinterLocalMaterial[]>([])
  const [localMaterialsBusy, setLocalMaterialsBusy] = useState(false)
  const [uploadBusy, setUploadBusy] = useState(false)
  const [serviceBusy, setServiceBusy] = useState(false)

  const selectedTask = useMemo(
    () => tasks.find(task => task.id === selectedTaskId) ?? tasks[0] ?? null,
    [selectedTaskId, tasks]
  )

  const selectedTaskVideos = useMemo(() => selectedTask?.videos.filter(isMoneyPrinterPreviewVideo) ?? [], [selectedTask])

  const updateForm = useCallback(<K extends keyof VideoGenerationForm>(key: K, value: VideoGenerationForm[K]) => {
    setForm(current => ({ ...current, [key]: value }))
  }, [])

  const updateConfig = useCallback(<K extends keyof MoneyPrinterConfigInput>(key: K, value: MoneyPrinterConfigInput[K]) => {
    setConfigForm(current => ({ ...current, [key]: value }))
  }, [])

  const refreshTasks = useCallback(async () => {
    const response = await moneyprinterClient.listTasks()

    if (!response.ok || !response.data) {
      notifyError(response.error?.message, 'Unable to load MoneyPrinter tasks')

      return
    }

    setTasks(response.data.tasks)

    if (!selectedTaskId && response.data.tasks[0]) {
      setSelectedTaskId(response.data.tasks[0].id)
    }
  }, [selectedTaskId])

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

  const checkHealth = useCallback(async () => {
    setServiceBusy(true)

    try {
      const response = await moneyprinterClient.getHealth()

      if (!response.ok || !response.data) {
        notifyError(response.error?.message, 'MoneyPrinter service health check failed')

        return
      }

      setHealth(response.data)
      notify({ kind: 'success', title: 'Video Studio health checked', message: response.data.message || '' })
    } finally {
      setServiceBusy(false)
    }
  }, [])

  const startService = useCallback(async () => {
    setServiceBusy(true)

    try {
      const response = await moneyprinterClient.startService()

      if (!response.ok || !response.data) {
        notifyError(response.error?.message, 'Unable to start MoneyPrinter service')

        return
      }

      setHealth(response.data)
      notify({ kind: 'success', title: 'MoneyPrinter service started', message: response.data.message || '' })
    } finally {
      setServiceBusy(false)
    }
  }, [])

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
      setConfigForm(current => ({ ...current, apiKey: '', coverrApiKey: '', pexelsApiKey: '', pixabayApiKey: '' }))
      notify({
        kind: 'success',
        message: 'Secrets were written to ignored external/MoneyPrinterTurbo/config.toml.',
        title: 'MoneyPrinter config saved'
      })
    } finally {
      setConfigBusy(false)
    }
  }, [configForm])

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
              Hermes 原生视频生成工作台。首版复刻 MoneyPrinterTurbo 高频流程：主题 → 文案 → 语音 → 素材 → 字幕 → 合成 → 预览；页面调用独立 capability API，不进入 Hermes Agent Core。
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
        <div className="mt-4 grid gap-2 sm:grid-cols-7">
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

      <div className="grid min-h-0 flex-1 gap-0 overflow-hidden lg:grid-cols-[minmax(22rem,28rem)_minmax(18rem,1fr)_minmax(20rem,32rem)]">
        <form className="min-h-0 overflow-y-auto border-r border-(--ui-stroke-secondary) p-5" onSubmit={event => event.preventDefault()}>
          <div className="space-y-5">
            <div className="rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3">
              <div className="mb-3 flex items-center justify-between gap-2">
                <div>
                  <div className="text-xs font-semibold text-(--ui-text-primary)">配置 API Keys</div>
                  <div className="mt-1 text-[0.6875rem] text-(--ui-text-tertiary)">
                    密钥只写入已忽略的本地 config.toml，不进入聊天或 git。
                  </div>
                </div>
                <Button disabled={configBusy} onClick={saveConfig} size="xs" type="button" variant="secondary">
                  {configBusy ? 'Saving…' : 'Save'}
                </Button>
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
                <span className={fieldLabelClass()}>Provider API Key</span>
                <Input
                  onChange={event => updateConfig('apiKey', event.target.value)}
                  placeholder="Paste in UI only; never send it in chat"
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
                  <span className={fieldLabelClass()}>Pexels Key</span>
                  <Input onChange={event => updateConfig('pexelsApiKey', event.target.value)} type="password" value={configForm.pexelsApiKey} />
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>Pixabay Key</span>
                  <Input onChange={event => updateConfig('pixabayApiKey', event.target.value)} type="password" value={configForm.pixabayApiKey} />
                </label>
                <label className="block space-y-2">
                  <span className={fieldLabelClass()}>Coverr Key</span>
                  <Input onChange={event => updateConfig('coverrApiKey', event.target.value)} type="password" value={configForm.coverrApiKey} />
                </label>
              </div>
            </div>

            <label className="block space-y-2">
              <span className={fieldLabelClass()}>视频主题</span>
              <Input
                onChange={event => updateForm('videoSubject', event.target.value)}
                placeholder="例如：30 秒介绍 Hermes Agent 的桌面端工作流"
                value={form.videoSubject}
              />
            </label>

            <label className="block space-y-2">
              <span className={fieldLabelClass()}>视频文案</span>
              <Textarea
                className="min-h-32"
                onChange={event => updateForm('videoScript', event.target.value)}
                placeholder="可留空，由 MoneyPrinterTurbo 根据主题生成；也可以人工编辑完整脚本。"
                value={form.videoScript}
              />
            </label>

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

            <label className="block space-y-2">
              <span className={fieldLabelClass()}>语音</span>
              <Input onChange={event => updateForm('voiceName', event.target.value)} value={form.voiceName} />
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

            <div className="rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3">
              <div className="mb-3 text-xs font-semibold text-(--ui-text-primary)">字幕与合成</div>
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
                    checked={form.matchMaterialsToScript}
                    onChange={event => updateForm('matchMaterialsToScript', event.target.checked)}
                    type="checkbox"
                  />
                  按文案匹配素材
                </label>
              </div>
            </div>

            <Button className="w-full" disabled={busy} onClick={createVideo} type="button">
              {busy ? 'Creating…' : 'Generate Video'}
            </Button>
          </div>
        </form>

        <section className="min-h-0 overflow-y-auto border-r border-(--ui-stroke-secondary) p-5">
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

        <aside className="min-h-0 overflow-y-auto p-5">
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
