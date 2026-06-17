import { useStore } from '@nanostores/react'
import { useCallback, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { getSessionMessages, transcribeAudio } from '@/hermes'
import { chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { AudioLines, Brain, FolderOpen, Mic, Search, Send, Sparkles } from '@/lib/icons'
import { playSpeechText } from '@/lib/voice-playback'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { $currentCwd, workspaceCwdForNewSession } from '@/store/session'

import { useVoiceRecorder } from '../chat/composer/hooks/use-voice-recorder'

import { nextCompanionAvatarState, type CompanionAvatarState } from './avatar-state'
import { classifyCompanionIntent } from './intent'
import { buildFileSearchActions, filePathToExternalUrl, type CompanionAction } from './local-actions'

type CompanionMessage = {
  actions?: CompanionAction[]
  id: string
  role: 'assistant' | 'system' | 'user'
  text: string
}

interface CompanionSessionRef {
  profile: null | string
  runtimeId: string
  storedId: null | string
}

interface SessionCreateLike {
  session_id: string
  stored_session_id?: null | string
}

interface FileSearchResult {
  matches: string[]
  reply: string
}

interface LaunchAppResult {
  ok: boolean
  appId?: string
  label?: string
  error?: string
  message?: string
}

interface OpenKnownFolderResult {
  ok: boolean
  folderId?: string
  label?: string
  path?: string
  error?: string
  message?: string
}

const SEARCH_SKIP_DIRS = new Set(['.git', '.next', '.turbo', 'build', 'dist', 'node_modules', 'release'])

function createMessage(role: CompanionMessage['role'], text: string, actions?: CompanionAction[]): CompanionMessage {
  return {
    actions,
    id: `${role}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    role,
    text
  }
}

function delay(ms: number) {
  return new Promise(resolve => window.setTimeout(resolve, ms))
}

function avatarSheetPosition(state: CompanionAvatarState): string {
  switch (state) {
    case 'listening':
      return '66.666% 0%'
    case 'thinking':
      return '33.333% 100%'
    case 'speaking':
      return '33.333% 0%'
    case 'acting':
      return '100% 100%'
    case 'idle':
    default:
      return '0% 33.333%'
  }
}

async function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.addEventListener('load', () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result)
      } else {
        reject(new Error('无法读取录音数据'))
      }
    })
    reader.addEventListener('error', () => reject(reader.error || new Error('无法读取录音数据')))
    reader.readAsDataURL(blob)
  })
}

async function resolveSearchRoot(currentCwd: string): Promise<string> {
  const trimmed = currentCwd.trim()

  if (trimmed) {
    return trimmed
  }

  const result = await window.hermesDesktop.settings.getDefaultProjectDir()

  if (result.resolvedCwd) {
    return result.resolvedCwd
  }

  return workspaceCwdForNewSession()
}

async function searchFiles(root: string, query: string): Promise<string[]> {
  const normalizedQuery = query.trim().toLowerCase()

  if (!normalizedQuery) {
    return []
  }

  const matches: string[] = []
  let scanned = 0

  const visit = async (dirPath: string, depth: number): Promise<void> => {
    if (depth < 0 || scanned >= 220 || matches.length >= 6) {
      return
    }

    const result = await window.hermesDesktop.readDir(dirPath)

    for (const entry of result.entries) {
      if (scanned >= 220 || matches.length >= 6) {
        return
      }

      scanned += 1

      if (entry.name.toLowerCase().includes(normalizedQuery)) {
        matches.push(entry.path)
      }

      if (entry.isDirectory && depth > 0 && !SEARCH_SKIP_DIRS.has(entry.name.toLowerCase())) {
        await visit(entry.path, depth - 1)
      }
    }
  }

  await visit(root, 3)

  return matches
}

function pickPreferredVoice(voices: SpeechSynthesisVoice[]): null | SpeechSynthesisVoice {
  return (
    voices.find(voice => /^zh(-|_)?/i.test(voice.lang) && /female|xiaoxiao|xiaoyi|meijia|siao/i.test(voice.name)) ||
    voices.find(voice => /^zh(-|_)?/i.test(voice.lang)) ||
    null
  )
}

async function speakWithFallback(text: string): Promise<void> {
  try {
    await playSpeechText(text, { source: 'voice-conversation' })
    return
  } catch {
    // Fall through to browser speech synthesis when backend TTS is unavailable.
  }

  if (!('speechSynthesis' in window)) {
    return
  }

  await new Promise<void>(resolve => {
    const utterance = new SpeechSynthesisUtterance(text)
    const preferredVoice = pickPreferredVoice(window.speechSynthesis.getVoices())

    utterance.lang = preferredVoice?.lang || 'zh-CN'
    utterance.voice = preferredVoice
    utterance.rate = 1
    utterance.pitch = 1.08
    utterance.onend = () => resolve()
    utterance.onerror = () => resolve()
    window.speechSynthesis.cancel()
    window.speechSynthesis.speak(utterance)
  })
}

function summarizeSearchMatches(query: string, matches: string[]): string {
  if (!matches.length) {
    return `我先在当前工作区附近帮你找了一圈，还没有看到和“${query}”匹配的文件。你可以再告诉我更完整一点的名字。`
  }

  const summary = matches.slice(0, 4).map(path => `- ${path}`).join('\n')

  return `我在当前工作区附近找到了这些结果：\n${summary}`
}

export function CompanionView({
  requestGateway
}: {
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}) {
  const currentCwd = useStore($currentCwd)
  const activeGatewayProfile = useStore($activeGatewayProfile)
  const sessionRef = useRef<CompanionSessionRef | null>(null)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [errorText, setErrorText] = useState<null | string>(null)
  const [messages, setMessages] = useState<CompanionMessage[]>([
    createMessage(
      'assistant',
      '你好，我是 NOVA Companion。你可以直接和我说话，我会尽量用温柔自然的方式回应你，也可以先帮你在当前项目里找文件。'
    )
  ])
  const [avatarState, setAvatarState] = useState<CompanionAvatarState>('idle')

  const appendMessage = useCallback((role: CompanionMessage['role'], text: string, actions?: CompanionAction[]) => {
    setMessages(prev => [...prev, createMessage(role, text, actions)])
  }, [])

  const ensureSession = useCallback(async () => {
    if (sessionRef.current) {
      return sessionRef.current
    }

    const cwd = await resolveSearchRoot(currentCwd)
    const profile = activeGatewayProfile || null
    const created = await requestGateway<SessionCreateLike>('session.create', {
      cols: 96,
      ...(cwd ? { cwd } : {}),
      ...(profile ? { profile } : {})
    })

    sessionRef.current = {
      profile,
      runtimeId: created.session_id,
      storedId: created.stored_session_id ?? null
    }

    return sessionRef.current
  }, [activeGatewayProfile, currentCwd, requestGateway])

  const waitForAssistantReply = useCallback(async (session: CompanionSessionRef, baselineAssistantCount: number) => {
    let stableRounds = 0
    let lastAssistantText = ''

    for (let attempt = 0; attempt < 36; attempt += 1) {
      const response = await getSessionMessages(session.storedId ?? session.runtimeId, session.profile ?? undefined)
      const chatMessages = toChatMessages(response.messages)
      const assistantMessages = chatMessages.filter(message => message.role === 'assistant')
      const latestAssistant = assistantMessages.at(-1)
      const latestText = latestAssistant ? chatMessageText(latestAssistant).trim() : ''

      if (assistantMessages.length > baselineAssistantCount && latestText) {
        if (latestText === lastAssistantText) {
          stableRounds += 1
        } else {
          lastAssistantText = latestText
          stableRounds = 0
        }

        if (stableRounds >= 1) {
          return latestText
        }
      }

      await delay(1200)
    }

    return lastAssistantText
  }, [])

  const handleFileSearchIntent = useCallback(
    async (query: string): Promise<FileSearchResult> => {
      const root = await resolveSearchRoot(currentCwd)
      const matches = await searchFiles(root, query)

      return {
        matches,
        reply: summarizeSearchMatches(query, matches)
      }
    },
    [currentCwd]
  )

  const handleOpenAppIntent = useCallback(async (target: string) => {
    const result = await window.hermesDesktop.launchApp(target)

    if (result.ok) {
      return `好的，我已经帮你打开${result.label || `“${target}”`}。`
    }

    if (result.error === 'unsupported-app') {
      return '这一版我先支持几个安全白名单应用：记事本、资源管理器、计算器、Chrome、Edge。你可以直接说“打开记事本”或“打开资源管理器”。'
    }

    return result.message || `我尝试打开“${target}”时没有成功。`
  }, [])

  const handleOpenFolderIntent = useCallback(async (target: string) => {
    const result: OpenKnownFolderResult = await window.hermesDesktop.openKnownFolder(target)

    if (result.ok) {
      return `好的，我已经帮你打开${result.label || `“${target}”`}。`
    }

    if (result.error === 'unsupported-folder') {
      return '这一版我先支持几个常用系统目录：桌面、下载、文档、主页、音乐、图片、视频。你可以直接说“打开桌面”或“打开下载”。'
    }

    return result.message || `我尝试打开“${target}”时没有成功。`
  }, [])

  const speakReply = useCallback(async (text: string, state: CompanionAvatarState = 'thinking') => {
    setAvatarState(nextCompanionAvatarState(state, 'speech-start'))
    await speakWithFallback(text)
    setAvatarState(nextCompanionAvatarState('speaking', 'reset'))
  }, [])

  const openLocalAction = useCallback(async (action: CompanionAction) => {
    setAvatarState(nextCompanionAvatarState('idle', 'action-start'))
    await window.hermesDesktop.openExternal(filePathToExternalUrl(action.target))
    setAvatarState(nextCompanionAvatarState('acting', 'reset'))
  }, [])

  const submitPrompt = useCallback(
    async (rawText: string) => {
      const text = rawText.trim()

      if (!text || busy) {
        return
      }

      setErrorText(null)
      setBusy(true)
      setInput('')
      appendMessage('user', text)

      try {
        const intent = classifyCompanionIntent(text)

        if (intent.kind === 'find-file') {
          setAvatarState(nextCompanionAvatarState('idle', 'action-start'))
          const localResult = await handleFileSearchIntent(intent.target)
          appendMessage('assistant', localResult.reply, buildFileSearchActions(localResult.matches))
          await speakReply(localResult.reply, 'acting')
          return
        }

        if (intent.kind === 'open-app') {
          setAvatarState(nextCompanionAvatarState('idle', 'action-start'))
          const localReply = await handleOpenAppIntent(intent.target)
          appendMessage('assistant', localReply)
          await speakReply(localReply, 'acting')
          return
        }

        if (intent.kind === 'open-folder') {
          setAvatarState(nextCompanionAvatarState('idle', 'action-start'))
          const localReply = await handleOpenFolderIntent(intent.target)
          appendMessage('assistant', localReply)
          await speakReply(localReply, 'acting')
          return
        }

        setAvatarState(nextCompanionAvatarState('idle', 'think-start'))
        const session = await ensureSession()
        const before = await getSessionMessages(session.storedId ?? session.runtimeId, session.profile ?? undefined)
        const baselineAssistantCount = toChatMessages(before.messages).filter(message => message.role === 'assistant').length

        await requestGateway('prompt.submit', {
          session_id: session.runtimeId,
          text
        })

        const reply =
          (await waitForAssistantReply(session, baselineAssistantCount)) ||
          '我收到你的话了，不过这一轮回复还没有完整回来。你可以再和我说一遍。'

        appendMessage('assistant', reply)
        await speakReply(reply)
      } catch (error) {
        const message = error instanceof Error ? error.message : 'NOVA Companion 暂时没有成功回应。'

        setErrorText(message)
        appendMessage('system', `发生了一点问题：${message}`)
        notifyError(error, 'NOVA Companion 运行失败')
        setAvatarState(nextCompanionAvatarState('thinking', 'reset'))
      } finally {
        setBusy(false)
      }
    },
    [
      appendMessage,
      busy,
      ensureSession,
      handleFileSearchIntent,
      handleOpenAppIntent,
      handleOpenFolderIntent,
      requestGateway,
      speakReply,
      waitForAssistantReply
    ]
  )

  const transcribeVoiceAudio = useCallback(async (audio: Blob) => {
    const dataUrl = await blobToDataUrl(audio)
    const result = await transcribeAudio(dataUrl, audio.type)

    return result.transcript
  }, [])

  const { dictate, voiceActivityState, voiceStatus } = useVoiceRecorder({
    maxRecordingSeconds: 45,
    onTranscribeAudio: transcribeVoiceAudio,
    focusInput: () => document.getElementById('nova-companion-input')?.focus(),
    onTranscript: transcript => {
      void submitPrompt(transcript)
    }
  })

  const assistantStatusLabel = useMemo(() => {
    if (busy) {
      return '正在思考'
    }

    if (voiceStatus === 'recording') {
      return '正在倾听'
    }

    if (voiceStatus === 'transcribing') {
      return '正在转写'
    }

    return '待命中'
  }, [busy, voiceStatus])

  const latestAssistantReply = useMemo(
    () => [...messages].reverse().find(message => message.role === 'assistant')?.text ?? '',
    [messages]
  )

  const handleVoiceClick = useCallback(() => {
    setAvatarState(nextCompanionAvatarState('idle', voiceStatus === 'idle' ? 'listen-start' : 'reset'))
    dictate()
  }, [dictate, voiceStatus])

  const quickSearchWorkspace = useCallback(async () => {
    const root = await resolveSearchRoot(currentCwd)
    appendMessage('assistant', `当前我会优先围绕这个位置帮助你工作：\n${root}`)
  }, [appendMessage, currentCwd])

  return (
    <div className="relative flex h-full min-h-0 flex-col overflow-hidden bg-[radial-gradient(circle_at_top,_rgba(255,170,219,0.28),_transparent_26%),linear-gradient(180deg,_#0c1025_0%,_#151e43_42%,_#0b1330_100%)] text-white">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(135deg,rgba(130,245,255,0.08),transparent_28%,rgba(255,160,223,0.12)_66%,transparent)]" />

      <div className="relative flex min-h-0 flex-1 gap-6 p-6">
        <section className="relative flex min-w-[22rem] max-w-[28rem] flex-1 flex-col overflow-hidden rounded-[2rem] border border-white/12 bg-white/8 p-5 shadow-[0_30px_120px_rgba(8,10,24,0.45)] backdrop-blur-xl">
          <div className="absolute inset-0 rounded-[2rem] border border-white/6" />

          <div className="relative flex items-center justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.35em] text-cyan-100/70">NOVA Companion</p>
              <h1 className="mt-2 text-3xl font-semibold text-white">温柔陪伴型桌面助手</h1>
            </div>
            <div className="rounded-full border border-cyan-200/25 bg-cyan-200/10 px-3 py-1 text-xs text-cyan-100">
              {assistantStatusLabel}
            </div>
          </div>

          <div className="relative mt-6 flex flex-1 items-center justify-center overflow-hidden rounded-[1.8rem] border border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.10),rgba(255,255,255,0.04))]">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(255,163,214,0.28),_transparent_32%),radial-gradient(circle_at_bottom_right,_rgba(121,233,255,0.20),_transparent_30%)]" />

            <div className="absolute inset-x-6 top-6 z-20 flex items-center justify-between text-xs text-white/75">
              <span className="rounded-full border border-white/12 bg-black/15 px-3 py-1">混合语音模式</span>
              <span className="rounded-full border border-white/12 bg-black/15 px-3 py-1">普通话 · 温柔陪伴</span>
            </div>

            <div className="relative z-10 h-[32rem] w-full max-w-[22rem]">
              <div
                className="absolute inset-x-0 top-0 mx-auto h-full w-full rounded-[1.7rem] bg-cover bg-no-repeat shadow-[0_24px_80px_rgba(255,112,192,0.16)] transition-transform duration-300"
                style={{
                  backgroundImage: "url('/companion/reference-sheet.jpg')",
                  backgroundPosition: avatarSheetPosition(avatarState),
                  backgroundSize: '400% 300%',
                  transform:
                    voiceStatus === 'recording' ? 'translateY(-6px) scale(1.02)' : busy ? 'translateY(-2px)' : 'none'
                }}
              />

              <div className="absolute inset-x-8 bottom-5 rounded-[1.4rem] border border-white/10 bg-black/20 px-4 py-3 text-sm text-white/85 backdrop-blur-md">
                <div className="flex items-center justify-between gap-3">
                  <span className="font-medium">
                    {voiceStatus === 'recording' ? '我在听你说话' : '可以直接开口和我聊天'}
                  </span>
                  <Sparkles className="size-4 text-pink-200" />
                </div>
                <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/10">
                  <div
                    className="h-full rounded-full bg-[linear-gradient(90deg,#ff9bd6,#88efff)] transition-[width] duration-150"
                    style={{
                      width: `${Math.min(100, Math.max(8, Math.round(voiceActivityState.level * 100)))}%`
                    }}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="mt-5 grid grid-cols-2 gap-3">
            <Button
              className="h-12 rounded-2xl border border-cyan-200/20 bg-cyan-200/12 text-cyan-50 hover:bg-cyan-200/20"
              onClick={handleVoiceClick}
              type="button"
            >
              <Mic className="size-4" />
              {voiceStatus === 'idle' ? '开始语音' : '结束语音'}
            </Button>
            <Button
              className="h-12 rounded-2xl border border-white/14 bg-white/10 text-white hover:bg-white/16"
              onClick={() => void quickSearchWorkspace()}
              type="button"
              variant="ghost"
            >
              <FolderOpen className="size-4" />
              当前工作区
            </Button>
          </div>
        </section>

        <section className="relative flex min-h-0 flex-[1.25] flex-col overflow-hidden rounded-[2rem] border border-white/12 bg-black/18 p-5 shadow-[0_30px_120px_rgba(8,10,24,0.45)] backdrop-blur-xl">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm text-white/65">互动记录</p>
              <h2 className="mt-1 text-2xl font-semibold text-white">像真人一样自然对话</h2>
            </div>
            <div className="flex items-center gap-2 text-xs text-white/60">
              <AudioLines className="size-4 text-cyan-200" />
              <span>在线优先，失败时自动降级</span>
            </div>
          </div>

          <div className="mt-5 flex flex-wrap gap-2">
            {[
              { icon: Brain, label: '陪聊问答' },
              { icon: Search, label: '当前项目找文件' },
              { icon: Mic, label: '普通话语音互动' }
            ].map(item => (
              <div
                className="flex items-center gap-2 rounded-full border border-white/10 bg-white/8 px-3 py-2 text-xs text-white/75"
                key={item.label}
              >
                <item.icon className="size-3.5 text-pink-200" />
                <span>{item.label}</span>
              </div>
            ))}
          </div>

          <div className="mt-5 min-h-0 flex-1 overflow-y-auto rounded-[1.6rem] border border-white/10 bg-black/18 p-4">
            <div className="space-y-3">
              {messages.map(message => (
                <div
                  className={
                    message.role === 'user'
                      ? 'ml-auto max-w-[80%] rounded-[1.4rem] rounded-br-md bg-[linear-gradient(135deg,#ff94d2,#8be8ff)] px-4 py-3 text-sm text-slate-950 shadow-[0_12px_40px_rgba(255,146,210,0.18)]'
                      : message.role === 'assistant'
                        ? 'max-w-[84%] rounded-[1.4rem] rounded-bl-md border border-white/10 bg-white/10 px-4 py-3 text-sm text-white/90'
                        : 'max-w-[84%] rounded-[1.2rem] border border-amber-200/15 bg-amber-100/10 px-4 py-3 text-sm text-amber-50/85'
                  }
                  key={message.id}
                >
                  <div className="mb-1 text-[0.68rem] uppercase tracking-[0.24em] text-white/45">
                    {message.role === 'user' ? '你' : message.role === 'assistant' ? 'NOVA' : '系统'}
                  </div>
                  <div className="whitespace-pre-wrap leading-6">{message.text}</div>
                  {message.actions?.length ? (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {message.actions.map(action => (
                        <Button
                          className="rounded-full border border-white/12 bg-white/6 text-white hover:bg-white/12"
                          key={action.id}
                          onClick={() => void openLocalAction(action)}
                          size="sm"
                          type="button"
                          variant="outline"
                        >
                          <FolderOpen className="size-3.5" />
                          {action.label}
                        </Button>
                      ))}
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 rounded-[1.6rem] border border-white/10 bg-white/8 p-3">
            <Textarea
              className="min-h-28 resize-none border-0 bg-transparent text-sm text-white placeholder:text-white/35 focus-visible:ring-0"
              disabled={busy}
              id="nova-companion-input"
              onChange={event => setInput(event.target.value)}
              placeholder="比如：陪我聊聊今天的心情 / 帮我找合同.docx / 给我一点工作上的鼓励"
              value={input}
            />
            <div className="mt-3 flex items-center justify-between gap-4">
              <div className="line-clamp-2 text-xs text-white/45">
                {errorText
                  ? `最近一次问题：${errorText}`
                  : latestAssistantReply
                    ? `最近回复：${latestAssistantReply.slice(0, 48)}`
                    : '准备好了'}
              </div>
              <Button
                className="h-11 rounded-full bg-[linear-gradient(135deg,#ff8fce,#82efff)] px-5 text-sm font-medium text-slate-950 hover:opacity-95"
                disabled={busy || !input.trim()}
                onClick={() => void submitPrompt(input)}
                type="button"
              >
                <Send className="size-4" />
                {busy ? '回应中...' : '发送给 NOVA'}
              </Button>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
