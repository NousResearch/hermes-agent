import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  configFormFromSummary,
  defaultVideoGenerationForm,
  isMoneyPrinterPreviewVideo,
  moneyprinterClient,
  resolveMoneyPrinterMediaUrl,
  scriptTextFromResult,
  termsTextFromResult,
  toCreateVideoPayload,
  toGenerateScriptPayload,
  toGenerateTermsPayload,
  toMiniMaxCloneVoicePayload,
  toMiniMaxMusicPayload,
  toMiniMaxTtsPayload,
  videoLibraryClient,
  videoLibraryApiPath,
  videoStudioApiPath
} from './moneyprinter-client'

afterEach(() => {
  vi.restoreAllMocks()
})

describe('moneyprinter video studio client mapping', () => {
  it('maps the default form to the MoneyPrinter create-video payload', () => {
    const payload = toCreateVideoPayload({
      ...defaultVideoGenerationForm,
      videoSubject: '上海早晨的咖啡店',
      videoScript: '第一幕，咖啡香气升起。',
      matchMaterialsToScript: true
    })

    expect(payload).toMatchObject({
      video_subject: '上海早晨的咖啡店',
      video_script: '第一幕，咖啡香气升起。',
      video_aspect: '9:16',
      video_concat_mode: 'random',
      video_count: 1,
      video_source: 'pexels',
      match_materials_to_script: true,
      paragraph_number: 1,
      voice_name: 'zh-CN-XiaoxiaoNeural-Female',
      subtitle_enabled: true,
      bgm_type: 'random'
    })
  })

  it('maps fine-grained script, terms, subtitle, and transition controls', () => {
    const form = {
      ...defaultVideoGenerationForm,
      customSystemPrompt: '你是短视频编导',
      paragraphNumber: 3,
      searchTermsAmount: 7,
      textBackgroundColor: 'false',
      videoConcatMode: 'sequential' as const,
      videoScript: '第一幕，咖啡香气升起。',
      videoScriptPrompt: '开头 3 秒抛出痛点',
      videoSubject: '上海早晨的咖啡店',
      videoTerms: 'coffee shop\nmorning, 上海',
      videoTransitionMode: 'FadeIn' as const
    }

    expect(toGenerateScriptPayload(form)).toMatchObject({
      custom_system_prompt: '你是短视频编导',
      paragraph_number: 3,
      video_script_prompt: '开头 3 秒抛出痛点',
      video_subject: '上海早晨的咖啡店'
    })
    expect(toGenerateTermsPayload(form)).toMatchObject({
      amount: 7,
      video_script: '第一幕，咖啡香气升起。',
      video_subject: '上海早晨的咖啡店'
    })

    const payload = toCreateVideoPayload(form)
    expect(payload.video_terms).toEqual(['coffee shop', 'morning', '上海'])
    expect(payload.video_concat_mode).toBe('sequential')
    expect(payload.video_transition_mode).toBe('FadeIn')
    expect(payload.text_background_color).toBe(false)
    expect(payload.custom_system_prompt).toBe('你是短视频编导')
    expect(payload.video_script_prompt).toBe('开头 3 秒抛出痛点')
  })

  it('hydrates visible config fields without returning saved secret values', () => {
    const form = configFormFromSummary({
      apiKeyConfigured: true,
      baseUrl: 'https://openrouter.ai/api/v1',
      configExists: true,
      llmProvider: 'openai',
      materialProviders: { coverr: true, pexels: true, pixabay: false },
      minimax: {
        apiKeyConfigured: true,
        baseUrl: 'https://api.minimax.io/v1',
        musicModel: 'music-2.6-free',
        t2aModel: 'speech-2.8-hd',
        voiceCloneModel: 'speech-2.8-hd'
      },
      modelConfigured: true,
      modelName: 'openrouter/auto'
    })

    expect(form).toMatchObject({
      apiKey: '',
      baseUrl: 'https://openrouter.ai/api/v1',
      coverrApiKey: '',
      llmProvider: 'openai',
      minimaxApiKey: '',
      minimaxBaseUrl: 'https://api.minimax.io/v1',
      minimaxMusicModel: 'music-2.6-free',
      minimaxT2aModel: 'speech-2.8-hd',
      minimaxVoiceCloneModel: 'speech-2.8-hd',
      modelName: 'openrouter/auto',
      pexelsApiKey: '',
      pixabayApiKey: ''
    })
  })

  it('extracts generated script and terms into editable text fields', () => {
    expect(scriptTextFromResult({ video_script: '生成的文案' })).toBe('生成的文案')
    expect(termsTextFromResult({ video_terms: ['coffee', 'morning'] })).toBe('coffee\nmorning')
  })

  it('keeps capability API paths under the moneyprinter namespace', () => {
    expect(videoStudioApiPath('/tasks/task-1')).toBe('/api/capabilities/moneyprinter/tasks/task-1')
    expect(videoStudioApiPath('videos')).toBe('/api/capabilities/moneyprinter/videos')
    expect(videoStudioApiPath('materials')).toBe('/api/capabilities/moneyprinter/materials')
  })

  it('keeps material-library API paths under the video-library namespace', () => {
    expect(videoLibraryApiPath('/assets')).toBe('/api/capabilities/video-library/assets')
    expect(videoLibraryApiPath('clips?tag=\u7ad6\u5c4f')).toBe('/api/capabilities/video-library/clips?tag=\u7ad6\u5c4f')
  })

  it('sends the selected library id on named-library requests', async () => {
    const api = vi.fn().mockResolvedValue({ data: {}, error: null, ok: true })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })

    await videoLibraryClient.listLibraries()
    await videoLibraryClient.listAssets('beef-noodle')
    await videoLibraryClient.listClips('beef-noodle', { limit: 5, query: '热气 牛肉' })
    await videoLibraryClient.replaceClipTags('beef-noodle', 'clip-1', ['人工确认'])
    await videoLibraryClient.createTimeline('beef-noodle', ['clip-1'], '9:16', [{ text: '后厨现煮' }])

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/api/capabilities/video-library/libraries' })
    )
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/api/capabilities/video-library/assets?library_id=beef-noodle' })
    )
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/capabilities/video-library/clips?library_id=beef-noodle&query=%E7%83%AD%E6%B0%94+%E7%89%9B%E8%82%89&limit=5'
      })
    )
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ body: { libraryId: 'beef-noodle', tags: ['人工确认'] } })
    )
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        body: {
          aspect: '9:16',
          clipIds: ['clip-1'],
          libraryId: 'beef-noodle',
          script: [{ text: '后厨现煮' }]
        }
      })
    )
  })

  it('maps selected local materials to MoneyPrinter video_materials', () => {
    const payload = toCreateVideoPayload({
      ...defaultVideoGenerationForm,
      localMaterials: ['clip-a.mp4', ' image-b.png '],
      videoSource: 'local',
      videoSubject: '本地素材测试'
    })

    expect(payload.video_source).toBe('local')
    expect(payload.video_materials).toEqual([
      { duration: 0, provider: 'local', url: 'clip-a.mp4' },
      { duration: 0, provider: 'local', url: 'image-b.png' }
    ])
  })

  it('routes capability media URLs through the authenticated Electron media protocol', () => {
    expect(
      resolveMoneyPrinterMediaUrl(
        '/api/capabilities/moneyprinter/stream/task-1/final video.mp4',
        'http://127.0.0.1:8080/'
      )
    ).toBe(
      'hermes-media://gateway/%2Fapi%2Fcapabilities%2Fmoneyprinter%2Fstream%2Ftask-1%2Ffinal%20video.mp4'
    )

    expect(resolveMoneyPrinterMediaUrl('/api/capabilities/moneyprinter/download/task-1/combined-1.mp4')).toBe(
      'hermes-media://gateway/%2Fapi%2Fcapabilities%2Fmoneyprinter%2Fdownload%2Ftask-1%2Fcombined-1.mp4'
    )

    expect(resolveMoneyPrinterMediaUrl('https://cdn.example/final-1.mp4')).toBe('https://cdn.example/final-1.mp4')
  })

  it('identifies playable final/combined outputs and rejects MoviePy temp files', () => {
    expect(isMoneyPrinterPreviewVideo({ name: 'final-1.mp4' })).toBe(true)
    expect(isMoneyPrinterPreviewVideo({ name: 'combined-1.mp4' })).toBe(true)
    expect(isMoneyPrinterPreviewVideo({ name: 'final-1TEMP_MPY_wvf_snd.mp4' })).toBe(false)
  })

  it('maps MiniMax voice clone inputs to adapter payload fields', () => {
    expect(
      toMiniMaxCloneVoicePayload({
        activate: true,
        cloneAudio: { contentBase64: 'data:audio/wav;base64,ZmFrZQ==', filename: 'clone.wav' },
        model: 'speech-2.8-hd',
        promptAudio: { filename: 'prompt.mp3', sourcePath: '/tmp/prompt.mp3' },
        promptText: '参考音频文本',
        trialText: '试听文本',
        voiceId: 'MiniMaxDemo001'
      })
    ).toEqual({
      activate: true,
      clone_audio: { contentBase64: 'data:audio/wav;base64,ZmFrZQ==', filename: 'clone.wav' },
      model: 'speech-2.8-hd',
      prompt_audio: { filename: 'prompt.mp3', sourcePath: '/tmp/prompt.mp3' },
      prompt_text: '参考音频文本',
      trial_text: '试听文本',
      voice_id: 'MiniMaxDemo001'
    })
  })

  it('maps an existing MiniMax voice preview to TTS payload fields', () => {
    expect(
      toMiniMaxTtsPayload({
        model: 'speech-2.8-turbo',
        text: '테스트입니다.',
        voiceId: 'Korean_GentleBoss'
      })
    ).toEqual({
      model: 'speech-2.8-turbo',
      save_as_custom_audio: true,
      speed: 1,
      text: '테스트입니다.',
      voice_id: 'Korean_GentleBoss',
      volume: 1
    })
  })

  it('uses endpoint-specific timeouts for long MiniMax requests', async () => {
    const api = vi.fn().mockResolvedValue({ data: { audio: null }, error: null, ok: true })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })

    await moneyprinterClient.generateMiniMaxTts({
      model: 'speech-2.8-turbo',
      text: '테스트입니다.',
      voiceId: 'Korean_GentleBoss'
    })

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/capabilities/moneyprinter/minimax/tts',
        timeoutMs: 180_000
      })
    )
  })

  it('maps MiniMax music inputs to adapter payload fields', () => {
    expect(
      toMiniMaxMusicPayload({
        isInstrumental: true,
        lyrics: '',
        lyricsOptimizer: false,
        model: 'music-2.6-free',
        prompt: '科技感短视频开场',
        saveAsBgm: true
      })
    ).toEqual({
      is_instrumental: true,
      lyrics: '',
      lyrics_optimizer: false,
      model: 'music-2.6-free',
      prompt: '科技感短视频开场',
      save_as_bgm: true
    })
  })
})
