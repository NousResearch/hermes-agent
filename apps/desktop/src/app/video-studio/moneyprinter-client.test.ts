import { describe, expect, it } from 'vitest'

import {
  defaultVideoGenerationForm,
  isMoneyPrinterPreviewVideo,
  resolveMoneyPrinterMediaUrl,
  toCreateVideoPayload,
  videoStudioApiPath
} from './moneyprinter-client'

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
      video_count: 1,
      video_source: 'pexels',
      match_materials_to_script: true,
      voice_name: 'zh-CN-XiaoxiaoNeural-Female',
      subtitle_enabled: true,
      bgm_type: 'random'
    })
  })

  it('keeps capability API paths under the moneyprinter namespace', () => {
    expect(videoStudioApiPath('/tasks/task-1')).toBe('/api/capabilities/moneyprinter/tasks/task-1')
    expect(videoStudioApiPath('videos')).toBe('/api/capabilities/moneyprinter/videos')
    expect(videoStudioApiPath('materials')).toBe('/api/capabilities/moneyprinter/materials')
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

  it('rewrites capability media URLs to the MoneyPrinter sidecar API', () => {
    expect(
      resolveMoneyPrinterMediaUrl(
        '/api/capabilities/moneyprinter/stream/task-1/final video.mp4',
        'http://127.0.0.1:8080/'
      )
    ).toBe('http://127.0.0.1:8080/api/v1/stream/task-1/final%20video.mp4')

    expect(resolveMoneyPrinterMediaUrl('/api/capabilities/moneyprinter/download/task-1/combined-1.mp4')).toBe(
      'http://127.0.0.1:8080/api/v1/download/task-1/combined-1.mp4'
    )

    expect(resolveMoneyPrinterMediaUrl('https://cdn.example/final-1.mp4')).toBe('https://cdn.example/final-1.mp4')
  })

  it('identifies playable final/combined outputs and rejects MoviePy temp files', () => {
    expect(isMoneyPrinterPreviewVideo({ name: 'final-1.mp4' })).toBe(true)
    expect(isMoneyPrinterPreviewVideo({ name: 'combined-1.mp4' })).toBe(true)
    expect(isMoneyPrinterPreviewVideo({ name: 'final-1TEMP_MPY_wvf_snd.mp4' })).toBe(false)
  })
})
