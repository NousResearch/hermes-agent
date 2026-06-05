import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { setVoicePlaybackState } from '@/store/voice-playback'

import { VoiceActivity, VoicePlaybackActivity } from './voice-activity'

afterEach(() => {
  cleanup()
  setVoicePlaybackState({
    audioElement: null,
    messageId: null,
    sequence: 0,
    source: null,
    status: 'idle'
  })
})

function renderInChinese(children: React.ReactNode) {
  return render(
    <I18nProvider configClient={null} initialLanguage="zh">
      {children}
    </I18nProvider>
  )
}

describe('voice activity i18n', () => {
  it('renders recorder activity labels from the Chinese catalog', () => {
    const { rerender } = renderInChinese(
      <VoiceActivity state={{ elapsedSeconds: 8, level: 0.6, status: 'recording' }} />
    )

    expect(screen.getByRole('status').textContent).toContain('听写中')
    expect(screen.getByRole('status').textContent).toContain('0:08')

    rerender(
      <I18nProvider configClient={null} initialLanguage="zh">
        <VoiceActivity state={{ elapsedSeconds: 65, level: 0, status: 'transcribing' }} />
      </I18nProvider>
    )

    expect(screen.getByRole('status').textContent).toContain('转写中')
    expect(screen.getByRole('status').textContent).toContain('1:05')
  })

  it('renders playback activity labels and stop button from the Chinese catalog', () => {
    setVoicePlaybackState({
      audioElement: null,
      messageId: 'message-1',
      sequence: 1,
      source: 'voice-conversation',
      status: 'preparing'
    })

    const { rerender } = renderInChinese(<VoicePlaybackActivity />)

    expect(screen.getByRole('status').textContent).toContain('正在准备音频')
    expect(screen.getByRole('button', { name: /停止/ })).toBeTruthy()

    setVoicePlaybackState({
      audioElement: null,
      messageId: 'message-1',
      sequence: 2,
      source: 'voice-conversation',
      status: 'speaking'
    })
    rerender(
      <I18nProvider configClient={null} initialLanguage="zh">
        <VoicePlaybackActivity />
      </I18nProvider>
    )

    expect(screen.getByRole('status').textContent).toContain('正在播放回复')

    setVoicePlaybackState({
      audioElement: null,
      messageId: 'message-2',
      sequence: 3,
      source: 'read-aloud',
      status: 'speaking'
    })
    rerender(
      <I18nProvider configClient={null} initialLanguage="zh">
        <VoicePlaybackActivity />
      </I18nProvider>
    )

    expect(screen.getByRole('status').textContent).toContain('正在朗读')
  })
})
