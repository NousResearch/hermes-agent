import { cleanup, render } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { toChatMessages } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import type { SessionMessage } from '@/types/hermes'

import { stubThreadEnvironment, ThreadRuntime } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
afterEach(cleanup)

const notProcessed = 'Your request was not processed. Send it again if you still want me to carry it out.'
const partial = 'This turn did not complete. Some actions may already have run; verify their effects before resending.'

const automated =
  'An automated delivery was interrupted and was not run again. Some actions may already have run; verify their effects.'

function transcript(rows: SessionMessage[]) {
  const messages = toChatMessages(rows).map(toRuntimeMessage)

  return render(
    <I18nProvider configClient={null} initialLocale="ru">
      <ThreadRuntime messages={messages}>
        <Thread />
      </ThreadRuntime>
    </I18nProvider>
  )
}

it.each([
  [notProcessed, 'Не удалось обработать запрос. Если он ещё актуален, отправьте его снова.'],
  [partial, 'Запрос не завершён. Некоторые действия могли выполниться. Проверьте результат перед повторной отправкой.'],
  [
    automated,
    'Автоматическая задача прервалась и повторно не запускалась. Проверьте, какие действия успели выполниться.'
  ],
  ['Additional failure information', 'Additional failure information']
])('renders the saved failure notice in the interface language: %s', (content, expected) => {
  const view = transcript([{ role: 'assistant', display_kind: 'failed_turn', content }])
  expect(view.getByText(expected)).toBeTruthy()
})

it('preserves the same wording in ordinary system messages', () => {
  const view = transcript([{ role: 'system', content: notProcessed }])
  expect(view.getByText(notProcessed)).toBeTruthy()
})

it('localizes the default refusal notice while preserving the submitted text', () => {
  const view = transcript([
    {
      role: 'user',
      display_kind: 'content_refused',
      display_metadata: JSON.stringify({
        content_refused: { notice: "This message was refused by the model's safety filter — try a different query." }
      }),
      content: 'my original question'
    }
  ])

  expect(
    view.getByText('Это сообщение отклонено фильтром безопасности модели — попробуйте другой запрос.')
  ).toBeTruthy()
  expect(view.getByText('my original question')).toBeTruthy()
})

it('localizes the steering label and preserves the instruction', () => {
  const view = transcript([{ role: 'system', content: 'steer:please continue' }])
  expect(view.getByText('Указание принято')).toBeTruthy()
  expect(view.getByText('please continue')).toBeTruthy()
})
