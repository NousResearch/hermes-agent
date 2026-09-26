import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { captureFollowUpSelection } from './passage'

const selectContents = (node: Node) => {
  const range = document.createRange()
  range.selectNodeContents(node)
  window.getSelection()?.addRange(range)
}

describe('captureFollowUpSelection', () => {
  let viewport: HTMLDivElement
  let assistantBody: HTMLDivElement
  let userBody: HTMLDivElement

  beforeEach(() => {
    viewport = document.createElement('div')
    viewport.innerHTML = `
      <div data-role="assistant" data-slot="aui_assistant-message-root">
        <div data-slot="aui_assistant-message-content">the answer <button type="button">Copy</button></div>
      </div>
      <div data-role="user" data-slot="aui_user-message-root"><div>my question</div></div>
    `
    document.body.append(viewport)
    assistantBody = viewport.querySelector('[data-slot="aui_assistant-message-content"]') as HTMLDivElement
    userBody = viewport.querySelector('[data-role="user"] div') as HTMLDivElement
  })

  afterEach(() => {
    window.getSelection()?.removeAllRanges()
    viewport.remove()
  })

  it('captures a passage out of an assistant message with its side', () => {
    const range = document.createRange()
    range.setStart(assistantBody.firstChild!, 4)
    range.setEnd(assistantBody.firstChild!, 10)
    window.getSelection()?.addRange(range)

    expect(captureFollowUpSelection(window.getSelection(), viewport)).toEqual({
      passage: 'answer',
      source: 'assistant'
    })
  })

  it('attributes a passage from the reader own message', () => {
    selectContents(userBody)

    expect(captureFollowUpSelection(window.getSelection(), viewport)).toEqual({
      passage: 'my question',
      source: 'user'
    })
  })

  it('refuses a collapsed caret', () => {
    const range = document.createRange()
    range.setStart(assistantBody.firstChild!, 2)
    range.collapse(true)
    window.getSelection()?.addRange(range)

    expect(captureFollowUpSelection(window.getSelection(), viewport)).toBeNull()
  })

  it('refuses a selection that reaches outside the transcript', () => {
    const outside = document.createElement('div')
    outside.textContent = 'composer draft'
    document.body.append(outside)
    selectContents(outside)

    expect(captureFollowUpSelection(window.getSelection(), viewport)).toBeNull()

    outside.remove()
  })

  it('refuses a selection spanning two messages (no single side to attribute)', () => {
    const range = document.createRange()
    range.setStart(assistantBody.firstChild!, 0)
    range.setEnd(userBody.firstChild!, 4)
    window.getSelection()?.addRange(range)

    expect(captureFollowUpSelection(window.getSelection(), viewport)).toBeNull()
  })

  it('refuses a passage inside a control that owns its selection', () => {
    const control = document.createElement('button')
    control.textContent = 'Copy answer'
    assistantBody.append(control)
    selectContents(control)

    expect(captureFollowUpSelection(window.getSelection(), viewport)).toBeNull()
  })

  it('refuses whitespace-only text', () => {
    const blank = document.createElement('div')
    blank.textContent = '   \n  '
    assistantBody.append(blank)
    selectContents(blank)

    expect(captureFollowUpSelection(window.getSelection(), viewport)).toBeNull()
  })

  it('normalizes the passage it captures', () => {
    const multi = document.createElement('div')
    multi.textContent = 'line one   \n\n\n\nline two  '
    assistantBody.append(multi)
    selectContents(multi)

    expect(captureFollowUpSelection(window.getSelection(), viewport)?.passage).toBe('line one\n\nline two')
  })
})
