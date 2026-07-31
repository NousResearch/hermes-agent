import { describe, expect, it } from 'vitest'

import type { PickedElement, PickedRegion } from './element-picker'
import { formatAnnotationMessage, formatAnnotationSessionMessage } from './message-format'

const baseElement: PickedElement = {
  classes: ['submit', 'primary'],
  id: '',
  pageTitle: '登录页',
  pageUrl: 'http://localhost:3000/login',
  rect: { height: 32, width: 80, x: 10, y: 20 },
  scrollX: 0,
  scrollY: 0,
  selector: '#app > button.submit.primary',
  tagName: 'BUTTON',
  text: '登录'
}

const baseRegion: PickedRegion = {
  pageTitle: '仪表盘',
  pageUrl: 'http://localhost:3000/dashboard',
  rect: { height: 200, width: 400, x: 50, y: 60 },
  scrollX: 0,
  scrollY: 0
}

describe('formatAnnotationMessage', () => {
  it('formats an element annotation with comment', () => {
    const msg = formatAnnotationMessage({
      comment: '按钮颜色应该是品牌蓝',
      kind: 'element',
      target: baseElement
    })

    expect(msg).toContain('[预览标注]')
    expect(msg).toContain('http://localhost:3000/login')
    expect(msg).toContain('`#app > button.submit.primary`')
    expect(msg).toContain('<BUTTON>')
    expect(msg).toContain('"登录"')
    expect(msg).toContain('按钮颜色应该是品牌蓝')
  })

  it('includes element id when present', () => {
    const msg = formatAnnotationMessage({
      comment: 'x',
      kind: 'element',
      target: { ...baseElement, id: 'login-btn', selector: '#login-btn' }
    })

    expect(msg).toContain('#login-btn')
  })

  it('truncates very long comments gracefully', () => {
    const longComment = '很长'.repeat(500)
    const msg = formatAnnotationMessage({
      comment: longComment,
      kind: 'element',
      target: baseElement
    })

    expect(msg.length).toBeLessThan(longComment.length + 600)
    expect(msg).toContain('[预览标注]')
  })

  it('formats a region annotation without selector', () => {
    const msg = formatAnnotationMessage({
      comment: '这一片间距不对',
      kind: 'region',
      target: baseRegion
    })

    expect(msg).toContain('[预览标注]')
    expect(msg).toContain('http://localhost:3000/dashboard')
    expect(msg).toContain('400×200')
    expect(msg).toContain('这一片间距不对')
    expect(msg).not.toContain('selector')
  })

  it('falls back to placeholder when comment is empty', () => {
    const msg = formatAnnotationMessage({
      comment: '   ',
      kind: 'element',
      target: baseElement
    })

    expect(msg).toContain('[预览标注]')
    expect(msg.trim().length).toBeGreaterThan(20)
  })
})


describe('formatAnnotationSessionMessage', () => {
  it('formats multiple items with numbered markers and shared page header', () => {
    const msg = formatAnnotationSessionMessage([
      { comment: '按钮颜色不对', kind: 'element', number: 1, target: baseElement },
      { comment: '这块布局乱了', kind: 'region', number: 2, target: baseRegion }
    ])

    expect(msg).toContain('[预览标注] 页面反馈（共 2 处）')
    expect(msg).toContain('① 元素标注')
    expect(msg).toContain('② 区域标注')
    expect(msg).toContain('`#app > button.submit.primary`')
    expect(msg).toContain('按钮颜色不对')
    expect(msg).toContain('这块布局乱了')
    // Element block carries the selector; region block does not.
    const regionBlock = msg.split('② 区域标注')[1]
    expect(regionBlock).not.toContain('选择器')
    expect(regionBlock).toContain('400×200px')
  })

  it('falls back to (n) markers beyond ⑩ and returns empty for no items', () => {
    expect(formatAnnotationSessionMessage([])).toBe('')

    const items = Array.from({ length: 11 }, (_, i) => ({
      comment: `c${i + 1}`,
      kind: 'region' as const,
      number: i + 1,
      target: baseRegion
    }))
    const msg = formatAnnotationSessionMessage(items)
    expect(msg).toContain('⑩')
    expect(msg).toContain('(11)')
  })
})
