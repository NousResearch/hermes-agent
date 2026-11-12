import { beforeEach, describe, expect, it } from 'vitest'

import { $workspaceOwnerKey } from '@/components/pane-shell/workspace-scope'

import { capturePreviewAnnotateDestination } from './handoff'

const groupSurface = (group: string, composerKey: string, ownerKey: string) => {
  const element = document.createElement('div')
  element.dataset.previewAnnotateDestination = 'group'
  element.dataset.previewAnnotateGroup = group
  element.dataset.previewAnnotateComposerKey = composerKey
  element.dataset.previewAnnotateOwnerKey = ownerKey
  document.body.append(element)

  return element
}

const withRect = (element: Element, left: number, top: number, width: number, height: number) => {
  Object.defineProperty(element, 'getBoundingClientRect', {
    configurable: true,
    value: () => ({
      bottom: top + height,
      height,
      left,
      right: left + width,
      top,
      width,
      x: left,
      y: top,
      toJSON: () => ({})
    })
  })
}

beforeEach(() => {
  document.body.innerHTML = ''
  window.sessionStorage.clear()
  $workspaceOwnerKey.set(null)
})

describe('capturePreviewAnnotateDestination', () => {
  it('pins the group owned by the active workspace when multiple groups are visible', () => {
    groupSurface('Sales', 'id:sales', 'group:sales')
    groupSurface('Design', 'id:design', 'group:design')
    $workspaceOwnerKey.set('group:design')

    const destination = capturePreviewAnnotateDestination()

    expect(destination).toMatchObject({
      composerKey: 'id:design',
      group: 'Design',
      kind: 'group'
    })
  })

  it('pins the visible group nearest the Browser pop-out control when rooms are side by side', () => {
    const sales = groupSurface('Sales', 'id:sales', 'group:sales')
    const design = groupSurface('Design', 'id:design', 'group:design')
    const popOut = document.createElement('button')

    document.body.append(popOut)
    withRect(sales, 100, 0, 500, 700)
    withRect(design, 620, 0, 500, 700)
    withRect(popOut, 1160, 40, 24, 24)
    $workspaceOwnerKey.set('group:sales')

    const destination = capturePreviewAnnotateDestination(popOut)

    expect(destination).toMatchObject({
      composerKey: 'id:design',
      group: 'Design',
      kind: 'group'
    })
  })

  it('fails closed instead of choosing the first group when several are visible and no owner matches', () => {
    groupSurface('Sales', 'id:sales', 'group:sales')
    groupSurface('Design', 'id:design', 'group:design')

    expect(capturePreviewAnnotateDestination()).toBeNull()
  })
})
