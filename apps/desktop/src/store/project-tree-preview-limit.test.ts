import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $projectTreePreviewLimit,
  clampProjectTreePreviewLimit,
  PROJECT_TREE_PREVIEW_DEFAULT,
  PROJECT_TREE_PREVIEW_MAX,
  PROJECT_TREE_PREVIEW_MIN,
  setProjectTreePreviewLimit
} from './project-tree-preview-limit'

const KEY = 'hermes.desktop.projectTreePreviewLimit.v1'

beforeEach(() => {
  window.localStorage.clear()
  $projectTreePreviewLimit.set(PROJECT_TREE_PREVIEW_DEFAULT)
})

describe('project tree preview limit', () => {
  it('ships the historical three-row preview', () => {
    expect(PROJECT_TREE_PREVIEW_DEFAULT).toBe(3)
    expect($projectTreePreviewLimit.get()).toBe(PROJECT_TREE_PREVIEW_DEFAULT)
  })

  it('stores a chosen count and mirrors it into localStorage', () => {
    expect(setProjectTreePreviewLimit(12)).toBe(12)
    expect($projectTreePreviewLimit.get()).toBe(12)
    expect(window.localStorage.getItem(KEY)).toBe('12')
  })

  it('clamps into the supported window instead of forwarding an unbounded value', () => {
    expect(setProjectTreePreviewLimit(0)).toBe(PROJECT_TREE_PREVIEW_MIN)
    expect(setProjectTreePreviewLimit(PROJECT_TREE_PREVIEW_MAX + 500)).toBe(PROJECT_TREE_PREVIEW_MAX)
    expect(setProjectTreePreviewLimit(7.9)).toBe(7)
  })

  it('falls back to the default for a non-finite value', () => {
    expect(clampProjectTreePreviewLimit(Number.NaN)).toBe(PROJECT_TREE_PREVIEW_DEFAULT)
    expect(clampProjectTreePreviewLimit(Number.POSITIVE_INFINITY)).toBe(PROJECT_TREE_PREVIEW_DEFAULT)
  })

  it('restores a stored count on load and clamps a tampered one', async () => {
    window.localStorage.setItem(KEY, '25')
    vi.resetModules()

    const fresh = await import('./project-tree-preview-limit')

    expect(fresh.$projectTreePreviewLimit.get()).toBe(25)

    window.localStorage.setItem(KEY, '999999')
    vi.resetModules()

    const clamped = await import('./project-tree-preview-limit')

    expect(clamped.$projectTreePreviewLimit.get()).toBe(clamped.PROJECT_TREE_PREVIEW_MAX)
  })
})
