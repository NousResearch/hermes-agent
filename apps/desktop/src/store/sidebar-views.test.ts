import { beforeEach, describe, expect, it } from 'vitest'

import {
  $sidebarCardRows,
  $sidebarGrouping,
  $sidebarOrdering,
  $sidebarPrFilter,
  $sidebarProfileFilter,
  $sidebarProjectFilter,
  $sidebarRowMeta,
  $sidebarShowArchived,
  $sidebarStatusFilter,
  resetSidebarView,
  setSidebarCardRows,
  setSidebarGrouping,
  setSidebarOrdering,
  setSidebarSessionOrderIds,
  setSidebarShowArchived,
  toggleSidebarPrFilter,
  toggleSidebarProfileFilter,
  toggleSidebarProjectFilter,
  toggleSidebarRowMeta,
  toggleSidebarStatusFilter
} from './layout'
import { $activeGatewayProfile, $showAllProfiles, ALL_PROFILES } from './profile'
import {
  $activeSavedSidebarViewId,
  $savedSidebarViews,
  applySavedSidebarView,
  deleteSavedSidebarView,
  renameSavedSidebarView,
  saveCurrentSidebarView,
  sidebarViewsCodec,
  updateSavedSidebarView
} from './sidebar-views'

beforeEach(() => {
  $activeGatewayProfile.set('default')
  $showAllProfiles.set(false)
  resetSidebarView()
  $savedSidebarViews.set({ version: 1, views: [] })
})

describe('saved sidebar views', () => {
  it('captures every filter-menu knob, including profile scope and manual order', () => {
    $showAllProfiles.set(true)
    setSidebarGrouping('none')
    setSidebarOrdering('manual')
    setSidebarSessionOrderIds(['session-b', 'session-a'])
    toggleSidebarRowMeta('updated')
    toggleSidebarRowMeta('tokens')
    setSidebarCardRows(true)
    toggleSidebarStatusFilter('working')
    toggleSidebarProjectFilter('project-a')
    toggleSidebarProfileFilter('research')
    toggleSidebarPrFilter('open')
    setSidebarShowArchived(true)

    const view = saveCurrentSidebarView('Needs review', { id: 'view-1', now: 100 })

    expect(view).toEqual({
      createdAt: 100,
      id: 'view-1',
      name: 'Needs review',
      state: {
        cardRows: true,
        filters: {
          profiles: ['research'],
          projects: ['project-a'],
          pullRequests: ['open'],
          showArchived: true,
          statuses: ['working']
        },
        grouping: 'none',
        manualOrderIds: ['session-b', 'session-a'],
        ordering: 'manual',
        profileScope: ALL_PROFILES,
        rowMeta: ['preview', 'tokens']
      },
      updatedAt: 100
    })
    expect($activeSavedSidebarViewId.get()).toBe('view-1')
  })

  it('restores a saved view after every current setting has changed', () => {
    $showAllProfiles.set(true)
    setSidebarGrouping('status')
    setSidebarOrdering('cost')
    toggleSidebarRowMeta('cost')
    setSidebarCardRows(true)
    toggleSidebarStatusFilter('needs-input')
    toggleSidebarProjectFilter('project-a')
    toggleSidebarProfileFilter('default')
    toggleSidebarPrFilter('merged')
    setSidebarShowArchived(true)
    saveCurrentSidebarView('Triage', { id: 'triage', now: 100 })

    resetSidebarView()
    $showAllProfiles.set(false)

    expect(applySavedSidebarView('triage')).toBe(true)
    expect($showAllProfiles.get()).toBe(true)
    expect($sidebarGrouping.get()).toBe('status')
    expect($sidebarOrdering.get()).toBe('cost')
    expect($sidebarRowMeta.get()).toEqual(['preview', 'updated', 'cost'])
    expect($sidebarCardRows.get()).toBe(true)
    expect($sidebarStatusFilter.get()).toEqual(['needs-input'])
    expect($sidebarProjectFilter.get()).toEqual(['project-a'])
    expect($sidebarProfileFilter.get()).toEqual(['default'])
    expect($sidebarPrFilter.get()).toEqual(['merged'])
    expect($sidebarShowArchived.get()).toBe(true)
    expect($activeSavedSidebarViewId.get()).toBe('triage')
  })

  it('renames and deletes by stable id without rewriting creation metadata', () => {
    saveCurrentSidebarView('Old name', { id: 'stable-id', now: 100 })

    expect(renameSavedSidebarView('stable-id', 'New name', 200)).toBe(true)
    expect($savedSidebarViews.get().views[0]).toMatchObject({
      createdAt: 100,
      id: 'stable-id',
      name: 'New name',
      updatedAt: 200
    })

    expect(deleteSavedSidebarView('stable-id')).toBe(true)
    expect($savedSidebarViews.get().views).toEqual([])
  })

  it('updates a saved view from the current configuration without changing its identity or name', () => {
    saveCurrentSidebarView('Review queue', { id: 'stable-id', now: 100 })

    $showAllProfiles.set(true)
    setSidebarGrouping('none')
    setSidebarOrdering('tokens')
    toggleSidebarRowMeta('profile')
    toggleSidebarProjectFilter('project-a')

    expect(updateSavedSidebarView('stable-id', 200)).toBe(true)
    expect($savedSidebarViews.get().views[0]).toEqual({
      createdAt: 100,
      id: 'stable-id',
      name: 'Review queue',
      state: {
        cardRows: false,
        filters: {
          profiles: [],
          projects: ['project-a'],
          pullRequests: [],
          showArchived: false,
          statuses: []
        },
        grouping: 'none',
        manualOrderIds: [],
        ordering: 'tokens',
        profileScope: ALL_PROFILES,
        rowMeta: ['preview', 'updated', 'profile']
      },
      updatedAt: 200
    })
    expect($activeSavedSidebarViewId.get()).toBe('stable-id')
  })

  it('stops identifying a view as active as soon as the current configuration drifts', () => {
    saveCurrentSidebarView('Default', { id: 'default-view', now: 100 })
    expect($activeSavedSidebarViewId.get()).toBe('default-view')

    toggleSidebarStatusFilter('unread')

    expect($activeSavedSidebarViewId.get()).toBeNull()
  })

  it('keeps the explicitly selected id when multiple views share a configuration', () => {
    saveCurrentSidebarView('First', { id: 'first', now: 100 })
    saveCurrentSidebarView('Second', { id: 'second', now: 200 })

    expect($activeSavedSidebarViewId.get()).toBe('second')

    applySavedSidebarView('first')

    expect($activeSavedSidebarViewId.get()).toBe('first')
  })

  it('sanitizes persisted data at the codec boundary', () => {
    const decoded = sidebarViewsCodec.decode(
      JSON.stringify({
        version: 1,
        views: [
          {
            createdAt: 10,
            id: 'valid',
            name: '  Saved  ',
            state: {
              cardRows: true,
              filters: {
                profiles: ['default', 3],
                projects: ['project-a', null],
                pullRequests: ['open', 'bogus'],
                showArchived: false,
                statuses: ['working', 'bogus']
              },
              grouping: 'none',
              manualOrderIds: ['session-a', 2],
              ordering: 'manual',
              profileScope: ALL_PROFILES,
              rowMeta: ['tokens', 'bogus']
            },
            updatedAt: 20
          },
          { id: '', name: 'Broken' }
        ]
      })
    )

    expect(decoded).toEqual({
      version: 1,
      views: [
        {
          createdAt: 10,
          id: 'valid',
          name: 'Saved',
          state: {
            cardRows: true,
            filters: {
              profiles: ['default'],
              projects: ['project-a'],
              pullRequests: ['open'],
              showArchived: false,
              statuses: ['working']
            },
            grouping: 'none',
            manualOrderIds: ['session-a'],
            ordering: 'manual',
            profileScope: ALL_PROFILES,
            rowMeta: ['tokens']
          },
          updatedAt: 20
        }
      ]
    })
  })
})
