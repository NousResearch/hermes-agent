/** Review B's preserved real source replies, not hand-built cursor substitutes. */
import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { isGroupFilesCursorError, parseGroupFilesPage, validateGroupFilesContinuation } from './group-files-client'
import { fileItem, filePage } from './group-files-test-fixtures'
import { SharedFilesDialog } from './group-files-view'
import wire from './group-files-wire-fixture.json'
import { translateBots } from './i18n-test-helper'

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, usePluginI18n: () => translateBots }
})
afterEach(cleanup)

it('accepts the actual 2068-character Unicode-query cursor and its continuation', () => {
  expect(wire.long_query_page.next_cursor.length).toBe(2068)
  const first = parseGroupFilesPage(wire.long_query_page)
  const second = parseGroupFilesPage(wire.long_query_continuation)
  expect(first.items).toHaveLength(8)
  expect(second.items).toHaveLength(1)
  expect(() => validateGroupFilesContinuation(first, second)).not.toThrow()
})

it("accepts Review A's independent actual 2076-character source cursor", () => {
  expect(wire.review_a_unicode_page.next_cursor.length).toBe(2076)
  expect(parseGroupFilesPage(wire.review_a_unicode_page).items).toHaveLength(8)
})

it('keeps search focused/reachable on the actual sparse empty first page and the older file page', async () => {
  const first = parseGroupFilesPage(wire.sparse_first)
  const next = parseGroupFilesPage(wire.sparse_second)
  const loadPage = vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(next)
  render(
    <SharedFilesDialog
      availability="available"
      group="Core"
      latestSeq={0}
      loadPage={loadPage}
      onClose={vi.fn()}
      open
      roomId="room-1"
    />
  )
  expect(screen.getByRole('textbox', { name: 'Search files' })).toBe(globalThis.document.activeElement)
  await screen.findByText('No files on this page')
  fireEvent.click(screen.getByRole('button', { name: 'Older' }))
  await screen.findByText('oldest.txt')
  expect(screen.getByRole('textbox', { name: 'Search files' })).toBeTruthy()
})

it('validates the full v2 manifest order both within a page and across its boundary', () => {
  const first = parseGroupFilesPage({
    ...filePage([], true),
    latest_seq: 21,
    items: [{ ...fileItem(20, 'first', 9), manifest_index: 0 }]
  })

  const second = parseGroupFilesPage({
    ...filePage(),
    latest_seq: 21,
    items: [{ ...fileItem(20, 'second', 1), manifest_index: 1 }]
  })

  expect(first.latestFileSeq).toBe(21)
  expect(() => validateGroupFilesContinuation(first, second)).not.toThrow()
  expect(() => validateGroupFilesContinuation(second, first)).toThrow()
  expect(() =>
    parseGroupFilesPage({
      ...filePage(),
      items: [
        { ...fileItem(20, 'first', 9), manifest_index: 0 },
        { ...fileItem(20, 'second', 1), manifest_index: 1 }
      ]
    })
  ).not.toThrow()
  expect(() =>
    parseGroupFilesPage({
      ...filePage(),
      items: [
        { ...fileItem(20, 'first', 1), manifest_index: 1 },
        { ...fileItem(20, 'second', 9), manifest_index: 0 }
      ]
    })
  ).toThrow()
})

it.each([-1, 8, '1', null, 0.5])('rejects invalid manifest_index %s without guessing a position', manifest_index => {
  expect(() => parseGroupFilesPage({ ...filePage(), items: [{ ...fileItem(), manifest_index }] })).toThrow()
})

it.each([-1, '21', null, 0.5])('rejects invalid latest_seq %s', latest_seq => {
  expect(() => parseGroupFilesPage({ ...filePage(), latest_seq })).toThrow()
})

it('permits only the explicit all-legacy prototype fallback, not mixed ordering metadata', () => {
  expect(parseGroupFilesPage(filePage()).items[0].manifestIndex).toBeUndefined()
  expect(parseGroupFilesPage(filePage()).latestFileSeq).toBeUndefined()
  expect(() =>
    parseGroupFilesPage({ ...filePage(), items: [fileItem(20), { ...fileItem(19), manifest_index: 0 }] })
  ).toThrow()
  const first = parseGroupFilesPage({ ...filePage([], true), items: [{ ...fileItem(20), manifest_index: 0 }] })
  expect(() => validateGroupFilesContinuation(first, parseGroupFilesPage(filePage([fileItem(19)])))).toThrow()
})

it('honors the typed cursor-reset receipt without depending on its prose', () => {
  expect(
    isGroupFilesCursorError({
      code: 4143,
      message: 'Changed',
      data: {
        reason: 'attachment_cursor_reset_required',
        reset_required: true,
        action: 'return_to_latest'
      }
    })
  ).toBe(true)
})
