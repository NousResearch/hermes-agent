/* global Element, Node, document */

import {
  Badge,
  Button,
  Codicon,
  EmptyState,
  ErrorState,
  Input,
  KEYBINDS_AREA,
  PALETTE_AREA,
  Streamdown,
  host,
  useValue
} from '@hermes/plugin-sdk'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const ID = 'markdown-viewer'
const WORKSPACE_ID = 'markdown-viewer.workspace'
const MARKDOWN_PATH_RE = /\.(?:md|markdown|mdx)$/i

function messageFrom(error) {
  if (error && typeof error === 'object') {
    if (typeof error.detail === 'string') return error.detail
    if (typeof error.message === 'string') return error.message
  }
  return String(error || 'Unknown error')
}

function rootLabel(root) {
  const parts = String(root || '')
    .split(/[\\/]/)
    .filter(Boolean)
  return parts.at(-1) || root || 'No active Project'
}

function formatBytes(bytes) {
  const value = Number(bytes || 0)
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(value >= 10 * 1024 ? 0 : 1)} KB`
  return `${(value / (1024 * 1024)).toFixed(1)} MB`
}

export function decodedHrefPath(href) {
  const raw = String(href || '').trim()
  try {
    if (raw.startsWith('#media:')) return decodeURIComponent(raw.slice('#media:'.length))
    if (raw.startsWith('file://')) {
      const path = decodeURIComponent(new URL(raw).pathname)
      return /^\/[a-z]:\//i.test(path) ? path.slice(1) : path
    }
    if (/^[a-z]:[\\/]/i.test(raw) || raw.startsWith('\\\\')) return decodeURIComponent(raw)
    if (!/^[a-z][a-z\d+.-]*:/i.test(raw)) return decodeURIComponent(raw)
  } catch {
    return ''
  }
  return ''
}

function markdownSuffixAfter(node) {
  const nextSibling = node?.nextSibling
  return nextSibling?.nodeType === Node.TEXT_NODE
    ? String(nextSibling.textContent || '').match(/^([^\n]*?\.(?:md|markdown|mdx))(?=\s|$)/i)?.[1] || ''
    : ''
}

function completeLegacyPath(path, node) {
  if (MARKDOWN_PATH_RE.test(path)) return path
  const suffix = markdownSuffixAfter(node)
  return suffix ? path + suffix : path
}

export function markdownPathFromClickTarget(target) {
  if (!(target instanceof Element)) return null

  const button = target.closest('button')
  if (button) {
    const card = button.parentElement
    const cardButtons = card ? Array.from(card.children).filter(child => child.tagName === 'BUTTON') : []
    if (button !== cardButtons.at(-1)) return null

    const titledPath = card?.querySelector('span[title]')?.getAttribute('title') || ''
    const path = completeLegacyPath(titledPath, card)
    return MARKDOWN_PATH_RE.test(path) ? path : null
  }

  const anchor = target.closest('a')
  if (!anchor) return null

  const path = completeLegacyPath(decodedHrefPath(anchor.getAttribute('href')), anchor)
  return MARKDOWN_PATH_RE.test(path) ? path : null
}

function installTranscriptLinkBridge(ctx) {
  const onClick = event => {
    if (event.defaultPrevented || event.button !== 0) return
    const path = markdownPathFromClickTarget(event.target)
    if (!path) return
    event.preventDefault()
    event.stopImmediatePropagation()
    openMarkdownWorkspace(ctx, path)
  }

  document.addEventListener('click', onClick, true)
  ctx.onDispose(() => document.removeEventListener('click', onClick, true))
}

function MarkdownWorkspace({ ctx, initialPath }) {
  const root = useValue(host.state.cwd)
  const focusedStoredSessionId = useValue(host.state.focusedStoredSessionId)
  const focusedSessionProfile = useValue(host.state.focusedSessionProfile)
  const [files, setFiles] = useState([])
  const [query, setQuery] = useState('')
  const [manualPath, setManualPath] = useState('')
  const [selected, setSelected] = useState(null)
  const [document, setDocument] = useState(null)
  const [listLoading, setListLoading] = useState(false)
  const [docLoading, setDocLoading] = useState(false)
  const [listError, setListError] = useState('')
  const [docError, setDocError] = useState('')
  const listGeneration = useRef(0)
  const documentGeneration = useRef(0)

  const loadList = useCallback(async () => {
    if (!root) {
      setFiles([])
      setListError('Open a Project session to browse its Markdown files.')
      return
    }
    const generation = ++listGeneration.current
    setListLoading(true)
    setListError('')
    try {
      const result = await ctx.rest(`/files?root=${encodeURIComponent(root)}&limit=2000`)
      if (generation !== listGeneration.current) return
      setFiles(Array.isArray(result?.files) ? result.files : [])
      if (result?.truncated) {
        setListError('Showing the first 2,000 Markdown files. Narrow the Project or type a relative path.')
      }
    } catch (error) {
      if (generation !== listGeneration.current) return
      setFiles([])
      setListError(messageFrom(error))
    } finally {
      if (generation === listGeneration.current) setListLoading(false)
    }
  }, [ctx, root])

  const loadFile = useCallback(
    async (path, quiet = false) => {
      const requested = String(path || '').trim()
      if (!root || !requested) return
      const generation = ++documentGeneration.current
      if (!quiet) setDocLoading(true)
      if (!quiet) setDocError('')
      try {
        const result = await ctx.rest(`/read?root=${encodeURIComponent(root)}&path=${encodeURIComponent(requested)}`)
        if (generation !== documentGeneration.current) return
        setSelected(result.relative)
        setManualPath(result.relative)
        setDocument(result)
      } catch (error) {
        if (generation !== documentGeneration.current) return
        if (!quiet) {
          setDocError(messageFrom(error))
          setDocument(null)
        }
      } finally {
        if (generation === documentGeneration.current && !quiet) setDocLoading(false)
      }
    },
    [ctx, root]
  )

  useEffect(() => {
    listGeneration.current += 1
    documentGeneration.current += 1
    setSelected(null)
    setDocument(null)
    setDocError('')
    setManualPath('')
    setQuery('')
    void loadList()
    if (initialPath) void loadFile(initialPath)
  }, [root, focusedStoredSessionId, focusedSessionProfile, initialPath, loadFile, loadList])

  useEffect(() => {
    if (!selected) return undefined
    const timer = setInterval(() => void loadFile(selected, true), 3000)
    return () => clearInterval(timer)
  }, [loadFile, selected])

  const visibleFiles = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase()
    if (!needle) return files
    return files.filter(file =>
      String(file.relative || '')
        .toLocaleLowerCase()
        .includes(needle)
    )
  }, [files, query])

  const openManual = useCallback(() => void loadFile(manualPath), [loadFile, manualPath])

  return jsxs('div', {
    className: 'flex h-full min-h-0 flex-col text-sm',
    'data-markdown-viewer': 'root',
    children: [
      jsxs('header', {
        className: 'flex shrink-0 items-center gap-2 border-b border-border px-3 py-2',
        children: [
          jsx(Codicon, { name: 'markdown', size: '1rem' }),
          jsxs('div', {
            className: 'min-w-0 flex-1',
            children: [
              jsx('div', { className: 'truncate font-medium', children: rootLabel(root) }),
              jsx('div', {
                className: 'truncate text-[0.6875rem] text-(--ui-text-tertiary)',
                title: root,
                children: root || 'No working directory'
              })
            ]
          }),
          jsx(Badge, { variant: 'outline', children: focusedSessionProfile || 'default' }),
          jsx(Button, {
            'aria-label': 'Refresh Markdown files',
            onClick: () => void loadList(),
            size: 'icon-sm',
            title: 'Refresh files',
            variant: 'ghost',
            children: jsx(Codicon, { name: 'refresh', size: '0.85rem' })
          })
        ]
      }),
      jsxs('div', {
        className: 'flex min-h-0 flex-1',
        children: [
          jsxs('aside', {
            className: 'flex w-72 shrink-0 flex-col border-r border-border',
            children: [
              jsxs('div', {
                className: 'grid shrink-0 gap-2 border-b border-border p-2',
                children: [
                  jsx(Input, {
                    'aria-label': 'Filter Markdown files',
                    onChange: event => setQuery(event.target.value),
                    placeholder: 'Filter Markdown files…',
                    value: query
                  }),
                  jsxs('div', {
                    className: 'flex gap-1.5',
                    children: [
                      jsx(Input, {
                        'aria-label': 'Open relative Markdown path',
                        className: 'min-w-0 flex-1 font-mono text-xs',
                        onChange: event => setManualPath(event.target.value),
                        onKeyDown: event => {
                          if (event.key === 'Enter') openManual()
                        },
                        placeholder: 'path/to/file.md',
                        value: manualPath
                      }),
                      jsx(Button, {
                        disabled: !manualPath.trim(),
                        onClick: openManual,
                        size: 'sm',
                        variant: 'outline',
                        children: 'Open'
                      })
                    ]
                  })
                ]
              }),
              jsxs('div', {
                className:
                  'flex shrink-0 items-center justify-between px-2 py-1.5 text-[0.6875rem] text-(--ui-text-tertiary)',
                children: [
                  jsx('span', { children: `${visibleFiles.length} file${visibleFiles.length === 1 ? '' : 's'}` }),
                  listLoading ? jsx('span', { children: 'Refreshing…' }) : null
                ]
              }),
              listError
                ? jsx('div', {
                    className: 'border-y border-border px-2 py-1.5 text-xs text-(--ui-text-secondary)',
                    children: listError
                  })
                : null,
              jsx('div', {
                className: 'min-h-0 flex-1 overflow-y-auto px-1 pb-2',
                children:
                  visibleFiles.length > 0
                    ? visibleFiles.map(file =>
                        jsxs(
                          'button',
                          {
                            className:
                              selected === file.relative
                                ? 'flex w-full items-start gap-2 rounded-md bg-(--chrome-action-hover) px-2 py-1.5 text-left'
                                : 'flex w-full items-start gap-2 rounded-md px-2 py-1.5 text-left hover:bg-(--chrome-action-hover)',
                            onClick: () => void loadFile(file.relative),
                            title: file.relative,
                            type: 'button',
                            children: [
                              jsx(Codicon, { className: 'mt-0.5 shrink-0', name: 'markdown', size: '0.8rem' }),
                              jsxs('span', {
                                className: 'min-w-0 flex-1',
                                children: [
                                  jsx('span', { className: 'block truncate text-xs', children: file.relative }),
                                  jsx('span', {
                                    className: 'block text-[0.625rem] text-(--ui-text-quaternary)',
                                    children: formatBytes(file.size)
                                  })
                                ]
                              })
                            ]
                          },
                          file.path
                        )
                      )
                    : !listLoading && !listError
                      ? jsx(EmptyState, {
                          description: query ? 'Try a different filter.' : 'No Markdown files were found.',
                          title: query ? 'No matches' : 'No Markdown files'
                        })
                      : null
              })
            ]
          }),
          jsx('main', {
            className: 'min-h-0 min-w-0 flex-1 overflow-y-auto',
            children: docLoading
              ? jsx('div', {
                  className: 'grid h-full place-items-center text-(--ui-text-tertiary)',
                  children: 'Opening Markdown…'
                })
              : docError
                ? jsx(ErrorState, { description: docError, title: 'Could not open Markdown' })
                : document
                  ? jsxs('article', {
                      className: 'mx-auto w-full max-w-4xl px-8 py-7',
                      children: [
                        jsxs('div', {
                          className: 'mb-6 flex items-center justify-between gap-3 border-b border-border pb-3',
                          children: [
                            jsx('div', {
                              className: 'min-w-0 truncate font-mono text-xs text-(--ui-text-secondary)',
                              title: document.path,
                              children: document.relative
                            }),
                            jsx('div', {
                              className: 'shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)',
                              children: formatBytes(document.size)
                            })
                          ]
                        }),
                        jsx('div', {
                          className:
                            'text-foreground [&_a]:underline [&_a]:underline-offset-2 [&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-3 [&_blockquote]:text-(--ui-text-secondary) [&_code]:font-mono [&_h1]:text-3xl [&_h1]:font-bold [&_h2]:mt-6 [&_h2]:text-2xl [&_h2]:font-semibold [&_h3]:mt-5 [&_h3]:text-xl [&_h3]:font-semibold [&_hr]:my-6 [&_hr]:border-border [&_li]:my-1 [&_ol]:list-decimal [&_ol]:pl-6 [&_p]:my-3 [&_pre]:my-4 [&_pre]:overflow-x-auto [&_pre]:rounded-md [&_pre]:border [&_pre]:border-border [&_pre]:bg-card [&_pre]:p-3 [&_table]:my-4 [&_table]:w-full [&_table]:border-collapse [&_td]:border [&_td]:border-border [&_td]:p-2 [&_th]:border [&_th]:border-border [&_th]:p-2 [&_ul]:list-disc [&_ul]:pl-6',
                          children: jsx(Streamdown, { mode: 'static', children: document.text })
                        })
                      ]
                    })
                  : jsx(EmptyState, {
                      description: 'Choose a file from the active Project or enter a relative path.',
                      title: 'Open a Markdown file'
                    })
          })
        ]
      })
    ]
  })
}

function openMarkdownWorkspace(ctx, initialPath = '') {
  if (typeof host.openWorkspace !== 'function') {
    host.notify({ kind: 'error', message: 'Update Hermes Desktop to use the Markdown Viewer workspace.' })
    return
  }
  host.openWorkspace(WORKSPACE_ID, {
    minWidth: 760,
    render: () => jsx(MarkdownWorkspace, { ctx, initialPath }),
    title: 'Markdown'
  })
}

export default {
  id: ID,
  name: 'Markdown Viewer',
  description: 'Browse and render Markdown files from the active Project.',
  register(ctx) {
    installTranscriptLinkBridge(ctx)
    ctx.registerMany([
      {
        id: 'open-command',
        area: PALETTE_AREA,
        data: {
          id: 'markdown-viewer.open',
          keywords: ['markdown', 'md', 'preview', 'file', 'project'],
          label: 'Markdown Viewer: Open Project File',
          run: () => openMarkdownWorkspace(ctx)
        }
      },
      {
        id: 'open-keybind',
        area: KEYBINDS_AREA,
        data: {
          category: 'view',
          defaults: [],
          id: 'markdown-viewer.open',
          label: 'Markdown Viewer: Open Project File',
          run: () => openMarkdownWorkspace(ctx)
        }
      }
    ])
  }
}
