/**
 * Preview Translator — opt-in status-bar chip that translates the focused
 * preview Browser tab in place. Guest access goes through host.preview
 * (the same script runner as drive_preview), never document.querySelector('webview').
 */
import {
  Badge,
  Button,
  Codicon,
  host,
  KEYBINDS_AREA,
  PALETTE_AREA,
  Popover,
  PopoverContent,
  PopoverTrigger,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  STATUSBAR_AREAS
} from '@hermes/plugin-sdk'
import { Fragment, useEffect, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const ID = 'preview-translator'
const STORAGE_LANG = 'targetLang'
const STORAGE_MODEL = 'model'

const LANGUAGES = [
  { id: 'English', name: 'English' },
  { id: 'Spanish', name: 'Spanish' },
  { id: 'Catalan', name: 'Catalan' },
  { id: 'German', name: 'German' },
  { id: 'French', name: 'French' },
  { id: 'Italian', name: 'Italian' },
  { id: 'Portuguese', name: 'Portuguese' },
  { id: 'Dutch', name: 'Dutch' },
  { id: 'Russian', name: 'Russian' },
  { id: 'Japanese', name: 'Japanese' },
  { id: 'Chinese', name: 'Chinese' }
]

let rest = null
let storage = null

const EXTRACT_SCRIPT = `
(function() {
  if (!document.body) return [];
  window.__hermes_node_map = new Map();
  window.__hermes_orig_map = window.__hermes_orig_map || new Map();
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
    acceptNode: function(node) {
      const parent = node.parentElement;
      if (!parent) return NodeFilter.FILTER_REJECT;
      const tag = parent.tagName.toLowerCase();
      if (['script','style','noscript','textarea','code','pre','svg','math'].includes(tag)) return NodeFilter.FILTER_REJECT;
      const text = node.nodeValue ? node.nodeValue.trim() : '';
      if (!text || text.length < 2) return NodeFilter.FILTER_SKIP;
      if (/^[\\d\\s.,:;!?()\\[\\]{}#@$%&*+\\-_=<>/\\\\|'"]+$/.test(text)) return NodeFilter.FILTER_SKIP;
      return NodeFilter.FILTER_ACCEPT;
    }
  });
  const items = [];
  let node;
  let id = 0;
  while ((node = walker.nextNode())) {
    if (!window.__hermes_orig_map.has(node)) window.__hermes_orig_map.set(node, node.nodeValue);
    window.__hermes_node_map.set(id, node);
    items.push({ id: id++, text: node.nodeValue });
  }
  return items;
})()
`

const RESTORE_SCRIPT = `
(function() {
  if (!window.__hermes_orig_map) return 0;
  let restored = 0;
  for (const [node, orig] of window.__hermes_orig_map.entries()) {
    if (node) { node.nodeValue = orig; restored++; }
  }
  return restored;
})()
`

function buildApplyScript(translations) {
  return `
(function(translations) {
  if (!window.__hermes_node_map) return 0;
  let updated = 0;
  translations.forEach(function(trans, id) {
    const node = window.__hermes_node_map.get(id);
    if (node && trans != null) { node.nodeValue = trans; updated++; }
  });
  return updated;
})(${JSON.stringify(translations)})
`
}

async function runInGuest(code) {
  if (!host.preview?.isAvailable()) {
    throw new Error('No preview page is open.')
  }

  return host.preview.executeJavaScript(code)
}

async function runTranslation({ targetLang, selectedModel, setStatus }) {
  if (!host.preview?.isAvailable()) {
    host.notify({ kind: 'warning', title: 'Translate Preview', message: 'Open a page in the preview browser first.' })

    return
  }

  if (!rest) {
    host.notify({
      kind: 'error',
      title: 'Translate Preview',
      message: 'Translation backend is unavailable. Restart the gateway if you just updated.'
    })

    return
  }

  try {
    setStatus('Extracting page text…')
    const items = await runInGuest(EXTRACT_SCRIPT)

    if (!items || items.length === 0) {
      setStatus('No translatable text on this page.')
      host.notify({ kind: 'info', title: 'Translate Preview', message: 'No translatable text found on this page.' })

      return
    }

    const texts = items.map(item => item.text)
    setStatus(`Translating ${texts.length} strings with ${selectedModel || 'auto'}…`)
    const res = await rest('/translate', {
      body: { model: selectedModel, target_lang: targetLang, texts },
      method: 'POST',
      timeoutMs: 180000
    })

    if (!res || !Array.isArray(res.translations)) {
      throw new Error('Invalid response from translation API')
    }

    setStatus('Writing translations into the page…')
    const updatedCount = await runInGuest(buildApplyScript(res.translations))
    setStatus(`Translated ${updatedCount} elements → ${targetLang}`)
    host.notify({
      kind: 'success',
      message: `${updatedCount} elements → ${targetLang} (${res.model_used})`,
      title: 'Page translated'
    })
  } catch (err) {
    setStatus(`Error: ${err.message || String(err)}`)
    host.notifyError(err, 'Translation failed')
  }
}

async function runRestore({ setStatus }) {
  if (!host.preview?.isAvailable()) {
    host.notify({ kind: 'warning', title: 'Restore Original', message: 'Open a page in the preview browser first.' })

    return
  }

  try {
    const restoredCount = await runInGuest(RESTORE_SCRIPT)
    setStatus(`Restored ${restoredCount} original strings.`)
    host.notify({ kind: 'info', message: `Restored ${restoredCount} text nodes.`, title: 'Original restored' })
  } catch (err) {
    setStatus(`Error: ${err.message || String(err)}`)
    host.notifyError(err, 'Restore failed')
  }
}

function persist(lang, model) {
  if (!storage) {
    return
  }

  void storage.set(STORAGE_LANG, lang)
  void storage.set(STORAGE_MODEL, model)
}

function TranslatorPanel() {
  const [targetLang, setTargetLang] = useState('English')
  const [selectedModel, setSelectedModel] = useState('auto')
  const [models, setModels] = useState([{ id: 'auto', name: 'Auto (session / main model)' }])
  const [status, setStatus] = useState('Ready')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (storage) {
      Promise.all([storage.get(STORAGE_LANG), storage.get(STORAGE_MODEL)])
        .then(([lang, model]) => {
          if (typeof lang === 'string' && lang) {
            setTargetLang(lang)
          }

          if (typeof model === 'string' && model) {
            setSelectedModel(model)
          }
        })
        .catch(() => {})
    }

    if (!rest) {
      return
    }

    rest('/models')
      .then(data => {
        if (data && Array.isArray(data.models) && data.models.length) {
          setModels(data.models)
        }
      })
      .catch(() => {})
  }, [])

  const onLang = value => {
    setTargetLang(value)
    persist(value, selectedModel)
  }
  const onModel = value => {
    setSelectedModel(value)
    persist(targetLang, value)
  }

  return jsxs('div', {
    children: [
      jsxs('div', {
        className: 'flex items-center justify-between border-b border-(--ui-stroke-secondary) pb-2',
        children: [
          jsxs('div', {
            children: [jsx(Codicon, { name: 'globe', size: '0.85rem' }), 'Preview Translator'],
            className: 'flex items-center gap-1.5 font-medium text-foreground'
          }),
          jsx(Badge, { children: 'LLM', variant: 'outline' })
        ]
      }),
      jsxs('div', {
        children: [
          jsx('label', {
            children: 'Target language',
            className: 'text-[0.65rem] font-medium uppercase text-(--ui-text-tertiary)'
          }),
          jsx(Select, {
            children: jsxs(Fragment, {
              children: [
                jsx(SelectTrigger, { children: jsx(SelectValue, {}), className: 'h-7 text-xs' }),
                jsx(SelectContent, {
                  children: LANGUAGES.map(lang => jsx(SelectItem, { children: lang.name, value: lang.id }, lang.id))
                })
              ]
            }),
            onValueChange: onLang,
            value: targetLang
          })
        ],
        className: 'flex flex-col gap-1'
      }),
      jsxs('div', {
        children: [
          jsx('label', {
            children: 'Translation model',
            className: 'text-[0.65rem] font-medium uppercase text-(--ui-text-tertiary)'
          }),
          jsx(Select, {
            children: jsxs(Fragment, {
              children: [
                jsx(SelectTrigger, { children: jsx(SelectValue, {}), className: 'h-7 text-xs' }),
                jsx(SelectContent, {
                  children: models.map(m => jsx(SelectItem, { children: m.name, value: m.id }, m.id))
                })
              ]
            }),
            onValueChange: onModel,
            value: selectedModel
          })
        ],
        className: 'flex flex-col gap-1'
      }),
      jsxs('div', {
        children: [
          jsx(Button, {
            children: busy ? 'Working…' : 'Translate',
            className: 'flex-1 text-[0.7rem] uppercase',
            disabled: busy,
            onClick: async () => {
              setBusy(true)
              await runTranslation({ selectedModel, setStatus, targetLang })
              setBusy(false)
            },
            size: 'sm'
          }),
          jsx(Button, {
            children: 'Restore',
            className: 'text-[0.7rem] uppercase',
            disabled: busy,
            onClick: async () => {
              setBusy(true)
              await runRestore({ setStatus })
              setBusy(false)
            },
            size: 'sm',
            variant: 'outline'
          })
        ],
        className: 'flex gap-2 pt-1'
      }),
      jsx('div', { children: status, className: 'text-[0.68rem] text-(--ui-text-secondary)' })
    ],
    className: 'flex w-72 flex-col gap-3 p-3 text-xs'
  })
}

function StatusWidget() {
  const [open, setOpen] = useState(false)

  return jsxs(Popover, {
    children: [
      jsx(PopoverTrigger, {
        asChild: true,
        children: jsxs(Button, {
          children: [jsx(Codicon, { name: 'globe', size: '0.75rem' }), jsx('span', { children: 'Translate' })],
          className: 'gap-1 px-1.5 text-[0.6875rem] text-(--ui-text-tertiary) hover:text-foreground',
          size: 'micro',
          title: 'Translate the preview webpage',
          variant: 'ghost'
        })
      }),
      jsx(PopoverContent, {
        align: 'end',
        children: jsx(TranslatorPanel, {}),
        className: 'p-0',
        side: 'top'
      })
    ],
    onOpenChange: setOpen,
    open
  })
}

function currentPrefs() {
  return Promise.all([
    storage ? storage.get(STORAGE_LANG) : Promise.resolve(null),
    storage ? storage.get(STORAGE_MODEL) : Promise.resolve(null)
  ]).then(([lang, model]) => ({
    selectedModel: typeof model === 'string' && model ? model : 'auto',
    targetLang: typeof lang === 'string' && lang ? lang : 'English'
  }))
}

export default {
  defaultEnabled: false,
  description: 'In-page translation for the preview browser. Pick the LLM in the status-bar Translate menu.',
  id: ID,
  name: 'Preview Translator',
  register(ctx) {
    rest = ctx.rest
    storage = ctx.storage

    ctx.register({
      area: STATUSBAR_AREAS.right,
      id: 'status-bar-translator',
      order: 15,
      render: () => jsx(StatusWidget, {})
    })

    const translateNow = () => {
      currentPrefs().then(prefs => runTranslation({ ...prefs, setStatus: () => {} }))
    }
    const restoreNow = () => runRestore({ setStatus: () => {} })

    ctx.registerMany([
      {
        area: PALETTE_AREA,
        data: {
          action: 'previewTranslator.translate',
          id: 'previewTranslator.translate',
          keywords: ['translate', 'page', 'preview', 'language', 'spanish', 'catalan'],
          label: 'Translate preview page',
          run: translateNow
        },
        id: 'translate-page'
      },
      {
        area: PALETTE_AREA,
        data: {
          id: 'previewTranslator.restore',
          keywords: ['restore', 'original', 'translate', 'preview'],
          label: 'Restore original preview page',
          run: restoreNow
        },
        id: 'restore-original'
      },
      {
        area: KEYBINDS_AREA,
        data: {
          category: 'view',
          defaults: ['mod+alt+t'],
          id: 'previewTranslator.translate',
          label: 'Translate preview page',
          run: translateNow
        },
        id: 'translate-keybind'
      }
    ])
  }
}
