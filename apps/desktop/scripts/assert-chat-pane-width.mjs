#!/usr/bin/env node
// CW-02 contract: main chat thread + composer fill the pane.
// Source-level (no Electron). Gutters (padding-inline) are allowed;
// a leftover 100%-2rem / leftover max-width column is not.
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const styles = fs.readFileSync(path.join(root, 'src/styles.css'), 'utf8')
const fallback = fs.readFileSync(path.join(root, 'src/app/chat/composer/index.tsx'), 'utf8')
const thread = fs.readFileSync(path.join(root, 'src/components/assistant-ui/thread/list.tsx'), 'utf8')
const errors = []

const assert = (ok, msg) => {
  if (!ok) errors.push(msg)
}

assert(
  /--composer-width:\s*100%;/.test(styles),
  '--composer-width must stay 100% of the chat pane',
)
assert(
  /\[data-slot='aui_thread-content'\] \{[\s\S]*?width:\s*100%;[\s\S]*?max-width:\s*var\(--composer-width\);/.test(
    styles,
  ),
  "[data-slot='aui_thread-content'] must be width 100% / max-width var(--composer-width)",
)
assert(
  /padding-inline:\s*1\.5rem;/.test(styles),
  'thread padding-inline gutters must remain',
)

const dockBlock = styles.match(/\[data-slot='composer-dock'\] \{[\s\S]*?\n\}/)
assert(Boolean(dockBlock), 'composer-dock rule missing')
if (dockBlock) {
  assert(
    !/100%\s*-\s*2rem/.test(dockBlock[0]),
    'composer-dock must not min() against 100%-2rem leftover column',
  )
  assert(
    /width:\s*calc\(var\(--composer-width\)\s*\+\s*10px\)/.test(dockBlock[0]),
    'composer-dock width must track --composer-width (+10px grab)',
  )
}

assert(
  !/w-\[min\(var\(--composer-width\),calc\(100%-2rem\)\)\]/.test(fallback),
  'ChatBarFallback must not use leftover 100%-2rem width',
)
assert(
  /w-\(--composer-width\)/.test(fallback),
  'ChatBarFallback must use w-(--composer-width)',
)
assert(
  /max-w-\(--composer-width\)/.test(thread) && /data-slot="aui_thread-content"/.test(thread),
  'thread list must mount max-w-(--composer-width) on aui_thread-content',
)
assert(
  /\[data-slot='tool-block'\]\[data-delegate-card\] \{[\s\S]*?max-width:\s*75%;/.test(styles),
  'delegation cards may stay max-width 75%',
)
assert(
  /--sidebar-width:/.test(styles),
  'sidebar still has its own width token (must not steal chat-pane fill)',
)

if (errors.length) {
  console.error('CW-02 chat pane width contract failed:')
  for (const e of errors) console.error(' -', e)
  process.exit(1)
}
console.log('CW-02 chat pane width contract: ok')
