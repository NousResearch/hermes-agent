import { useCallback, useEffect, useMemo, useState } from 'react'
import { type NodeRendererProps, Tree } from 'react-arborist'

import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'
import type { BrowserTabState } from '@/store/browser'
import { GUEST_HANDLE_HELPERS_SOURCE, highlightGuestElement, inspectGuestElement, runGuestScript } from '@/store/browser-bridge'
import { type SelectedElement, setBrowserSelection } from '@/store/browser-guest-state'

import { getFileTreeDndManager } from '../../right-sidebar/files/dnd-manager'

/**
 * Component / DOM tree panel.
 *
 * Walks the guest page (document.body down) into a JSON-safe tree via
 * `runGuestScript`, labels each node with its best-effort React component name
 * (falling back to the tag), renders it with react-arborist, and on select
 * highlights + inspects the node so it flows into the shared browser selection.
 */

const INDENT = 12
const MAX_TREE_HEIGHT = 176
const MAX_TREE_NODES = 3000
const ROW_HEIGHT = 22

interface TreeNodeData {
  attrs?: { class?: string; id?: string }
  children: TreeNodeData[]
  handle: string
  id: string
  name: string
  tag: string
}

interface TreeSnapshot {
  react: boolean
  root: TreeNodeData
  truncated: boolean
}

interface InspectDetail {
  attributes?: Record<string, string>
  className?: string
  cssPath?: string
  htmlPreview?: string
  layout?: { height: number; width: number; x: number; y: number }
  ref?: string
  role?: string
  stableRef?: string
  styles?: Record<string, string>
  tag?: string
  text?: string
}

// Built once at module load. `GUEST_HANDLE_HELPERS_SOURCE` supplies
// `HERMES_HANDLE_ATTR`, `hermesResolveHandle`, etc. inside the IIFE. NAMES ONLY
// are read off the React fiber — fibers/props are never serialized (they cycle).
// O(nodes * depth) fiber walk; nodes are capped at MAX so it stays cheap.

const TREE_BUILDER_IIFE = `(() => {
  ${GUEST_HANDLE_HELPERS_SOURCE}
  var MAX = ${MAX_TREE_NODES};
  var count = 0;
  var truncated = false;
  var hasReact = false;
  var resolveReactName = function (el) {
    var keys = Object.keys(el);
    var key = null;
    for (var i = 0; i < keys.length; i += 1) {
      if (/^__reactFiber\\$/.test(keys[i]) || /^__reactInternalInstance\\$/.test(keys[i])) { key = keys[i]; break; }
    }
    if (!key) { return null; }
    hasReact = true;
    var fiber = el[key];
    var guard = 0;
    while (fiber && guard < 2000) {
      var type = fiber.type;
      if (typeof type === 'function') { return type.displayName || type.name || null; }
      if (type && typeof type === 'object') {
        var inner = type;
        var unwrap = 0;
        while (inner && typeof inner === 'object' && (inner.type || inner.render) && unwrap < 12) {
          inner = inner.type || inner.render;
          unwrap += 1;
        }
        if (typeof inner === 'function') { return inner.displayName || inner.name || null; }
      }
      fiber = fiber.return;
      guard += 1;
    }
    return null;
  };
  var shouldSkip = function (el) {
    var tag = el.tagName;
    if (tag === 'SCRIPT' || tag === 'STYLE') { return true; }
    if (el.hasAttribute('data-hermes-highlight') || el.hasAttribute('data-hermes-overlay')) { return true; }
    return false;
  };
  var build = function (el, id) {
    count += 1;
    var tag = el.tagName ? el.tagName.toLowerCase() : 'node';
    var name = tag;
    try { var reactName = resolveReactName(el); if (reactName) { name = reactName; } } catch (e) {}
    try { el.setAttribute(HERMES_HANDLE_ATTR, id); } catch (e) {}
    var attrs = {};
    var rawId = el.getAttribute ? el.getAttribute('id') : null;
    if (rawId) { attrs.id = String(rawId).slice(0, 100); }
    var rawClass = el.getAttribute ? el.getAttribute('class') : null;
    if (rawClass) { attrs.class = String(rawClass).slice(0, 200); }
    var children = [];
    var kids = el.children || [];
    for (var j = 0; j < kids.length; j += 1) {
      if (count >= MAX) { truncated = true; break; }
      var child = kids[j];
      if (shouldSkip(child)) { continue; }
      children.push(build(child, id + '.' + j));
    }
    return { attrs: attrs, children: children, handle: id, id: id, name: name, tag: tag };
  };
  var body = document.body;
  var root = body ? build(body, '0') : { attrs: {}, children: [], handle: '0', id: '0', name: 'body', tag: 'body' };
  return { react: hasReact, root: root, truncated: truncated };
})()`

function isTreeNode(value: unknown): value is TreeNodeData {
  if (!value || typeof value !== 'object') {
    return false
  }

  const node = value as Record<string, unknown>

  return (
    Array.isArray(node.children) &&
    typeof node.handle === 'string' &&
    typeof node.id === 'string' &&
    typeof node.name === 'string' &&
    typeof node.tag === 'string'
  )
}

function isTreeSnapshot(value: unknown): value is TreeSnapshot {
  if (!value || typeof value !== 'object') {
    return false
  }

  return isTreeNode((value as { root?: unknown }).root)
}

function countNodes(node: TreeNodeData): number {
  let total = 1

  for (const child of node.children) {
    total += countNodes(child)
  }

  return total
}

function toSelectedElement(detail: InspectDetail, node: TreeNodeData, url: string): SelectedElement {
  const layout = detail.layout

  return {
    at: Date.now(),
    attributes: detail.attributes,
    className: detail.className,
    componentName: node.name === node.tag ? undefined : node.name,
    cssPath: detail.cssPath,
    htmlPreview: detail.htmlPreview,
    layout: layout ? { height: layout.height, width: layout.width, x: layout.x, y: layout.y } : undefined,
    ref: typeof detail.ref === 'string' && detail.ref ? detail.ref : node.handle,
    role: detail.role,
    stableRef: detail.stableRef,
    styles: detail.styles,
    tag: typeof detail.tag === 'string' ? detail.tag : node.tag,
    text: detail.text,
    url
  }
}

function NodeRow({ node, style }: NodeRendererProps<TreeNodeData>) {
  const data = node.data
  const showTag = data.name !== data.tag

  return (
    <div
      aria-expanded={node.isLeaf ? undefined : node.isOpen}
      className={cn(
        'flex h-full cursor-pointer select-none items-center gap-1 rounded-sm pr-2 hover:bg-(--ui-row-hover-background) hover:text-foreground',
        node.isSelected && 'bg-(--ui-row-active-background) text-foreground'
      )}
      style={style}
      title={data.attrs?.id ? `${data.tag}#${data.attrs.id}` : data.tag}
    >
      <span
        aria-hidden
        className="grid w-3.5 shrink-0 place-items-center text-(--ui-text-tertiary)"
        onClick={event => {
          event.stopPropagation()
          node.toggle()
        }}
      >
        {node.isLeaf ? null : <Codicon name={node.isOpen ? 'chevron-down' : 'chevron-right'} size="0.7rem" />}
      </span>
      <span className="truncate text-(--ui-text-primary)">{data.name}</span>
      {showTag ? <span className="shrink-0 text-(--ui-text-quaternary)">{data.tag}</span> : null}
    </div>
  )
}

export function ComponentTreePanel({ tab }: { tab: BrowserTabState }) {
  const [snapshot, setSnapshot] = useState<null | TreeSnapshot>(null)
  const [error, setError] = useState<null | string>(null)
  const [loading, setLoading] = useState(true)
  const [open, setOpen] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)

    try {
      const result = await runGuestScript(tab.id, TREE_BUILDER_IIFE, 'observe')

      if (!isTreeSnapshot(result)) {
        throw new Error('The page tree could not be read.')
      }

      setSnapshot(result)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
      setSnapshot(null)
    } finally {
      setLoading(false)
    }
    // tab.url isn't read in the body, but it must retrigger load so the tree
    // rebuilds when the page navigates (the prior handles go stale).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab.id, tab.url])

  useEffect(() => {
    void load()
  }, [load])

  const handleSelect = useCallback(
    (node: TreeNodeData) => {
      void highlightGuestElement(tab.id, node.handle).catch(() => undefined)
      void inspectGuestElement(tab.id, node.handle)
        .then(detail => {
          if (detail && typeof detail === 'object') {
            setBrowserSelection(tab.id, toSelectedElement(detail as InspectDetail, node, tab.url))
          }
        })
        .catch(() => undefined)
    },
    [tab.id, tab.url]
  )

  const treeHeight = useMemo(() => {
    if (!snapshot) {
      return ROW_HEIGHT
    }

    return Math.min(Math.max(countNodes(snapshot.root), 1) * ROW_HEIGHT, MAX_TREE_HEIGHT)
  }, [snapshot])

  return (
    <div className="border-t border-(--ui-stroke-tertiary) px-3 py-2 text-[0.68rem] text-(--ui-text-secondary)">
      <div className="mb-2 flex items-center justify-between gap-2 text-(--ui-text-tertiary)">
        <button
          aria-expanded={open}
          className="flex items-center gap-1 font-medium uppercase tracking-wide hover:text-foreground"
          onClick={() => setOpen(value => !value)}
          type="button"
        >
          <Codicon name={open ? 'chevron-down' : 'chevron-right'} size="0.7rem" />
          Components
        </button>
        <button
          className="rounded-md px-2 py-1 hover:bg-(--ui-control-hover-background) hover:text-foreground disabled:opacity-40"
          disabled={loading}
          onClick={() => void load()}
          type="button"
        >
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>
      <div className={open ? undefined : 'hidden'}>
      {error ? (
        <div className="rounded-md bg-(--ui-editor-surface-background) px-2 py-2" role="alert">
          <p className="text-(--ui-red)">Could not read the page tree.</p>
          <p className="mt-1 break-words text-(--ui-text-tertiary)">{error}</p>
          <button
            className="mt-2 rounded-md border border-(--ui-stroke-tertiary) px-2 py-1 hover:bg-(--ui-control-hover-background) hover:text-foreground"
            onClick={() => void load()}
            type="button"
          >
            Retry
          </button>
        </div>
      ) : loading && !snapshot ? (
        <p className="text-(--ui-text-tertiary)">Reading the component tree…</p>
      ) : !snapshot || snapshot.root.children.length === 0 ? (
        <p className="text-(--ui-text-tertiary)">No elements found on this page.</p>
      ) : (
        <>
          {snapshot.react ? null : (
            <p className="mb-1 text-(--ui-text-tertiary)">No React detected — showing DOM tree (best-effort).</p>
          )}
          {snapshot.truncated ? (
            <p className="mb-1 text-(--ui-text-quaternary)">Large page — tree truncated to {MAX_TREE_NODES} nodes.</p>
          ) : null}
          <div className="max-h-44 overflow-auto">
            <Tree<TreeNodeData>
              childrenAccessor={node => (node.children.length > 0 ? node.children : null)}
              data={[snapshot.root]}
              disableDrag
              disableDrop
              dndManager={getFileTreeDndManager()}
              height={treeHeight}
              indent={INDENT}
              onSelect={nodes => {
                const selected = nodes[0]

                if (selected?.data) {
                  handleSelect(selected.data)
                }
              }}
              openByDefault
              rowHeight={ROW_HEIGHT}
              width="100%"
            >
              {NodeRow}
            </Tree>
          </div>
        </>
      )}
      </div>
    </div>
  )
}
