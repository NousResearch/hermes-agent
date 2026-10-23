import { host } from '@hermes/plugin-sdk'
import { Children, cloneElement, isValidElement, type ReactNode } from 'react'

const ABSOLUTE_PATH =
  /(?:^|(?<=[\s([{"'`]))(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+\\[^\\/\s]+\\|\/(?!\/))[\p{L}\p{N}_.~+@/\\-]+/gu

function isFileReference(path: string): boolean {
  const body = path.replace(/^[A-Za-z]:/, '')

  return body.split(/[\\/]/).filter(Boolean).length > 1 || /\.[\p{L}\p{N}]{1,12}$/u.test(body)
}

function linkText(text: string): ReactNode {
  const nodes: ReactNode[] = []
  let cursor = 0

  for (const match of text.matchAll(ABSOLUTE_PATH)) {
    const raw = match[0]
    const path = raw.replace(/\.+$/, '')

    if (!isFileReference(path)) {continue}
    nodes.push(text.slice(cursor, match.index))
    nodes.push(
      <button
        aria-label={`Reveal ${path} in file tree`}
        className="ref kanban-filepath-link cursor-pointer"
        key={`${match.index}-${path}`}
        onClick={() => host.revealFileInTree(path)}
        title={path}
        type="button"
      >
        {path}
      </button>
    )
    // Keep punctuation outside the button instead of consuming it with the path.
    cursor = match.index + path.length
  }

  nodes.push(text.slice(cursor))

  return nodes
}

interface InlineProps {
  children?: ReactNode
  code?: string
  href?: string
  node?: { tagName?: string }
}

/** Decorate rendered inline nodes, not Markdown source: links and fenced code
 * keep their own semantics. Inline code and emphasis retain their wrapper. */
export function decorateFilePaths(children: ReactNode): ReactNode {
  return Children.map(children, child => {
    if (typeof child === 'string') {return linkText(child)}

    if (!isValidElement<InlineProps>(child)) {return child}
    const tag = typeof child.type === 'string' ? child.type : child.props.node?.tagName

    if (child.props.href || tag === 'a' || tag === 'pre' || child.props.code !== undefined) {return child}

    if (child.props.children === undefined) {return child}

    return cloneElement(child, {}, decorateFilePaths(child.props.children))
  })
}
