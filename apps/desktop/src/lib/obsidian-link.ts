type MarkdownNode = {
  children?: MarkdownNode[]
  type?: string
  url?: string
}

const OBSIDIAN_FRAGMENT_PREFIX = '#obsidian:'

/**
 * Carry Markdown-authored Obsidian URIs through rehype-sanitize as inert
 * fragments. The renderer restores them only for an explicit user click.
 */
export function remarkObsidianLinks() {
  return (tree: MarkdownNode) => {
    const walk = (node: MarkdownNode) => {
      if (node.type === 'link' && node.url && /^obsidian:/i.test(node.url)) {
        node.url = `${OBSIDIAN_FRAGMENT_PREFIX}${encodeURIComponent(node.url)}`
      }

      node.children?.forEach(walk)
    }

    walk(tree)
  }
}

export function obsidianHrefFromMarkdownHref(href?: string): string | null {
  if (!href?.startsWith(OBSIDIAN_FRAGMENT_PREFIX)) {
    return null
  }

  try {
    const decoded = decodeURIComponent(href.slice(OBSIDIAN_FRAGMENT_PREFIX.length))

    return /^obsidian:/i.test(decoded) ? decoded : null
  } catch {
    return null
  }
}
