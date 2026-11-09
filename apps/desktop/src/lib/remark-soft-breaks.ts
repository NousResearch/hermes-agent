export interface MdastNode {
  children?: MdastNode[]
  type: string
  value?: string
}

/** Render single newlines in prose as line breaks, the way chat replies read
 *  (poems, addresses, one-item-per-line lists). Only `text` nodes are split,
 *  so code, inline code and math keep their own whitespace. */
export function remarkSoftBreaks() {
  return (tree: MdastNode) => {
    const visit = (node: MdastNode) => {
      if (!node.children) {
        return
      }

      node.children = node.children.flatMap(child => {
        if (child.type !== 'text' || !child.value?.includes('\n')) {
          visit(child)

          return [child]
        }

        return child.value
          .split(/[ \t]*\r?\n[ \t]*/)
          .flatMap((value, index): MdastNode[] => [
            ...(index > 0 ? [{ type: 'break' }] : []),
            ...(value ? [{ type: 'text', value }] : [])
          ])
      })
    }

    visit(tree)
  }
}
