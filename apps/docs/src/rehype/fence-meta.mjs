function walk(node, visit) {
  visit(node);
  if (node.children) for (const child of node.children) walk(child, visit);
}

/** Carry fence metadata through rehypeRaw via the highlighter's native
 * attribute fallback; the raw re-parse drops HAST data.meta. */
export function rehypeFenceMeta() {
  return (tree) => {
    walk(tree, (node) => {
      if (node.type !== 'element' || node.tagName !== 'pre') return;
      const code = (node.children ?? []).find((child) => child.type === 'element' && child.tagName === 'code');
      if (code?.data?.meta) code.properties.metastring = code.data.meta;
    });
  };
}
