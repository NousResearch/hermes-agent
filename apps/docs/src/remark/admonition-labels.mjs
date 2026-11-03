function walk(node, visit) {
  visit(node);
  if (node.children) for (const child of node.children) walk(child, visit);
}

// Docusaurus translated six default labels but left its legacy aliases
// (important, success) untranslated. Explicit titles remain authored.
const BASE_LABELS = {
  note: 'note',
  tip: 'tip',
  info: 'info',
  warning: 'warning',
  caution: 'caution',
  danger: 'danger',
  important: 'important',
  success: 'success',
};

const LOCALE_LABELS = {
  'zh-Hans': {
    note: '备注',
    tip: '提示',
    info: '信息',
    warning: '注意',
    caution: '警告',
    danger: '危险',
  },
};

/** Locale comes from the doc's path segment, e.g. `content/zh-Hans/...`. */
export function remarkAdmonitionLabels() {
  return (tree, file) => {
    const localeLabels = LOCALE_LABELS[
      String(file.path ?? '')
        .split(/[\\/]/)
        .find((segment) => Object.hasOwn(LOCALE_LABELS, segment))
    ] ?? {};
    const labels = { ...BASE_LABELS, ...localeLabels };
    walk(tree, (node) => {
      if (node.type !== 'containerDirective' || !Object.hasOwn(labels, node.name)) return;
      const hasTitle = (node.children ?? []).some((child) => child.data?.directiveLabel);
      if (hasTitle) return;
      node.children.unshift({
        type: 'paragraph',
        data: { directiveLabel: true },
        children: [{ type: 'text', value: labels[node.name] }],
      });
    });
  };
}
