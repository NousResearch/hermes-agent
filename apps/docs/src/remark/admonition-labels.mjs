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

/** The default label for an untitled callout, in the locale being viewed. */
export function calloutLabel(type, locale) {
  return LOCALE_LABELS[locale]?.[type] ?? BASE_LABELS[type] ?? type;
}

/**
 * Give untitled callouts a default label. The label is a component, not text:
 * a zh-Hans route without its own translation renders the English source, so
 * the locale is known only when the page renders, never from the file path.
 */
export function remarkAdmonitionLabels() {
  return (tree) => {
    walk(tree, (node) => {
      if (node.type !== 'containerDirective' || !Object.hasOwn(BASE_LABELS, node.name)) return;
      const hasTitle = (node.children ?? []).some((child) => child.data?.directiveLabel);
      if (hasTitle) return;
      node.children.unshift({
        type: 'paragraph',
        data: { directiveLabel: true },
        children: [{
          type: 'mdxJsxTextElement',
          name: 'DefaultCalloutLabel',
          attributes: [{ type: 'mdxJsxAttribute', name: 'type', value: node.name }],
          children: [],
        }],
      });
    });
  };
}
