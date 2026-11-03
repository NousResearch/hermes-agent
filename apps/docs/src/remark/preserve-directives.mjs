const DIRECTIVE_MARKERS = { textDirective: ':', leafDirective: '::', containerDirective: ':::' };

function text(value) {
  return { type: 'text', value };
}

function paragraph(children) {
  return { type: 'paragraph', children };
}

/** Source-equivalent `{key=value …}` longhand for directive attributes. */
function attributesToSource(attributes) {
  const parts = [];
  for (const [key, value] of Object.entries(attributes ?? {})) {
    if (value === undefined || value === null || value === '') parts.push(key);
    else if (/[\s{}]/.test(value)) parts.push(`${key}=${JSON.stringify(value)}`);
    else parts.push(`${key}=${value}`);
  }
  return parts.length ? `{${parts.join(' ')}}` : '';
}

/** `:name[children]{attrs}` (or its bare `:name` form) as inline nodes.
 * Only for directives whose children are inline content (text and leaf). */
function inlineSource(node, marker, attributes) {
  const children = node.children ?? [];
  if (!children.length) return [text(`${marker}${node.name}${attributes}`)];
  return [text(`${marker}${node.name}[`), ...children, text(`]${attributes}`)];
}

/** The source form of a directive node the pipeline did not consume. */
function restore(node) {
  const marker = DIRECTIVE_MARKERS[node.type];
  const attributes = attributesToSource(node.attributes);
  if (node.type === 'textDirective') return inlineSource(node, marker, attributes);
  if (node.type === 'leafDirective') return [paragraph(inlineSource(node, marker, attributes))];
  // containerDirective: `[label]` is the first paragraph flagged as the label;
  // every other child is block content, not part of the marker line.
  const label = (node.children ?? []).find((child) => child.type === 'paragraph' && child.data?.directiveLabel);
  const body = (node.children ?? []).filter((child) => child !== label);
  const open = label
    ? [text(`${marker}${node.name}[`), ...label.children, text(`]${attributes}`)]
    : [text(`${marker}${node.name}${attributes}`)];
  return [paragraph(open), ...body, paragraph([text(marker)])];
}

/**
 * Directives nothing else consumes render as empty boxes and swallow their
 * text (`9:00` → `9`, `host:9090` headings lose their port and anchor), so
 * turn every surviving directive back into its source text. Configure after
 * remarkDirectiveAdmonition: known admonition containers are already
 * Callout components by then and are left untouched.
 */
export function remarkPreserveDirectives() {
  return (tree) => {
    function visit(node) {
      if (!node.children) return;
      const kept = [];
      for (const child of node.children) {
        // Restore nested directives first so spliced bodies are already clean.
        visit(child);
        if (Object.hasOwn(DIRECTIVE_MARKERS, child.type)) kept.push(...restore(child));
        else kept.push(child);
      }
      node.children = kept;
    }
    visit(tree);
  };
}
