const DIRECTIVE_MARKERS = { textDirective: ':', leafDirective: '::', containerDirective: ':::' };

function text(value) {
  return { type: 'text', value };
}

function paragraph(children) {
  return { type: 'paragraph', children };
}

function offsets(node) {
  const start = node?.position?.start?.offset;
  const end = node?.position?.end?.offset;
  return Number.isInteger(start) && Number.isInteger(end) ? [start, end] : null;
}

/** Source-equivalent `{key=value …}` longhand, for nodes without source positions. */
function attributesToSource(attributes) {
  const parts = [];
  for (const [key, value] of Object.entries(attributes ?? {})) {
    if (value === undefined || value === null || value === '') parts.push(key);
    else if (/[\s{}]/.test(value)) parts.push(`${key}=${JSON.stringify(value)}`);
    else parts.push(`${key}=${value}`);
  }
  return parts.length ? `{${parts.join(' ')}}` : '';
}

/**
 * The marker line around a directive's (already restored) children, sliced
 * from the source so attributes, empty `[]` and spacing stay exactly as
 * written. Without positions, fall back to rebuilding it from the parse.
 */
function markerText(node, source, inner) {
  const marker = DIRECTIVE_MARKERS[node.type];
  const range = offsets(node);
  if (source && range) {
    const [start, end] = range;
    const lineEnd = node.type === 'containerDirective' ? source.indexOf('\n', start) : -1;
    const stop = lineEnd === -1 || lineEnd > end ? end : lineEnd;
    const innerRange = offsets(inner);
    if (innerRange) return [source.slice(start, innerRange[0]), source.slice(innerRange[1], stop)];
    return [source.slice(start, stop), ''];
  }
  const attributes = attributesToSource(node.attributes);
  return inner ? [`${marker}${node.name}[`, `]${attributes}`] : [`${marker}${node.name}${attributes}`, ''];
}

/** Children of an inline directive span one source range: its `[label]`. */
function inlineSpan(children) {
  const ranges = children.map(offsets);
  if (!children.length || ranges.some((range) => !range)) return null;
  return { position: { start: { offset: ranges[0][0] }, end: { offset: ranges.at(-1)[1] } } };
}

/** The source form of a directive node the pipeline did not consume. */
function restore(node, source) {
  if (node.type !== 'containerDirective') {
    const children = node.children ?? [];
    const [open, close] = markerText(node, source, inlineSpan(children));
    const nodes = [text(open), ...children, ...(close ? [text(close)] : [])];
    return node.type === 'leafDirective' ? [paragraph(nodes)] : nodes;
  }
  // containerDirective: `[label]` is the first paragraph flagged as the label;
  // every other child is block content, not part of the marker line.
  const label = (node.children ?? []).find((child) => child.type === 'paragraph' && child.data?.directiveLabel);
  const body = (node.children ?? []).filter((child) => child !== label);
  const [open, close] = markerText(node, source, label && inlineSpan(label.children));
  const opening = paragraph([text(open), ...(label?.children ?? []), ...(close ? [text(close)] : [])]);
  // Only a container the author closed gets a closing marker back.
  const range = offsets(node);
  const closed = !source || !range || /(^|\n)[ \t]*:{3,}[ \t]*$/.test(source.slice(range[0], range[1]).trimEnd());
  return [opening, ...body, ...(closed ? [paragraph([text(DIRECTIVE_MARKERS.containerDirective)])] : [])];
}

/**
 * Directives nothing else consumes render as empty boxes and swallow their
 * text (`9:00` → `9`, `host:9090` headings lose their port and anchor), so
 * turn every surviving directive back into its source text. Configure after
 * remarkDirectiveAdmonition: known admonition containers are already
 * Callout components by then and are left untouched.
 */
export function remarkPreserveDirectives() {
  return (tree, file) => {
    const source = typeof file?.value === 'string' ? file.value : file?.value ? String(file.value) : '';
    function visit(node) {
      if (!node.children) return;
      const kept = [];
      for (const child of node.children) {
        // Restore nested directives first so spliced bodies are already clean.
        visit(child);
        if (Object.hasOwn(DIRECTIVE_MARKERS, child.type)) kept.push(...restore(child, source));
        else kept.push(child);
      }
      node.children = kept;
    }
    visit(tree);
  };
}
