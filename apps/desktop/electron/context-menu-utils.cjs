const FILE_TREE_CONTEXT_SELECTOR = '[data-hermes-file-tree-path]'

function coordinate(value) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : 0
}

function buildFileTreeContextLookupScript(x, y) {
  const px = coordinate(x)
  const py = coordinate(y)

  return `(() => {
    const target = document.elementFromPoint(${JSON.stringify(px)}, ${JSON.stringify(py)});
    const node = target && typeof target.closest === 'function'
      ? target.closest(${JSON.stringify(FILE_TREE_CONTEXT_SELECTOR)})
      : null;
    if (!node) return null;
    const path = node.getAttribute('data-hermes-file-tree-path') || '';
    if (!path) return null;
    return {
      path,
      name: node.getAttribute('data-hermes-file-tree-name') || '',
      isDirectory: node.getAttribute('data-hermes-file-tree-is-directory') === 'true'
    };
  })()`
}

function normalizeFileTreeContext(value) {
  if (!value || typeof value !== 'object') {
    return null
  }

  const rawPath = typeof value.path === 'string' ? value.path : ''
  const path = rawPath.trim()

  if (!path) {
    return null
  }

  return {
    path,
    name: typeof value.name === 'string' ? value.name : '',
    isDirectory: value.isDirectory === true
  }
}

module.exports = {
  FILE_TREE_CONTEXT_SELECTOR,
  buildFileTreeContextLookupScript,
  normalizeFileTreeContext
}
