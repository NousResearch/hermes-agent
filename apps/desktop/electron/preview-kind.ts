const HTML_EXTENSIONS = new Set(['.html', '.htm'])
const PDF_EXTENSIONS = new Set(['.pdf'])
const MODEL3D_EXTENSIONS = new Set(['.3mf', '.glb', '.gltf', '.obj', '.ply', '.stl'])

export type PreviewKind = 'binary' | 'html' | 'image' | 'model3d' | 'pdf' | 'text'

/** Right-rail preview kind for a local file, from its extension, MIME type and binary sniff. */
export function previewKindFor(ext: string, mimeType: string, binary: boolean): PreviewKind {
  if (HTML_EXTENSIONS.has(ext)) {
    return 'html'
  }

  if (mimeType.startsWith('image/')) {
    return 'image'
  }

  if (PDF_EXTENSIONS.has(ext) || mimeType === 'application/pdf') {
    return 'pdf'
  }

  if (MODEL3D_EXTENSIONS.has(ext)) {
    return 'model3d'
  }

  return binary ? 'binary' : 'text'
}
