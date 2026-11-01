/** Office files the desktop preview rail can render. Legacy `.doc` is omitted:
 * neither the Word nor the spreadsheet renderer reads OLE Word. */

export type OfficePreviewKind = 'document' | 'presentation' | 'spreadsheet'

const OFFICE_KIND_BY_EXT: Record<string, OfficePreviewKind> = {
  '.csv': 'spreadsheet',
  '.docx': 'document',
  '.ods': 'spreadsheet',
  '.odt': 'document',
  '.pptx': 'presentation',
  '.xls': 'spreadsheet',
  '.xlsx': 'spreadsheet'
}

/** `ext` includes the dot, as `path.extname` returns it. */
export function officePreviewKind(ext: string): OfficePreviewKind | null {
  return OFFICE_KIND_BY_EXT[ext.toLowerCase()] ?? null
}

/** Last extension of a filesystem path. A `#` or `?` suffix is literal, not a
 * URL fragment, so `report.xlsx#notes` is not an office file. */
export function officePreviewKindFromPath(value: string | undefined): OfficePreviewKind | null {
  if (!value) {
    return null
  }

  const match = /(\.[^./\\?#]+)$/.exec(value)

  return match ? officePreviewKind(match[1]) : null
}
