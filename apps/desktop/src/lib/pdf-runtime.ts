import workerUrl from 'pdfjs-dist/legacy/build/pdf.worker.min.mjs?url'

export const pdfWorkerUrl = workerUrl

let runtimePromise: ReturnType<typeof importPdfRuntime> | null = null

function importPdfRuntime() {
  return import('pdfjs-dist/legacy/build/pdf.mjs')
}

/** One compatibility runtime for document loading, page rendering, and text layers. */
export function loadPdfRuntime() {
  runtimePromise ??= importPdfRuntime()

  return runtimePromise
}
