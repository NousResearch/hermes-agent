import chiSimDataUrl from '@tesseract.js-data/chi_sim/4.0.0_best_int/chi_sim.traineddata.gz?url'
import chiTraDataUrl from '@tesseract.js-data/chi_tra/4.0.0_best_int/chi_tra.traineddata.gz?url'
import engDataUrl from '@tesseract.js-data/eng/4.0.0_best_int/eng.traineddata.gz?url'
import jpnDataUrl from '@tesseract.js-data/jpn/4.0.0_best_int/jpn.traineddata.gz?url'
import type { Worker } from 'tesseract.js'
import tesseractCoreUrl from 'tesseract.js-core/tesseract-core-lstm.wasm.js?url'
import tesseractWorkerUrl from 'tesseract.js/dist/worker.min.js?url'

export type OcrLanguage = 'chi_sim' | 'chi_tra' | 'eng' | 'jpn'

export interface OcrWord {
  confidence: number
  height: number
  text: string
  width: number
  x: number
  y: number
}

const languageUrls: Record<OcrLanguage, string> = {
  chi_sim: chiSimDataUrl,
  chi_tra: chiTraDataUrl,
  eng: engDataUrl,
  jpn: jpnDataUrl
}

const OCR_CACHE_VERSION = 2
const OCR_CACHE_LIMIT = 64
let activeWorker: { language: OcrLanguage; worker: Promise<Worker> } | null = null
let recognitionQueue: Promise<void> = Promise.resolve()
const memoryCache = new Map<string, OcrWord[]>()
const recognitionJobs = new Map<string, Promise<OcrWord[]>>()

async function loadWorker(language: OcrLanguage): Promise<Worker> {
  if (activeWorker?.language === language) {
    return activeWorker.worker
  }

  if (activeWorker) {
    const previous = activeWorker.worker
    activeWorker = null
    await previous.then(worker => worker.terminate()).catch(() => undefined)
  }

  const created = Promise.all([fetch(languageUrls[language]), import('tesseract.js')])
    .then(([response, runtime]) => {
      if (!response.ok) {
        throw new Error(`Bundled OCR language data is unavailable (${response.status})`)
      }

      return Promise.all([response.arrayBuffer(), runtime])
    })
    .then(([data, runtime]) =>
      runtime.createWorker([{ code: language, data: new Uint8Array(data) }], undefined, {
        corePath: tesseractCoreUrl,
        workerPath: tesseractWorkerUrl
      })
    )

  activeWorker = { language, worker: created }

  try {
    return await created
  } catch (error) {
    if (activeWorker?.worker === created) {
      activeWorker = null
    }

    throw error
  }
}

export async function terminateOcrWorker(): Promise<void> {
  const current = activeWorker
  activeWorker = null
  await current?.worker.then(worker => worker.terminate()).catch(() => undefined)
}

function rememberMemory(key: string, words: OcrWord[]): void {
  memoryCache.delete(key)
  memoryCache.set(key, words)

  while (memoryCache.size > OCR_CACHE_LIMIT) {
    const oldest = memoryCache.keys().next().value

    if (typeof oldest !== 'string') {
      break
    }

    memoryCache.delete(oldest)
  }
}

export function defaultOcrLanguage(locale = navigator.language): OcrLanguage {
  const normalized = locale.toLowerCase()

  if (normalized.startsWith('ja')) {
    return 'jpn'
  }

  if (normalized.includes('hant') || normalized.includes('tw') || normalized.includes('hk')) {
    return 'chi_tra'
  }

  if (normalized.startsWith('zh')) {
    return 'chi_sim'
  }

  return 'eng'
}

async function readCache(key: string): Promise<OcrWord[] | null> {
  const inMemory = memoryCache.get(key)

  if (inMemory) {
    rememberMemory(key, inMemory)

    return inMemory
  }

  if (!('indexedDB' in globalThis)) {
    return null
  }

  return new Promise(resolve => {
    const request = indexedDB.open('hermes-desktop-ocr', OCR_CACHE_VERSION)

    request.onupgradeneeded = () => {
      if (request.result.objectStoreNames.contains('results')) {
        request.result.deleteObjectStore('results')
      }

      request.result.createObjectStore('results')
    }

    request.onerror = () => resolve(null)

    request.onsuccess = () => {
      const db = request.result
      const get = db.transaction('results').objectStore('results').get(key)
      get.onerror = () => resolve(null)

      get.onsuccess = () => {
        const record = get.result as { accessedAt?: unknown; version?: unknown; words?: unknown } | undefined

        const result =
          record?.version === OCR_CACHE_VERSION && Array.isArray(record.words) ? (record.words as OcrWord[]) : null

        if (result) {
          rememberMemory(key, result)
          db.transaction('results', 'readwrite')
            .objectStore('results')
            .put({ accessedAt: Date.now(), version: OCR_CACHE_VERSION, words: result }, key)
        }

        db.close()
        resolve(result)
      }
    }
  })
}

function writeCache(key: string, words: OcrWord[]): void {
  rememberMemory(key, words)

  if (!('indexedDB' in globalThis)) {
    return
  }

  const request = indexedDB.open('hermes-desktop-ocr', OCR_CACHE_VERSION)

  request.onupgradeneeded = () => {
    if (request.result.objectStoreNames.contains('results')) {
      request.result.deleteObjectStore('results')
    }

    request.result.createObjectStore('results')
  }

  request.onsuccess = () => {
    const db = request.result
    const transaction = db.transaction('results', 'readwrite')
    const store = transaction.objectStore('results')
    store.put({ accessedAt: Date.now(), version: OCR_CACHE_VERSION, words }, key)
    const all = store.getAll()
    const keys = store.getAllKeys()
    let records: Array<{ accessedAt?: unknown }> | null = null
    let recordKeys: IDBValidKey[] | null = null

    const prune = () => {
      if (!records || !recordKeys || records.length <= OCR_CACHE_LIMIT) {
        return
      }

      const ordered = records
        .map((record, index) => ({ accessedAt: Number(record?.accessedAt) || 0, key: recordKeys![index] }))
        .sort((a, b) => a.accessedAt - b.accessedAt)

      for (const entry of ordered.slice(0, ordered.length - OCR_CACHE_LIMIT)) {
        store.delete(entry.key)
      }
    }

    transaction.oncomplete = () => db.close()

    all.onsuccess = () => {
      records = all.result
      prune()
    }

    keys.onsuccess = () => {
      recordKeys = keys.result
      prune()
    }
  }
}

export async function recognizeImageText(
  image: string | HTMLCanvasElement,
  cacheKey: string,
  language: OcrLanguage
): Promise<OcrWord[]> {
  const key = `${cacheKey}:${language}`
  const cached = await readCache(key)

  if (cached) {
    return cached
  }

  const running = recognitionJobs.get(key)

  if (running) {
    return running
  }

  const recognition = recognitionQueue
    .catch(() => undefined)
    .then(() => loadWorker(language))
    .then(worker => worker.recognize(image, {}, { blocks: true, text: true }))
    .then(result =>
      (result.data.blocks ?? [])
        .flatMap(block => block.paragraphs)
        .flatMap(paragraph => paragraph.lines)
        .flatMap(line => line.words)
        .filter(word => word.text.trim())
        .map(word => ({
          confidence: word.confidence,
          height: word.bbox.y1 - word.bbox.y0,
          text: word.text,
          width: word.bbox.x1 - word.bbox.x0,
          x: word.bbox.x0,
          y: word.bbox.y0
        }))
    )

  recognitionJobs.set(key, recognition)
  recognitionQueue = recognition.then(
    () => undefined,
    () => undefined
  )

  try {
    const words = await recognition
    writeCache(key, words)

    return words
  } finally {
    recognitionJobs.delete(key)
  }
}

if (typeof window !== 'undefined') {
  window.addEventListener('pagehide', () => void terminateOcrWorker(), { once: true })
}
