'use strict'

const DEFAULT_ZOOM_FACTOR = 1
const ZOOM_FACTORS = Object.freeze([0.5, 0.67, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2])
const MIN_ZOOM_FACTOR = ZOOM_FACTORS[0]
const MAX_ZOOM_FACTOR = ZOOM_FACTORS[ZOOM_FACTORS.length - 1]

function normalizeZoomFactor(value) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return DEFAULT_ZOOM_FACTOR
  return Math.round(parsed * 100) / 100
}

function clampZoomFactor(value) {
  const normalized = normalizeZoomFactor(value)
  if (normalized < MIN_ZOOM_FACTOR) return MIN_ZOOM_FACTOR
  if (normalized > MAX_ZOOM_FACTOR) return MAX_ZOOM_FACTOR
  return normalized
}

function getZoomPercent(value) {
  return Math.round(clampZoomFactor(value) * 100)
}

function getNextZoomFactor(current, direction) {
  const factor = clampZoomFactor(current)
  const step = direction >= 0 ? 1 : -1

  if (step > 0) {
    return ZOOM_FACTORS.find(next => next > factor + 0.001) ?? MAX_ZOOM_FACTOR
  }

  for (let index = ZOOM_FACTORS.length - 1; index >= 0; index -= 1) {
    if (ZOOM_FACTORS[index] < factor - 0.001) {
      return ZOOM_FACTORS[index]
    }
  }

  return MIN_ZOOM_FACTOR
}

function isZoomAccelerator(input) {
  if (!input || input.alt || (!input.control && !input.meta)) {
    return null
  }

  const key = String(input.key || '').toLowerCase()
  const code = String(input.code || '')

  if (key === '+' || key === '=' || code === 'Equal' || code === 'NumpadAdd') {
    return 'in'
  }

  if (key === '-' || key === '_' || code === 'Minus' || code === 'NumpadSubtract') {
    return 'out'
  }

  if (key === '0' || code === 'Digit0' || code === 'Numpad0') {
    return 'reset'
  }

  return null
}

function createZoomController({ fs, configPath, getWindow, onChanged } = {}) {
  let factor = DEFAULT_ZOOM_FACTOR

  const readPersistedFactor = () => {
    if (!fs || !configPath) return DEFAULT_ZOOM_FACTOR

    try {
      const parsed = JSON.parse(fs.readFileSync(configPath, 'utf8'))
      return clampZoomFactor(parsed?.factor)
    } catch {
      return DEFAULT_ZOOM_FACTOR
    }
  }

  const persistFactor = next => {
    if (!fs || !configPath) return

    try {
      fs.mkdirSync(require('node:path').dirname(configPath), { recursive: true })
      fs.writeFileSync(configPath, `${JSON.stringify({ factor: next }, null, 2)}\n`)
    } catch {
      // Zoom should still work for this session if persistence fails.
    }
  }

  const applyToWindow = targetWindow => {
    const win = targetWindow || getWindow?.()
    const webContents = win?.webContents
    if (!webContents || webContents.isDestroyed?.()) return
    webContents.setZoomFactor(factor)
  }

  const emitChanged = () => {
    const payload = { factor, percent: getZoomPercent(factor) }
    onChanged?.(payload)
    return payload
  }

  const setFactor = (value, { persist = true } = {}) => {
    factor = clampZoomFactor(value)
    applyToWindow()
    if (persist) persistFactor(factor)
    return emitChanged()
  }

  const reset = () => setFactor(DEFAULT_ZOOM_FACTOR)
  const adjust = direction => setFactor(getNextZoomFactor(factor, direction))
  const getState = () => ({ factor, percent: getZoomPercent(factor) })
  const load = () => setFactor(readPersistedFactor(), { persist: false })

  const handleBeforeInput = (event, input) => {
    const action = isZoomAccelerator(input)
    if (!action) return false

    event?.preventDefault?.()
    if (action === 'in') adjust(1)
    if (action === 'out') adjust(-1)
    if (action === 'reset') reset()
    return true
  }

  return { adjust, applyToWindow, getState, handleBeforeInput, load, reset, setFactor }
}

module.exports = {
  DEFAULT_ZOOM_FACTOR,
  MAX_ZOOM_FACTOR,
  MIN_ZOOM_FACTOR,
  ZOOM_FACTORS,
  clampZoomFactor,
  createZoomController,
  getNextZoomFactor,
  getZoomPercent,
  isZoomAccelerator,
  normalizeZoomFactor
}
