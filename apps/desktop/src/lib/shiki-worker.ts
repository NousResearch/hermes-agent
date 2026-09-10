/**
 * Shiki Worker - Off-main-thread tokenization for syntax highlighting
 * 
 * This worker handles Shiki's codeToTokens calls to prevent main thread blocking
 * when rendering large diffs or code blocks.
 */

import type { BundledLanguage, ThemedToken } from 'shiki'

// Message types for worker communication
export interface WorkerMessage<T = unknown> {
  type: 'tokenize' | 'tokenizeChunk' | 'cancel' | 'init' | 'reinit'
  id: number
  payload?: T
}

export interface InitPayload {
  language?: BundledLanguage
  theme?: 'github-dark-dimmed' | 'github-light-default'
}

export interface ReinitPayload {
  language?: BundledLanguage
  theme?: 'github-dark-dimmed' | 'github-light-default'
}

export interface TokenizePayload {
  code: string
  language: BundledLanguage
  theme: 'github-dark-dimmed' | 'github-light-default'
}

export interface TokenizeChunkPayload {
  code: string
  language: BundledLanguage
  theme: 'github-dark-dimmed' | 'github-light-default'
  chunkIndex: number
}

export interface WorkerResponse<T = unknown> {
  type: 'result' | 'error' | 'progress' | 'init'
  id: number
  payload?: T
  error?: string
  success?: boolean
}

export interface TokenizeResult {
  tokens: ThemedToken[][]
}

export interface TokenizeChunkResult {
  tokens: ThemedToken[][]
  chunkIndex: number
}

// Global state
let shikiLoaded = false
let currentLanguage: BundledLanguage = 'txt' as BundledLanguage
let currentTheme: 'github-dark-dimmed' | 'github-light-default' = 'github-light-default'
let codeToTokens: (code: string, options: { lang: BundledLanguage; theme: string }) => Promise<{ tokens: ThemedToken[][] }>
const pendingRequests = new Map<number, (response: WorkerResponse) => void>()

// Initialize Shiki in the worker
async function initShiki(messageId: number, language?: BundledLanguage, theme?: 'github-dark-dimmed' | 'github-light-default') {
  if (shikiLoaded) {
    if (language) {currentLanguage = language}

    if (theme) {currentTheme = theme}
    self.postMessage({ type: 'init', id: messageId, success: true } satisfies WorkerResponse)

    return
  }
  
  try {
    const shiki = await import('shiki')
    codeToTokens = shiki.codeToTokens
    shikiLoaded = true

    if (language) {currentLanguage = language}

    if (theme) {currentTheme = theme}
    self.postMessage({ type: 'init', id: messageId, success: true } satisfies WorkerResponse)
  } catch (error) {
    self.postMessage({ 
      type: 'init', 
      id: messageId,
      success: false, 
      error: error instanceof Error ? error.message : 'Failed to load Shiki' 
    } satisfies WorkerResponse)
  }
}

// Handle tokenization request
async function handleTokenize(message: WorkerMessage<TokenizePayload>) {
  if (!shikiLoaded) {
    await initShiki(message.id, message.payload?.language, message.payload?.theme)
  }
  
  if (!codeToTokens) {
    pendingRequests.get(message.id)?.({
      type: 'error',
      id: message.id,
      error: 'Shiki not initialized'
    } satisfies WorkerResponse)

    return
  }
  
  try {
    const { code, language, theme } = message.payload!
    const result = await codeToTokens(code, { lang: language, theme })
    
    pendingRequests.get(message.id)?.({
      type: 'result',
      id: message.id,
      payload: { tokens: result.tokens } satisfies TokenizeResult
    } satisfies WorkerResponse)
  } catch (error) {
    pendingRequests.get(message.id)?.({
      type: 'error',
      id: message.id,
      error: error instanceof Error ? error.message : 'Tokenization failed'
    } satisfies WorkerResponse)
  }
}

// Handle chunk tokenization request
async function handleTokenizeChunk(message: WorkerMessage<TokenizeChunkPayload>) {
  if (!shikiLoaded) {
    await initShiki(message.id, message.payload?.language, message.payload?.theme)
  }
  
  if (!codeToTokens) {
    pendingRequests.get(message.id)?.({
      type: 'error',
      id: message.id,
      error: 'Shiki not initialized'
    } satisfies WorkerResponse)

    return
  }
  
  try {
    const { code, language, theme, chunkIndex } = message.payload!
    const result = await codeToTokens(code, { lang: language, theme })
    
    pendingRequests.get(message.id)?.({
      type: 'result',
      id: message.id,
      payload: { 
        tokens: result.tokens,
        chunkIndex 
      } satisfies TokenizeChunkResult
    } satisfies WorkerResponse)
  } catch (error) {
    pendingRequests.get(message.id)?.({
      type: 'error',
      id: message.id,
      error: error instanceof Error ? error.message : 'Chunk tokenization failed'
    } satisfies WorkerResponse)
  }
}

// Handle cancel request
function handleCancel(messageId: number) {
  // Remove from pending and reject
  const callback = pendingRequests.get(messageId)

  if (callback) {
    callback({
      type: 'error',
      id: messageId,
      error: 'Cancelled'
    } satisfies WorkerResponse)
    pendingRequests.delete(messageId)
  }
}

// Handle reinit - reinitialize with new language/theme
async function handleReinit(messageId: number, language?: BundledLanguage, theme?: 'github-dark-dimmed' | 'github-light-default') {
  // Reset shiki state to reload with new config
  shikiLoaded = false

  if (language) {currentLanguage = language}

  if (theme) {currentTheme = theme}
  
  // Reject all pending requests
  pendingRequests.forEach((callback, id) => {
    callback({
      type: 'error',
      id,
      error: 'Worker reinitializing'
    } satisfies WorkerResponse)
  })
  pendingRequests.clear()
  
  await initShiki(messageId, language, theme)
}

// Message handler
self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const message = event.data
  
  switch (message.type) {
    case 'init':
      await initShiki(message.id, (message.payload as InitPayload | undefined)?.language, (message.payload as InitPayload | undefined)?.theme)

      break
      
    case 'tokenize':
      pendingRequests.set(message.id, (response) => {
        self.postMessage(response)
      })
      await handleTokenize(message as WorkerMessage<TokenizePayload>)

      break
      
    case 'tokenizeChunk':
      pendingRequests.set(message.id, (response) => {
        self.postMessage(response)
      })
      await handleTokenizeChunk(message as WorkerMessage<TokenizeChunkPayload>)

      break
      
    case 'cancel':
      handleCancel(message.id)

      break
      
    case 'reinit':
      await handleReinit(message.id, (message.payload as ReinitPayload | undefined)?.language, (message.payload as ReinitPayload | undefined)?.theme)

      break
  }
}

// Export types for TypeScript consumers
export type { ThemedToken }