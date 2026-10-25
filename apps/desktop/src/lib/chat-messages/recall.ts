export const MEMORY_RECALL_MESSAGE_ID_PREFIX = 'memory-recall-'

export const isMemoryRecallMessageId = (id: string): boolean => id.startsWith(MEMORY_RECALL_MESSAGE_ID_PREFIX)
