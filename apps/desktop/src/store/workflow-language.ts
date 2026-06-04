import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

export type WorkflowLanguage = 'en' | 'zh'

export const WORKFLOW_LANGUAGE_STORAGE_KEY = 'hermes.desktop.workflowLanguage.v1'
export const WORKFLOW_LANGUAGES: readonly WorkflowLanguage[] = ['en', 'zh']

export function normalizeWorkflowLanguage(value: null | string | undefined): WorkflowLanguage {
  return value === 'zh' ? 'zh' : 'en'
}

export const $workflowLanguage = atom<WorkflowLanguage>(
  normalizeWorkflowLanguage(storedString(WORKFLOW_LANGUAGE_STORAGE_KEY))
)

$workflowLanguage.subscribe(language => persistString(WORKFLOW_LANGUAGE_STORAGE_KEY, language))

export function setWorkflowLanguage(language: WorkflowLanguage) {
  $workflowLanguage.set(normalizeWorkflowLanguage(language))
}
