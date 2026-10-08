import type { QuestionnaireTranslations } from '@/i18n/types_questionnaire'
import { accentsFor } from '@/themes/accents'

import { chosenName, chosenTask, connectorOptions, pluginOptions, type StepId } from './flow'
import type { QuestionnaireState } from './store'

function firstAndMore(labels: string[], none: string, copy: QuestionnaireTranslations): string {
  if (labels.length === 0) {
    return none
  }

  return labels.length === 1 ? labels[0] : copy.summary.more(labels[0], labels.length - 1)
}

type Summary = (state: QuestionnaireState, copy: QuestionnaireTranslations) => string

const SUMMARIES = {
  accent: ({ answers }, copy) =>
    answers.accent?.startsWith('custom:')
      ? copy.summary.custom
      : (accentsFor(true).find(swatch => swatch.id === (answers.accent ?? 'nous'))?.name ?? copy.summary.custom),
  apps: ({ answers, facts }, copy) =>
    firstAndMore(
      pluginOptions(facts)
        .filter(option => answers.apps.includes(option.id))
        .map(option => option.label(copy)),
      copy.summary.noApps,
      copy
    ),
  connectors: ({ answers, facts }, copy) =>
    firstAndMore(
      connectorOptions(facts)
        .filter(option => answers.connectors.includes(option.id))
        .map(option => option.label(copy)),
      copy.summary.noConnectors,
      copy
    ),
  layout: ({ answers }, copy) => (answers.layout === 'terminal-deck' ? copy.layout.elite : copy.layout.basic),
  local: ({ answers, facts }, copy) =>
    answers.local === 'yes' && facts.local ? copy.summary.local(facts.local.name) : copy.summary.localNo,
  name: ({ answers }, copy) => chosenName(answers) ?? copy.skipped,
  task: ({ answers, facts }, copy) => chosenTask(facts, answers)?.label(copy) ?? copy.skipped,
  tour: ({ answers }, copy) => (answers.tour === 'quick' ? copy.summary.tour : copy.summary.noTour)
} satisfies Record<StepId, Summary>

/** The trail chip's text for a passed step. */
export function answerSummary(stepId: StepId, state: QuestionnaireState, copy: QuestionnaireTranslations): string {
  return state.answers.skipped.includes(stepId) ? copy.skipped : SUMMARIES[stepId](state, copy)
}
