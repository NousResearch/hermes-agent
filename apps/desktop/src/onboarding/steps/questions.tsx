import { useStore } from '@nanostores/react'

import { useI18n } from '@/i18n'
import { MessageQuestion } from '@/lib/icons'

import { firstName, stepOptions, STEPS } from '../flow'
import { $questionnaire, confirmStep, setAnswers, skipStep } from '../store'
import { QuestionPills } from '../visuals/question-pills'

import { StepCard } from './step-card'

export function NameStep() {
  const { t } = useI18n()
  const { answers, facts } = useStore($questionnaire)
  const suggested = firstName(facts.machine?.full_name)
  const name = answers.name ?? suggested ?? ''

  const choices = stepOptions(STEPS.name, facts, answers).map(option => ({
    id: option.id,
    label: option.label(t.questionnaire)
  }))

  return (
    <StepCard
      canConfirm={Boolean(name.trim())}
      icon={MessageQuestion}
      kind={t.assistant.clarify.oneQuestion}
      onConfirm={() => {
        setAnswers({ name: name.trim() })
        confirmStep('name')
      }}
      onSkip={() => skipStep('name')}
      title={t.questionnaire.name.greeting}
    >
      <QuestionPills
        choices={choices}
        onOther={value => setAnswers({ name: value.trim() ? value : (suggested ?? '') })}
        onPick={id => setAnswers({ name: id })}
        other={name === suggested ? '' : name}
        otherPlaceholder={t.questionnaire.name.placeholder}
        picked={name === suggested ? suggested : null}
        question={t.questionnaire.name.question}
        stepId="name"
      />
    </StepCard>
  )
}

export function TaskStep() {
  const { t } = useI18n()
  const { answers, facts } = useStore($questionnaire)
  const options = stepOptions(STEPS.task, facts, answers)
  const task = answers.task
  const typed = task?.id === 'other' ? (task.text ?? '') : ''
  const picked = task && options.some(option => option.id === task.id) ? task.id : null

  return (
    <StepCard
      canConfirm={Boolean(picked || typed.trim())}
      icon={MessageQuestion}
      kind={t.assistant.clarify.oneQuestion}
      onConfirm={() => confirmStep('task')}
      onSkip={() => skipStep('task')}
      title={t.questionnaire.task.title}
    >
      <QuestionPills
        choices={options.map(option => ({ detail: option.ask, id: option.id, label: option.label(t.questionnaire) }))}
        onOther={text => setAnswers({ task: text.trim() ? { id: 'other', text } : undefined })}
        onPick={id => setAnswers({ task: { id } })}
        other={typed}
        picked={picked}
        question={t.questionnaire.task.question}
        stepId="task"
      />
    </StepCard>
  )
}

export function TourStep() {
  const { t } = useI18n()
  const { answers, facts } = useStore($questionnaire)
  const options = stepOptions(STEPS.tour, facts, answers)

  return (
    <StepCard
      canConfirm={answers.tour !== undefined}
      icon={MessageQuestion}
      kind={t.assistant.clarify.oneQuestion}
      onConfirm={() => confirmStep('tour')}
      onSkip={() => {
        setAnswers({ tour: 'none' })
        skipStep('tour')
      }}
      title={t.questionnaire.tour.title}
    >
      <QuestionPills
        choices={options.map(option => ({
          detail: option.detail?.(t.questionnaire),
          id: option.id,
          label: option.label(t.questionnaire)
        }))}
        onPick={id => setAnswers({ tour: id === 'quick' ? 'quick' : 'none' })}
        picked={answers.tour ?? null}
        question={t.questionnaire.tour.question}
        stepId="tour"
      />
    </StepCard>
  )
}
