/**
 * The first-run questionnaire (plan PR3). `Questionnaire` is the controller wiring mounts: it runs
 * the due check once the gateway is open, reads the facts while the questionnaire is open, and lends
 * the handoff what only the app shell has. `QuestionnaireScreen` is the first state of the one
 * first-run overlay (`components/onboarding`, D22); the picker, ready screen and confirm screen are its others.
 */

import { useStore } from '@nanostores/react'
import { AnimatePresence, LayoutGroup } from 'motion/react'
import { atom } from 'nanostores'
import { type ComponentType, useEffect } from 'react'

import { Button } from '@/components/ui/button'
import { FadeScroll } from '@/components/ui/fade-scroll'
import { Progress } from '@/components/ui/progress'
import { useI18n } from '@/i18n'
import { $freeTierStatus } from '@/store/free-tier'
import { notifyError } from '@/store/notifications'
import { markQuestionnaireDecided } from '@/store/onboarding-presence'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { isMainWindow } from '@/store/windows'

import { decideQuestionnaire, mayRunHere, type OnboardingRequester, setRun } from './due'
import { defaultFactSources, freeAccountState, watchFacts } from './facts'
import type { StepId } from './flow'
import { finish, type HandoffDeps, openFirstChat, skipSetup } from './handoff'
import { startQuestionnaireQuickstart } from './status-items'
import { AccentStep, AppsStep, ConnectorsStep, LayoutStep, LocalStep } from './steps/looks'
import { NameStep, TaskStep, TourStep } from './steps/questions'
import { ReviewStep } from './steps/review'
import {
  $questionnaire,
  $questionnaireOpen,
  closeQuestionnaire,
  goBack,
  goToStep,
  passedSteps,
  type QuestionnaireState,
  setFacts,
  setPending
} from './store'
import { answerSummary } from './summary'
import { runQuickTourWhenTargetsPaint } from './tour'
import { StepTransition, TrailChip } from './visuals/motion'

/** What the handoff needs from the app shell; the overlay adds its own readiness round. */
type QuestionnaireHost = Omit<HandoffDeps, 'launchProfile' | 'refreshReadiness'>

const $questionnaireHost = atom<null | QuestionnaireHost>(null)

interface QuestionnaireProps {
  enabled: boolean
  openDefaultChat: HandoffDeps['openDefaultChat']
  openLandedChat: HandoffDeps['openLandedChat']
  requestGateway: OnboardingRequester
}

export function Questionnaire({ enabled, openDefaultChat, openLandedChat, requestGateway }: QuestionnaireProps) {
  const open = useStore($questionnaireOpen)

  useEffect(() => {
    // Other windows start decided (store/onboarding-presence); only the main one runs the due check.
    if (!isMainWindow()) {
      return
    }

    if (enabled) {
      // Anywhere else the questionnaire is decided at once, without an RPC.
      if (mayRunHere()) {
        void decideQuestionnaire(requestGateway)
      } else {
        markQuestionnaireDecided()
      }
    }
  }, [enabled, requestGateway])

  useEffect(() => {
    $questionnaireHost.set({
      openDefaultChat,
      openLandedChat,
      request: requestGateway,
      runTour: runQuickTourWhenTargetsPaint,
      startQuickstart: startQuestionnaireQuickstart
    })
  }, [openDefaultChat, openLandedChat, requestGateway])

  useEffect(() => (open ? watchFacts(defaultFactSources(requestGateway), setFacts) : undefined), [open, requestGateway])

  return null
}

const STEP_VIEWS = {
  accent: AccentStep,
  apps: AppsStep,
  connectors: ConnectorsStep,
  layout: LayoutStep,
  local: LocalStep,
  name: NameStep,
  task: TaskStep,
  tour: TourStep
} satisfies Record<StepId, ComponentType>

function Trail({ state }: { state: QuestionnaireState }) {
  const { t } = useI18n()
  const passed = passedSteps(state)
  const locked = state.pending !== null

  if (passed.length === 0) {
    return null
  }

  return (
    <nav aria-label={t.questionnaire.trailLabel} className="flex flex-wrap gap-1.5">
      {passed.map(stepId => (
        <TrailChip
          disabled={locked}
          key={stepId}
          label={t.questionnaire.changeAnswer}
          onClick={() => goToStep(stepId)}
          stepId={stepId}
        >
          {answerSummary(stepId, state, t.questionnaire)}
        </TrailChip>
      ))}
    </nav>
  )
}

function Preparing() {
  const { t } = useI18n()

  return (
    <div className="grid gap-3" role="status">
      <p className="text-sm font-medium">{t.onboarding.starting}</p>
      <p className="text-sm text-muted-foreground">{t.onboarding.preparingInstall}</p>
      <Progress animated aria-label={t.onboarding.starting} indeterminate />
    </div>
  )
}

/** The questionnaire inside the first-run overlay. `refreshReadiness` is the overlay's own readiness round. */
export function QuestionnaireScreen({ refreshReadiness }: { refreshReadiness: () => Promise<void> }) {
  const { t } = useI18n()
  const state = useStore($questionnaire)
  const host = useStore($questionnaireHost)
  const { pending, stepId, view } = state

  // D23: a free account that failed for good hands over to the picker at once, mid-question or not.
  useEffect(() => {
    if (!host) {
      return
    }

    // A lost save leaves setup due on the next launch, so the toast keeps the write one click away.
    const saveFailed = (error: unknown) =>
      notifyError(error, t.questionnaire.saveFailed, {
        action: { label: t.questionnaire.retry, onClick: () => void setRun(host.request, false).catch(saveFailed) },
        id: 'questionnaire-save'
      })

    return $freeTierStatus.subscribe(status => {
      const current = $questionnaire.get()

      if (freeAccountState(status) === 'failed' && current.phase === 'shown' && current.pending === null) {
        closeQuestionnaire('failed')
        void setRun(host.request, false)
          .catch(saveFailed)
          .finally(() => void refreshReadiness())
      }
    })
  }, [host, refreshReadiness, t])

  if (!host) {
    return null
  }

  const deps: HandoffDeps = {
    ...host,
    launchProfile: normalizeProfileKey($activeGatewayProfile.get()),
    refreshReadiness
  }

  const { answers, facts } = state

  // The flags are saved and the questionnaire is gone, so the confirmed answers ride on the toast.
  const firstChatFailed = (error: unknown) =>
    notifyError(error, t.questionnaire.start.failed, {
      action: { label: t.questionnaire.retry, onClick: () => void openFirstChat(facts, answers, deps).catch(firstChatFailed) },
      id: 'questionnaire-first-chat'
    })

  const start = () => {
    void finish(facts, answers, deps).catch(error => {
      if ($questionnaire.get().phase === 'done') {
        firstChatFailed(error)

        return
      }

      setPending(null)
      notifyError(error, t.questionnaire.start.failed)
    })
  }

  const skip = () =>
    void skipSetup(deps).catch(error => {
      setPending(null)
      notifyError(error, t.questionnaire.settings.failed)
    })

  if (view === 'preparing') {
    return <Preparing />
  }

  const Step = stepId ? STEP_VIEWS[stepId] : null
  const canGoBack = view === 'review' || passedSteps(state).length > 0

  return (
    <LayoutGroup>
      {/* The card is capped at the overlay's height: the answers scroll, Skip setup and Back stay in view. */}
      <div className="flex min-h-0 flex-col gap-5">
        <FadeScroll className="-m-1 min-h-0 p-1" maxHeight="none">
          <div className="grid gap-5">
            <Trail state={state} />
            <AnimatePresence initial={false} mode="wait">
              <StepTransition key={view === 'review' ? 'review' : (stepId ?? 'none')}>
                {view === 'review' ? <ReviewStep onStart={start} /> : Step ? <Step /> : null}
              </StepTransition>
            </AnimatePresence>
          </div>
        </FadeScroll>
        <div className="flex shrink-0 items-center justify-between gap-3">
          <Button disabled={pending !== null} onClick={skip} size="xs" type="button" variant="text">
            {t.questionnaire.skipSetup}
          </Button>
          {canGoBack ? (
            <Button disabled={pending !== null} onClick={goBack} size="xs" type="button" variant="text">
              {t.questionnaire.back}
            </Button>
          ) : null}
        </div>
      </div>
    </LayoutGroup>
  )
}
