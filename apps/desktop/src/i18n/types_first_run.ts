/** First-run onboarding: the setup_beat cards and their receipts, Skip setup, the local-model offer
 *  at the end, and the Settings row that runs it again. */
export interface FirstRunCopy {
  /** Receipt labels, one per beat. */
  beats: {
    name: string
    theme: string
    layout: string
    connectors: string
    plugins: string
    tour: string
    build: string
  }
  namePlaceholder: string
  /** The OS account's full name, offered as a row on the name card. */
  useName: (name: string) => string
  install: string
  installed: string
  loading: string
  noPlugins: string
  /** A receipt for a card that settled with nothing chosen (no apps connected, no plugin installed). */
  noneChosen: string
  noAnswer: string
  /** A card answer the backend refused; the model brings the card back. */
  answerRefused: string
  /** The tour card's options by id; the backend's labels are English. */
  tours: {
    quick: string
    full: string
    skip: string
  }
  skippedAll: string
  skip: string
  skipTip: string
  skipFailed: string
  /** A slash command sent in the onboarding chat before setup ends: it does not run. */
  slashOffTitle: string
  slashOffBody: string
  localModel: {
    title: string
    text: (model: string, size: string) => string
    setUp: string
    notNow: string
  }
  /** The first-run intro typed into the empty setup chat, two lines. The backend's scripted
   *  greeting (_GREETINGS in agent/onboarding_text.py) is the two joined: keep them equal. */
  intro: {
    greeting: string
    plan: string
  }
  rerun: {
    title: string
    description: string
    action: string
    failed: string
  }
}
