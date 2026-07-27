// Fun starter ideas for the new-project dialog. Pills prefill IDEA.md; the set
// shown is a random handful from this pool (reshuffled on open / via the dice),
// so creating a project always feels a little playful. Pure content — edit
// freely, order doesn't matter.

export interface ProjectIdeaTemplate {
  emoji: string
  label: string
  idea: string
}

export const PROJECT_IDEA_TEMPLATES: ProjectIdeaTemplate[] = [
  {
    emoji: '🎮',
    label: 'Game jam',
    idea: 'A tiny browser game built in a weekend.\n\n- One core mechanic, juicy feedback\n- No build step — single HTML/JS file\n- Playable in under 60 seconds'
  },
  {
    emoji: '📚',
    label: 'Novel',
    idea: 'A novel-in-progress.\n\n- Track chapters, characters, and timeline\n- Daily word-count goal\n- Keep research notes beside the draft'
  },
  {
    emoji: '🤖',
    label: 'Discord bot',
    idea: 'A Discord bot for a small community.\n\n- Slash commands + a fun daily ritual\n- Lightweight persistence\n- Deploy somewhere free'
  },
  {
    emoji: '📊',
    label: 'Data viz',
    idea: 'An interactive visualization of a dataset I care about.\n\n- Pick the dataset and the one question it answers\n- Clean → chart → annotate\n- Shareable as a single page'
  },
  {
    emoji: '🎨',
    label: 'Generative art',
    idea: 'A generative art piece.\n\n- One algorithm, lots of seeds\n- Export high-res stills\n- A gallery of the best outputs'
  },
  {
    emoji: '🍳',
    label: 'Recipe box',
    idea: 'A personal recipe collection.\n\n- Searchable by ingredient and mood\n- Scale servings on the fly\n- Auto-build a shopping list'
  },
  {
    emoji: '🧪',
    label: 'Research log',
    idea: 'A research notebook for an open question.\n\n- Log experiments, results, and dead ends\n- Cite sources inline\n- Weekly synthesis of what I learned'
  },
  {
    emoji: '💸',
    label: 'Budget tracker',
    idea: 'A no-nonsense budget tracker.\n\n- Import transactions, tag them fast\n- Monthly burn vs. plan\n- One chart that tells the truth'
  },
  {
    emoji: '🌱',
    label: 'Habit tracker',
    idea: 'A habit tracker that actually sticks.\n\n- A handful of daily checkboxes\n- Streaks without guilt\n- A calm weekly review'
  },
  {
    emoji: '🗺️',
    label: 'Trip planner',
    idea: 'A trip planner for an upcoming adventure.\n\n- Day-by-day itinerary\n- Map of pins + notes\n- Packing + budget checklist'
  },
  {
    emoji: '🎵',
    label: 'Music toy',
    idea: 'A little music-making toy.\n\n- One instrument or sequencer\n- Web Audio, no installs\n- Record + share a loop'
  },
  {
    emoji: '🧩',
    label: 'Puzzle maker',
    idea: 'A generator for a puzzle I love.\n\n- Procedurally make solvable puzzles\n- Difficulty dial\n- Printable + playable'
  },
  {
    emoji: '📝',
    label: 'Digital garden',
    idea: 'A digital garden / personal wiki.\n\n- Atomic notes that link to each other\n- Grows over time, never "done"\n- Publish the public ones'
  },
  {
    emoji: '🛰️',
    label: 'API wrapper',
    idea: 'A clean wrapper around an API I keep reaching for.\n\n- Typed client + sensible defaults\n- One example per endpoint\n- Publish it'
  },
  {
    emoji: '🏋️',
    label: 'Workout plan',
    idea: 'A workout planner / logger.\n\n- Build a weekly split\n- Log sets fast on mobile\n- Track progress over months'
  },
  {
    emoji: '🧠',
    label: 'Flashcards',
    idea: 'A spaced-repetition flashcard app.\n\n- Quick card capture\n- Simple SM-2 scheduling\n- A daily review that fits in 5 minutes'
  },
  {
    emoji: '✍️',
    label: 'Screenplay',
    idea: 'A short screenplay.\n\n- Logline → beats → scenes\n- Proper format, distraction-free\n- A table read by the end'
  },
  {
    emoji: '🔭',
    label: 'Learn-by-building',
    idea: "A project to learn a thing I've been avoiding.\n\n- Smallest real thing that teaches it\n- Notes on every gotcha\n- A writeup when it works"
  }
]

// A shuffled slice of the pool — the pills shown at any moment.
export function randomIdeaTemplates(count = 6): ProjectIdeaTemplate[] {
  const pool = [...PROJECT_IDEA_TEMPLATES]

  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))

    ;[pool[i], pool[j]] = [pool[j], pool[i]]
  }

  return pool.slice(0, Math.min(count, pool.length))
}
