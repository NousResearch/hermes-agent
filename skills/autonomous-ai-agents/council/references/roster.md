# Council roster

Every member reasons via a distinct `reasoning_method`. Never seat two members
sharing one — method diversity is what the council buys (DMAD, arXiv:2410.12853).

## The 18 members

| Member | Figure | Domain | Reasoning method | Polarity |
|--------|--------|--------|------------------|----------|
| `council-aristotle` | Aristotle | Categorization & structure | taxonomic-decomposition | Classifies everything |
| `council-socrates` | Socrates | Assumption destruction | elenchic-questioning | Questions everything |
| `council-sun-tzu` | Sun Tzu | Adversarial strategy | adversarial-simulation | Reads terrain & competition |
| `council-ada` | Ada Lovelace | Formal systems & abstraction | formal-stepwise-verification | What can/can't be mechanized |
| `council-aurelius` | Marcus Aurelius | Resilience & moral clarity | negative-visualization | Control vs acceptance |
| `council-machiavelli` | Machiavelli | Power dynamics & realpolitik | incentive-backward-induction | How actors actually behave |
| `council-lao-tzu` | Lao Tzu | Non-action & emergence | via-negativa | When less is more |
| `council-feynman` | Feynman | First-principles debugging | first-principles-reconstruction | Refuses unexplained complexity |
| `council-torvalds` | Linus Torvalds | Pragmatic engineering | empirical-reduction-to-practice | Ship it or shut up |
| `council-musashi` | Miyamoto Musashi | Strategic timing | timing-tempo-analysis | The decisive strike |
| `council-watts` | Alan Watts | Perspective & reframing | frame-dissolution | Dissolves false problems |
| `council-karpathy` | Andrej Karpathy | Neural network intuition & empirical ML | gradient-empiricism | How models actually learn and fail |
| `council-sutskever` | Ilya Sutskever | Scaling frontier & AI safety | scaling-extrapolation | When capability becomes risk |
| `council-kahneman` | Daniel Kahneman | Cognitive bias & decision science | bias-audit-system2 | Your own thinking is the first error |
| `council-meadows` | Donella Meadows | Systems thinking & feedback loops | causal-loop-mapping | Redesign the system, not the symptom |
| `council-munger` | Charlie Munger | Multi-model reasoning & economics | multi-model-inversion | Invert — what guarantees failure? |
| `council-taleb` | Nassim Taleb | Antifragility & tail risk | tail-stress-testing | Design for the tail, not the average |
| `council-rams` | Dieter Rams | User-centered design | subtractive-essentialism | Less, but better — the user decides |

## Polarity pairs

Members who structurally disagree. Used for `--duo`, and for detecting a panel
that has stacked one side of a tension.

- **Socrates vs Feynman** — Destroys top-down vs rebuilds bottom-up
- **Aristotle vs Lao Tzu** — Classifies everything vs structure IS the problem
- **Sun Tzu vs Aurelius** — Wins external games vs governs the internal one
- **Ada vs Machiavelli** — Formal purity vs messy human incentives
- **Torvalds vs Watts** — Ships concrete solutions vs questions whether the problem exists
- **Musashi vs Torvalds** — Waits for the perfect moment vs ships it now
- **Karpathy vs Sutskever** — Build it, observe it, iterate vs pause, research, ensure safety first
- **Karpathy vs Ada** — Empirical ML intuition vs formal systems theory
- **Kahneman vs Feynman** — Your cognition is the first error vs trust first-principles reasoning
- **Meadows vs Torvalds** — Redesign the feedback loop vs fix the symptom and ship
- **Munger vs Aristotle** — Multi-model lattice vs single taxonomic system
- **Taleb vs Karpathy** — Hidden catastrophic tails vs smooth empirical scaling curves
- **Rams vs Ada** — What the user needs vs what computation can do
- **Sutskever vs Machiavelli** — Safety ideals vs industry incentives
- **Socrates vs Watts** — Destroys assumptions vs dissolves the frame

## Triads

Match the problem against these keywords for auto-selection.

| Keyword | Triad | Rationale |
|---------|-------|-----------|
| `architecture` | Aristotle + Ada + Feynman | Classify + formalize + simplicity-test |
| `strategy` | Sun Tzu + Machiavelli + Aurelius | Terrain + incentives + moral grounding |
| `ethics` | Aurelius + Socrates + Lao Tzu | Duty + questioning + natural order |
| `debugging` | Feynman + Socrates + Ada | Bottom-up + assumption testing + formal verification |
| `innovation` | Ada + Lao Tzu + Aristotle | Abstraction + emergence + classification |
| `conflict` | Socrates + Machiavelli + Aurelius | Expose + predict + ground |
| `complexity` | Lao Tzu + Aristotle + Ada | Emergence + categories + formalism |
| `risk` | Sun Tzu + Aurelius + Feynman | Threats + resilience + empirical verification |
| `shipping` | Torvalds + Musashi + Feynman | Pragmatism + timing + first-principles |
| `product` | Torvalds + Machiavelli + Watts | Ship it + incentives + reframing |
| `founder` | Musashi + Sun Tzu + Torvalds | Timing + terrain + engineering reality |
| `ai` | Karpathy + Sutskever + Ada | Empirical ML + scaling frontier + formal limits |
| `ai-product` | Karpathy + Torvalds + Machiavelli | ML capability + shipping pragmatism + incentives |
| `ai-safety` | Sutskever + Aurelius + Socrates | Safety frontier + moral clarity + assumption destruction |
| `decision` | Kahneman + Munger + Aurelius | Bias detection + inversion + moral clarity |
| `systems` | Meadows + Lao Tzu + Aristotle | Feedback loops + emergence + categories |
| `uncertainty` | Taleb + Sun Tzu + Sutskever | Tail risk + terrain + scaling frontier |
| `design` | Rams + Torvalds + Watts | User clarity + maintainability + reframing |
| `economics` | Munger + Machiavelli + Sun Tzu | Models + incentives + competition |
| `bias` | Kahneman + Socrates + Watts | Cognitive bias + assumption destruction + frame audit |

## Duo pairs

For `--duo`. Match the problem keywords; fall back to the default pair.

| Keywords | Pair | Tension |
|----------|------|---------|
| architecture, structure, categories | Aristotle vs Lao Tzu | Classification vs emergence |
| shipping, execution, release | Torvalds vs Musashi | Ship now vs wait for timing |
| strategy, competition, market | Sun Tzu vs Aurelius | External victory vs internal governance |
| formalization, systems, abstraction | Ada vs Machiavelli | Formal purity vs human messiness |
| framing, purpose, meaning | Socrates vs Watts | Destroy assumptions vs dissolve the frame |
| engineering, theory, pragmatism | Torvalds vs Watts | Build it vs question if it should exist |
| ai, ml, neural, model, training | Karpathy vs Sutskever | Build and iterate vs pause and ensure safety |
| ai-safety, alignment, risk | Sutskever vs Machiavelli | Safety ideals vs industry incentives |
| decision, bias, thinking, judgment | Kahneman vs Feynman | Your cognition is the error vs trust first-principles |
| systems, feedback, complexity, loops | Meadows vs Torvalds | Redesign the system vs fix the symptom |
| economics, investment, models, moat | Munger vs Aristotle | Multi-model lattice vs single taxonomy |
| risk, uncertainty, fragility, tail | Taleb vs Karpathy | Hidden tails vs smooth empirical curves |
| design, user, usability, ux | Rams vs Ada | What the user needs vs what computation can do |
| default (no keyword match) | Socrates vs Feynman | Top-down questioning vs bottom-up rebuilding |

## Profiles

### `execution-lean` — 5 members

Fast decision-to-action loops. **Torvalds, Feynman, Sun Tzu, Aurelius, Ada.**

- `ship-now` → Torvalds + Feynman + Aurelius
- `launch-strategy` → Sun Tzu + Torvalds + Machiavelli
- `stability` → Ada + Feynman + Aurelius

### `exploration-orthogonal` — 12 members

Discovery and unknown-unknowns reduction. **Socrates, Feynman, Sun Tzu,
Machiavelli, Ada, Lao Tzu, Aurelius, Torvalds, Karpathy, Sutskever, Kahneman,
Meadows.**

- `unknowns` → Socrates + Lao Tzu + Feynman
- `market-entry` → Sun Tzu + Machiavelli + Aurelius
- `system-design` → Ada + Feynman + Torvalds
- `reframing` → Socrates + Lao Tzu + Ada
- `ai-frontier` → Karpathy + Sutskever + Ada
- `blind-spots` → Kahneman + Meadows + Socrates

### `classic` — all 18 members

The full roster with the domain triads above. Expensive: 18 seats × rounds.
Reserve it for decisions that are costly to reverse.
