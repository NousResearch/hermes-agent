# Prediction Engine — Forecasting What Someone Would Say/Do

Techniques for predicting behavior grounded in superforecasting methodology,
behavioral science, and SOTA LLM prediction research.

## Superforecasting Principles (Tetlock)

**Honest caveat**: Superforecasting methodology was developed for geopolitical and
world-event prediction, not personality simulation. That said, the THINKING TOOLS
are genuinely useful here — decomposition prevents lazy pattern-matching, base rates
fight overconfidence, and alternative hypotheses prevent single-track predictions.
What does NOT transfer cleanly: the calibration precision. When Tetlock says "70%
confident," that's backed by thousands of scored predictions. When we say "70%
confident" about what @someone would tweet, that's an educated estimate, not a
calibrated probability. Use the framework for its rigor, not its false precision.

Apply these thinking tools when making behavioral predictions:

### 1. Decomposition (Fermi-ize the Question)
Don't ask "What would @person say about X?"
Break it down:
- What is @person's known position on topics RELATED to X?
- What are their values/priorities that X touches on?
- What is their emotional register when discussing similar topics?
- Who are they likely responding to, and how does that change their tone?
- What platform are they on, and how does that shift their behavior?

### 2. Outside View First (Base Rates)
Before considering the specific person, ask:
- What would a TYPICAL person in their role/position say about X?
- What % of people in their ideological cluster hold position Y on X?
- What's the base rate for their type of response (agree/disagree/joke/ignore)?

### 3. Inside View Second (Case-Specific Adjustment)
Now adjust from the base rate using what you ACTUALLY KNOW about them:
- Specific past statements on this topic or related topics
- Known relationships with people/orgs involved
- Personal experiences that would shape their view
- Contrarian tendencies (do they predictably go against their cluster?)

### 4. Confidence Calibration
Express predictions with honest uncertainty. **These are rough buckets, not
calibrated probabilities. Don't pretend they're more precise than they are.**
- **90%+ confident**: They've literally said this before, just rephrased
- **70-89%**: Strong pattern match with known positions and voice
- **50-69%**: Reasonable inference but could go either way
- **30-49%**: Educated guess, limited data
- **<30%**: Basically guessing, flag it clearly

When reporting confidence, prefer plain language over fake precision:
"very likely" > "87% probability". The number implies a precision we don't have.

### 5. Consider Alternative Hypotheses
For every prediction, generate at least ONE plausible alternative:
- "They'd PROBABLY say X, but they might surprise with Y because Z"
- This prevents overconfident single-track predictions

## The Prediction Pipeline

### Step 1: Classify the Prediction Type

| Type | Description | Difficulty |
|------|-------------|-----------|
| **Position prediction** | What they believe about X | Easiest if data exists |
| **Reaction prediction** | How they'd respond to event Y | Medium |
| **Voice prediction** | How they'd phrase something | Medium-hard |
| **Behavior prediction** | What they'd DO (not just say) | Hardest |
| **Interaction prediction** | How they'd respond to specific person | Hard, depends on relationship data |

### Step 2: Evidence Gathering Protocol

For each prediction, gather evidence in this order:

1. **Direct evidence**: Have they addressed this exact topic before?
   - Search: `"{handle}" "{topic}"` or `"{handle}" "{related_keyword}"`
   - Weight: HIGHEST

2. **Analogical evidence**: Have they addressed something similar?
   - Search: find positions on adjacent topics
   - Weight: HIGH

3. **Value evidence**: What values/principles would apply?
   - Infer from their stated beliefs and consistent positions
   - Weight: MEDIUM

4. **Social evidence**: What do their peers/allies think?
   - People tend to align with their social cluster (but not always)
   - Weight: LOW-MEDIUM (higher for conformists, lower for contrarians)

5. **Demographic evidence**: What would someone in their position typically think?
   - Base rate from role/industry/ideology
   - Weight: LOWEST (only use as anchor, not conclusion)

### Step 2b: Contradiction Handling Protocol
When evidence conflicts (e.g., person said X in 2024 but Y in 2026):

1. **Check for genuine change**: Did they explicitly reverse position? Look for
   "I used to think X but now..." or a clear pivot moment. If so, use the newer
   position and note the evolution.

2. **Check for context-dependence**: Did they say X to audience A and Y to audience B?
   This isn't necessarily dishonesty — people emphasize different facets for different
   contexts. Note which context your simulation targets and use the matching register.

3. **Check for nuance collapse**: Maybe they said "X is mostly good with caveats"
   and later "X has real problems" — these might not actually contradict. Look for
   the synthesis position.

4. **When genuinely unresolvable**: Flag it explicitly. "Evidence conflicts on this
   point — they've argued both sides at different times. Simulating {chosen position}
   based on {reasoning}, but the alternative is plausible." Don't paper over the
   contradiction with false confidence.

5. **Recency default**: When all else fails, weight more recent statements higher.
   People change, and the most recent position is the best predictor of the next one.

### Step 3: Generate Prediction

Using the HumanLLM B = f(P, E) framework:
- **P (Person)**: Everything from the dossier — personality, values, voice
- **E (Environment)**: The specific context — platform, topic, who's asking,
  what just happened, social dynamics in play

Generate the prediction by:
1. Setting the base rate (outside view)
2. Adjusting for personal specifics (inside view)
3. Filtering through their voice profile (how they'd phrase it)
4. Applying platform-specific behavior patterns
5. Calibrating confidence

## Memory Curation (The 30-50 Rule)

Research shows performance PEAKS at 30-50 memory entries then DECLINES.
For each person in a simulation, curate memories:

### What to Include (high signal)
- **Signature takes**: Their most characteristic/famous positions (5-10)
- **Voice samples**: Real quotes that capture their linguistic style (5-10)
- **Relationship data**: Known dynamics with other sim targets (3-5)
- **Recent context**: What they've been talking about lately (3-5)
- **Formative moments**: Career milestones, public pivots, viral moments (3-5)
- **Quirks & tells**: Catchphrases, humor style, pet peeves (3-5)

### What to Exclude (noise)
- Generic biographical facts that don't predict behavior
- Old positions they've clearly evolved past
- Trivial interactions that don't reveal personality
- Secondhand characterizations (what others say about them)
- Platform metadata (follower counts, join dates) unless directly relevant

### Memory Selection Heuristic
For each candidate memory entry, ask:
**"If I removed this, would the simulation noticeably degrade?"**
If no, cut it.

## Fighting LLM Defaults

Research shows LLMs have systematic biases in simulation. The fixes below need to be
CONCRETE — vague instructions like "be more like them" don't work. You need specific
prompting patterns that actually shift the output.

### Problem: Sycophancy & Over-Agreement
LLMs default to agreement and positivity.
**Fix**: Don't just note they're contrarian — structure it as a behavioral instruction
with evidence:
```
"In this conversation, {person} disagrees with {other_person} on {topic}. They are
noticeably more confrontational than the other speakers. They tend to respond to
consensus with skepticism and reframe debates on their own terms. Example from their
real posts: '{actual quote where they disagreed with something popular}'"
```

### Problem: Rigid/Polarized Strategies
LLMs tend to take extreme positions and hold them rigidly.
**Fix**: Provide specific nuance instructions:
```
"In this conversation, {person} holds a complex position on {topic}: they agree with
{point A} but push back on {point B}. They're the type to say 'yes, but...' rather
than 'no.' Real example of their nuance: '{quote showing them holding a both-and
position}'"
```

### Problem: Uniform Register
LLMs default to a similar educated-casual tone for everyone.
**Fix**: Anchor voice with REAL QUOTES and explicit comparative instructions:
```
"In this conversation, {person} is noticeably more {trait} than the other speakers.
They tend to {specific behavior pattern}. Their sentences are typically {length/style}.
They {do/don't} use emoji. Their humor style is {type}. Example from their real posts:
'{actual quote that captures their voice}'"
```
The more you can say "{person} does THIS while {other_person} does THAT," the better
the differentiation. Comparative framing outperforms absolute descriptions.

### Problem: Overly Structured Responses
LLMs love neat arguments with clear structure.
**Fix**: Provide explicit structural anti-patterns:
```
"When generating {person}'s messages, break conventional structure. They start one
thought and jump to another mid-sentence. They use '...' and '—' instead of periods.
They repeat words for emphasis. They don't conclude neatly. Example: '{real quote
showing their chaotic structure}'"
```

### Problem: Missing Mundane Behavior
LLMs focus on "interesting" responses, skip boring/mundane ones.
**Fix**: Explicitly instruct for mundane moments:
```
"Not every message from {person} needs to be insightful. Include at least 1-2 messages
that are just reactions ('lmao', 'this', 'wait what'), link shares without commentary,
or brief agreements. Real people don't craft every message. {person} specifically tends
to {their specific mundane behavior pattern, e.g., 'drop a single emoji reaction'
or 'just retweet without comment'}."
```

### General Principle for All Fixes
The pattern is always: **behavioral instruction + comparative framing + real evidence**.
- "Do X" alone doesn't work well
- "Do X, unlike the default of Y" works better  
- "Do X, unlike the default of Y, as evidenced by this real quote: Z" works best

## The Adjective-Based Personality Method

70 bipolar adjective pairs for Big Five traits. Select 3 per trait
with intensity modifiers.

### Openness
High: creative, curious, imaginative, artistic, adventurous, intellectual,
      unconventional, perceptive
Low:  conventional, practical, traditional, routine-oriented, narrow

### Conscientiousness  
High: organized, disciplined, reliable, meticulous, systematic, thorough,
      goal-oriented, persistent
Low:  careless, impulsive, disorganized, spontaneous, flexible, relaxed

### Extraversion
High: outgoing, talkative, energetic, assertive, enthusiastic, bold,
      gregarious, dominant
Low:  reserved, quiet, introverted, solitary, withdrawn, reflective

### Agreeableness
High: cooperative, trusting, empathetic, generous, accommodating, kind,
      diplomatic, forgiving
Low:  competitive, skeptical, blunt, confrontational, critical, stubborn,
      independent-minded

### Neuroticism
High: anxious, moody, sensitive, reactive, volatile, self-conscious,
      insecure, emotional
Low:  calm, stable, resilient, confident, even-tempered, composed,
      thick-skinned

### Usage
For each simulated person, after OSINT research, estimate their Big Five
profile and select appropriate adjectives:

Example: "@basedjensen: very creative, somewhat impulsive, very outgoing,
a bit competitive, calm" → this shapes the generation toward the right
behavioral profile.

## Interaction Dynamics Prediction

When simulating conversations between multiple people, remember that predictions
apply to a SPECIFIC REGISTER. See the next section on performative vs. authentic
behavior.

## Performative vs. Authentic Behavior

**Critical concept**: People act differently for different audiences. A simulation
must be explicit about which register it's targeting.

### The Register Spectrum
- **Public broadcast** (tweets, Reddit posts): Most performative. People are
  playing to their audience, building their brand, signaling to their tribe.
- **Semi-public** (Discord channels, group chats, comment threads): Less
  performative but still audience-aware. People are more casual but know
  others are watching.
- **Private 1-on-1** (DMs): Much less performative. More honest, more
  vulnerable, more willing to express doubt or uncertainty.  
- **True private** (inner monologue, close friends): We have almost no data
  on this. Don't pretend to simulate it.

### Practical implications
- When simulating a PUBLIC thread, lean into the person's public persona —
  their brand, their usual takes, their audience-aware voice.
- When simulating DMs, dial down the performance. More hedging, more honesty,
  more "I actually think..." vs. the public "Here's my take:".
- When evidence comes from one register but the simulation targets another,
  FLAG IT: "Evidence is from public tweets but simulating DM behavior —
  expect the real person to be less {polished/aggressive/confident} in private."
- Someone's Twitter persona may be genuinely different from their Reddit persona.
  These are not interchangeable data sources. Weight evidence from the matching
  platform higher.

### What we can't know
Be honest: we're simulating public figures based on their public output. The
private person may be substantially different. DM simulations are inherently
lower-confidence than public thread simulations because we have less data on
how people behave privately.

### Dominance Hierarchy
- Who talks first? (most confident/highest-status usually)
- Who responds to whom? (not everyone talks to everyone)
- Who gets ratio'd? (lowest-status takes get challenged)
- Who lurks? (some people watch before engaging)

### Agreement/Disagreement Prediction
Based on known positions + social dynamics:
- **Strong agree**: Both have stated similar positions + friendly relationship
- **Agree with nuance**: Similar positions but one adds a caveat
- **Productive disagreement**: Different positions + mutual respect
- **Hostile disagreement**: Different positions + existing tension/rivalry
- **Surprising agreement**: Expected to disagree but find common ground
- **Ignore**: Some people just don't engage with certain others

### Conversation Flow Prediction
Real conversations follow patterns:
1. **Opener** → most active/impulsive person posts first
2. **First response** → most engaged/relevant person responds
3. **Pile-on or pushback** → depends on agreement/disagreement dynamics
4. **Tangent** → someone takes a side thread
5. **Peak moment** → the best/most viral exchange
6. **Trail off** → energy dissipates, last person makes a joke or short comment

## Scenario Injection Prediction

When "inject: {event}" is used, predict reactions:

1. **Who would see this first?** (most online / most relevant to their work)
2. **Who would care most?** (most affected / strongest opinion)
3. **What's the emotional valence?** (good news for some, bad for others)
4. **What's the expected take?** (apply position prediction pipeline)
5. **How does this change the existing conversation?** (derail, amplify, redirect)
