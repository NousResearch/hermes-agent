# Anti-Slop Reference — Mechanical Detection for Simulation Output

Source: NousResearch/autonovel ANTI-SLOP.md + slop-forensics + EQ-Bench Slop Score
Adapted for personality simulation: slop in simulated speech is a dead giveaway that
the output is LLM-generated, not human-generated. EVERY simulated utterance must pass
this filter or the simulation fails the "indistinguishable from real" standard.

## Why This Matters More for Simulation Than Normal Writing

Normal LLM output that's a bit sloppy is fine — you know it's AI.
Simulated speech that contains slop BREAKS THE ILLUSION. If @eigenrobot's
simulated tweet contains "delve" or "it's worth noting," anyone who follows
him would instantly know it's fake. Slop detection is the minimum viable
authenticity check.

## Tier 1: Kill on Sight — SCAN AND AUTO-STRIP

These words almost never appear in casual human writing, especially on Twitter.
If ANY appear in simulated tweets/posts, the simulation has failed.

REGEX SCAN LIST (case-insensitive):
```
delve|utilize|leverage\b.*\b(as verb)|facilitate|elucidate|embark|
endeavor|encompass|multifaceted|tapestry|testament|paradigm|
synergy|synergize|holistic|catalyze|catalyst|juxtapose|
nuanced\b|realm\b|landscape\b(metaphorical)|myriad|plethora
```

On detection: REWRITE the sentence using the human alternative.
Do not just swap the word — the sentence structure around slop words
is usually sloppy too.

## Tier 2: Suspicious in Clusters — COUNT PER PERSON

These are fine alone. Three in one person's simulated output = rewrite.

```
robust|comprehensive|seamless|cutting-edge|innovative|streamline|
empower|foster|enhance|elevate|optimize|scalable|pivotal|intricate|
profound|resonate|underscore|harness|navigate\b(metaphorical)|
cultivate|bolster|galvanize|cornerstone|game-changer
```

Count per simulated person. If count >= 3: flag and rewrite.

## Tier 3: Filler Phrases — DELETE ALL

These add zero information. No human tweets these.

SCAN LIST (match as substrings):
```
- "it's worth noting"
- "important to note"  
- "notably"
- "interestingly"
- "let's dive into"
- "let's explore"
- "as we can see"
- "as mentioned earlier"
- "in conclusion"
- "to summarize"
- "furthermore"
- "moreover"
- "additionally" (at start of sentence)
- "in today's"
- "it goes without saying"
- "when it comes to"
- "in the realm of"
- "one might argue"
- "it could be suggested"
- "this begs the question"
- "a comprehensive approach"
- "a holistic approach"  
- "a nuanced approach"
- "not just X, but Y" (the #1 LLM rhetorical crutch)
```

## Rhetorical Slop — The Hardest to Catch

These pass vocabulary checks and mechanical verification but still read as
LLM-generated because the STRUCTURE is too polished. This is the deepest
layer of slop — the instruct model's training to produce "satisfying" output.

### Parallel Antithesis
"The most X are... The most Y are..."
"It's not about X. It's about Y."
Every simulated tweet that contains a balanced two-part rhetorical structure
should be checked: would this person actually construct that parallelism,
or would they just say the second half and trust you to get it?
FIX: delete the setup. Keep only the punchline half.

### "Not X, Not Y, But Z" / "Not Just X, But Y"
The #1 LLM rhetorical crutch. Appears in almost every simulation.
FIX: just say Z. Delete the negations.

### "Show Me X and I'll Show You Y"
Rhetorical formula that reads like a book blurb or TED talk.
No one tweets like this unless they're deliberately performing rhetoric.
FIX: state it flat. "Every community that works has a shared enemy" not
"Show me a thriving community and I'll show you..."

### Clean Escalating Lists
"First it was A, then B, then C, now D" — four perfectly escalating steps.
Real people do 2 steps and trail off, or skip to the end, or lose the thread.
FIX: cut to 2 steps max. Or break the pattern: "first A, then B, and then
somehow we ended up at D and nobody noticed"

### Academic Abstraction in Casual Voice
Words like "instrumentalized" "coordinate human behavior" "recursive loop"
in a tweet from someone who writes casually. The vocabulary is from papers,
not from posting.
FIX: use the word they'd actually reach for. "coordinate human behavior" →
"get people to do stuff." If the plain version sounds dumb, maybe the take
itself is thinner than the fancy words made it seem.

### The "Every Tweet Is A Banger" Problem
The deepest slop: every simulated utterance is GOOD. Considered. Structured.
Satisfying. Real twitter feeds are 70% mid, 20% boring, 10% brilliant.
The simulation should include:
- Half-finished thoughts ("idk if this makes sense but")
- Trailing off ("wait actually nvm")
- Boring logistical tweets ("anyone know a good dentist in brooklyn")
- Self-interruptions ("ok this is getting long")
- Acknowledgments that add nothing ("lol yeah" "hmm" "fair")
If every tweet in the simulation could be screenshot'd as a banger,
the simulation is too polished to be real.

## Structural Slop Patterns — CHECK IN SIMULATION OUTPUT

### Pattern: Identical Sentence Structure Across Speakers
If two or more simulated people use the same sentence structure
(e.g., "The thing about X is Y"), the simulation has failed voice
differentiation. Real people have different syntactic habits.

### Pattern: Topic Sentence Machine
If a simulated post follows: topic sentence → elaboration → example → wrap-up,
it's LLM structure, not human. Real tweets are: punchline first, or tangent,
or one-liner, or trailing thought.

### Pattern: Symmetry Addiction
If the conversation has neat equal turns, balanced perspectives, everyone
getting the same number of posts — that's not real. Real conversations
are asymmetric. Someone dominates. Someone lurks. Someone gets interrupted.

### Pattern: The Hedge Parade
"This approach may potentially help improve..." — no human tweets like this.
Either commit to the statement or don't make it.

### Pattern: Em Dash Overload
Count em dashes (—) per person. If >2 per post on average, flag it.
Most people use them sparingly or not at all.

### Pattern: Sycophantic Agreement Flow
If the conversation flows: A says thing → B says "great point, and also..." →
C says "building on that..." — that's instruct-model conversation, not human.
Real conversations have: disagreement, misunderstanding, tangents, ignoring,
one-upping, and sometimes just "lol."

### Pattern: Uniform Register
If all simulated people sound like they're writing at the same education level
with the same formality — the simulation failed. Real people have wildly different
registers. A shitposter and an academic should sound nothing alike.

## Integration: Mechanical Slop Scan

Run BEFORE subjective discriminator scoring, alongside emoji/length/caps checks.

```
For each simulated utterance:
  1. Scan for Tier 1 words → auto-rewrite if found
  2. Count Tier 2 words per person → flag if >= 3
  3. Scan for Tier 3 filler phrases → auto-delete
  4. Check for structural patterns:
     - Same sentence structure across speakers?
     - Topic-sentence-machine structure?
     - Symmetric turn-taking?
     - Hedge parade?
     - Em dash count?
     - Sycophantic flow?
  5. If ANY Tier 1 found or ANY structural pattern detected: 
     FAIL the utterance and regenerate
```

This scan is MECHANICAL. It cannot be vibes-scored. The words are either
there or they're not. Run it every time, no exceptions.
