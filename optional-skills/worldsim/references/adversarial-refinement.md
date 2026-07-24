# Adversarial Refinement — GAN-Style Accuracy Convergence

Three self-improving loops that push simulation accuracy toward reality.
This is what separates "creative roleplay" from "predictive simulation."

## Philosophy

A GAN has a generator and a discriminator locked in a game.
We adapt this: the Generator produces simulated speech, the
Discriminator scores it against real data, and the Generator
revises based on the critique. Multiple rounds = convergence.

The key insight: we have REAL DATA from the targets. Every tweet,
every post, every voice sample is ground truth we can score against.
Most simulators throw away this advantage by generating in one shot.

## Approach 1: Discriminator Loop (Real-Time Refinement)

Run AFTER initial simulation generation. 2-3 rounds.

### Round Flow
```
GENERATE → DISCRIMINATE → CRITIQUE → REGENERATE → DISCRIMINATE → ...
```

### Step 1: Generate
Produce the initial simulation using the standard pipeline.

### Step 2a: Mechanical Verification (MANDATORY — runs BEFORE subjective scoring)

These checks are QUANTITATIVE. They compare numbers from real data to numbers
from simulated output. They cannot be hand-waved. Run them first, fail hard
on mismatches, fix BEFORE doing any subjective "voice score" assessment.

The generator and discriminator share the same brain (the LLM). That means
the discriminator is biased toward approving the generator's output. Mechanical
checks are the circuit breaker that prevents collapse.

**EMOJI FREQUENCY CHECK**
```
1. Count emoji in last 30 real tweets → emoji_rate = tweets_with_emoji / total
2. Count emoji in simulated utterances for this person
3. If simulated emoji rate > real emoji rate + 10%: FAIL. Remove emoji.
4. Check WHICH emoji they use. If simulated uses emoji not in their real set: FAIL.
5. Check WHERE they use emoji: originals vs replies vs both?
   Bio emoji ≠ tweet emoji. Many people have emoji in bio, zero in posts.
```

**SENTENCE LENGTH CHECK**
```
1. Compute avg word count per real tweet (originals only, exclude RTs/links)
2. Compute avg word count per simulated utterance for this person
3. If simulated avg differs by >40% from real avg: FAIL. Adjust length.
   (e.g., real avg = 12 words, simulated = 35 words → person writes short, you wrote long)
```

**CAPITALIZATION CHECK**
```
1. Count % of real tweets starting with lowercase letter
2. Count % of simulated utterances starting with lowercase
3. If mismatch >20%: FAIL. Fix capitalization.
   (Most TPOT people are lowercase-first. Instruct models default to uppercase.)
```

**PUNCTUATION PATTERN CHECK**
```
1. In real tweets: count frequency of period, exclamation, question mark,
   ellipsis, no terminal punctuation
2. Compare to simulated. Key tells:
   - Do they end tweets with periods? (many people don't)
   - Do they use "!!" or "!!!"? (some do, most don't)
   - Do they trail off with "..."?
3. If simulated adds punctuation the person doesn't use: FAIL.
```

**REPLY/ORIGINAL RATIO CHECK**
```
1. From their real tweet data: what % are replies vs originals?
2. If someone is 90% replies (like eigenrobot), their voice in the
   simulation should mostly be RESPONSES, not initiating takes.
3. If a reply-heavy person is simulated as a take-launcher: FAIL.
```

**VOCABULARY SPOT CHECK**
```
1. From simulated text, extract 3 distinctive words/phrases
2. Search: do these words/phrases appear in their real tweets?
3. If you're putting words in their mouth they've never used: FLAG.
   (Not auto-fail — people use new words — but flag for review)
```

**RHETORICAL SLOP SCAN**
```
1. Scan for parallel antithesis: "The most X... The most Y..."
   "It's not about X. It's about Y." → FAIL if found. Keep only the punchline half.
2. Scan for "Not X, not Y, but Z" / "Not just X, but Y" → FAIL. Just say Z.
3. Scan for "Show me X and I'll show you Y" → FAIL. State it flat.
4. Count escalating list steps (first A, then B, then C, now D).
   If 4+ clean steps: FAIL. Cut to 2 or break the pattern.
5. Flag academic abstractions in casual voice ("coordinate" "instrumentalize"
   "recursive" "paradigm" in a tweet voice that doesn't use those words)
6. THE BANGER CHECK: read all utterances for one person sequentially.
   If every single one could be screenshot'd as a standalone banger: FAIL.
   Real feeds are 70% mid. Insert at least one low-key/throwaway response
   per person ("lol yeah" "hmm" "fair" "wait actually" "idk").
```

Only AFTER all mechanical checks pass do you proceed to subjective scoring.
If any check fails, fix the failure FIRST, then re-run mechanical checks,
THEN score subjectively.

### Step 2b: Discriminate (subjective, AFTER mechanical checks pass)
For each simulated utterance, run these checks against real data:

**Voice Match Score** — Does it SOUND like them?
- Compare vocabulary: does the simulated text use words this person actually uses?
- Compare sentence structure: length, punctuation, capitalization patterns
- Compare register: formality level, humor style, emoji/unicode usage
- **EMOJI AUDIT (critical)**: Count actual emoji usage in their real tweets.
  Most people use emoji FAR less than instruct models assume. A "warm" person
  ≠ emoji user. Check: what % of their real tweets contain emoji? Which specific
  emoji do they use? Are they in originals or only replies? Bio emoji ≠ tweet emoji.
  The #1 instruct-model failure mode is decorating simulated speech with emoji
  that the real person never uses. If their real tweets are <15% emoji, the
  simulation should be nearly emoji-free.
- Method: Show the discriminator 5 REAL posts and the simulated post.
  Ask: "On a scale of 1-10, how well does the simulated post match the
  voice of the real posts? What specific elements are wrong?"

**Position Match Score** — Does it say what they'd ACTUALLY say?
- Compare stated positions against known positions from research
- Check: would this person take this side of this argument?
- Check: would they frame it this way? (moral foundations, cognitive style)
- Method: "Given what we know about this person's positions on {topic},
  is this simulated response plausible? What would they actually say differently?"

**Interaction Match Score** — Does the conversation FLOW realistically?
- Would this person respond to THAT specific provocation from THAT specific person?
- Is the social dynamic right? (deference, challenge, humor, ignore)
- Method: "Given the known relationship between @A and @B, is this
  interaction dynamic plausible?"

### Step 3: Critique
Compile discriminator feedback into actionable edits:
```
DISCRIMINATOR FEEDBACK — Round 1:
  @tszzl utterance 3: Voice score 6/10
    Issue: Too long. Roon posts in fragments, not paragraphs.
    Fix: Break into 2-3 shorter tweets. Remove conjunctions.
  
  @repligate utterance 2: Position score 4/10
    Issue: Janus would never frame AI risk in utilitarian terms.
    They use phenomenological/consciousness-first framing.
    Fix: Reframe through the lens of simulacra theory.
```

### Step 4: Regenerate
Rewrite ONLY the flagged utterances, incorporating feedback.
Keep utterances that scored 8+ unchanged.

### Step 5: Re-Discriminate
Score again. If all utterances hit 7+, stop. If not, one more round.
Hard cap at 3 rounds to prevent infinite loops.

### Implementation
```
For each simulated utterance:
  1. Pull 5 real posts from the person (random sample from voice data)
  2. Present real posts + simulated post to the LLM-as-discriminator
  3. Ask for: voice score (1-10), specific mismatches, suggested edits
  4. If score < 7, regenerate with the critique as context
  5. Re-score
```

## Approach 2: Held-Out Prediction Test (Ground Truth Calibration)

The most rigorous accuracy measure. Run BEFORE simulation to calibrate
the model, or AFTER to validate.

### Method
1. Pull N recent original tweets from each target
2. Split: older half = "context" (voice training), newer half = "ground truth"
3. Give the simulator ONLY the context tweets
4. Ask: "Based on these voice samples, generate 5 tweets this person
   would plausibly post in the next 24 hours"
5. Compare generated tweets to the held-out ground truth
6. Score on: topic overlap, voice fidelity, register match, originality

### Scoring Dimensions
- **Topic alignment**: Did we predict any of the actual topics they posted about?
  (Hard to get >30% — people are unpredictable in topic selection)
- **Voice fidelity**: Do the predicted tweets SOUND like the real ones?
  (Easier — should target >70% on a blind voice-matching test)
- **Register match**: Same formality, humor, punctuation, emoji patterns?
  (Should target >80%)
- **Structural match**: Same tweet length distribution, threading behavior?
  (Should target >70%)

### What This Tells You
- If voice fidelity is low: your dossier voice profile is wrong. Re-research.
- If topics don't overlap: that's EXPECTED. Content is unpredictable.
  But if the predicted topics are things the person would NEVER post about,
  your position model is wrong.
- If register doesn't match: your linguistic analysis missed something.
  Go back to the raw tweets and look for patterns you overlooked.

### Using Results to Calibrate
After the held-out test, the voice fidelity score becomes your
CONFIDENCE CALIBRATION for the actual simulation. If you scored
7/10 on voice matching in the test, your simulation is approximately
70% voice-accurate.

## Approach 3: Historical Replay (Hardest, Most Rigorous)

Find a REAL conversation thread between the simulation targets.
Simulate it blind. Diff against reality.

### Method
1. Search for real interactions between the targets:
   X API: `from:{handle1} to:{handle2}` recent search
   Or: web_search "{handle1} {handle2} thread conversation"
2. Find a substantive conversation (not just "lol" replies)
3. Extract the TOPIC and FIRST POST of the real conversation
4. Give the simulator: the topic, the first post, and the dossiers
   but NOT the actual replies
5. Simulate how the conversation would go
6. Compare simulated replies to actual replies
7. Score: position accuracy, voice accuracy, dynamic accuracy

### Scoring
- **Position accuracy**: Did the simulated person take the same stance
  as the real person? (Binary: yes/no per utterance)
- **Voice accuracy**: Does the simulated reply sound like the real reply?
  (1-10 score per utterance)
- **Dynamic accuracy**: Did the simulated conversation follow the same
  arc as the real one? (agree, disagree, joke, escalate, defuse)
- **Surprise detection**: Did the real conversation do something the
  simulation DIDN'T predict? (This reveals model blind spots)

### When To Use
- Before launching a high-fidelity simulation, find one real interaction
  to use as calibration
- If the historical replay scores <50% position accuracy, the dossiers
  need more research
- If voice scores <60%, the voice profiles need more real quote anchoring

## Approach 4: Comparative Discrimination (Tournament Style)

Generate 3 different versions of the same utterance for a person.
Mix in 2 REAL posts from them. Ask: "Which of these 5 posts are real?"

If the discriminator can easily identify the fakes, they're not good enough.
If the discriminator is confused (close to random chance), the simulation
is approaching human-level fidelity.

### Method
1. Generate 3 simulated tweets for @person on a given topic
2. Pull 2 real tweets from @person on a similar topic
3. Shuffle all 5
4. Ask: "These are 5 posts attributed to @person. 2 are real, 3 are
   simulated. Which 2 are real? Explain your reasoning."
5. Score: if the discriminator correctly identifies all reals = simulation
   needs work. If it misidentifies any = simulation is convincing.

### Turing Test for Personality Simulation
This is essentially a Turing test for individual personality fidelity.
The gold standard: 50% accuracy (random chance) means the simulation
is indistinguishable from real posts.

## Integration Into Pipeline

### Minimum (fidelity 50+)
After Phase 3 simulation, run ONE round of Approach 1 (discriminator loop).
Score each utterance against 3 real posts. Regenerate anything below 6/10.

### Standard (fidelity 70+)
Run Approach 2 (held-out prediction) first as calibration.
Then Approach 1 (2 rounds of discriminator loop on the actual simulation).

### Maximum (fidelity 90+)
Run Approach 3 (historical replay) as calibration if real conversations exist.
Run Approach 2 (held-out prediction) for voice calibration.
Run Approach 1 (3 rounds of discriminator loop).
Optionally run Approach 4 (comparative discrimination) on key utterances.

## Key Principles

1. **Real data is the reward signal.** Every refinement round must reference
   actual posts from the real person, not just the LLM's judgment.
2. **Voice is easier to match than content.** Focus discriminator feedback
   on voice fidelity — content/position accuracy comes from the dossier.
3. **Diminishing returns after 3 rounds.** The LLM starts overfitting to
   its own critique. Stop at 3 rounds max.
4. **Separate scores for separate dimensions.** Don't collapse voice +
   position + dynamics into one number. Keep them distinct so you know
   WHERE the simulation is weak.
5. **Document the scores.** After refinement, append to the simulation
   output: "Voice fidelity: X/10, Position accuracy: X/10, Rounds: N"
