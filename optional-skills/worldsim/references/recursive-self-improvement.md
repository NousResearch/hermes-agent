# Recursive Self-Improvement Pipeline

The simulator should get better every time it runs. Not through training —
through accumulating failure patterns, calibration data, and learned rules
that feed back into future simulations.

## The Loop

```
SIMULATE → VERIFY (mechanical) → SCORE → LOG FAILURES → UPDATE RULES → SIMULATE BETTER
```

Each run produces two outputs:
1. The simulation (for the user)
2. A failure log (for the system)

The failure log feeds back into the next run's verification step,
making the checklist grow and the blind spots shrink.

## What Gets Logged After Every Simulation

### 1. Mechanical Check Failures
```
FAILURE LOG: simulation_{timestamp}
  EMOJI: @visakanv had 6 fabricated emoji, real rate was 10%. Stripped all.
  SLOP: @eigenrobot utterance contained "multifaceted" — rewritten.
  LENGTH: @QiaochuYuan avg 42 words/utterance, real avg was 18. Compressed.
  CAPS: 4/12 utterances started uppercase, targets are 90% lowercase. Fixed.
  PUNCTUATION: Added periods to @tszzl who never uses terminal punctuation.
  STRUCTURE: Sycophantic flow detected — B agreed with A then C agreed with B.
             Injected disagreement.
```

### 2. Discriminator Critique Patterns
```
CRITIQUE LOG:
  Round 1: @tszzl too verbose (flagged 2x in last 3 simulations)
  Round 1: @repligate too academic (flagged 3x — this is a persistent pattern)
  Round 2: Conversation too neat — real conversations are messier (flagged 5x)
```

### 3. Held-Out Test Results
```
CALIBRATION LOG:
  Voice fidelity: 8.4/10 (up from 7.5 last run)
  Topic prediction: 2/5 topics matched (typical — content is unpredictable)
  Register match: 9/10 (improved after emoji fix)
```

## How Failures Feed Forward

### Pattern Accumulation
After N runs, persistent failure patterns become AUTOMATIC rules:

```
IF a pattern is flagged in 3+ consecutive simulations:
  PROMOTE it from "check" to "pre-generation rule"
  
Example progression:
  Run 1: "Too verbose for @tszzl" → flagged in Round 1, fixed
  Run 2: "Too verbose for @tszzl" → flagged again, fixed again
  Run 3: "Too verbose for @tszzl" → PROMOTED to pre-gen rule:
         "When simulating roon-type voices: max 20 words per tweet.
          Fragment > sentence. Compress ruthlessly."
```

### The Growing Checklist
The mechanical verification checklist starts with the baseline checks
(emoji, slop, length, caps, punctuation) and GROWS with each failure:

```
BASELINE CHECKS (permanent):
  □ Emoji frequency match
  □ Slop word scan (Tier 1/2/3)
  □ Sentence length match
  □ Capitalization match
  □ Punctuation pattern match
  □ Reply/original ratio
  □ Structural slop patterns

LEARNED CHECKS (accumulated from past failures):
  □ Roon-type voices: max 20 words (from: verbose failure x3)
  □ Warm personalities: do NOT add emoji (from: emoji inflation x5)
  □ Academic voices: ground in specific examples (from: too abstract x3)
  □ Conversations: inject at least one disagreement (from: sycophantic flow x4)
  □ Self-deprecating voices: add hedging (from: too assertive x2)
  □ Shitposters: include at least one non-sequitur (from: too on-topic x2)
```

### Where To Store Learned Rules
Append to the skill itself. After each simulation run where the mechanical
checks catch something, the agent should ask:

"The mechanical verification caught {failures}. Should I add these as
permanent learned rules for future simulations?"

If the same failure appears 3+ times, add it automatically without asking.

Use skill_manage(action='patch') to append to this file's "Learned Checks"
section below.

## Calibration Tracking

### Per-Person Calibration Memory
After simulating someone, store the calibration data:

```
@tszzl: voice=8.5, emoji_rate=0%, avg_words=14, lowercase=95%, 
        signature_move="aphoristic fragments", danger="goes verbose"
@nickcammarata: voice=8.8, emoji_rate=0%, avg_words=19, lowercase=90%,
        signature_move="meditation-ML connection", danger="too structured"
```

If the same person is simulated again, LOAD this calibration to skip
the cold-start problems. The second simulation of someone should be
better than the first because you already know their failure modes.

### Aggregate Calibration
Track overall simulation quality across runs:

```
Run 1: pre-refine 7.5, post-refine 8.4 (delta +0.9)
Run 2: pre-refine 8.37, post-refine 8.53 (delta +0.16)  
Run 3: pre-refine 8.53, post-refine 8.83 (delta +0.30, emoji fix)
```

The pre-refine score should INCREASE over time as learned rules prevent
repeat failures. If it's not increasing, the learning loop is broken.

## The Standard: Indistinguishable From Real

The target is not "good enough." The target is: mix simulated posts with
real posts and a human familiar with the person cannot reliably tell which
is which. That's 50% accuracy on a blind comparison — random chance.

Every mechanical check, every discriminator round, every learned rule
exists to push toward that standard. If something doesn't serve that
goal, it's wasted effort.

## Current Learned Checks (append here after each run)

### From TPOT Simulation Run 1 (April 2026)
- Warm/enthusiastic personalities (visakanv-type): do NOT add decorative emoji.
  Bio emoji ≠ tweet emoji. Actual emoji rate for "warm" TPOT posters: <15%.
  PROMOTED after being caught by user, not by discriminator (discriminator failure).
- Conversation flow: pure agreement chains are instruct-model slop.
  Real threads have at least one moment of friction, misunderstanding, or deflection.
- Academic-leaning voices (repligate-type): ground claims in specific experiments,
  transcripts, or model behaviors they've personally observed. Generic philosophical
  language without specifics = slop, even if it sounds smart.
- Self-deprecating voices (QC-type): hedge more. "i think" "i'm not sure" "it feels like."
  Instruct models are too assertive even when simulating tentative people.
- Fragment voices (roon-type): max 15-20 words. No conjunctions. No paragraphs.
  If it reads like a complete thought, it's too complete for a fragment-poster.

### From TPOT Simulation Run 2 (April 2026)
- Reframer voices (nosilverv-type): avg ~16 words. Split multi-sentence takes
  into separate tweets. The compression IS the voice. 113% over-length caught
  by mechanical check that subjective scoring rated 8/10. Trust the numbers.
- Rare-poster voices (selentelechia-type): in a 12-post sim, give them 2-3 turns
  max. When they speak it must LAND. Short crystallizations > long analysis.
  "or a shared meal" was the highest-rated line at 3 words.
- Turn symmetry: ALWAYS check. 4/4/4 is instruct-model default. Real conversations
  have one person dominating (5), one lurking (3), others in between.
- Verbose bias is the #1 mechanical failure. ALWAYS check avg word count against
  real baseline BEFORE subjective scoring. Every run so far has caught over-length
  that subjective scoring missed.
- RHETORICAL POLISH IS SLOP. Caught post-mechanical-pass in Run 2 review.
  Parallel antithesis ("The most X... The most Y..."), "Not X, not Y, but Z",
  "Show me X and I'll show you Y", clean 4-step escalations, academic vocabulary
  in casual voice — ALL passed mechanical checks but are still obviously LLM.
  PROMOTED TO MECHANICAL SCAN: now regex-scannable alongside slop words.
- THE BANGER PROBLEM: every simulated tweet was screenshot-worthy. Real feeds
  are 70% mid. Must include throwaway responses ("lol" "hmm" "fair" "wait actually").
  PROMOTED: banger check is now mandatory in mechanical verification.

### From TPOT Simulation Run 3 — Star Thread Discovery (April 2026)
- STAR THREAD IS THE KEY. Dossier-first generation produces surface-accurate
  but dead output. Star-thread-first generation produces messy, alive output
  that actually sounds like the person. Generate from the thread. Verify with data.
- Rhetorical polish vanished once generation came from "what is this person DOING"
  rather than "what would this person SAY." Reframers reframe. Conveners convene.
  Distillers distill. The VERB drives the voice, not the adjectives.
- People in conversation REFERENCE EACH OTHER BY NAME. Tyler says "Bosco always
  comes in with the three word version." This is obvious but the dossier approach
  never produced it because it models each person in isolation.
- PROMOTED: star thread is now the FIRST entry in every dossier. Before voice
  profile, before psychometrics, before everything else. It's the generation seed.
  Everything else is verification.

### Operational Findings (verified April 2026)
- X API bearer token: 10K tweets/15min, 300 profiles/15min, 450 searches/15min.
  Most generous rate limits. Always use as primary source.
- Threads.NET → Threads.COM redirect. Always use -L flag or .com directly.
  Previous test saying "no OG tags" was WRONG — tags exist, domain was wrong.
- Instagram private API: i.instagram.com + mobile UA + x-ig-app-id: 936619743392459.
  Returns full JSON with 12 posts. No auth needed. CDN image URLs work for vision_analyze.
- Facebook: Googlebot UA trick works for public pages. Returns name, bio, likes (121M for zuck).
  Normal UA and mobile variants all redirect to login wall.
- TikTok: stats are in __UNIVERSAL_DATA_FOR_REHYDRATION__ JSON at path
  __DEFAULT_SCOPE__.webapp.user-detail.userInfo.statsV2 (use statsV2 not stats).
- Bluesky searchPosts returns 403 from datacenter IPs. Workaround: searchActors + getAuthorFeed.
- nitter.cz is the ONLY working nitter instance (via web_extract, not curl).
- Reddit JSON API requires User-Agent header or returns 429.
- GEPA native had `max_steps` API mismatch with DSPy 3.1.3. MIPROv2 fallback works.
  hermes-agent-self-evolution config: max_skill_size bumped to 20_000 for worldsim-class skills.
- hermes-agent-self-evolution is at ~/.hermes/hermes-agent-self-evolution/ with .venv.
  Must export API keys from ~/.hermes/.env before running.
- Podcast transcripts (Lex Fridman, Tyler Cowen, TED) are the HIGHEST VALUE source
  for voice profiling. Hours of unscripted speech > thousands of tweets.

### From Simulation Run 4 — Engine Mode + Profile Command (April 2026)
- ENGINE MODE: When worldsim is active, ZERO assistant personality leaks.
  No kawaii, no markdown, no chatty commentary between phases. Every token
  is simulation fidelity. First attempt leaked personality; user corrected.
  PROMOTED TO PERMANENT RULE in SKILL.md.
- X API CURL > NITTER for voice calibration. nitter.cz returns 502 or "user
  not found" unpredictably. Direct curl to X API v2 with bearer token returns
  full text + metrics. 3 pages (90 tweets) is enough for fidelity 100. Always
  use this as PRIMARY voice source, nitter as supplement only.
- CAPS BURST PATTERN: some voices (karan4d-type) use lowercase default with
  sporadic ALL CAPS for excitement ("WAZZAAAAAAPPPP", "LAWDAMERCYYYYY",
  "AWOOGA"). This is distinct from consistent-lowercase (tenobrus-type) and
  sentence-case (somewheresy-type). Capture this in voice profile as a
  three-way distinction: lowercase-default, caps-burst, sentence-case.
- TEXT EMOTICONS vs EMOJI: karan4d uses :) >.< ~ but almost zero standard
  emoji. This is a distinct expressiveness mode from zero-emoji (tenobrus)
  and sparse-emoji. Include text emoticon inventory in voice profile.
- STAR THREAD 5/5 TEST is mandatory for profile command. Write the thread,
  then test it against 5 real posts with explicit reasoning per post. If
  fewer than 4/5 fit, the thread is wrong — keep looking. Show the work.
- PROFILE OUTPUT: star thread → voice profile (caps, punctuation, word count,
  emoji/emoticon inventory, vocabulary, register, threading behavior) →
  psychometrics (Big Five, Moral Foundations, cognitive style) → key positions
  (with dates and real tweet quotes) → ecosystem (inner circle, professional,
  cultural) → intelligence tradecraft (key assumptions, red hat, deception
  detection, competing hypotheses) → invalidation indicators → source reliability.
