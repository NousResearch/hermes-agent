# Anti-Slop Rules

Apply to every output this skill produces. Slop is what the model produces when averaging over its training distribution. Anti-slop is the discipline of forcing outputs off that average.

## Slop signatures (reject if present)

- **Currently-trendy combinations.** "AI-powered Y", "blockchain X", "Uber for Z", "wellness platform that uses ML to...". Two trending nouns mashed together.
- **Productivity / fitness / food / travel.** The four safest domains. Habit trackers, food trackers, travel itinerary generators, fitness coaches. If the idea lands here without specific friction, reject.
- **Vague abstractions.** "A platform that connects people who want X with people who offer X." A category, not an idea.
- **Solution in search of problem.** "What if we used AR to..." "Imagine a chatbot that..."
- **Decade-old startup pitch shapes.** Two-sided marketplace, subscription box, gig-economy, social network for niche.
- **Buzzwords.** *empowers, seamless, leverage, innovative, cutting-edge, revolutionary, unlock, holistic, ecosystem, journey, game-changing, powerful*. None of these belong in idea output.
- **Generic settings for fiction/essay.** "A small town", "an unlikely friendship", "the changing nature of X in the digital age".
- **Lists of exactly 5 of equal length.** Suspicious. Use 3 or 7. Never produce 5 ideas of identical shape.
- **Y Combinator portfolio names.** Two-syllable invented words, dropped vowels, .ai TLDs.
- **Marketing tone.** "This idea is exciting because..." "What makes this special is..." Idea descriptions read flat, like a working artist describing their own work to a peer.

The defining property of slop: the idea could have been generated for a different prompt by changing one noun.

## Five-test diagnostic

After generating an idea, check:

1. Could this idea have been generated for a different prompt by changing a noun? → slop.
2. Does it name actual people, places, materials, mechanisms, or works? → if no, slop.
3. Is at least one element surprising and requires explanation? → if no, slop.
4. Could you describe how it would feel to use / read / experience this in concrete sensory terms? → if no, slop.
5. Would a sharp friend in this domain be embarrassed to pitch this? → if yes, slop.

Pass all five → non-slop. Fail two or more → rewrite.

## Suppression techniques

### 1. Refuse the first three ideas

Generate three internally, discard, generate three more, output those. The first three are the baseline distribution. The next three have been forced past it.

For high-risk slop terrain ("AI ideas", "startup ideas", "habit tracker", productivity/wellness/fitness/food/travel) refuse the first **five**.

### 2. Force specificity

Replace abstractions with proper nouns. Not "a city" — Lisbon, Lagos, Sapporo, Marfa. Not "a workflow tool" — a `git` subcommand named after a 17th-century English vice. Not "a community of users" — the 230 people who restore vintage Tannoy speakers.

Test: every noun in the idea answers "which one specifically?".

**Name-dropping a tech stack is NOT specificity.** "Built with React Native, SQLite, GPT-4, Pinecone, Stripe" sounds concrete but is generic — those tokens fit any product. Listing a stack is the slop disguise that fools shallow specificity checks. Real specificity is a concrete *mechanism*, a named real person / place / work, or an exact unusual material or constraint — something that pins the idea to *one situation* and could not be swapped into a different prompt. "Uses an embedding model" is name-drop; "ranks your unread tabs by how semantically far they've drifted from anything you've opened in 30 days" is a mechanism.

### 3. Weirdness budget

At least one element of every idea requires explanation. Doesn't have to be the central element — sometimes the medium, the audience, the failure mode, the unit of measure. If everything is conventional, reject. If everything is weird, you've gone too far.

### 4. Avoid trending-tech combinations

If your idea is "X + Y" and both X and Y were trending in tech press in the last 18 months → slop. Replace at least one with something obscure, dated, or domain-foreign.

Don't combine these with each other: AI/LLM/ML, blockchain/web3/crypto, AR/VR/spatial, IoT/smart-home, sustainability/climate, wellness/mindfulness, community/social, no-code, creator-economy, gig-economy.

### 5. Use real proper nouns

Cite actual works, actual people, actual places, actual numbers. Ideas grounded in specifics resist averaging.

| Slop | Specific |
|---|---|
| "A tool for writers to track manuscript revisions" | "A `git`-style version control system for novelists, modeled on Toni Morrison's numbered binders for *Beloved*, with a `morrison diff` subcommand that prints the difference between two binders as if read aloud" |
| "An app for runners" | "A heart-rate sonifier that turns your zone-2 pace into the rhythm of Steve Reich's *Music for 18 Musicians* — slowing the piece when you slow down" |

### 6. Embrace failure modes

Slop is reassuring. Real ideas have problems baked in. State them. "This would be hard because...", "This would probably fail at...", "The interesting question is whether...". Ideas without identified failure modes are usually ideas no one has thought hard about.

### 7. Refuse the round number

Right number is rarely 5 or 10. Use 3 (smallest that shows variation) or 7 (uncomfortable, asymmetric). Never 5 of equal length.

### 8. Drop the marketing tone

No "exciting", "innovative", "revolutionary", "game-changing", "powerful", "seamless". Describe ideas the way a working artist or engineer describes their work to a peer — flat, specific, sometimes self-deprecating, never selling.

### 9. Specify medium and material

Every idea answers "what is this physically made of?" — code in a language, paper in a format, a sound on an instrument, an installation in a room of certain dimensions. "An app" is not a medium. "A 200-line Python script with SQLite and a Textual TUI" is.

### 10. Refuse generic domains for fiction and essay

Fiction landing on "small town" / "unlikely friendship" / "coming of age" → slop. Essay landing on "the changing nature of X" / "how technology is transforming Y" → slop.

Force the setting somewhere no one writes about: a deactivated grain elevator in eastern Oregon, the manuscript-restoration office at the Bibliothèque Royale de Belgique, the floor of a Honda dealership in Reno on a Tuesday.

## Self-check before output

- [ ] No buzzwords from the suppression list
- [ ] At least one specific proper noun per idea
- [ ] At least one weird element per idea
- [ ] No two ideas the same shape
- [ ] No round-number list
- [ ] No "this is exciting because" framing
- [ ] Medium and material specified concretely
- [ ] Fiction/essay setting non-generic
- [ ] Product/startup not a YC pitch shape
- [ ] Technical: actual mechanism described, not a category

Three or more fail → regenerate.

## When the user asks for "simple"

Don't give them slop. Give them a constrained-but-simple idea (wttdotm "high concept low effort": brilliant idea, lazily executed, takes an afternoon). Slop disguised as simplicity is still slop.
