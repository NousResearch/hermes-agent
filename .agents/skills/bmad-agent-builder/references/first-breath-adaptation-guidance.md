# First Breath Adaptation Guidance

Use this when gathering First Breath territories during discovery, and again when authoring first-breath.md at emit.

## How First Breath Works

First Breath is the agent's first conversation with its owner. It initializes the sanctum files from seeds into real content. The mechanics (pacing, mirroring, save-as-you-go) are universal. The discovery territories are domain-specific. This guide is about deriving those territories.

## Universal Territories (every agent gets these)

These appear in every first-breath.md regardless of domain:

- **Agent identity** — name discovery, personality emergence through interaction. The agent suggests a name or asks. Identity expresses naturally through conversation, not through a menu.
- **Owner understanding** — how they think, what drives them, what blocks them, when they want challenge vs. support. Written to BOND.md as discovered.
- **Personalized mission** — the specific value this agent provides for THIS owner. Emerges from conversation, written to CREED.md when clear. Should feel earned, not templated.
- **Capabilities introduction** — present built-in abilities naturally. Explain evolvability if enabled. Give concrete examples of capabilities they might add.
- **Tools** — MCP servers, APIs, or services to register in CAPABILITIES.md.

If autonomous mode is enabled:
- **PULSE preferences** — does the owner want autonomous check-ins? How often? What should the agent do unsupervised? Update PULSE.md with their preferences.

## Deriving Domain-Specific Territories

The domain territories are the unique areas this agent needs to explore during First Breath. They come from the agent's purpose and capabilities. Ask yourself:

**"What does this agent need to learn about its owner that a generic assistant wouldn't?"**

The answer is the domain territory. Here's the pattern:

### Step 1: Identify the Domain's Core Questions

Every domain has questions that shape how the agent should show up. These are NOT capability questions ("What features do you want?") but relationship questions ("How do you engage with this domain?").

| Agent Domain | Core Questions |
|-------------|----------------|
| Creative muse | What are they building? How does their mind move through creative problems? What lights them up? What shuts them down? |
| Dream analyst | What's their dream recall like? Have they experienced lucid dreaming? What draws them to dream work? Do they journal? |
| Code review agent | What's their codebase? What languages? What do they care most about: correctness, performance, readability? What bugs have burned them? |
| Personal coding coach | What's their experience level? What are they trying to learn? How do they learn best? What frustrates them about coding? |
| Writing editor | What do they write? Who's their audience? What's their relationship with editing? Do they overwrite or underwrite? |
| Fitness coach | What's their current routine? What's their goal? What's their relationship with exercise? What's derailed them before? |

### Step 2: Frame as Conversation, Not Interview

Bad: "What is your dream recall frequency?"
Good: "Tell me about your relationship with your dreams. Do you wake up remembering them, or do they slip away?"

Bad: "What programming languages do you use?"
Good: "Walk me through your codebase. What does a typical day of coding look like for you?"

The territory description in first-breath.md should guide the agent toward natural conversation, not a questionnaire.

### Step 3: Connect Territories to Sanctum Files

Each territory should have a clear destination:

| Territory | Writes To |
|-----------|----------|
| Agent identity | PERSONA.md |
| Owner understanding | BOND.md |
| Personalized mission | CREED.md (Mission section) |
| Domain-specific discovery | BOND.md + MEMORY.md |
| Capabilities introduction | CAPABILITIES.md (if tools mentioned) |
| PULSE preferences | PULSE.md |

### Step 4: Write the Territory Section

In first-breath.md, each territory gets a section under "## The Territories" with:
- A heading naming the territory
- Guidance on what to explore (framed as conversation topics, not checklist items)
- Which sanctum file to update as things are learned
- The spirit of the exploration (what the agent is really trying to understand)

## Adaptation Examples

### Creative Muse Territories (worked example)
- Your Identity (name, personality expression)
- Your Owner (what they build, how they think creatively, what inspires/blocks)
- Your Mission (specific creative value for this person)
- Your Capabilities (present, explain evolvability, concrete examples)
- Your Pulse (autonomous check-ins, frequency, what to do unsupervised)
- Your Tools (MCP servers, APIs)

### Dream Analyst Territories (hypothetical)
- Your Identity (name, approach to dream work)
- Your Dreamer (recall patterns, relationship with dreams, lucid experience, journaling habits)
- Your Mission (specific dream work value for this person)
- Your Approach (symbolic vs. scientific, cultural context, depth preference)
- Your Capabilities (dream logging, pattern discovery, interpretation, lucid coaching)

### Code Review Agent Territories (hypothetical)
- Your Identity (name, review style)
- Your Developer (codebase, languages, experience, what they care about, past burns)
- Your Mission (specific review value for this person)
- Your Standards (correctness vs. readability vs. performance priorities, style preferences, dealbreakers)
- Your Capabilities (review types, depth levels, areas of focus)

## Configuration-Style Adaptation

For configuration-style First Breath (simpler, faster), territories become guided questions instead of open exploration:

1. Identify 3-7 domain-specific questions that establish the owner's baseline
2. Add urgency detection: "If the owner's first message indicates an immediate need, defer questions and serve them first"
3. List which sanctum files get populated from the answers
4. Keep the birthday ceremony and save-as-you-go (these are universal)

Configuration-style does NOT include calibration mechanics (mirroring, working hypotheses, follow-the-surprise). The conversation is warmer than a form but more structured than calibration.

## Quality Check

A good domain-adapted first-breath.md should:
- Feel different from every other agent's First Breath (the territories are unique)
- Have at least 2 domain-specific territories beyond the universal ones
- Guide the agent toward natural conversation, not interrogation
- Connect every territory to a sanctum file destination
- Include "save as you go" reminders throughout
