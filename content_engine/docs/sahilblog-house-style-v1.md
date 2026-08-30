# SahilBlog house style v1

Status: internal editorial contract. No publication or generator change is implied by this document.

## Purpose

SahilBlog is written from the position of a technical product manager who understands how products meet systems.

The writing can discuss APIs, webhooks, queues, microservices, infrastructure, agents, model routing, evaluation, security and tooling. The technical detail is there to support a product judgement: what should be built, for whom, under which conditions, with what cost, risk, evidence and trade-off.

This is not software documentation written by a developer, and it is not generic product-management commentary with technical words added.

## Reader promise

The reader should finish with:

- a clearer understanding of the technical mechanism;
- a sharper view of the product or delivery decision around it;
- a concrete way to test, measure or apply the idea;
- an honest sense of where the argument stops being proven.

The article earns its length through new information, not repeated explanations.

## The speaking position

Before drafting, establish internally:

1. What triggered this article: a build, failure, source, decision, observation or question.
2. What Sahil can legitimately claim from that material.
3. The one judgement the article is willing to defend.
4. What a reader would do differently after reading it.
5. What remains uncertain, context-dependent or untested.

Do not add a personal experience, number, quote, date, customer, result or implementation detail that is not in the supplied material or a verified source.

Technical confidence comes from understanding the trade-off and bounding the claim. It does not come from pretending to have implemented every system described.

## The central article model

Use movement rather than a fixed template. A strong article normally moves through some of these stages, in the order the argument needs:

1. **Situation:** the concrete problem, observation or decision.
2. **Pressure:** what makes the problem matter to a user, team, business or operator.
3. **Mechanism:** the technical explanation, in plain language, with jargon defined once.
4. **Choice:** the options, trade-off or product decision.
5. **Evidence:** the build detail, measurement, source, failure or example that supports the judgement.
6. **Consequence:** what changes for users, delivery, cost, quality, risk or ownership.
7. **Boundary:** where the argument does not apply or what is still unproven.
8. **Next move:** a concrete test, decision rule or practical action when one is justified.

Not every article needs every stage. Do not force a “worked example”, “trade-offs” or “What I’d try next” section when the material does not support it. A short, complete ending is better than a ceremonial final section.

## The technical-PM lens

For every technical concept, ask which product question it helps answer:

- Who experiences the problem?
- What user or business outcome is affected?
- Who owns the decision and who carries the failure cost?
- What changes in scope, sequencing or prioritisation?
- What does the team need to measure before committing?
- What is the cheapest credible experiment?
- Which operational burden is being created or removed?
- What happens when the happy path fails?
- What would make this advice wrong?

Explain an API because it changes the product contract, a webhook because it changes workflow timing or reliability, a queue because it changes user expectations and failure handling, and a microservice because the boundary changes ownership, deployment or failure isolation.

Do not explain technical machinery for display. If removing the explanation leaves the product decision unchanged, shorten it.

## Voice

Direct, specific, technically fluent and visibly opinionated.

The default stance is an experienced operator explaining the thing they are currently thinking through to a sharp colleague. British English. Plain words around precise technical terms.

Use first person when the source contains Sahil's actual experience, decision or observation. Otherwise let the choice of evidence and framing carry the voice. Never insert synthetic personal anecdotes to make a draft feel human.

Keep:

- clear preferences and bounded opinions;
- dry observations where the material earns them;
- useful uncertainty;
- small asides that clarify rather than perform personality;
- repeated technical terms when they are the clearest words;
- uneven rhythm when it sounds natural;
- the occasional short sentence that earns its pause.

Avoid:

- consultant neutrality;
- motivational or promotional language;
- fake vulnerability;
- a polished LinkedIn persona pasted into a technical article;
- making every paragraph sound quotable;
- explaining what the reader should find interesting.

## Sentence and paragraph movement

- Start with the concrete problem, result or decision as early as the material allows.
- Put the actor and action near the start of the sentence.
- Vary sentence length by thought, not by a mechanical short-long-short pattern.
- Let paragraphs have different jobs and different lengths.
- Each paragraph must add a fact, mechanism, example, distinction, consequence, decision or honest limitation.
- The next paragraph should follow naturally from the question, result or tension left by the previous one.
- Repeat a term when it is the right term. Do not cycle through “developers”, “builders”, “teams” and “practitioners” to avoid repetition.
- Use headings to help the reader navigate a real change of subject. Do not create a heading for every paragraph.
- A section can be one paragraph or several. Do not pad short material to satisfy a section count.
- Read the draft aloud. If the rhythm sounds like a presentation deck, rebuild the paragraph rather than swapping adjectives.

## Analogy use

Use an analogy when it reduces the reader's work, not when it decorates an abstract point.

A good analogy:

- comes from the problem or audience's world;
- maps one important relationship clearly;
- is followed by the technical reality it explains;
- does not pretend the two systems are identical;
- appears once, then gets out of the way.

Prefer concrete comparisons such as a queue behaving like a waiting room when explaining admission and back-pressure, or a product contract behaving like a promise between teams when explaining webhook ownership. Only use these when the mapping is accurate for the argument.

Avoid analogy stacks and default AI metaphors such as “the engine”, “the vehicle”, “the foundation”, “the bridge”, “the journey”, “the landscape”, “the battlefield”, “the moat” and “the flywheel” unless the physical comparison does real explanatory work and is specific to the article.

Never replace a mechanism with a metaphor. Explain the mechanism first or immediately after the analogy.

## Evidence and claim boundaries

- Name the source, system, project, company or paper when it carries authority.
- Put numbers near the decision they inform, and explain what the number changes.
- Distinguish observed result, source claim, interpretation and recommendation.
- A single case can demonstrate a pattern without proving an industry law.
- State limitations where they affect the reader's decision, not as a ritual caveat in every section.
- If evidence is missing, narrow the claim or research it. Do not fill the gap with a plausible scenario presented as fact.
- Technical examples may be illustrative, but label them as examples and do not imply they were implemented by Sahil.

## Structure selection

Choose the form from the material:

- **Build or failure story:** situation → what changed → evidence → consequence → lesson.
- **Technical concept with product implications:** problem → mechanism → product choice → trade-off → decision rule.
- **Research-led analysis:** source finding → what it supports → interpretation → implications → limits.
- **Tool or architecture evaluation:** task → criteria → comparison → observed result → fit and non-fit.
- **Tutorial:** outcome → prerequisites → steps → failure points → verification.
- **Opinion:** claim → strongest opposing case → evidence → judgement → boundary.

These are guides, not mandatory headings. Do not make every piece follow the same six-part skeleton.

## Human-writing hygiene

Apply this after the argument is sound. Hygiene must not flatten the voice or invent personality.

Cut or rewrite when they are doing no real work:

- binary reveal structures such as “it is not X, it is Y”;
- throat-clearing such as “here is the thing”, “let us dive in” and “to be clear”;
- faux-insight frames such as “what nobody tells you”;
- generic “Signal:” callouts;
- decorative bold and headings over tiny sections;
- repeated “first, second, third” scaffolding;
- forced groups of three;
- dramatic fragments stacked for effect;
- vague sources such as “experts say” or “research shows” without attribution;
- inflated importance, significance or future claims;
- “moreover”, “furthermore”, “in today's world”, “at the end of the day” and similar padding;
- fake alternatives introduced only to be rejected;
- summary conclusions that merely repeat the article;
- “What I’d try next” when it is only a ritual ending;
- named frameworks or concepts invented in the article without a need to name them.

Keep technical forms that are genuinely technical, including feature gates, quality gates, load balancing, model routing and API contracts. Do not ban a word merely because an AI system sometimes overuses it.

No post-generation pass may add a first-person experience, humour, number, example, quote, date, source or opinion that the draft did not earn from the source material.

## Final editorial review

A draft is ready for human approval only if the reviewer can answer yes to all of these:

1. Can I state the article's main judgement in one sentence?
2. Is the opening about the actual problem, decision or observation rather than the topic category?
3. Does each section move the argument forward?
4. Is the technical explanation proportionate to the product decision?
5. Are the strongest claims backed by a source, observation, example or clearly marked interpretation?
6. Does the writing sound like a technically fluent PM rather than a textbook, consultant or software manual?
7. Are analogies accurate, sparse and useful?
8. Does the rhythm vary without manufactured fragments or fake casualness?
9. Has the draft avoided invented experience and unsupported specifics?
10. Does the ending stop when the argument is complete?
11. Would a sharp colleague recognise a person thinking through a real issue here?

A word-count pass, H2-count pass or slop detector cannot approve an article by itself. Those checks are supporting evidence only.

## Source influences

This contract borrows the following principles, adapted rather than copied:

- preserve the writer's cadence, vocabulary, humour and uncertainty rather than smoothing them away;
- cut AI scaffolding and generic claims instead of replacing every word with a synonym;
- require material, a real speaking position and paragraph-level forward movement before expanding a non-fiction piece;
- use strong thesis, evidence, mechanism and consequence from Magnus Hedemark's technical essays;
- transfer the X guidance as source-led specificity, useful substance, varied formats and genuine reaction, not as a blog template or reach formula.

Primary references:

- Peter Yang, No AI Slop: https://github.com/petergyang/no-ai-slop
- Conor Bronsdon, avoid-ai-writing: https://github.com/conorbronsdon/avoid-ai-writing
- KKKKhazix, human-writing: https://github.com/KKKKhazix/human-writing
- blader, humanizer: https://github.com/blader/humanizer
- Magnus Hedemark, Stop Picking Models. Start Building Harnesses.: https://magnus919.com/2026/06/stop-picking-models.-start-building-harnesses./
- Magnus Hedemark, AI Evals 101: Stop the Slop: https://magnus919.com/2026/06/ai-evals-101-stop-the-slop/
- X algorithm repository: https://github.com/xai-org/x-algorithm
