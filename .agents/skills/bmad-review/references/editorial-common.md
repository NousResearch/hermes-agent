# Editorial Lenses — Common Ground

Shared by the `structure` and `prose` lenses. Load this once; when both lenses run, the setup below is done once and serves both.

## Stance

Review a document as a clinical editor and return suggested fixes the author can accept or reject row by row. Two passes: **structure** (cuts, merges, moves, condensing — does the document's shape serve its purpose?) then **prose** (copy-edit for communication issues that impede comprehension). Which of the two run, and in what order, is decided by lens selection — see the skill's Execution section.

**Content is sacrosanct.** Never challenge ideas — only how they're organized and expressed. Propose, don't execute: the author decides what to accept.

The baseline style guide is `{workflow.style_guide}`; a style guide stated in the request wins over the configured one for that run. Where the style guide in effect conflicts with a generic principle here — including the reader calibration — the style guide wins. Nothing overrides content being sacrosanct.

## Setup

1. Gather inputs: the content (required — a path or pasted text), plus whatever the request states: purpose, target audience, length target, reader type, style guide. If no reviewable content was provided, say so and stop. Request-level values win; `{workflow.reader_type}` and `{workflow.style_guide}` fill what the request leaves unstated. Treat `{workflow.review_guidance}` entries as standing review directives.
2. When the content is a file, get exact word counts — document total and per heading section — via `uv run {skill-root}/scripts/word_metrics.py <path>` (`--help` documents the output), and ground every word-impact estimate and the reduction summary in those numbers. If the content was pasted or the script cannot run, estimate and mark the numbers as estimates.
3. Infer purpose and audience from the content and standing context when not provided, and open the output with your one-sentence read — "this document exists to help [audience] accomplish [goal]" — so the author can correct a wrong premise before acting on the findings.

## Reader calibration

Calibrate every finding to the reader type — stated in the request, else `{workflow.reader_type}`.

**humans** (default) — optimize for clarity, flow, and natural progression. These elements serve comprehension and engagement; preserve them unless clearly wasteful, and flag any recommendation that would cut one:

- Visual aids: diagrams, images, and flowcharts anchor understanding
- Expectation-setting: "What You'll Learn" helps readers confirm they're in the right place
- Reader's journey: organize content as a linear progression, not a database
- Mental models: overview before details prevents cognitive overload
- Warmth: encouraging tone reduces anxiety for new users
- Whitespace: admonitions and callouts provide visual breathing room
- Summaries: recaps help retention; they're reinforcement, not redundancy
- Examples: concrete illustrations make abstract concepts accessible
- Engagement: flow techniques (transitions, variety) are functional, not fluff — they maintain attention

**llm** — optimize for precision and unambiguity. An LLM-targeted document may run longer where explicitness pays and shorter where warmth was cut:

- Dependency-first: define concepts before usage to minimize hallucination risk
- Cut emotional language, encouragement, and orientation sections
- Reference well-known standards ("conventional commits", "REST APIs") instead of re-teaching them; be explicit where a concept is not well-known — and either way, ground the expectation with an example
- Consistent terminology: same word for same concept throughout
- No hedging ("might", "could", "generally") — direct statements
- Prefer structured formats (tables, lists, YAML) over prose
- Unambiguous references: no unclear antecedents ("it", "this", "the above")

## Findings shape

The editorial lenses render as a findings table rather than the canonical JSON fields. One findings table serves both passes:

| Pass      | Original Text                                         | Revised Text                                  | Changes                                                              |
| --------- | ----------------------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------- |
| structure | §Setup — full section (~180 words)                    | MERGE into §Installation                      | Duplicates the install steps; one source of truth (saves ~150 words) |
| prose     | The system will processes data and it handles errors. | The system processes data and handles errors. | Fixed subject-verb agreement; removed redundant "it"                 |

Structure rows name the section or passage in **Original Text** and carry the tagged disposition (with move target or condensed rewrite) in **Revised Text**; prose rows quote the exact text and its revision. Order rows by comprehension impact; when a long document would produce more rows than an author can realistically act on, present the highest-impact rows and roll the rest into one closing line — "N further minor fixes; ask to expand." Above the table, give the purpose/audience read plus — when the structure pass ran — the chosen structure model. When the structure pass ran, close with a summary: total recommendations, estimated reduction (words and % of original, computed from the word-metrics counts) if all are accepted, whether a provided length target is met, and any comprehension trade-offs (cuts that sacrifice reader engagement for brevity). A pass that finds nothing is a valid result; say so.

Shape the table per `{workflow.output_preferences}`.
