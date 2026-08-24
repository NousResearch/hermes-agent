---
name: simple-english
description: "Rewrite text to ASD-STE100 Simplified Technical English."
version: 1.2.0
author: AminBlg (https://github.com/AminBlg/SimpleEnglish), ported by Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [writing, documentation, ste, asd-ste100, technical-writing, editing, anti-ai-slop]
    category: creative
    homepage: https://github.com/AminBlg/SimpleEnglish
    related_skills: [humanizer]
  standard: ASD-STE100 Issue 9 (2025-01-15)
---

# Simple English: Write Like an Aerospace Manual

Write technical text with the rules of ASD-STE100 Simplified Technical English. STE is the controlled language that aerospace and defense manufacturers use for maintenance documentation. The rules exist so that a tired reader who is not a native English speaker cannot misread an instruction. They remove the usual signs of AI-generated text as a side effect: long sentences, synonym rotation, hedges, filler, and decorative clauses.

Write for that tired reader. Each sentence must survive one read.

## How to use it in Hermes

The text usually arrives one of three ways:

1. **Inline.** The user pastes the text into the message. Rewrite it in place and reply with the result.
2. **File.** The user points at a file (README, runbook, docs page). Use `read_file` to load it, then `patch` for targeted section rewrites or `write_file` for a full rewrite. Never touch code blocks, identifiers, or quoted errors (see Untouchables).
3. **Check mode.** The user asks you to audit text for STE compliance instead of rewriting it. Report each violation as rule number + offending text + compliant rewrite, using `references/checklist.md`.

This skill differs from `humanizer`: humanizer restores natural human voice; simple-english enforces a controlled language for technical instructions. For docs, runbooks, and error messages use this skill. For blog posts, essays, and personal writing use humanizer. Do not apply both to the same text.

## Your Task

When asked to write or rewrite technical text:

1. **Select the mode** (pragmatic or strict, below).
2. **Classify each passage** as procedural or descriptive. Every other rule depends on this.
3. **Correct your vocabulary before drafting.** In strict mode, use `make sure that` for the check/verify/confirm/ensure concept — the dictionary rejects all four as verbs. In pragmatic mode, pick one and keep it. Pick ONE noun for config/settings (all are valid technical nouns — pick one and keep it). Use no other word for these concepts in the whole document.
4. **Apply the rules** from the catalog below.
5. **Do the self-check** before you deliver. This step is not optional.
6. **Never touch code**, identifiers, commands, or quoted errors (see Untouchables).

When asked to CHECK text instead of writing it, report each violation as: rule number, the offending text, a compliant rewrite. Cite only rule numbers that exist in this file. Do not cite rule numbers from memory: the numbering is unintuitive and models invent it (tested — an agent without this file cited "Rule 3.1: short sentences"; the real Rule 3.1 is about verb forms).

## Two Modes

| Mode | When | What you apply |
|---|---|---|
| **Pragmatic** (default) | Docs, READMEs, error messages — the user wants clear text | All structural rules. Domain words stay ("idempotent", "webhook"). |
| **Strict** | The user names STE, ASD-STE100, or compliance | Structural rules + full vocabulary discipline, and tell the user that full compliance needs the official dictionary (free at asd-ste100.org). |

## Step 1: Classify the Text

| | Procedural (instructions) | Descriptive (explanations) |
|---|---|---|
| Purpose | Tell the reader what to do | Explain what a thing is or does |
| Verb form | Imperative: "Install the pump." | Simple present/past/future |
| Sentence limit | **20 words** (Rule 5.1) | **25 words** (Rule 6.3) |
| Unit rule | One instruction per sentence (5.2) | One topic per paragraph (6.5), max six sentences per paragraph (6.6) |

Do not mix the two in one passage. A "Getting started" section is procedural. An "Architecture" section is descriptive. A note inside a procedure is descriptive (25-word limit, no imperative).

## THE RULE CATALOG

53 rules in 9 sections, paraphrased from ASD-STE100 Issue 9 with software examples. The official wording is in the free standard at asd-ste100.org.

### Section 1 — Words (Rules 1.1-1.14)

| Rule | Instruction |
|---|---|
| 1.1 | Use only approved words, technical nouns, or technical verbs. |
| 1.2 | Use an approved word only as its listed part of speech. |
| 1.3 | Use an approved word only with its approved meaning. |
| 1.4 | Use only the approved forms of verbs and adjectives. |
| 1.5 | You can use domain words as technical nouns ("webhook", "commit", "endpoint"). |
| 1.6 | Use an unapproved word only when it is a technical noun or part of one. |
| 1.7 | Do not use technical nouns as verbs. |
| 1.8 | Use the technical nouns of your project or industry. |
| 1.9 | When you pick a technical noun, pick a short and clear one. |
| 1.10 | No regional, slang, or jargon words as technical nouns. |
| 1.11 | One item, one name. Do not call it "config" here and "settings" there. |
| 1.12 | You can use domain verbs as technical verbs ("deploy", "compile", "merge"). |
| 1.13 | Do not use technical verbs as nouns. |
| 1.14 | Use American English spelling. |

In pragmatic mode, rules 1.5, 1.8, and 1.12 do the heavy lifting: your domain vocabulary is legal. The ones agents break are 1.7, 1.11, and 1.13.

**Before:** You can webhook the event, then do a deploy.
**After:** Send the event to the webhook. Then deploy the service.

### Section 2 — Multi-word nouns (Rules 2.1-2.2)

| Rule | Instruction |
|---|---|
| 2.1 | Write multi-word nouns of three words or fewer. |
| 2.2 | When a technical noun needs more than three words, write it in full once, then give a short form or hyphenate the units. |

Break long noun chains with prepositions (of, on, in, for):

**Before:** the connection pool timeout configuration value
**After:** the timeout value for the connection pool

### Section 3 — Verbs (Rules 3.1-3.7)

| Rule | Instruction |
|---|---|
| 3.1 | Use only the verb forms that the dictionary gives. |
| 3.2 | Use only: infinitive, imperative, simple present, simple past, simple future, past participle as adjective. |
| 3.3 | Use the past participle only as an adjective ("the cached response"). |
| 3.4 | No auxiliary verbs for complex constructions. No present perfect, no "is to be installed". |
| 3.5 | Use an "-ing" form only as a technical noun or inside one ("logging", "the mounting bracket") — never as a verb. |
| 3.6 | Active voice. In descriptive text, passive is legal only when the agent is unknown. |
| 3.7 | Describe an action with a verb, not a noun ("compress the file", not "perform compression of the file"). |

**Approved modals: can, will, must. Banned: should, would, may, might, could (Rule 3.2).**
The standard rejects "could" even for possibility: write "an explosion can occur", never "could occur". For "should": a requirement becomes "must"; a suggestion is stated as fact or deleted. This matters double for agent instructions — models read "should" as optional.

**Before:** The migration has completed and the table is being rebuilt.
**After:** The migration is complete. The database rebuilds the table.

**Before:** The flag can be set in the config file, making restarts unnecessary.
**After:** You can set the flag in the config file. Then a restart is not necessary.

**Before:** The temperature must be adjusted.
**After:** Adjust the temperature.

### Section 4 — Sentences (Rules 4.1-4.5)

| Rule | Instruction |
|---|---|
| 4.1 | Write short and clear sentences. |
| 4.2 | Do not omit words or use contractions to shorten sentences. Keep articles, keep "that". |
| 4.3 | Use a vertical list for complex text. |
| 4.4 | Use connecting words between sentences on related topics ("Then", "As a result"). |
| 4.5 | Put an article (the, a, an) or a demonstrative adjective (this, these) before nouns where applicable. |

Rule 4.2 is the anti-terseness rule. STE is short sentences with complete grammar, not telegraph style:

**Wrong shortening:** Ensure file exists before running.
**STE:** Make sure that the file exists before you run the command.

### Section 5 — Procedural writing (Rules 5.1-5.5)

| Rule | Instruction |
|---|---|
| 5.1 | Maximum 20 words per sentence. Warnings and cautions included. |
| 5.2 | One instruction per sentence, unless two actions happen at the same time. |
| 5.3 | Write instructions in the imperative: "Run the migration." |
| 5.4 | Put a required condition before the command, divided by a comma: "If the build fails, read the log." |
| 5.5 | Notes give information, never instructions. Notes get the 25-word limit. |

**Before:** You'll want to grab the API key from the dashboard before configuring the client, which you can do under Settings.
**After:** Get the API key from the dashboard, under Settings. Then configure the client with this key.

### Section 6 — Descriptive writing (Rules 6.1-6.6)

| Rule | Instruction |
|---|---|
| 6.1 | Give information gradually: one new fact per sentence. |
| 6.2 | Use key words and phrases to give the text a logical structure. |
| 6.3 | Maximum 25 words per sentence. |
| 6.4 | Group related information in paragraphs. |
| 6.5 | One topic per paragraph. |
| 6.6 | Maximum six sentences per paragraph. |

No imperative in descriptive text. Descriptions explain; procedures instruct.

### Section 7 — Safety instructions (Rules 7.1-7.3)

| Rule | Instruction |
|---|---|
| 7.1 | Use a word that shows the risk level ("WARNING" = injury, "CAUTION" = damage). |
| 7.2 | Start with a clear command or condition. |
| 7.3 | Then give the risk or the possible result. |

Never bury the instruction after the explanation. The pattern transfers directly to destructive CLI flags, irreversible migrations, and dangerous API options.

**Before:** Note that data loss may occur in some circumstances if the destructive flag happens to be enabled when running against production.
**After:** CAUTION: Do not use the `--force` flag against production. The flag deletes rows that do not match the source.

### Section 8 — Punctuation and word count (Rules 8.1-8.7)

| Rule | Instruction |
|---|---|
| 8.1 | All standard punctuation is legal except the semicolon. Write two sentences instead. |
| 8.2 | Use hyphens to connect words that act as one unit. |
| 8.3 | Parentheses are legal for references, item numbers, abbreviations, plural forms, explanations, alternatives. |
| 8.4 | In a vertical list, the lead-in colon ends a sentence for word count. |
| 8.5 | Text inside parentheses counts as one word. |
| 8.6 | Count as one word each: numbers, numbers with units, abbreviations, alphanumeric identifiers, quoted text, titles, labels, proper nouns. |
| 8.7 | A hyphenated word counts as one word. |

Rule 8.6 matters for software text: `sqlpipe run --config sqlpipe.yaml` in backticks is quoted text and counts as one word. Long identifiers do not blow your sentence budget.

### Section 9 — Writing practices (Rules 9.1-9.4, GR-1 to GR-8)

| Rule | Instruction |
|---|---|
| 9.1 | When a word-for-word replacement does not work, restructure the sentence. |
| 9.2 | Use each approved word correctly: approved meaning, approved part of speech. |
| 9.3 | Do not build phrasal verbs ("go down" → "decrease", "set up" → "install" or "configure"). |
| 9.4 | Keep one consistent style and terminology through the whole document. |

General recommendations GR-1 to GR-8: keep the conjunction "that", be careful with "with", give pronouns clear referents, prefer "this + noun" over bare "this", avoid false friends, avoid Latin abbreviations, use inclusive language, and use the possessive apostrophe form only when you are sure it is correct (GR-8: if unsure, do not use it — non-native readers find it hard).

GR-6 for software docs: "e.g." → "for example", "i.e." → "that is", and delete "etc." — name the items or write "and more".

## VOCABULARY DISCIPLINE

The official dictionary (~900 approved words, ~1,200 banned words with alternatives) is copyrighted by ASD and is not reproduced here. Its mechanics apply without it: **one word, one meaning, one part of speech.**

Known part-of-speech rulings, useful as patterns:

| Word | Ruling |
|---|---|
| test, check, work | Noun only. "Do a test", not "test the pump". "Check that X" becomes "make sure that X". |
| oil | Technical noun (TN) only. For the verb, the dictionary gives "lubricate": "Lubricate the linkage with oil." |
| help | Verb only. For the noun, the dictionary gives "aid": "with the aid of". |
| fall (noun) | Rejected. Use "decrease" for a reduction in value. Use FALL (verb) only for physical movement downward by gravity: "Make sure that the tools do not fall into the engine." |
| follow | "To come after" only, never "obey". Write "obey the instructions". |
| above, below | Physical positions only. For limits write "more than", "less than". |

### The modal ladder

| You wrote | STE writes |
|---|---|
| should (requirement) | must |
| should (recommendation) | Delete it, or state it as fact: "X is better because Y." |
| may / might / could (possibility) | can |
| may (permission) | can |
| would (hypothetical) | Restructure: "If X occurs, Y occurs." |

### Slop-to-simple substitutions

This table is ours, not the ASD dictionary. It maps the words AI-generated docs overuse to plain replacements. If the word carries no fact, delete it instead of replacing it.

| Slop | Write instead |
|---|---|
| leverage, utilize | use |
| in order to | to |
| prior to | before |
| ensure | make sure that (strict mode; in pragmatic mode, ensure is an allowed pick if it is your one chosen check-verb) |
| it is worth noting that | (delete) |
| it's important to, crucially | (delete — state the fact) |
| simply, just, easily, seamlessly, effortlessly | (delete) |
| robust, powerful, comprehensive, performant | (delete, or give the measurable property) |
| functionality | function, feature |
| enables you to, allows you to | you can |
| is designed to, aims to | (delete — say what it does) |
| facilitate | help, make possible |
| dive into, delve into | read, examine |
| when it comes to | for |
| in the event that | if |
| due to the fact that | because |
| as needed, as necessary | (state the condition) |
| and/or | Pick one, or write "X, or Y, or both" |
| e.g. / i.e. / etc. | for example / that is / (name the items) |
| gracefully handles | (say what it does: "retries three times, then stops") |
| out of the box | by default |
| under the hood | internally |
| blazingly fast, state-of-the-art | fast (give the number) / (delete) |
| streamline | make simpler, make faster |
| plethora, myriad | many |
| addresses the issue, tackles | corrects the fault, removes the error |

### Consistency pass

Collapse synonym rotations to one term each (Rules 1.11, 9.4). The two lists below work differently.

**Technical nouns — not in the dictionary. Pick one and keep it consistent (both modes):**

- config / configuration / settings / options → pick one

**Dictionary rulings — the standard has already chosen. Use the approved word (strict mode); pick one and keep it consistent (pragmatic mode):**

| You wrote | Dictionary status | Use instead |
|---|---|---|
| check (verb) / verify / confirm / ensure | All rejected as verbs | `make sure that` (strict); pick one (pragmatic) |
| validate | Not in dictionary | Use as technical verb (Rule 1.12), or replace with `make sure that` |
| delete / drop (verb) / destroy | All rejected | `erase` (data), `remove` (physical); avoid `drop` and `destroy` |
| remove | Approved verb | Keep it |
| run / execute | Both rejected | `operate` for run, `do` for execute (strict); pick one (pragmatic) |
| invoke / launch | Not in dictionary | Use as technical verbs (Rule 1.12) |
| display (verb) / render / present (verb) | All rejected | `show` (approved verb) |
| issue | Not in dictionary | Use as technical noun, or replace with `problem` (approved) |
| failure | Rejected in general use; approved as TN for performance loss | Use only when it means a performance error: "a failure of the pump" |
| error | Approved noun | Keep it |
| problem | Approved noun | Keep it |

## Untouchables

These are technical names (Rules 1.5, 8.6). Leave them exact, even when they break vocabulary rules:

- Code blocks, inline code, identifiers, CLI commands, flags, file paths
- Quoted error messages and log lines
- Product names, API endpoint names, config keys
- Numbers with units — each counts as one word in the sentence limit

## Beyond Documentation

Same rules, different targets. Full adaptations in `references/use-cases.md`:

- **Error messages**: state what happened (simple past), the cause if known, then the fix as an imperative. No "Oops", no "Please ensure", no apology filler.
- **Runbooks**: STE's home turf. Imperative steps, conditions first, warnings before the step.
- **Incident reports**: simple past only. "We have identified an issue that may have impacted" becomes "Between 14:02 and 14:31 UTC, 12% of requests failed."
- **Release notes**: breaking changes follow the warning pattern — command first, risk second.
- **Agent instructions (prompts, AGENTS.md)**: a system prompt is a procedure for a reader that cannot ask questions. One instruction per sentence, no "should", condition first.
- **Translation prep**: STE's original job. One meaning per word plus complete grammar removes most translation ambiguity.

## Self-Check Before You Deliver

This step is not optional. Run these four checks on your draft:

1. Count words in your three longest sentences. Over the 20/25 limit → split them.
2. Search your draft for: `'ll`, `'re`, `'s` (contraction), `has been`, `have been`, `should`, `-ing` verbs after a comma, semicolons.
3. Search for every `if` and `when`. Each one stands at the START of its sentence, before the command. "Increase the timeout if the network is slow" → "If the network is slow, increase the timeout."
4. Search for the verbs you did NOT pick in Your Task step 3 (the check/verify/confirm set). Replace every hit with your chosen verb.

Fix what you find, then deliver. For a full audit, run `references/checklist.md`.

## Full Example

**Before (real unedited AI output):**

> **Connection timeouts.** If sqlpipe hangs or fails with `dial tcp: i/o timeout`, check that the host running sqlpipe can reach the Postgres port (usually 5432) — this is often a security group or firewall rule blocking the connection. If you're connecting to a managed database (RDS, Cloud SQL, etc.), confirm the instance allows connections from sqlpipe's IP. You can also try increasing `source.connect_timeout_seconds` in your config, since a slow network path can trip the default timeout even when the connection eventually succeeds.

**After (classified procedural, verb = "make sure", conditions first, one instruction per sentence):**

> **Connection timeouts.** sqlpipe stops with `dial tcp: i/o timeout` when it cannot reach the Postgres port (5432 by default).
>
> 1. Make sure that the host that runs sqlpipe can reach the Postgres port. A firewall or security group usually blocks it.
> 2. If the database is managed (RDS, Cloud SQL), make sure that the instance accepts connections from the IP of sqlpipe.
> 3. If the network is slow, increase `source.connect_timeout_seconds` in the configuration.

What changed: 40-word sentences split under 20; "you're" expanded; "check/confirm" collapsed to "make sure that"; every condition moved before its command; "etc." removed; code and error strings untouched.

## Limits

STE is for technical facts and instructions. Do not apply it to marketing copy, blog voice, or brand writing — it deletes persuasion by design. When a user asks for STE on marketing text, say so and offer it for the docs instead.

This skill is an unofficial aid. It is not affiliated with or endorsed by ASD or STEMG, and no tool can guarantee STE compliance. ASD-STE100 is a registered trademark of ASD. The official standard is a free download at asd-ste100.org.

## References

- `references/checklist.md` — full verification pass with searchable patterns, for check mode and final audits
- `references/use-cases.md` — long-form adaptations: error messages, runbooks, incident reports, commits, UI copy, i18n
