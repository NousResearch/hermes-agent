# Verification checklist

Run this pass on every draft before you deliver it. The checks are ordered from mechanical to judgment.

## Mechanical checks (searchable)

Search the draft for each pattern. Every hit outside code blocks and quoted text is a violation.

| Search for | Violation | Fix |
|---|---|---|
| `'ll`, `'re`, `'ve`, `n't`, `it's` | Contraction (Rule 4.2) | Expand it. |
| `has been`, `have been`, `had been` | Present/past perfect (Rule 3.4) | Simple past or simple present. |
| `has` / `have` + past participle | Present perfect (Rule 3.4) | Simple past. |
| `should`, `would`, `may`, `might`, `could` | Unapproved modal (Rule 3.2) | See the modal ladder in SKILL.md. |
| `is being`, `are being`, `was being` | Progressive passive (Rules 3.4, 3.5) | Active, simple tense. |
| `, making`, `, allowing`, `, enabling`, `, ensuring` | "-ing" clause as verb (Rule 3.5) | New sentence with a real subject. |
| `;` | Semicolon (Rule 8.1) | Two sentences. |
| `e.g.`, `i.e.`, `etc.` | Latin abbreviation (GR-6) | "for example", "that is", name the items. |
| `simply`, `easily`, `seamlessly`, `robust` | Filler (no fact) | Delete. |
| ` if `, ` when ` (mid-sentence) | Trailing condition (Rule 5.4) | Move the condition to the start of the sentence, add a comma. |

## Countable checks

1. **Sentence length.** Count words in each sentence. Procedural limit: 20. Descriptive limit: 25. Notes: 25.
   Backticked commands, numbers with units, and identifiers count as one word each (Rule 8.6).
2. **Paragraph size.** Maximum six sentences per paragraph (Rule 6.6).
3. **Multi-word nouns.** Any noun chain over three words → break it with prepositions (Rule 2.1).
4. **Instructions per sentence.** One, unless the actions are simultaneous (Rule 5.2).

## Judgment checks

5. **Classification.** Is each passage cleanly procedural or descriptive? Procedures in imperative, descriptions never in imperative.
6. **Voice.** Any passive sentence: is the agent truly unknown, and is the passage descriptive? Otherwise make it active (Rule 3.6).
7. **Condition placement.** Every "if/when" stands before its command, with a comma (Rule 5.4).
8. **Synonym rotation.** One term per concept across the whole document (Rules 1.11, 9.4). Scan for check/verify/confirm, config/settings, run/execute.
9. **Warnings.** Command or condition first, risk second (Rules 7.2, 7.3).
10. **Completeness.** Articles present, "that" present after "make sure", no telegraph style (Rule 4.2).
11. **Untouchables intact.** Code, identifiers, quoted errors, and proper nouns are unchanged.

## When reporting violations (check mode)

For each violation give: the rule number, the offending text, and a compliant rewrite. Cite only rule numbers that appear in SKILL.md.
End the report with this statement when the user asked for STE compliance: "No tool can guarantee ASD-STE100 compliance. Final approval rests with the writer. The official standard is a free download at asd-ste100.org."
