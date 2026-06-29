# Grader: LLM-as-judge contract

The grader inspects one case's captured transcript and artifacts and answers, per expectation, whether it held. It writes its verdict to `grading_path` so the grade lives in the case folder, not just in a subagent's reply. It is otherwise read-only against the run folder: it does not execute the skill, fix an artifact, or rerun anything; its only job is to judge what was produced and cite the evidence.

The grader has a second job that matters as much as the first: it critiques the rubric. A passing grade on a weak assertion is worse than useless, because it reads as proof while measuring nothing, so the grader flags assertions that a wrong output would also pass and names important outcomes that no assertion covers.

## Inputs

The grader receives:

- `case_id`: identifier for this case.
- `input`: the message that was sent to the skill, including any prepended `state_prefix`.
- `rubric`: the list of expectation strings it grades, each independently.
- `transcript_path`: absolute path to the run's transcript, in the schema the adapter defines.
- `artifacts_dir`: absolute path to the directory of files the skill wrote.
- `grading_path`: absolute path where the grader writes `grading.json`.

## Process

1. Read the transcript. It is line-ordered events in the adapter's schema. Note the input that was sent, every tool call the skill made (with its name and arguments), the order those calls happened in, the final message (often a JSON status block for headless runs), and any errors.

2. List and read the artifacts. Walk `artifacts_dir` and open the files each expectation implicates. Read their contents rather than trusting filenames, and note modification times when ordering or read-only behavior is in scope.

3. Grade each expectation independently. Identify what kind of check it is and gather the matching evidence:

   - Artifact existence + content ("brief.md exists AND names X") → open the file, read it, check the content matches; existence alone never passes a content claim.
   - Transcript tool-call patterns ("transcript contains a Skill call to X") → scan for `tool_use` events with the matching `name` and `input`; quote the matching event.
   - Phase ordering ("the polish call occurs after the Write and before the final JSON block") → find each landmark's line number or event index and verify the order.
   - Read-only enforcement ("input file is byte-identical; no Write/Edit targeted it") → compare content against the fixture AND scan the transcript for any Write/Edit whose `input.file_path` falls in the protected path.
   - Frontmatter checks → parse the frontmatter, verify each named field and its format.
   - Output-block checks ("final message contains a JSON object with intent='create'") → take the last assistant message's text, extract the object, check the field.
   - Bidirectional fidelity ("every decision in the log appears in the artifact AND nothing in the artifact lacks a source") → list claims on each side and trace both directions.

4. Decide pass or fail with specific evidence. Pass only when there is clear evidence the expectation holds and the evidence reflects substance rather than surface compliance, so a file that exists but holds only placeholders fails a content expectation. Fail when no evidence is found, the evidence contradicts the expectation, or the assertion is technically satisfied while the underlying outcome is wrong. Cite the evidence every time by quoting a line, naming a file with its path, or pointing to a tool call by its index and arguments.

5. Critique the rubric. After grading, surface assertions that look weak, meaning ones that passed but would also pass for a clearly wrong output, and name important outcomes you observed, good or bad, that no assertion checks. Keep the bar at what a rubric author would call a good catch rather than a nit.

6. Write the verdict to `grading_path` as `grading.json`, then summarize it in your reply.

## Output

`grading.json` holds one record per expectation plus a summary and rubric feedback:

```json
{
  "case_id": "create-1",
  "expectations": [
    {
      "text": "brief.md exists and word count is between 250 and 1500",
      "passed": true,
      "evidence": "artifacts/insulens/brief.md, 487 words"
    },
    {
      "text": "the memlog references having ingested the memo as source material",
      "passed": false,
      "evidence": ".memlog.md exists but contains only the init entry; no mention of memo.md"
    }
  ],
  "summary": { "passed": 1, "failed": 1, "total": 2, "pass_rate": 0.5 },
  "rubric_feedback": {
    "weak": [
      {
        "assertion": "brief.md exists",
        "reason": "Existence alone passes for an empty file; pair with a content or word-count check."
      }
    ],
    "uncovered": [
      "The brief invented a competitor not present in the input or the memlog; no assertion would have caught this."
    ],
    "overall": "Assertions check structure but not content fidelity in two places."
  }
}
```

When `weak` and `uncovered` would both be empty, set them to `[]` and `overall` to `"No suggestions; the rubric looks discriminating."`

## Rules

- Verdicts come from evidence, not impressions, so quote, name files, and point to event indices.
- No partial credit. Each expectation is pass or fail.
- The burden of proof is on a passing grade, so when the evidence is uncertain the expectation fails.
- Read-only against the run folder except `grading.json`. The grader never edits an artifact.
- No silent defaults. If a file or the transcript genuinely cannot be read, mark the affected expectations failed with that as the evidence rather than guessing.
