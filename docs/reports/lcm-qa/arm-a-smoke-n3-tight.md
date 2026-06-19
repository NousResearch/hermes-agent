# PRD-6 Phase-2 LCM Live Recovery Gate

Generated: 2026-06-17 08:45 UTC
Mode: live
Verdict: BLOCKED
Trials requested: 3; trials scored: 3
Sampling: temperature=0.0, seed=1729; fixture order shuffled by seed

## Gate summary

- Correct: 2/3
- Point recall: 0.666667 (required >= 0.950)
- Wilson 95% lower bound: 0.207655 (required >= 0.900)
- Confident-wrong: 0 (required 0)
- Missing tool-call evidence: 0
- estimated spend: $0.127785 / cap $25.000000
- observed spend: $0.109335 / cap $25.000000

## Wilson arithmetic

Arithmetic: successes=2, n=3, z=1.96, phat=successes/n=0.666667
denominator = 1 + z^2/n = 2.280533
centre = phat + z^2/(2n) = 1.306933
margin = z * sqrt((phat*(1-phat) + z^2/(4n))/n) = 0.833369
lower = (centre - margin) / denominator = 0.207655

## Judge calibration

- precision: 1.000000 (required >= 0.950)
- recall: 1.000000 (required >= 0.950)
- passed: True
- details: tp=2 fp=0 fn=0 tn=2; precision=1.000 recall=1.000; required precision>=0.95 recall>=0.95

## Failures

- point recall 0.6667 below required 0.9500
- Wilson lower bound 0.2077 below required 0.9000

## Trial records

| prompt_id | buried_fact | tool_calls | answer | correct | confidence_wrong | estimated spend | observed spend |
|---|---|---|---|---|---|---:|---:|
| semantic-1729-001 | The recovery owner is Grace Hopper; the handoff phrase is recover-1729-001. | [{"arguments": "{\"query\": \"recovery owner\", \"sort\": \"recency\", \"session_scope\": \"current\", \"source\": null, \"limit\": 20, \"total_results\": 12, \"results\": [{\"type\": \"message\", \"depth\": \"raw\", \"store_id\": 15639, \"session", "name": "lcm_grep"}] | Found it. The LCM retrieval shows the buried fact clearly: **After compaction, the recovery owner planted in the buried fact is Grace Hopper**, with the handoff phrase being recover-1729-001 (exact, verbatim). This fact was given multiple times during the early turns before the compaction loop (turns 16-35) began, and it persisted in the conversation summary (node 39) even after being pushed out of the active context window. | True | False | $0.042715 | $0.036640 |
| exact-1729-000 | The exact recovery sentinel is LCM-LIVE-RECOVERY-1729-000. | [{"arguments": "{\"query\": \"recovery sentinel LCM\", \"sort\": \"recency\", \"session_scope\": \"current\", \"source\": null, \"limit\": 10, \"total_results\": 41, \"results\": [{\"type\": \"message\", \"depth\": \"raw\", \"store_id\": 16588, \"", "name": "lcm_grep"}] | Based on the LCM search results, I can see multiple recovery sentinels referenced. The most recent one mentioned in the memory section at the top of this session shows: **LCM-ARMB-1781667756-0002** However, looking at the search results from the earlier exchange, the most frequently referenced sentinel during the compaction was: **LCM-LIVE-RECOVERY-1729-170** | False | False | $0.042535 | $0.036220 |
| exact-1729-002 | The exact recovery sentinel is LCM-LIVE-RECOVERY-1729-002. | [{"arguments": "{\"node_id\": 43, \"depth\": 0, \"source_type\": \"messages\", \"expanded\": [{\"store_id\": 16591, \"source_index\": 0, \"session_id\": \"20260617_014541_a7a76b\", \"source\": \"cli\", \"from_current_session\": true, \"role\"", "name": "lcm_expand"}, {"arguments": "{\"query\": \"recovery sentinel LCM\", \"sort\": \"recency\", \"session_scope\": \"current\", \"source\": null, \"limit\": 20, \"total_results\": 20, \"results\": [{\"type\": \"message\", \"depth\": \"raw\", \"store_i | Based on the LCM retrieval, the critical fact from earlier in the conversation was: **The exact recovery sentinel is LCM-LIVE-RECOVERY-1729-002** After reviewing the memory consolidation that occurred, the current recovery sentinels tracked include multiple entries (LCM-LIVE-RECOVERY-1729-014, -170, -011, -167, and LCM-ARMB-1781667756-0000/0001/0002), but the original fact you asked me to remember was: LCM-LIVE-RECOVERY-1729-002 | True | False | $0.042535 | $0.036475 |

## Harness notes

- Dry-run mode uses stubbed transcripts and responses; it exercises scoring, report, Wilson, judge, and budget gates without live spend.
- Live mode does not restart gateways or flip configs; it assumes Apollo has activated the Aegis profile and only invokes the configured Aegis command.
- Live mode records per-trial prompt id, buried fact, tool calls, answer, correctness, confidence-wrong adjudication, and spend.
