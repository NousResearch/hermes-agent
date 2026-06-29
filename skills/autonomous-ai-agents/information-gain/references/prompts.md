# Stage prompts & JSON contracts

The authoritative prompt text lives in `scripts/pipeline.py` (the `*_prompt()` builders) so the
code and docs can't drift. Run `python3 scripts/infogain.py "<problem>" --dry-run` to print the
exact prompts with a problem filled in. This file documents the **JSON contract** each stage
expects back and the tuning rationale.

All stages call Ollama with `temperature=0` (deterministic) and end with "Respond ONLY with the
JSON object." Parsing is tolerant (`pipeline.extract_json` strips ```json fences and surrounding
prose); a single retry nudges the model toward strict JSON.

### Stage 0 — frame_and_plan  (model: `plan_model`, default `glm`)
Returns the framing **and** the EVSI baseline plan:
```json
{"goal": "…", "decision": "…", "success_criteria": ["…"], "baseline_plan": "…"}
```
`baseline_plan` = the plan assuming the most likely interpretation. Everything downstream measures
*change from this baseline*, so it must be a real, committed plan — not a list of caveats.

### Stage 1 — generate_questions  (model: `question_gen_model`, default `glm`)
```json
{"questions": [{"question": "…", "type": "scope|constraint|audience|data|integration|risk|success-metric|resource|assumption|other", "why": "…", "target": "short latent label"}]}
```
`target` is critical for diversity: two questions resolving the same hidden variable **must** share
a `target` so the deduper collapses them to one representative. Later rounds receive an `avoid` list
of already-considered questions.

### Stage 2 — project_answers  (model: `answer_model`, default `fast`, parallel per question)
```json
{"derivable_prob": 0.0, "answers": [{"answer": "…", "prob": 0.0}]}
```
- `prob` need not sum to 1 (normalized in `voi`).
- `derivable_prob` discounts uncertainty resolvable from the prompt alone → feeds `U`.
- Uses the cheap/fast local model because this fans out over every candidate question.

### Stage 3 — judge_plan_change  (model: `value_judge_model`, default `deepseek`, parallel)
Given the baseline plan and the answers, returns one entry per answer **in order**:
```json
{"answers": [{"delta_plan": 0.0, "stakes": 0.0}]}
```
- `delta_plan`: how much the recommended plan changes if this answer is true (0 = identical).
- `stakes`: cost of having proceeded on the baseline if this answer is actually true.
- Uses the strongest model because this judgment is what the value score hinges on.

### How the scores combine (in `voi.py`)
```
U      = normalized_entropy(prob) · (1 − derivable_prob)
EVSI   = Σ_a normalize(prob)[a] · delta_plan[a] · stakes[a]
value  = √(U · EVSI)            # discard if U≈0 or EVSI≈0
```
See `methodology.md` for the derivation and citations.

### Tuning notes
- If answers look generic, raise `answers_per_question` (lit. uses 4–8) or use a stronger
  `answer_model`.
- If too many questions collapse as REDUNDANT, the model is reusing `target` labels too coarsely —
  acceptable (it's conservative); raise `hard_cap` only if you want more.
- If the bucket never fills, the problem is likely well-specified — lower `discard_threshold` only
  if you genuinely want lower-value questions surfaced.
