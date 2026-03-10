---
name: adversarial-critique
description: Use the council to stress-test your work before publishing or deploying. Identifies blind spots through adversarial evaluation and safety gating.
version: 1.0.0
author: Hermes Ouroboros
license: MIT
metadata:
  hermes:
    tags: [Council, Critique, Safety, Quality, Review]
    related_skills: [multi-perspective-analysis, bayesian-synthesis]
---

# Adversarial Critique

Use `council_evaluate` and `council_gate` to stress-test your work before it goes live. The council's adversarial structure surfaces blind spots that self-review misses.

## When to Use

- **Before publishing**: Evaluate research, articles, or analysis
- **Before deploying**: Gate code deployments with safety review
- **Before sending**: Review messages or communications for tone and accuracy
- **During development**: Get structured feedback on design decisions
- **After completion**: Retrospective quality check on finished work

## Tools

### council_evaluate -- Quality Assessment
```
council_evaluate(
    content="[your work product]",
    question="[what task it was supposed to accomplish]",
    criteria=["accuracy", "depth", "falsifiability", "evidence"]
)
```

Returns per-persona feedback and an overall confidence score.

### council_gate -- Safety Check
```
council_gate(
    action="Deploy v2.0 to production with new authentication system",
    risk_level="high",
    context="New auth uses JWT tokens instead of session cookies. All tests pass."
)
```

Returns `allowed` (bool) + reasoning. Uses abbreviated council (Skeptic + Oracle + Arbiter) for speed.

**Risk level thresholds**:
- `low`: Allowed if confidence >= 30%
- `medium`: Allowed if confidence >= 50%
- `high`: Allowed if confidence >= 70%

## Workflow: Evaluate Then Improve

1. **Do your work**: Research, write, code, analyze
2. **Evaluate with council**: `council_evaluate(content=your_output)`
3. **Read the Skeptic's critique**: What did you miss? What assumptions did you make?
4. **Read the Contrarian's reframe**: Is there a better way to think about this?
5. **Revise**: Address the council's concerns
6. **Re-evaluate**: Run council_evaluate again on the revised version
7. **Track improvement**: Compare confidence scores between versions

## Skill Evolution Pattern

The council can help improve your other skills:

1. Complete a task using a skill (e.g., coding, research)
2. Run `council_evaluate` on the output
3. Read the Skeptic's critique carefully
4. If the critique reveals a systematic gap, update the skill:
   ```
   skill_manage(
       action="patch",
       name="coding-best-practices",
       content="## Error Handling\nAlways handle timeout errors in API calls..."
   )
   ```
5. Future uses of that skill now include the lesson learned

This creates a self-improving loop: council critique -> skill update -> better output -> higher council scores.

## Example: Research Critique

```
# Step 1: Research a topic
research_output = web_search("quantum computing threat to RSA encryption timeline")

# Step 2: Write analysis
analysis = "Based on current progress..."

# Step 3: Council critique
council_evaluate(
    content=analysis,
    question="Is quantum computing a near-term threat to RSA?",
    criteria=["accuracy", "evidence", "nuance", "completeness"]
)

# Step 4: Read feedback, revise, repeat
```

## Example: Deployment Gate

```
# Before deploying
council_gate(
    action="Push database migration that adds NOT NULL constraint to users.email",
    risk_level="high",
    context="Production has 50k rows. 12 rows have NULL email. Migration includes backfill."
)

# If allowed=false, review skeptic_concerns before proceeding
```

## Safety Philosophy

> Before irreversible actions (deploying code, sending external messages, deleting data),
> consider using council_gate to get a safety review.

The cost of a 30-second council review is far less than the cost of an irreversible mistake.
The gate is not a replacement for testing -- it's an additional perspective check.
