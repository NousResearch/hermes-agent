# Feature Impact Analysis: {{project_name}}

## Scoring

**Primary Persona (⭐):** High = 5 pts | Medium = 3 pts | Low = 1 pt  
**Other Personas:** High = 3 pts | Medium = 1 pt | Low = 0 pts

**Max Possible Score:** {{max_score}} (with {{persona_count}} personas)  
**Must Have Threshold:** {{must_have_threshold}}+ or Primary High (5)

---

## Prioritized Features

| Rank | Feature | Score | Decision |
| ---- | ------- | ----- | -------- |

{{#each sorted_features}}
| {{this.rank}} | {{this.name}} | {{this.score}} | {{this.decision}} |
{{/each}}

---

## Decisions

**Must Have MVP (Primary High OR Top Tier Score):**
{{#each must_have}}

- {{this.name}} ({{this.score}})
  {{/each}}

**Consider for MVP:**
{{#each consider}}

- {{this.name}} ({{this.score}})
  {{/each}}

**Defer (Nice-to-Have or Low Strategic Value):**
{{#each defer}}

- {{this.name}} ({{this.score}})
  {{/each}}

---

_Generated with Whiteport Design Studio framework_  
_Strategic input for Phase 4: UX Design and Phase 6: PRD/Development_
