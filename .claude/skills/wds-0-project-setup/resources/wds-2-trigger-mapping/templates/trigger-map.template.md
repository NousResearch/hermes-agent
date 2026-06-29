# Trigger Map Poster: {{project_name}}

> Visual overview connecting business goals to user psychology

**Created:** {{date}}
**Author:** {{user_name}}
**Methodology:** Based on Effect Mapping (Balic & Domingues), adapted for WDS framework

---

## Strategic Documents

This is the visual overview. For detailed documentation, see:

- **01-Business-Goals.md** - Full vision statements and SMART objectives
- **02-Target-Groups.md** - All personas with complete driving forces
- **03-Feature-Impact-Analysis.md** - Prioritized features with impact scores
- **04-08-\*.md** - Individual persona detail files

---

## Vision

{{vision_statement}}

---

## Business Objectives

{{#each objectives}}

### Objective {{@index + 1}}: {{this.statement}}

- **Metric:** {{this.metric}}
- **Target:** {{this.target}}
- **Timeline:** {{this.timeline}}
  {{/each}}

---

## Target Groups (Prioritized)

{{#each prioritized_groups}}

### {{@index + 1}}. {{this.name}}

**Priority Reasoning:** {{this.reasoning}}

> {{this.persona_summary}}

**Key Positive Drivers:**
{{#each this.positive_drivers}}

- {{this}}
  {{/each}}

**Key Negative Drivers:**
{{#each this.negative_drivers}}

- {{this}}
  {{/each}}

{{/each}}

---

## Trigger Map Visualization

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontFamily':'Inter, system-ui, sans-serif', 'fontSize':'14px'}}}%%
flowchart LR
    %% Business Goals (Left)
    {{#each business_goals}}
    BG{{@index}}["<br/>{{this.emoji}} {{this.title}}<br/><br/>{{#each this.points}}{{this}}<br/>{{/each}}<br/>"]
    {{/each}}
    
    %% Central Platform
    PLATFORM["<br/>{{platform_emoji}} {{platform_name}}<br/><br/>{{platform_tagline}}<br/><br/>{{platform_transformation}}<br/><br/>"]
    
    %% Target Groups
    {{#each target_groups}}
    TG{{@index}}["<br/>{{this.emoji}} {{this.name}}<br/>{{this.priority}}<br/><br/>{{#each this.profile}}{{this}}<br/>{{/each}}<br/>"]
    {{/each}}
    
    %% Driving Forces
    {{#each target_groups}}
    DF{{@index}}["<br/>{{this.emoji}} {{this.name}}'S DRIVERS<br/><br/>WANTS<br/>{{#each this.positive_drivers}}✅ {{this}}<br/>{{/each}}<br/>FEARS<br/>{{#each this.negative_drivers}}❌ {{this}}<br/>{{/each}}<br/>"]
    {{/each}}
    
    %% Connections
    {{#each business_goals}}
    BG{{@index}} --> PLATFORM
    {{/each}}
    {{#each target_groups}}
    PLATFORM --> TG{{@index}}
    TG{{@index}} --> DF{{@index}}
    {{/each}}

    %% Light Gray Styling with Dark Text
    classDef businessGoal fill:#f3f4f6,color:#1f2937,stroke:#d1d5db,stroke-width:2px
    classDef platform fill:#e5e7eb,color:#111827,stroke:#9ca3af,stroke-width:3px
    classDef targetGroup fill:#f9fafb,color:#1f2937,stroke:#d1d5db,stroke-width:2px
    classDef drivingForces fill:#f3f4f6,color:#1f2937,stroke:#d1d5db,stroke-width:2px
    
    {{#each business_goals}}
    class BG{{@index}} businessGoal
    {{/each}}
    class PLATFORM platform
    {{#each target_groups}}
    class TG{{@index}} targetGroup
    class DF{{@index}} drivingForces
    {{/each}}
```

---

## Design Focus Statement

{{focus_statement}}

**Primary Design Target:** {{top_group.name}}

**Must Address:**
{{#each must_address_drivers}}

- {{this}}
  {{/each}}

**Should Address:**
{{#each should_address_drivers}}

- {{this}}
  {{/each}}

---

## Cross-Group Patterns

### Shared Drivers

{{shared_drivers}}

### Unique Drivers

{{unique_drivers}}

{{#if conflicts}}

### Potential Tensions

{{conflicts}}
{{/if}}

---

## Next Steps

This Trigger Map Poster provides a quick reference. For detailed work:

- [ ] **Review detailed docs** - See 01-Business-Goals.md, 02-Target-Groups.md, 03-Feature-Impact-Analysis.md
- [ ] **Use for Feature Prioritization** - Reference feature impact scores
- [ ] **Guide UX Design** - Ensure designs address priority drivers
- [ ] **Validate with Users** - Test assumptions with real target group members
- [ ] **Update as Learnings Emerge** - This is a living document

---

_Generated with Whiteport Design Studio framework_
_Trigger Mapping methodology credits: Effect Mapping by Mijo Balic & Ingrid Domingues (inUse), adapted with negative driving forces_
