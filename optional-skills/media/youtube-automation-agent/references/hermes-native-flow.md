# Hermes-native flow

This document explains how the skill captures the essence of the upstream YouTube automation repo as a native Hermes workflow.

## Stage mapping

### 1. Strategy
Mirrors:
- `ContentStrategyAgent.generateContentStrategy()`

Hermes output should include:
- topic candidates
- selected topic
- angle
- target audience
- content type
- keywords
- best publish timing

### 2. Script
Mirrors:
- `ScriptWriterAgent.generateScript()`

Hermes output should include:
- title
- hook
- intro
- main sections
- CTA
- narration-ready full script
- duration estimate

### 3. Thumbnail
Mirrors:
- `ThumbnailDesignerAgent.generateThumbnail()`

Hermes output should include:
- primary text
- secondary text
- concept
- composition
- colors
- alternate thumbnail ideas

### 4. SEO
Mirrors:
- `SEOOptimizerAgent.optimize()`

Hermes output should include:
- optimized title
- description
- tags
- hashtags
- chapters
- category suggestion

### 5. Production
Mirrors:
- `ProductionManagementAgent.processContent()`

Hermes output should include:
- narration plan
- visual asset list
- caption plan
- assembly checklist
- automation-vs-manual split

### 6. Publishing
Mirrors:
- `PublishingSchedulingAgent.scheduleContent()` and `publishContent()`

Hermes output should include:
- publish time
- privacy status
- upload checklist
- metadata map
- QA checklist

### 7. Analytics
Mirrors:
- `AnalyticsOptimizationAgent.analyzeVideoPerformance()`

Hermes output should include:
- KPI plan
- review schedule
- thresholds
- title/thumbnail iteration triggers
- next-video feedback loop

## Why this matters

The upstream repo spreads these stages across multiple Node agents. This skill converts the same operational sequence into a persistent Hermes-run workflow so the user can progress stage by stage and keep context across sessions.
