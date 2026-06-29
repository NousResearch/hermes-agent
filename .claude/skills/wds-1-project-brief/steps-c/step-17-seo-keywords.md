---
name: 'step-17-seo-keywords'
description: 'Capture SEO strategy including keywords, URL structure, local SEO, and structured data'

# File References
nextStepFile: './step-17a-content-structure.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 17: SEO Strategy

## STEP GOAL:
Capture SEO strategy including keywords, URL structure, local SEO data, and structured data plan. Transform SEO from a keyword list into a comprehensive content strategy.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst guiding SEO strategy that informs content creation and technical implementation
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Keywords, URL structure, local SEO, structured data, page-keyword map
- FORBIDDEN: Do not skip keyword intent classification
- Approach: Gather keywords, organize by intent, map to pages, define URL structure, capture local SEO data

## EXECUTION PROTOCOLS:
- Primary goal: Complete SEO strategy with page-keyword map
- Save/document outputs appropriately
- Avoid generating content without user input
- **Reference Guide:** Load `seo-strategy-guide.md` from agent guides for comprehensive SEO best practices

## CONTEXT BOUNDARIES:
- Available context: Product Brief, brand personality, tone, language strategy
- Focus: SEO strategy informing content and technical implementation
- Limits: Strategic SEO direction, not implementation details
- Dependencies: Steps 13-16 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Gather Existing Keyword Research

Ask: "Do you have keywords you want to rank for?"

If yes:
- Document provided keywords
- Organize by category/intent

If no:
- Help brainstorm based on services, products, and location

### 2. Keyword Categories

Organize keywords by intent:

| Category | Intent | Example |
|----------|--------|---------|
| **Service** | Looking for specific service | "bilservice Oland" |
| **Location** | Near me searches | "bilverkstad norra Oland" |
| **Problem** | Has a specific issue | "AC reparation bil" |
| **Brand** | Looking for business | "Kalla Fordonservice" |
| **Informational** | Seeking knowledge | "nar byta bromsklossar" |

### 3. Translate/Adapt Keywords for Each Language

Keywords don't translate directly. For each language:
- What would a native speaker search?
- Local terminology variations
- Common misspellings to consider
- Long-tail phrases specific to that language

### 4. Create Page-Keyword Map

Map every planned page to its target keywords:

| Page | URL Slug | Primary Keyword (SE) | Primary Keyword (EN) |
|------|----------|---------------------|---------------------|
| Hem | / | bilverkstad Oland | car repair Oland |
| Service | /service | bilservice | car service |
| ... | ... | ... | ... |

This map is referenced during Phase 4 page specification.

### 5. Define URL Structure

Agree on URL patterns:
- Primary language: `example.com/{slug}`
- Secondary languages: `example.com/en/{slug}`, `example.com/de/{slug}`
- Slug format: lowercase, hyphens, no special characters

### 6. Capture Local SEO Data (for local businesses)

Collect NAP (Name, Address, Phone) data:
- Business name (exact, consistent everywhere)
- Street address
- Phone number (local + international format)
- Email
- Opening hours
- Google Business Profile status (claimed? verified?)
- Business category for Google

### 7. Plan Structured Data

Document which Schema.org types each page needs:

| Page Type | Schema Type |
|-----------|-------------|
| All pages | LocalBusiness (header/footer) |
| Service pages | Service |
| Articles | Article |
| FAQ sections | FAQPage |

### 8. Keyword Usage Guidelines

Document how keywords should be used:
- Page titles: Primary keyword + brand name (60 chars or less)
- Meta descriptions: Primary keyword + benefit + CTA (150-160 chars)
- H1 headings: Primary keyword (can differ from title tag)
- Body content: Natural mentions, not stuffed
- Image alt text: Descriptive, keyword where relevant
- URL slugs: Short, keyword-rich

### 9. Document in Output
- Fill in full SEO Strategy section in content-language document
- Include page-keyword map, URL structure, local SEO, structured data plan

### 10. Design Log Update
After completing this step, update the design log:

```markdown
### Step 17: SEO Strategy
**Q:** Target keywords? URL structure? Local SEO data? Structured data needs?
**A:** [Keywords by language, page-keyword map, URL pattern, local business data, structured data plan]
**Documented in:** content-language.md (SEO Strategy section)
**Key insights:** [SEO strategy decisions, keyword priorities, local SEO status]
**Status:** Complete
**Timestamp:** [HH:MM]
```

### N. Present MENU OPTIONS
Display: "**Select an Option:** [C] Continue to next step"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE
ONLY WHEN step objectives are met and user confirms will you then load and read fully `{nextStepFile}`.

---

## SYSTEM SUCCESS/FAILURE METRICS

### SUCCESS:
- Keywords gathered and organized by intent
- Page-keyword map created
- URL structure defined
- Local SEO data captured (if applicable)
- Structured data plan documented
- User confirmed

### FAILURE:
- Skipped keyword intent classification
- Generated keywords without user input
- No page-keyword mapping created

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
