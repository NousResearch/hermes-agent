---
name: 'step-31-multilingual'
description: 'Document multilingual requirements and technical SEO needs'

# File References
nextStepFile: './step-32-create-platform-document.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 31: Multilingual & SEO

## STEP GOAL:
Document language requirements and technical SEO needs for the platform.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst documenting multilingual setup and technical SEO requirements
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Language needs, URL structure, translation workflow, technical SEO, performance targets
- FORBIDDEN: Do not skip structured data planning
- Approach: Determine language needs, recommend URL structure, plan translation workflow, document SEO requirements
- **Reference Guide:** Load `seo-strategy-guide.md` from agent guides for comprehensive SEO best practices

## EXECUTION PROTOCOLS:
- Primary goal: Multilingual requirements and SEO technical specs documented
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Product Brief, language strategy (from content steps), technology stack
- Focus: Technical implementation of multilingual and SEO
- Limits: Platform-level requirements, not content-level SEO
- Dependencies: Step 30 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Determine Language Needs

Ask: "What languages does the site need to support?"
- Single language - simpler setup
- Multiple languages - requires plugin and strategy

### 2. If Multilingual:

**Recommend URL structure:**
```
example.com/           -> Primary language (Swedish)
example.com/en/        -> English
example.com/de/        -> German
```

**Plugin recommendation:**
- **Polylang** - Free tier works, good SEO support
- **WPML** - More features, paid
- **TranslatePress** - Visual translation

**Translation workflow:**
- Who translates? (client, translator, agency)
- Full translation or priority pages?
- Ongoing updates process

**Technical requirements:**
- hreflang tags (automatic with good plugins)
- Language switcher placement
- Default language handling

### 3. SEO Technical Requirements

Document needs:
- **Meta tags** - Title, description per page (all languages)
- **Structured data** - Schema.org markup per page type:
  - `LocalBusiness` / `AutoRepair` - Business identity (all pages)
  - `Service` - Individual service pages
  - `Article` - Blog/news articles
  - `FAQPage` - FAQ sections
  - `BreadcrumbList` - Navigation breadcrumbs
- **Sitemap** - XML sitemap generation (includes all language versions)
- **Robots.txt** - Crawl directives
- **Page speed** - Core Web Vitals targets (LCP < 2.5s, FID < 100ms, CLS < 0.1)
- **Mobile-first** - Responsive, mobile-optimized
- **Canonical URLs** - Self-referencing canonical on every page
- **hreflang tags** - Language alternates declared on every page
- **404 handling** - Custom 404 page with navigation

### 4. Performance Targets

Discuss realistic targets:
- Page load time goal (e.g., < 3 seconds on 4G)
- Core Web Vitals targets
- Mobile performance priority

### 5. Update Output Document
- Fill in Multilingual Requirements section
- Fill in SEO Requirements section
- Add Performance Targets

### 6. Design Log Update
After completing this step, update the design log:

```markdown
### Step 31: Multilingual & SEO
**Q:** Language needs? SEO technical requirements? Performance targets?
**A:** [User responses - summarized]
**Documented in:** platform-requirements.md (Multilingual, SEO sections)
**Key insights:** [Important decisions or revelations]
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
- Language requirements documented
- URL structure defined (if multilingual)
- Translation workflow planned
- SEO technical requirements documented
- Structured data plan created
- Performance targets set
- User confirmed

### FAILURE:
- Skipped structured data planning
- Generated SEO requirements without user input
- Did not document translation workflow

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
