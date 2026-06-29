---
name: 'step-02-setup-environment'
description: 'Verify the development environment is ready, all dependencies are installed, and the project runs cleanly before writing any new code'

# File References
nextStepFile: './step-03-implement.md'
---

# Step 2: Setup Environment

## STEP GOAL:

Verify the development environment is ready, all dependencies are installed, and the project runs cleanly before writing any new code.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are an Implementation Partner guiding structured development activities
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring software development methodology expertise, user brings domain knowledge and codebase familiarity
- ✅ Maintain clear and structured tone throughout

### Step-Specific Rules:

- 🎯 Focus only on verifying environment, installing dependencies, starting dev server, and establishing test baseline
- 🚫 FORBIDDEN to begin implementing features — that is the next step
- 💬 Approach: Walk through environment verification systematically with user
- 📋 Document any pre-existing issues so they are not confused with regressions later

## EXECUTION PROTOCOLS:

- 🎯 Confirm environment is clean and ready for implementation
- 💾 Document test baseline and any pre-existing issues in the dialog file
- 📖 Reference project configuration files and existing documentation
- 🚫 Do not write any feature code during this step

## CONTEXT BOUNDARIES:

- Available context: Implementation plan from Step 1; project configuration files
- Focus: Environment verification, dependency installation, baseline establishment
- Limits: No feature implementation
- Dependencies: Step 1 must be complete (plan exists, branch created)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Verify Tech Stack Requirements

Check that the project's required tooling is available:

- Runtime (Node.js version, Python version, etc.)
- Package manager (npm, yarn, pnpm, pip, etc.)
- Build tools (Vite, Webpack, Turbopack, etc.)
- Any CLI tools the project depends on

If version mismatches exist, resolve them now — not mid-implementation.

### 2. Install Dependencies

Run the project's install command. Watch the output for:

- Deprecation warnings (note but do not fix unless blocking)
- Peer dependency conflicts (resolve if they cause build failures)
- Missing system-level dependencies

```
npm install   # or yarn, pnpm install, pip install -r requirements.txt, etc.
```

### 3. Start the Development Server

Run the dev server and confirm:

- The project builds without errors
- The existing pages / routes load in the browser
- Hot reload or watch mode works
- No console errors on existing pages

### 4. Verify Design System Access

If the project uses a design system or design tokens, confirm you can access them:

- Token files (colors, spacing, typography) are present and importable
- Component library is installed and renders correctly
- Icon sets or asset libraries are available
- Any Figma-to-code output is up to date

### 5. Create Project Structure (If Needed)

If the spec requires new directories or organizational structure, create them now:

- New page directories
- New component directories
- Test file locations
- Any configuration files for new routes

Follow the existing project conventions for naming and placement.

### 6. Run Existing Tests to Establish Baseline

Run the full test suite before touching anything:

```
npm test   # or the project's equivalent
```

Record the result:

- **All pass:** Good baseline. Any future failure is from your changes.
- **Some fail:** Document which tests fail BEFORE you start. These are pre-existing failures and not your responsibility, but you need to know they exist so you do not accidentally claim them as regressions.
- **No tests exist:** Note this. Consider whether the spec requires tests (Step 03 will address test writing).

### 7. Verify Checklist

- [ ] Runtime and tooling versions verified
- [ ] Dependencies installed without blocking errors
- [ ] Dev server starts and existing pages load
- [ ] Design tokens / design system accessible
- [ ] New directories created (if needed)
- [ ] Existing test suite run, baseline documented
- [ ] Any pre-existing issues noted in dialog file

### 8. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 3: Implement"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the environment is verified clean, dependencies installed, and test baseline documented will you then load and read fully `{nextStepFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Runtime and tooling versions verified
- Dependencies installed without blocking errors
- Dev server starts and existing pages load
- Design tokens / design system accessible
- Existing test suite run, baseline documented
- Pre-existing issues noted

### ❌ SYSTEM FAILURE:
- Starting implementation with unresolved environment issues
- Not establishing test baseline
- Not documenting pre-existing failures
- Skipping dependency installation

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
