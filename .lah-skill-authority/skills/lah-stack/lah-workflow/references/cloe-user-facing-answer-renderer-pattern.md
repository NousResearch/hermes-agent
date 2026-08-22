# User-Facing Answer Renderer Pattern

## Problem

The CLOE cognitive pipeline produces structured context output through the
Brain Context Builder. This output contains:

- System prompts (`=== OPENCLAW TRUSTED SYSTEM POLICY ===`)
- Safety envelopes (`=== SAFETY ENVELOPE ===`)
- Context packs (`=== CLOE COGNITIVE CONTEXT PACK ===`)
- Internal metadata (`selected_collectors`, `intent_tags`, `compact_summary`)
- LLM instructions (`## LLM INSTRUCTIONS`)

These internal artifacts must never reach the user.

## Solution: Answer Renderer

Insert a rendering layer AFTER the Brain Context Builder and BEFORE the
provider completion:

```
Gateway → CLOE → Envelope → Brain Context Builder → Answer Renderer → Provider → User
```

### Implementation

File: `src/services/cloe/operator-answer-renderer.js`

Key functions:

- `renderOperatorAnswer(brainContext, { debug, language })` — main entry point
- `stripInternalArtifacts(text)` — regex-based artifact removal
- `containsInternalArtifacts(text)` — detection for debug mode
- `detectLanguage(prompt)` — fr/en/es/de detection

### NORMAL Mode

Strips all internal artifacts. Returns only the natural answer:

```javascript
const rendered = renderOperatorAnswer(brainContext, { debug: false });
// rendered.answer → cleaned text without any internal artifacts
```

### DEBUG Mode

Returns natural answer plus reasoning metadata:

```javascript
const rendered = renderOperatorAnswer(brainContext, { debug: true });
// rendered.debug → { mode, system_prompt_length, context_text_length,
//                     artifacts_stripped, system_prompt_excerpt, context_excerpt }
```

### Patterns to Strip

```javascript
const INTERNAL_PATTERNS = [
  /=== OPENCLAW TRUSTED SYSTEM POLICY ===[\s\S]*?(?=\n\n|$)/,
  /=== SAFETY ENVELOPE ===[\s\S]*?(?=\n\n|$)/,
  /=== CLOE COGNITIVE CONTEXT( PACK)? ===[\s\S]*?(?=\n\n|$)/,
  /## LLM INSTRUCTIONS[\s\S]*?(?=\n\n|$)/,
  /intent_tags[\s\S]*?(?=\n|$)/,
  /selected_collectors[\s\S]*?(?=\n|$)/,
  /compact_summary[\s\S]*?(?=\n|$)/,
  /cognitive_context_pack[\s\S]*?(?=\n|$)/,
  /attach_to_prompt[\s\S]*?(?=\n|$)/,
  /brain_context_builder[\s\S]*?(?=\n|$)/,
];
```

### Gateway Integration

In `buildCloeAssistantResponse()`:

```javascript
import { renderOperatorAnswer, detectLanguage } from '../cloe/operator-answer-renderer.js';

// ... build brainContext via buildOpenClawBrainContext ...

const rendered = renderOperatorAnswer(brainContext, {
  debug: false,
  language: detectLanguage(prompt)
});

return {
  title: 'CLOE Assistant',
  answer: rendered.answer,  // Clean, natural answer
  // ...
};
```

### Smoke Test

The 9 original trace prompts (French operator questions about identity,
capabilities, memory, roadmap, next move, blockers, Codex prompts,
LLM changes, devil's advocate) must ALL produce natural answers with
ZERO internal artifact violations.

Tools/CI: `tools/ci/cloe-real-operator-9-question-smoke-test-v2.mjs`

### Pitfalls

1. **Context pack formatting varies**: The Brain Context Builder may output
   `=== CLOE COGNITIVE CONTEXT PACK ===` or `--- CLOE COGNITIVE CONTEXT ---`.
   Patterns must cover both forms with `( PACK)?` optional suffix.

2. **Metadata fields change**: The `cognitiveContextPack` structure may add
   new metadata fields. The `containsInternalArtifacts()` detection list must
   be maintained alongside `INTERNAL_PATTERNS`.

3. **The `patch` tool double-escapes backslashes in regex**: After patching
   regex patterns in the renderer, normalize with Python:
   `c=c.replace('\\\\b', '\\b')`. See the main workflow skill's pitfalls.

4. **Debug mode should never be default**: Always default to `debug: false`.
   Debug mode exposes system prompt excerpts and context excerpts which are
   internal information.
