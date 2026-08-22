# Gateway Routing Fix Pattern — CLOE Intent Routing

## Problem

CLOE-relevant prompts don't reach the `cloe_assistant` intent because the
Gateway router's `classifyReadonlyConversationIntent` checks generic intents
(identity, capabilities, status, self-evolution, etc.) before
`cloe_assistant`. Prompts about the assistant's identity, capabilities,
roadmap, memory, or policy are caught by earlier intents and never reach
the CLOE runtime adapter.

Additionally, the safety gate (mutating/provider) is too broad — common words
like "database", "api", "model" in harmless questions trigger the safety
block before CLOE even sees them.

## Solution: CLOE Question Pre-Check

Add a CLOE-specific pre-check AFTER the safety gate but BEFORE all generic
intent checks. This catches CLOE-relevant questions before they match
generic patterns.

### Implementation pattern

```javascript
// In classifyReadonlyConversationIntent (readonly-conversation-router.js)

// ... safety gate (mutating/provider) ...

// CLOE question pre-check — runs before generic intents
// Catches CLOE-relevant QUESTIONS that may contain safety keywords
if (includesAny(normalized, [
  // Identity questions (FR and EN)
  /\b(who are you|what are you|qui es-tu|who am i talking to)\b/,
  /\b(qui es-tu|who are you|what are you)\b.*\b(assistan|type|rôle|censé|meant to)\b/,

  // Capabilities (FR and EN)
  /\b(what do you do|what can you do|que fais-tu|que peux-tu|à quoi sers-tu)\b/,
  /\b(capable|do for me|help me with)\b.*\b(today|aujourd'hui|faire|help|aide)\b/,

  // Roadmap and progress (FR and EN)
  /\b(roadmap|where are we|où en est-on|où sommes-nous|how far along)\b/,
  /\b(what's next|what should I do next|next move|next step|prochaine mission)\b/,

  // Policy questions (not commands)
  /\b(how do i|comment|que dois-je)\b.*\b(change|changer|switch|modify)\b/,

  // Codex/Hermes handoff
  /\b(prepare|prépare|create|generate|draft)\b.*\b(prompt|codex|hermes)\b/,

  // Devil advocate / critical review
  /\b(devil|honestly|frank|critical)\b.*\b(advocate|direction|chloé|cloe|path|design|approach)\b/,

  // Memory questions
  /\b(do you have memory|do you remember|as-tu une mémoire|what memory|your memory)\b/,

  // Blockers and continuity
  /\b(what's blocking|what is blocking|qu'est-ce qui bloque|current blocker)\b/,
  /\b(continuity|continuité|session|previous conversation|contexte|follow-up)\b/
])) {
  return { intent: 'cloe_assistant', normalized };
}

// ... existing generic intent checks (identity, status, capabilities, etc.) ...
```

### Key design decisions

1. **Placement matters**: insert AFTER safety gate but BEFORE generic intents.
2. **Question-oriented patterns**: prioritize question words (`what`, `how`, `qui`, `que`)
   to avoid matching commands that should go through the safety gate.
3. **FR + EN coverage**: all patterns must work for both French and English prompts,
   since the scenario library uses English but real operators use French.
4. **Standalone patterns for identity**: `/^who are you$/` catches the exact
   phrase that was previously caught by the `identity` intent.
5. **Broad enough for context**: patterns like `/\b(roadmap|where are we)\b/`
   catch both "where are we on the roadmap" and "où en est-on dans la roadmap CLOE".

### Testing

Validate with the 9 canonical trace prompts:

```bash
node -e "
import('./src/services/gateway/readonly-conversation-router.js').then(m => {
  const tests = [
    'Salut, qui es-tu et quel type d\'assistante es-tu censée être ?',
    'Qu\'es-tu capable de faire aujourd\'hui...',
    'As-tu une mémoire ?...',
    'Où en est-on dans la roadmap CLOE ?',
    'Quel est le prochain move recommandé ?',
    'Qu\'est-ce qui bloque actuellement ?',
    'Prépare-moi un prompt Codex...',
    'Si je te demande de changer ton modèle LLM maintenant ?',
    'Devil advocate : est-ce qu\'on construit Chloé dans la bonne direction ?'
  ];
  for (const t of tests) {
    const r = m.classifyReadonlyConversationIntent(t);
    console.log(r.intent === 'cloe_assistant' ? '✅' : '❌', r.intent);
  }
});
```

All 9 must return `cloe_assistant`.

### Known side effects

- Identity questions ("Who are you?") now go to CLOE instead of the Gateway's
  hardcoded identity handler. This is intentional — CLOE should answer with
  its own context.
- Capability questions ("What can you do?") now go to CLOE instead of the
  Gateway's static capabilities list.
- Prompts containing safety keywords ("database", "api", "model") that are
  clearly QUESTIONS (not commands) are now correctly routed to CLOE.
- The `cluster_metrics` scenario note confirmed this pattern works at scale
  (18/36 scenario turns route to cloe_assistant vs 7/36 before).
