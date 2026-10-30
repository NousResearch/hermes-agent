# Creative direction: make state feel authored

Rive is most valuable when a small authored visual system reacts with continuity: a gesture has weight, a diagram retains relationships, an instrument communicates a real transition, or a character makes a response legible. Aesthetic delight can be the purpose. The aim is an intentional experience—not a runtime specimen enlarged into a hero and not every screen moving at once.

## Choose through a short path

1. **Use the brief.** Identify the primary surface and what the person should understand, do or feel. Sketch the still first. Do not ask what the context already answers.
2. **State the meaningful change.** Is it ordinary UI state/layout, a reusable authored vector/character state graph, or genuinely spatial 3D? Prefer native DOM/SVG for ordinary UI; consider Rive for authored interactive art; consider Three.js only when actual spatial exploration earns it and the project authorizes it.
3. **Make one recommendation.** Explain in one sentence why it fits. If a decision is required, show only the two or three choices at that fork, not a gallery of libraries/styles. 'Keep it static/native' is a valid result.
4. **Reveal the next requirement only when relevant.** For Rive: source/export access → typed asset contract → required renderer features → measured delivery. Do not expose plan menus, every runtime, shaders and rigging before knowing the need.
5. **Prototype one consequential beat.** Show entry, interaction, interruption and stable still. Judge it at actual delivery size. Continue only if it improves comprehension, pleasure, maintainability or reuse enough to justify cost.

During other work, surface at most one timely idea and do not interrupt urgent execution. For explicit creative/brainstorming briefs, recommend one strong direction at the ambition already requested; do not require another request to unlock cinematic or exploratory work. Reveal meaningful alternatives only if a decision remains. Static composition is a planning/fallback aid, not a utility test that an artwork must pass; full-screen Rive can be appropriate with accessible entry/exit and lifecycle safeguards. Progressive disclosure limits option overload, not imagination.

## Apply the project's design system

Use the actual brand, palette, typography, logo assets and motion contracts of
the project. Derive Rive colors from shared tokens rather than making a second
palette. Verify font rights and glyph support independently from host CSS.
Separate wrapper motion from asset motion so two clocks do not fight. A local
experiment does not silently become a canonical brand primitive.

## Craft process

- **Storyboard poses before timelines.** Entry, anticipation only when useful, meaningful action, settle, error, cancel, reduced. Immediate semantic acknowledgment should not wait for a flourish.
- **Design silhouettes and paths at the real size.** Geometry, weight, alignment, overlap and clipping must work before glow. A smaller, sharper diagram often beats an elaborate scene.
- **Choreograph attention.** One foreground beat; stagger supporting elements only to explain relationship. Author pauses and quiet states as deliberately as motion.
- **Preserve continuity on interruption.** Input should feel answered, not queued behind a canned movie. Test reverse, rapid changes, cancel and repeated activation. Do not spring every property by default.
- **Use exaggeration purposefully.** Character gestures can be expressive; operational status should stay precise. Restraint is a hierarchy choice, not a requirement that every creative artifact be timid.
- **Separate reusable structure from repeated decoration.** A stateful component should support meaningful variants. Copying the same loop into every section is not a system.
- **Review without the effect.** With motion off, renderer missing or loading stalled, the essential composition and meaning remain. For nonessential art, a deliberate still and optional replay can be the complete fallback.
- **Make feedback honest.** Distinguish agent critique, actual visual inspection, human taste feedback and user testing. Do not manufacture objective quality scores.

## Opportunity cards (proposals, not installed integrations)

### A. Agent handoff as a small working instrument

**Trigger:** a workflow genuinely moves an item between stages/agents and people lose track of who owns it. **Benefit:** make handoff/recovery readable without a large orchestration graph. **Interaction:** one workpiece travels through a compact authored instrument; selecting a stage reveals the existing DOM details. **Distinctive treatment:** a stable structural frame, one accent on active ownership, a distinct intervention beat. **Inputs:** validated host state with source timestamp, or clearly labeled synthetic fixtures. **Minimum:** one handoff, blocked state, cancellation and static poses. **Risk:** fake progress or animation reporting success; host remains truth. **Simpler alternative:** native status/list. **Value test:** does a person identify current owner and next action faster with or without the illustration?

### B. A diagram you can interrogate

**Trigger:** an explanation has two or three interacting mechanisms that static slides obscure. **Benefit:** show cause/effect and preserve spatial relationships. **Interaction:** semantic controls vary one condition; the authored vector mechanism reacts; DOM explains current state. **Distinctive treatment:** precise cutaway/mask and selective trace reveal rather than ubiquitous arrows. **Inputs:** reviewed mechanism and bounded parameter model. **Minimum:** one parameter, three valid poses, invalid-input handling, native reference. **Risk:** implying physical accuracy from illustration. **Alternative:** SVG diagram. **Value test:** can the viewer correctly predict the consequence after using it? For true spatial rotation/occlusion, evaluate Three.js separately instead.

### C. A configurator that feels like handling the product

**Trigger:** variants alter meaningful internal structure, not only a color swatch. **Benefit:** reduce disconnected screenshots and show what actually changes. **Interaction:** DOM selection changes nested artboard/appearance with continuous alignment. **Distinctive treatment:** a sparse exploded 2D cutaway or instrument face with deliberate settle. **Inputs:** owned product/vector assets and verified variant rules. **Minimum:** two variants and a static comparison; no checkout/actions inside the asset. **Risk:** artwork that promises unsupported product capabilities. **Alternative:** two SVGs/images. **Value test:** fewer selection errors or a clearer explanation; actual 3D fit/geometry needs a different tool.

### D. A responsive guide, not a mascot pasted everywhere

**Trigger:** onboarding/learning benefits from a character or abstract guide reacting to real choices. **Benefit:** warmth, motivation and memorable feedback. **Interaction:** a small expressive character has listening, considering, guiding and recovered poses, driven by explicit app events. **Distinctive treatment:** an original geometric visual vocabulary compatible with the chosen identity; richer expression without repurposing the product logo. **Inputs:** character rights, audience/voice brief and state contract. **Minimum:** three poses, interrupted response, keyboard-equivalent control, still version. **Risk:** infantilizing a professional interface or implying sentience/real progress. **Alternative:** microcopy and static illustration. **Value test:** actual audience preference and task clarity, not an invented engagement score.

### E. An interactive chapter specimen inside the design manual

**Trigger:** readers need to understand why a motion/state rule exists. **Benefit:** teach through manipulation rather than a long option catalog. **Interaction:** one recommended specimen with a small 'Try the failure' disclosure and one changed-context example. **Distinctive treatment:** expose the causal state and current data contract alongside the composed visual. **Inputs:** canonical rule, exact reference, licensed authored asset. **Minimum:** one approved pattern with native reference and a few deterministic poses. **Risk:** the demonstration silently becoming a new canonical primitive. **Alternative:** existing native field guide. **Value test:** can a fresh implementer reproduce the rule and identify the counterexample?

### F. A reusable recovery illustration across surfaces

**Trigger:** the same product repeatedly communicates connecting, unavailable, retrying and recovered across web/native clients. **Benefit:** one authored visual state system with consistent semantics. **Interaction:** host events select a deliberate pose/transition; actual Retry stays in native/HTML controls. **Distinctive treatment:** sparse instrument geometry, healthy quiet state, state colors only as meaningful exceptions. **Inputs:** platform-specific validated adapters, original asset and real failure definitions. **Minimum:** web proof then a separately built target-native proof. **Risk:** assuming runtime parity or reporting recovery from the asset's timer. **Alternative:** standard native state UI. **Value test:** consistency and reuse after measured device/accessibility cost.

### G. A compact authored brand story with meaningful branching

**Trigger:** a standalone explainer/presentation is meant to delight and reveal a concept, not operate critical controls. **Benefit:** expressive continuity that a linear video cannot respond to. **Interaction:** one meaningful branch changes how a short illustrative story resolves; replay is optional and reading is never delayed. **Distinctive treatment:** strong silhouettes, carefully timed reveals, confident stills and only one focal event. **Inputs:** clear story, original vector assets, rights-cleared sound only if explicitly opted in. **Minimum:** one branch and two endings; silent/static first. **Risk:** oversized payload, gratuitous loops, inaccessible story content. **Alternative:** short video or native SVG timeline. **Value test:** does interaction add an idea the static/video version cannot communicate as well?

## Adoption gate

Adopt a Rive supplement only after an actual owned/authorized asset, versioned contract and target behavior have been verified. Compare to the simpler alternative. Document source/export ownership, renderer cost, static/reduced path, and future editing burden. Do not confuse the official test fixture's third-party typography/colors with original artwork. Keep ambitious future features explicitly proposed until their concrete implementation passes.
