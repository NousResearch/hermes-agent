# prompt-refinement

Registers the `prompt_refinement` auxiliary model slot used by Hermes Desktop's
Prompt Refinement review (`apps/desktop/src/plugins/prompt-refinement/`).

- Opt-in per Hermes home: `hermes plugins enable prompt-refinement`.
- Pick the model under Settings > Model > Auxiliary > "Prompt refinement".
- Registers no tools, hooks or prompts. The refinement call is a stateless
  `llm.oneshot` request: no session, history, tools or skills.
- Not enabled: Desktop's refinement still works through the default
  auxiliary route; the slot just isn't selectable. (A backend that refuses an
  unregistered task makes the review fail closed and say how to enable this.)
