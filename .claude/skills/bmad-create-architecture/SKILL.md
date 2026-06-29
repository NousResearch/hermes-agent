---
name: bmad-create-architecture
description: 'DEPRECATED — consolidated into bmad-architecture create intent - this skill will be removed in v7 in favor of `bmad-architecture`.'
---

# DEPRECATED — forwards to bmad-architecture (create intent)

This skill was consolidated into `bmad-architecture`. It is retained as a thin compatibility shim so existing invocations by name and `_bmad/custom/bmad-create-architecture.toml` override files keep working. New work should invoke `bmad-architecture` directly — it detects create / update / validate intent from the conversation.

## On Activation

1. Resolve customization: `python3 {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --key workflow`. This picks up any `{project-root}/_bmad/custom/bmad-create-architecture.toml` and `bmad-create-architecture.user.toml` overrides for the legacy fields (`activation_steps_prepend`, `activation_steps_append`, `persistent_facts`, `on_complete`).

2. Load `{project-root}/_bmad/bmm/config.yaml` (and `config.user.yaml` if present) to resolve `{user_name}` and `{communication_language}`.

3. Emit a deprecation notice to the user in `{communication_language}`:

   > Notice: `bmad-create-architecture` is deprecated and will be removed in a future release. It now forwards to `bmad-architecture` with create intent. To silence this notice and access the full new customization surface (`spine_template`, `spine_output_path`, `run_folder_pattern`, `doc_standards`, `external_sources`, `external_handoffs`, `finalize_reviewers`), migrate `_bmad/custom/bmad-create-architecture.toml` to `_bmad/custom/bmad-architecture.toml` and invoke `bmad-architecture` directly next time. Customization fields that were in this version still remain in the new version and will be respected if present in `_bmad/custom/bmad-architecture.toml`, but the new version also supports additional fields that you can take advantage of by migrating.

4. Invoke `bmad-architecture` with the following context. Pass these as the activating context so `bmad-architecture` honors them instead of resolving its own customization from scratch:

   - **Intent:** `create` — skip `bmad-architecture`'s usual intent detection step.
   - **Pre-resolved legacy customization** — use these in place of resolving from `bmad-architecture`'s own `customize.toml` for the four legacy fields. For everything else (`spine_template`, `spine_output_path`, `run_folder_pattern`, `doc_standards`, `external_sources`, `external_handoffs`, `finalize_reviewers`), use `bmad-architecture`'s own defaults and overrides as normal:
     - `activation_steps_prepend` = the resolved value from step 1
     - `activation_steps_append` = the resolved value from step 1
     - `persistent_facts` = the resolved value from step 1
     - `on_complete` = the resolved value from step 1
   - **Original user input:** forward whatever the user said when invoking this skill verbatim.

   `bmad-architecture` takes the workflow from here. Do not execute any further steps in this shim.
