---
name: bmad-validate-prd
description: 'DEPRECATED — consolidated into bmad-prd validate intent - this skill will be removed in v7 in favor of `bmad-prd`.'
---

# DEPRECATED — forwards to bmad-prd (validate intent)

This skill was consolidated into `bmad-prd`. It is retained as a thin compatibility shim so existing invocations by name and `_bmad/custom/bmad-validate-prd.toml` override files keep working. New work should invoke `bmad-prd` directly — it detects create / update / validate intent from the conversation.

## On Activation

1. Resolve customization: `python3 {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --key workflow`. This picks up any `{project-root}/_bmad/custom/bmad-validate-prd.toml` and `bmad-validate-prd.user.toml` overrides for the legacy fields (`activation_steps_prepend`, `activation_steps_append`, `persistent_facts`, `on_complete`).

2. Load `{project-root}/_bmad/bmm/config.yaml` (and `config.user.yaml` if present) to resolve `{user_name}` and `{communication_language}`.

3. Emit a deprecation notice to the user in `{communication_language}`:

   > Notice: `bmad-validate-prd` is deprecated and will be removed in a future release. It now forwards to `bmad-prd` with validate intent. To silence this notice and access the full new customization surface (`prd_template`, `validation_checklist`, `doc_standards`, `external_sources`, `external_handoffs`, `output_dir`, `output_folder_name`), migrate `_bmad/custom/bmad-validate-prd.toml` to `_bmad/custom/bmad-prd.toml` and invoke `bmad-prd` directly next time. Customization fields that were in this version still remain in the new version and will be respected if present in `_bmad/custom/bmad-prd.toml`, but the new version also supports additional fields that you can take advantage of by migrating.

4. Invoke `bmad-prd` with the following context. Pass these as the activating context so `bmad-prd` honors them instead of resolving its own customization from scratch:

   - **Intent:** `validate` — skip `bmad-prd`'s usual intent detection step.
   - **Pre-resolved legacy customization** — use these in place of resolving from `bmad-prd`'s own `customize.toml` for the four legacy fields. For everything else (`prd_template`, `validation_checklist`, `validation_report_template`, `doc_standards`, `output_dir`, `output_folder_name`, `external_sources`, `external_handoffs`), use `bmad-prd`'s own defaults and overrides as normal:
     - `activation_steps_prepend` = the resolved value from step 1
     - `activation_steps_append` = the resolved value from step 1
     - `persistent_facts` = the resolved value from step 1
     - `on_complete` = the resolved value from step 1
   - **Original user input:** forward whatever the user said when invoking this skill verbatim (the target PRD path, etc.).

   `bmad-prd` takes the workflow from here. Do not execute any further steps in this shim.
