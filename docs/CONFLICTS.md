# Merge Conflict Guide

The single most important thing to read before pulling upstream changes. This fork modifies **user-facing strings** across dozens of files — the same files upstream modifies in every release.

## Conflict Risk Table

| File | Risk | What we changed | What upstream might change | Resolution strategy |
|---|---|---|---|---|
| `agent/prompt_builder.py` | 🔴 High | `DEFAULT_AGENT_IDENTITY` line: "AI Collaborator, built by Unitalk"; `HERMES_AGENT_HELP_GUIDANCE`: "Unitalk AI Collaborator" | Prompt guidance blocks, model lists, tool enforcement text | **Always accept ours** for identity lines. Upstream changes to guidance blocks: merge manually — upstream's behavioral changes are important, but our branding strings stay. |
| `agent/onboarding_text.py` | 🔴 High | `GREETING`: "I'm Unitalk"; `_GREETINGS` dict; `facts_block`: "computer running Unitalk"; `opening_prompt`: "Unitalk desktop app" | Beat questions, reminder logic, task lists, flow rules | **Accept ours for greeting/identity strings.** Upstream's logic changes (new beats, new reminder keys) must be merged carefully — the `prompt_builder` import and `_GREETINGS` dict are the collision points. |
| `apps/desktop/src/i18n/en_shared_metrics.ts` | 🟡 Medium | "Share with Unitalk", "stats to Unitalk" | New stats fields, wording changes | Accept upstream's structural changes (new fields) but **keep our "Unitalk" strings**. The `share` and `sending` keys are the ones we touch. |
| `apps/desktop/src/i18n/fr_shared_metrics.ts` | 🟡 Medium | "Partager avec Unitalk" | Same as above — stats fields evolve | Same strategy: upstream structure, our brand strings. |
| `apps/desktop/src/i18n/*_shared_metrics.ts` (all 8 locales) | 🟡 Medium | "Unitalk" / "Partager avec Unitalk" in each locale's `share`/`sending` keys | Locale additions, corrections | Accept upstream's non-consent strings. **Keep our consent strings.** |
| `apps/desktop/src/i18n/*_first_run.ts` (10 files) | 🟡 Medium | `intro.greeting`: "I'm Unitalk" in each locale | New beat labels, locale fixes | Accept upstream changes — **verify that the greeting still says "Unitalk"** after merge. |
| `apps/desktop/src/themes/presets.ts` | 🟡 Medium | 11 brand themes (not 13 geeky ones); `DEFAULT_SKIN_NAME = 'unitalk'`; `RETIRED_SKINS` set | New upstream themes, palette changes | Accept ours for `DEFAULT_SKIN_NAME` and `RETIRED_SKINS`. Upstream's new themes: add them to `BUILTIN_THEMES` manually. |
| `apps/desktop/src/components/onboarding/providers.tsx` | 🟡 Medium | `PROVIDER_ORDER` puts `nous` first; `HIDDEN_PROVIDER_IDS` hides `qwen-oauth`; browser-auth only | New providers added by upstream | Accept ours for `PROVIDER_ORDER` and `HIDDEN_PROVIDER_IDS`. Add new upstream providers to `PROVIDER_ORDER` (push them down). |
| `apps/desktop/src/plugins/welcome/` | 🟢 Low | Entire welcome plugin is ours — doesn't exist upstream | No conflict possible | No merge needed — **ours only.** |
| `agent/system_prompt.py` | 🟢 Low | Imports `DEFAULT_AGENT_IDENTITY` from `prompt_builder` (unchanged import name) | New guidance block imports, slot resolution logic | Accept upstream changes. The import name didn't change — only the string value in `prompt_builder.py` did. |
| `apps/desktop/src/components/onboarding/first-run/intro-copy.tsx` | 🟢 Low | Comment references `onboarding_text.py` (unchanged) | Typing animation tuning | Accept upstream. **Verify** intro greeting still resolves to "I'm Unitalk" at runtime — the component itself doesn't hardcode the string. |
| `agent/codex_responses_adapter.py` | 🟢 Low | Imports `DEFAULT_AGENT_IDENTITY` (unchanged import name) | Instruction assembly logic | Accept upstream. The import name didn't change. |
| `agent/transports/codex.py` | 🟢 Low | Imports `DEFAULT_AGENT_IDENTITY` (unchanged import name) | API request construction | Accept upstream. |
| `apps/desktop/src/i18n/en.ts`, `fr.ts`, etc. | 🟢 Low | "Unitalk" in onboarding descriptions, provider captions | New UI string keys | Accept upstream. **Grep for "Hermes" → "Unitalk" in user-facing text** after merge. |
| `agent/onboarding_session.py` | 🟢 Low | Imports from `onboarding_text` (unchanged module name) | Session lifecycle logic | Accept upstream. |
| `agent/onboarding_first_tasks.py` | 🟢 Low | No changes | Task definitions | Accept upstream. |
| `apps/shared/src/skin.ts` | 🟢 Low | `'welcome'` in skin list (existing) | New skin definitions | Accept upstream. |

## Protected Identifiers — Never Modify

These must **never** contain "Unitalk". Grep for them after every merge:

```
X-Hermes-Session-Token      X-Hermes-Signature-256
X-Hermes-Event              X-Hermes-Delivery
X-Hermes-Session-Id         HERMES_HOME
HERMES_DESKTOP_DEV_SERVER   HERMES_DESKTOP_CDP_PORT
hermes_cli                  hermes_state_*
hermes_constants            hermes_bootstrap
hermes_logging              hermes_platform
hermes_dir_acl              hermes_time
hermes_agent                ~/.hermes/
~/hermes-first-task/        hermes-agent.nousresearch.com/docs
NousResearchInc.HermesAgent
```

## Post-Merge Verification

After resolving conflicts, run the verification checks:

```bash
# 1. Grep for accidental identifier corruption
grep -r "unitalk_[a-z_]*" agent/ apps/ --include="*.py" | grep -v node_modules

# 2. Ensure protected identifiers still exist
grep -r "HERMES_HOME" agent/ --include="*.py"
grep -r "hermes-agent.nousresearch.com" agent/ --include="*.py"

# 3. Check user-facing "Hermes Agent" is gone from UI
grep -r "Hermes Agent" apps/desktop/src/ --include="*.ts" --include="*.tsx" | grep -v node_modules

# 4. Python syntax check
python3 -m py_compile agent/prompt_builder.py
python3 -m py_compile agent/onboarding_text.py

# 5. Run the branding script (idempotent)
./scripts/apply-unitalk-branding.sh
```

## Worst-Case Recovery

If a merge corrupts the branding beyond easy repair:

1. `git checkout -- agent/prompt_builder.py agent/onboarding_text.py`
2. `git checkout -- apps/desktop/src/themes/presets.ts`
3. `git checkout -- apps/desktop/src/components/onboarding/providers.tsx`
4. `git checkout -- apps/desktop/src/i18n/`
5. Run `./scripts/apply-unitalk-branding.sh`
6. Manually merge any new upstream features into the restored files