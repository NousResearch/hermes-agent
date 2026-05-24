# MemoryMunch Pre-E2E Fallow / Node Dependency Living Report

Status: living doc, pre-E2E gate
Source audit: `.hermes/review-packs/memorymunch-async-softwall-20260524/fallow-audit-20260524T144817/hermes-fallow-audit.json`
Fallow verdict: pass (new-only gate)
Dead-code/dependency issues: 15 total; introduced by our MemoryMunch work: 0; inherited/pre-existing: 15.

## Lame terms summary

- The 15 issues are not Python code bugs in MemoryMunch.
- They are package.json dependency hygiene findings: packages listed as dependencies but not seen by fallow as imported from known entry points.
- Fallow reported zero unresolved imports and zero unlisted dependencies. That means this audit did not find missing node modules.
- Because all 15 are `introduced=false`, they look inherited from Hermes / existing app areas, not caused by the MemoryMunch plugin work.
- Do not auto-remove them before E2E unless each owning app is built/tested; some can be runtime/plugin/config dependencies that static analysis cannot see.

## Node modules / missing modules answer

Fallow found no missing JS dependencies in code references: `unresolved_imports=0`, `unlisted_dependencies=0`, `type_only_dependencies=0`, `test_only_dependencies=0`.
The 15 fallow warnings are the opposite direction: possibly extra dependencies in package manifests.

A live `npm ls --depth=0` install-state check found separate local install gaps:

- Root package: `agent-browser` is installed, but root `node_modules` also has many extraneous transitive packages. Lame terms: root node_modules is dirty, not proof the app is broken.
- `ui-tui`: listed deps are installed.
- `web`: listed deps are installed.
- `website`: local `node_modules` is missing the Docusaurus/doc-site deps (`@docusaurus/core`, presets/theme, `@mdx-js/react`, `clsx`, `react`, `react-dom`, `typescript`, etc.). Lame terms: website build cannot be trusted until npm install is run in `website` or workspace install restores those packages.
- `scripts/whatsapp-bridge`: local `node_modules` is missing `@whiskeysockets/baileys`, `express`, `pino`, and `qrcode-terminal`. Lame terms: WhatsApp bridge cannot be locally smoke-tested until its npm install is restored.

Deep JS dependency audit recommended? Yes, but as a separate safe lane. For MemoryMunch Python E2E, the missing website/WhatsApp node modules are not on the plugin path. For full repo E2E, restore/install those package dirs first, then build/test before removing any dependency.

## The 15 issues

1. `agent-browser`
   - Type: unused dependency
   - File: `package.json:19`
   - Area affected: root/browser tooling
   - Introduced by us: no / inherited
   - Lame terms impact: may make installs heavier or stale; not evidence of a broken import.
   - Fix risk: low only after that area builds/tests without it; medium if removed blindly because static analysis may miss runtime/plugin usage.
   - Special caution: this looks like a runtime/tooling package that may be used indirectly, so do not remove without targeted runtime proof.

2. `@whiskeysockets/baileys`
   - Type: unused dependency
   - File: `scripts/whatsapp-bridge/package.json:11`
   - Area affected: WhatsApp bridge
   - Introduced by us: no / inherited
   - Lame terms impact: may make installs heavier or stale; not evidence of a broken import.
   - Fix risk: low only after that area builds/tests without it; medium if removed blindly because static analysis may miss runtime/plugin usage.
   - Special caution: this looks like a runtime/tooling package that may be used indirectly, so do not remove without targeted runtime proof.

3. `@hermes/ink`
   - Type: unused dependency
   - File: `ui-tui/package.json:19`
   - Area affected: Hermes terminal UI
   - Introduced by us: no / inherited
   - Lame terms impact: may make installs heavier or stale; not evidence of a broken import.
   - Fix risk: low only after that area builds/tests without it; medium if removed blindly because static analysis may miss runtime/plugin usage.
   - Special caution: this looks like a runtime/tooling package that may be used indirectly, so do not remove without targeted runtime proof.

4. `@observablehq/plot`
   - Type: unused dependency
   - File: `web/package.json:17`
   - Area affected: Hermes web app
   - Introduced by us: no / inherited
   - Lame terms impact: may make installs heavier or stale; not evidence of a broken import.
   - Fix risk: low only after that area builds/tests without it; medium if removed blindly because static analysis may miss runtime/plugin usage.

5. `@react-three/fiber`
   - Type: unused dependency
   - File: `web/package.json:18`
   - Area affected: Hermes web app
   - Introduced by us: no / inherited
   - Lame terms impact: may make installs heavier or stale; not evidence of a broken import.
   - Fix risk: low only after that area builds/tests without it; medium if removed blindly because static analysis may miss runtime/plugin usage.

6. `class-variance-authority`
   - Type: unused dependency
   - File: `web/package.json:25`
   - Area affected: Hermes web app
   - Introduced by us: no / inherited
   - Lame terms impact: may make installs heavier or stale; not evidence of a broken import.
   - Fix risk: low only after that area builds/tests without it; medium if removed blindly because static analysis may miss runtime/plugin usage.

7. `gsap`
   - Type: unused dependency
   - File: `web/package.json:27`
   - Area affected: Hermes web app
   - Introduced by us: no / inherited
   - Lame terms impact: may make installs heavier or stale; not evidence of a broken import.
   - Fix risk: low only after that area builds/tests without it; medium if removed blindly because static analysis may miss runtime/plugin usage.

8. `leva`
   - Type: unused dependency
   - File: `web/package.json:28`
   - Area affected: Hermes web app
   - Introduced by us: no / inherited
   - Lame terms impact: may make installs heavier or stale; not evidence of a broken import.
   - Fix risk: low only after that area builds/tests without it; medium if removed blindly because static analysis may miss runtime/plugin usage.

9. `unicode-animations`
   - Type: unused dependency
   - File: `web/package.json:35`
   - Area affected: Hermes web app
   - Introduced by us: no / inherited
   - Lame terms impact: may make installs heavier or stale; not evidence of a broken import.
   - Fix risk: low only after that area builds/tests without it; medium if removed blindly because static analysis may miss runtime/plugin usage.

10. `@mdx-js/react`
   - Type: unused dependency
   - File: `website/package.json:25`
   - Area affected: Docusaurus docs site
   - Introduced by us: no / inherited
   - Lame terms impact: may make installs heavier or stale; not evidence of a broken import.
   - Fix risk: low only after that area builds/tests without it; medium if removed blindly because static analysis may miss runtime/plugin usage.

11. `clsx`
   - Type: unused dependency
   - File: `website/package.json:26`
   - Area affected: Docusaurus docs site
   - Introduced by us: no / inherited
   - Lame terms impact: may make installs heavier or stale; not evidence of a broken import.
   - Fix risk: low only after that area builds/tests without it; medium if removed blindly because static analysis may miss runtime/plugin usage.

12. `@babel/plugin-syntax-jsx`
   - Type: unused devDependency
   - File: `ui-tui/package.json:30`
   - Area affected: Hermes terminal UI
   - Introduced by us: no / inherited
   - Lame terms impact: may make installs heavier or stale; not evidence of a broken import.
   - Fix risk: low only after that area builds/tests without it; medium if removed blindly because static analysis may miss runtime/plugin usage.

13. `babel-plugin-react-compiler`
   - Type: unused devDependency
   - File: `ui-tui/package.json:36`
   - Area affected: Hermes terminal UI
   - Introduced by us: no / inherited
   - Lame terms impact: may make installs heavier or stale; not evidence of a broken import.
   - Fix risk: low only after that area builds/tests without it; medium if removed blindly because static analysis may miss runtime/plugin usage.

14. `three`
   - Type: unused devDependency
   - File: `web/package.json:47`
   - Area affected: Hermes web app
   - Introduced by us: no / inherited
   - Lame terms impact: may make installs heavier or stale; not evidence of a broken import.
   - Fix risk: low only after that area builds/tests without it; medium if removed blindly because static analysis may miss runtime/plugin usage.

15. `@docusaurus/module-type-aliases`
   - Type: unused devDependency
   - File: `website/package.json:32`
   - Area affected: Docusaurus docs site
   - Introduced by us: no / inherited
   - Lame terms impact: may make installs heavier or stale; not evidence of a broken import.
   - Fix risk: low only after that area builds/tests without it; medium if removed blindly because static analysis may miss runtime/plugin usage.

## Recommended fix lane

1. Keep MemoryMunch E2E focused on Python plugin/watchdog/compaction tests.
2. For JS deps, run per-area build/test first: root npm check, `ui-tui` build/test/type-check, `web` build, `website` build/typecheck, WhatsApp bridge smoke if needed.
3. Only remove a dependency when the owning app still passes after removal.
4. If a dependency is intentionally runtime/config-loaded, add a fallow ignore with a comment instead of deleting it.
5. Re-run fallow after every dependency cleanup.

## Ownership call

Current evidence says these are inherited Hermes/app dependency hygiene items, not MemoryMunch plugin defects. MemoryMunch plugin is Python-only in `contrib/plugins/memorymunch` and does not add package.json dependencies.
