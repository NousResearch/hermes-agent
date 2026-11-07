# Netlify

Automated hosting: exact site select/create, site environment writes, official CLI build/deploy, and provider-confirmed URLs. Custom domain attachment, DNS and project visibility remain guided. The approved Netlify + Neon throwaway run on 2026-09-23 passed creation, env writes and deployment. After separately approved visibility UI changes, anonymous access and bundle scanning passed; API and browser CRUD/refresh supplied separate application acceptance evidence.

Exact-site cleanup later passed under its own approval, including an absent site and anonymous default-origin 404. Account logins, credential configuration and local evidence were retained. That cleanup used a supervised fixture helper, predating the built-in `golive teardown` flow. The built-in flow's own Netlify removal has since passed a disposable run: it deleted only the project golive had created and deployed in that same run, the account's site list counted 0 before the run, 1 during it and 0 after, and a second `teardown` planned nothing. The live results cover the tested apps and account scope.

## Login and selection

Ask the user to run `netlify login` in their own terminal when needed. The adapter reuses that login through the documented private current-user OAuth store, entirely in-process, and verifies the HTTPS identity matches the CLI. Never read or display that file using agent tools. Do not request a token in chat. The optional golive private `NETLIFY_AUTH_TOKEN` fallback must match the CLI principal; it is not a required second login.

Only if that fallback is needed, use `credentials --prompt NETLIFY_AUTH_TOKEN --json` on macOS.
The human enters the value in the private native dialog, never in arguments or chat. Their own
editor remains available if native entry is unavailable, unsupported, or preferred; follow
[How the human connects accounts](../SKILL.md#how-the-human-connects-accounts) for fallback,
replacement and cancellation.

Choose the exact site and team. `NETLIFY_ACCOUNT_ID` disambiguates teams. Site creation requires a specific approved team ID, a verified current Free plan and available documented site capacity. Paid, legacy Starter, unknown plans or unknown/exhausted capacity stop the automated write. Never upgrade or buy credits. [Login guide](https://docs.netlify.com/api-and-cli-guides/cli-guides/get-started-with-cli/), [Free plan](https://docs.netlify.com/manage/accounts-and-billing/billing/billing-for-credit-based-plans/credit-based-pricing-plans/).

## Environment and build boundary

Preview values target `deploy-preview`; production values target `production`. Public values use omitted scopes/Free defaults. Sensitive values remain write-only secrets with `builds`, `functions`, `runtime` scopes. Netlify's official Terraform provider explicitly documents this Free-plan exception; general documentation about paid custom scopes does not invalidate it. [Official env resource documentation](https://github.com/netlify/terraform-provider-netlify/blob/main/docs/resources/environment_variable.md), [implementation](https://github.com/netlify/terraform-provider-netlify/blob/main/internal/provider/environment_variable_resource.go).

Existing values are updated per context. Do not replace all contexts, widen scopes or downgrade an existing secret. Do not use `netlify api --data`, `env:set` argv or env-import files for secret transport; the adapter uses HTTPS bodies and headers and suppresses raw provider output.

Local Netlify builds see masked non-development secret values. Runtime-only server credentials fit this flow. If the application needs raw secrets during its build, explain the boundary and prepare a separately reviewed remote-build workflow; do not change secrets to readable values to get a passing build. [Secrets Controller](https://docs.netlify.com/build/environment-variables/secrets-controller/).

## Verify and report

The official CLI builds by default. Deployment targets the exact selected site with an explicit context; production adds `--prod`. Verify ready state, owning site ID and published production ID before reporting a URL. Only observed HTTPS Netlify default origins count; configured custom domains and guessed preview patterns do not.

Netlify may create projects with private visibility. The separate `netlify-public-access` check sends an anonymous homepage request after confirming a ready published production deployment. HTTP 401/403 or a Netlify access-control redirect produces a blocking exact-project handoff. Redirects are not followed, and 2xx proves only anonymous homepage access, not backend or UI correctness. Keep a successful deploy recorded as successful; visibility remediation does not require redeploying.

Open the exact project's **Project configuration → General → Visitor access → Project visibility**, and confirm its ID. If Netlify visitor protection blocks the intended public homepage, prepare **Private → Applies to: Previews only** to expose production while preserving private previews; the documented **Make public** action also preserves private previews. Do not select blanket Public or change team defaults. Show the exact change and get explicit approval before Save. Do not weaken application authentication or override a team-enforced restriction. Run `golive verify` afterward; only the passing anonymous check closes the handoff. No supported API, CLI or MCP visibility mutation exists. [Visibility documentation](https://docs.netlify.com/manage/security/secure-access-to-sites/project-visibility/), [official access-control guidance](https://github.com/netlify/context-and-tools/blob/main/skills/netlify-access-control/SKILL.md).

The official Netlify MCP exists, but is not used by the standalone adapter. Keep agent/plugin availability separate from adapter capability. [Netlify agent setup guides](https://docs.netlify.com/build/build-with-ai/agent-setup-guides/).

Report mock tests, source research and live evidence separately. Any live account write needs the exact golive plan approval first.
