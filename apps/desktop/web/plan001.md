# Hermes Desktop Web — Implementation Plan 001

> **For Hermes:** Implement this plan task-by-task with the existing Hermes Desktop build/test workflow. Do not copy or edit the Desktop renderer source. Keep the browser host adapter inside `apps/desktop/web`.

**Goal:** Make `hermes desktop-web` serve the complete Hermes Desktop React UI in a normal browser, using the existing Hermes Desktop/Vite/Tailwind dependency graph, replacing Electron’s `window.hermesDesktop` bridge with a browser adapter, and managing a dedicated headless Hermes backend child with the same lifecycle model as native Desktop.

**Architecture:** `apps/desktop/web` is a standalone web host around the existing Desktop renderer. Its entrypoint installs a browser implementation of `window.hermesDesktop` before dynamically importing the untouched `apps/desktop/src/main.tsx`; the adapter forwards backend-owned operations through the standalone host to the backend selected in Desktop Web settings. The web command is independent from `hermes dashboard`: both may run in parallel, and Desktop Web owns only its own UI process and port `13043`.

**Tech Stack:** Existing Desktop package dependencies only: React, React DOM, Vite, Tailwind CSS 4, React Compiler/Babel plugin, TypeScript, Vitest, and existing Hermes shared modules. No new runtime dependency, server framework, Node HTTP framework, PTY package, Electron package, or alternate frontend stack.

---

## 0. Non-negotiable scope and current repository state

### Allowed implementation area

The implementation files for this plan are limited to:

- `apps/desktop/web/**`
- `apps/desktop/tsconfig.web.json`

Do not edit:

- `apps/desktop/src/**`
- `apps/desktop/electron/**`
- the Desktop renderer `global.d.ts`
- Electron preload/main files
- Hermes Python runtime files
- Dashboard route implementation
- root dependency versions or unrelated packages

The upstream Desktop renderer remains authoritative and is imported in place. Do not copy `apps/desktop/src` into `apps/desktop/web`.

### CLI integration constraint

The branch already contains a `hermes_cli/subcommands/gui_web.py` parser and a `cmd_gui_web` implementation outside the allowed web folder. The current parser registers `desktop-web` with `gui-web` as an alias, but its options currently resemble native Desktop options (`--source`, `--build-only`, `--fake-boot`, etc.) rather than the requested Dashboard-style `--host`, `--port`, `--status`, `--no-open`, `--stop`, and `--skip-build` surface.

This plan assumes one of these is true before implementation begins:

1. the existing CLI glue is updated in a separately authorized change to pass the requested flags into the web host; or
2. the CLI glue already evolves to the required contract without changes from this plan.

If neither is true, the exact CLI acceptance criteria cannot be met while modifying only `apps/desktop/web` and `tsconfig.web.json`. Do not work around this by parsing unrelated process arguments from the browser bundle or by modifying Desktop `src`.

### Existing local state to preserve

The branch currently has staged/unstaged work in `apps/desktop/web`, including a web `preload.ts`, Vite config, package manifest, and deleted/replaced draft files. Before implementation:

```bash
git status --short
git diff -- apps/desktop/web apps/desktop/tsconfig.web.json
git diff --cached -- apps/desktop/web apps/desktop/tsconfig.web.json
```

Do not discard or restore unrelated user changes. Reconcile the existing web draft deliberately.

---

## 1. Behavioral contract

### 1.1 Renderer ownership

`apps/desktop/src/main.tsx` and all of its imported modules remain the UI authority. The web entrypoint must:

1. load the browser bridge;
2. install `window.hermesDesktop` exactly once;
3. dynamically import `../../src/main.tsx` or the equivalent absolute Desktop source entry;
4. avoid a static import that can evaluate Desktop code before the bridge is installed;
5. import no Electron module and no Node-only module into the browser graph.

The Desktop renderer must continue to own:

- layout and navigation;
- sessions and transcript rendering;
- profile and connection UI;
- project/file sidebar;
- settings panels;
- Git/review UI;
- themes and responsive behavior;
- gateway event rendering;
- loading/error states.

The web host must not create a second chat application or duplicate renderer state.

### 1.2 Backend ownership

Hermes Agent/Dashboard remains authoritative for:

- sessions and durable session IDs;
- live runtime IDs and gateway events;
- profile data;
- model/provider configuration;
- tool execution;
- filesystem policy and path validation;
- Git operations on the backend workspace;
- authentication and WebSocket ticket issuance;
- agent execution and streaming.

The browser adapter is a transport and host-capability adapter, not an agent implementation.

### 1.3 Default runtime

The command must use these defaults:

```text
host: 127.0.0.1
port: 13043
```

It starts a dedicated `hermes serve --host 127.0.0.1 --port 0 --isolated` child for each Desktop Web invocation. The child uses an OS-assigned private loopback port; Desktop Web exposes only port `13043`. The address `127.0.0.1:13001` may be used for local testing on this machine, but must not be hardcoded as the Desktop Web backend default. Desktop Web settings remain the renderer-facing configuration surface for connection/profile behavior.

### 1.4 Dashboard-style lifecycle flags

The final command contract is:

```text
hermes desktop-web [--host HOST] [--port PORT]
                    [--status]
                    [--no-open]
                    [--stop]
                    [--skip-build]
```

Expected behavior:

| Flag | Required behavior |
|---|---|
| no flags | build if required, start web host on `127.0.0.1:13043`, open browser unless disabled by environment/CLI policy |
| `--host` | bind the web host to the requested host; preserve Dashboard authentication requirements for non-loopback binds |
| `--port` | bind the web host to the requested port; reject invalid or privileged values with the same style of error as Dashboard |
| `--status` | report the running Desktop Web process/listener and exit without building or opening a browser |
| `--no-open` | start/serve without opening a browser |
| `--stop` | stop only the matching Desktop Web process, never arbitrary Dashboard/Serve/Electron processes |
| `--skip-build` | serve the existing web build; fail clearly if the required `dist` does not exist |

Do not silently treat `desktop-web` as native Electron Desktop. It must never invoke `electron`, `node-pty`, native updater, native windows, or the packaged Desktop executable.

---

## 2. Web host layout

The target layout is:

```text
apps/desktop/
├── src/                         # authoritative, untouched Desktop renderer
├── electron/                    # authoritative, untouched Electron host
├── tsconfig.web.json            # web-only typecheck project
└── web/
    ├── AGENTS.md
    ├── README.md
    ├── package.json              # uses existing Desktop dependency versions
    ├── vite.config.ts
    ├── index.html
    ├── src/
    │   ├── main.ts               # bridge-first browser bootstrap
    │   ├── preload.ts            # browser window.hermesDesktop implementation
    │   ├── unsupported.ts         # controlled unsupported capability errors
    │   ├── api.ts                 # typed fetch/error/auth helper if extracted
    │   ├── gateway.ts             # browser WebSocket/ticket adapter if extracted
    │   ├── files.ts               # browser file picker/upload adapter if extracted
    │   └── *.test.ts              # web adapter tests
    └── dist/                     # generated; ignored by Git if repository policy requires
```

Keep modules small. Extract pure URL, auth, error, response-unwrapping, and capability functions instead of placing all logic in one `preload.ts`.

---

## 3. Build-tool integration

### Task 1: Reconcile the web package manifest

**Objective:** Make the web package use the exact dependency versions already used by Desktop without adding a second toolchain.

**Files:**

- Modify: `apps/desktop/web/package.json`
- Modify: `apps/desktop/tsconfig.web.json`

**Requirements:**

- Keep the existing Desktop versions for React, React DOM, Vite, Tailwind, React Compiler, TypeScript, Vitest, and UI dependencies.
- Do not add a web server framework or a second HTTP implementation.
- Keep `build`, `dev`, `preview`, and `typecheck` scripts aligned with the Desktop workspace conventions.
- Add only scripts that can run with already-declared tools, such as `test` if Vitest is already available from the Desktop workspace.
- Keep `node-pty`, Electron Builder, Electron, and other native-only dependencies out of the web bundle. If they are inherited from the workspace package graph, explicitly exclude them from browser resolution rather than importing them.

**Verification:**

```bash
cd apps/desktop/web
npm run typecheck
```

Expected: no web-project TypeScript errors.

### Task 2: Complete `tsconfig.web.json`

**Objective:** Typecheck the browser host and authoritative shared types without typechecking Electron or the entire native Desktop host.

**Requirements:**

- Keep the config at `apps/desktop/tsconfig.web.json`.
- Include `web/src/**` and only the shared source files required by the web adapter.
- Include the Desktop `src/global.d.ts` type declarations for the bridge contract, but do not include `apps/desktop/electron/**`.
- Preserve aliases needed by the imported Desktop renderer:
  - `@/*` → `apps/desktop/src/*`
  - `@desktop/*` → `apps/desktop/src/*`
  - `@hermes/shared` and required subpaths → `apps/shared/src/**`
- Use the same module/module-resolution/JSX settings as Desktop wherever compatible.
- Keep browser libs (`DOM`, `DOM.Iterable`) and Node types only where Vite config/server tooling requires them; do not make browser source depend on Node globals.
- Ensure `rootDir`/`outDir` do not make generated declarations overwrite renderer or Electron output.

**Verification:**

```bash
cd apps/desktop
npx tsc --noEmit -p tsconfig.web.json
```

Expected: only web and explicitly included shared/type declarations are checked.

---

## 4. Direct renderer reuse

### Task 3: Implement bridge-first bootstrap

**Objective:** Start the untouched Desktop UI after installing the browser bridge.

**Files:**

- Modify/create: `apps/desktop/web/index.html`
- Modify/create: `apps/desktop/web/src/main.ts`

**Implementation shape:**

```ts
import { installWebHermesDesktop } from './preload'

installWebHermesDesktop()
await import('../../src/main')
```

Use a dynamic import or an equivalent guaranteed sequencing mechanism. Do not edit `apps/desktop/src/main.tsx` to accommodate the browser.

The bootstrap must:

- preserve the Desktop renderer’s own `styles.css` import;
- preserve Desktop’s asset paths and font loading;
- avoid a second React root;
- avoid importing Electron preload code;
- report initialization failures through the Desktop renderer’s existing error boundary or a minimal in-app error, not `alert()`.

### Task 4: Align Vite configuration with Desktop

**Objective:** Make Vite compile the full Desktop renderer source in place using the same aliases, React Compiler, Tailwind, asset handling, and dependency resolution conventions.

**Files:**

- Modify: `apps/desktop/web/vite.config.ts`

**Requirements:**

- Use existing Desktop Vite plugins and versions:
  - `@vitejs/plugin-react`
  - `@rolldown/plugin-babel`
  - `babel-plugin-react-compiler`
  - `@tailwindcss/vite`
- Import source directly from `apps/desktop/src` via aliases; never copy it.
- Permit Vite filesystem access to the repository root, Desktop source, shared source, and package locations only as required.
- Resolve one React/React DOM installation to avoid duplicate React runtime errors.
- Preserve Desktop’s special handling for dynamic `driver.js` raw imports, emoji assets, fonts, and any existing asset transforms that are necessary for the full renderer.
- Exclude Electron/Node-only imports from the browser graph. If a Desktop source module imports an Electron-only capability dynamically, provide an alias or browser-safe build branch only inside the web Vite configuration/adapter boundary; do not edit the source module.
- Keep browser base path `/` for the initial milestone unless the existing CLI contract already supplies a configurable base path.
- Use an empty explicit PostCSS plugin list if that is the Desktop build’s established Tailwind 4 pattern.

**Verification:**

```bash
cd apps/desktop/web
npm run build
```

Expected: the complete Desktop renderer transforms and produces `dist/index.html` plus assets without Electron or Node runtime imports in browser chunks.

---

## 5. Browser `window.hermesDesktop` contract

### Task 5: Build a typed request/error/auth adapter

**Objective:** Centralize HTTP behavior so all bridge methods preserve Dashboard authentication and safe error semantics.

**Files:**

- Create/modify: `apps/desktop/web/src/api.ts`
- Test: `apps/desktop/web/src/api.test.ts`

**Requirements:**

- Use same-origin `fetch()` with `credentials: 'include'`.
- Preserve HTTP methods and JSON body shapes exactly.
- Support JSON and multipart upload requests using browser `FormData`.
- Never expose cookies, tokens, API keys, or response bodies containing credentials to logs or UI errors.
- Parse JSON only when JSON is present; turn HTML reverse-proxy/login responses into safe status errors rather than JSON parse noise.
- Treat `401/403` as authentication failures and navigate only once to `/login?next=...`.
- Do not redirect repeatedly when concurrent startup requests receive `401`.
- Preserve structured status codes and generic error categories for the renderer.
- Apply profile and connection identity as explicit query/body fields according to the existing Desktop/API contract; never infer identity from display labels.
- Reject unsafe cross-origin state-changing requests at the host server boundary.

**Tests:**

- JSON success;
- empty success;
- JSON error;
- HTML error;
- one-time 401 redirect;
- profile and connection scope preservation;
- request timeout/abort classification;
- upload body construction without logging bytes.

### Task 6: Implement Gateway WebSocket parity

**Objective:** Make the renderer’s existing Gateway client connect to the authenticated backend through same-origin WebSocket transport.

**Files:**

- Create/modify: `apps/desktop/web/src/gateway.ts` or `preload.ts`
- Test: `apps/desktop/web/src/gateway.test.ts`

**Requirements:**

- Mint a fresh WebSocket ticket for OAuth/cookie-auth connections through the Dashboard ticket endpoint.
- Construct `ws://`/`wss://` from the current browser origin, preserving any configured path prefix.
- Attach `ticket`, `profile`, and `connectionId` only where the backend contract expects them.
- Never cache one-time tickets for reconnects.
- Distinguish `401/403` reauthentication from timeout/network/5xx transport failures.
- Preserve the Desktop renderer’s Gateway RPC framing, request IDs, event handling, timeout behavior, cancellation, and reconnect semantics. Do not create a second Gateway protocol implementation if the shared/Desktop client already provides one.
- Ensure a same-origin reverse proxy forwards `/api` WebSocket upgrades to the Dashboard backend.

**Tests:**

- ticket URL construction;
- fresh ticket on every dial;
- profile/connection routing;
- auth rejection classification;
- transient retry classification;
- malformed ticket response;
- reconnect cleanup and stale pending request handling.

### Task 7: Implement connection/profile registry parity

**Objective:** Preserve Desktop’s connection registry behavior in browser-compatible form while keeping secrets server-side.

**Files:**

- Modify/create: `apps/desktop/web/src/preload.ts`
- If extraction is needed: `apps/desktop/web/src/connection-registry.ts`
- Test: `apps/desktop/web/src/connection-registry.test.ts`

**Requirements:**

- Support the Desktop connection kinds relevant to the web host: local, remote, and cloud-shaped remote entries. Treat SSH as unsupported until a real server-side forwarding path exists.
- Keep stable connection IDs and profile names separate.
- Preserve duplicate-profile disambiguation using source identity, not only profile name.
- Return safe registry DTOs without token values or secret headers.
- Keep browser-entered credentials out of `localStorage`; use authenticated server-side storage or require re-entry after restart.
- Fail closed on unknown explicit `connectionId`; never silently fall back to local.
- Validate HTTP(S) URLs and reject embedded user/password credentials.
- Apply the upstream forbidden-header policy to custom forwarded headers:
  - reject `authorization`, `cookie`, `origin`, `host`, `connection`, `upgrade`, framing headers, and Hermes session-token overrides;
  - accept only valid HTTP header names and bounded values.
- Keep connection apply/re-home semantics aligned with Desktop: clear/reconnect gateway-bound state, do not hard reload for an ordinary connection switch.

**Tests:**

- local fallback;
- duplicate labels/IDs;
- duplicate profile names across sources;
- unknown connection failure;
- URL validation;
- forbidden header filtering;
- safe registry serialization;
- connection/profile route generation.

---

## 6. Filesystem, upload, Git, and browser capabilities

### Task 8: Implement filesystem bridge methods through Dashboard routes

**Objective:** Preserve the Desktop project tree/editor behavior against the VPS filesystem.

**Files:**

- Modify: `apps/desktop/web/src/preload.ts`
- Test: `apps/desktop/web/src/files.test.ts`

**Functional methods:**

```text
readDir             → GET /api/fs/list
readFileText        → GET /api/fs/read-text
readFileDataUrl     → GET /api/fs/read-data-url
readFileDataUrlForAttach → GET /api/fs/read-data-url or bounded upload fallback
writeTextFile       → POST /api/fs/write-text
gitRoot             → GET /api/fs/git-root
settings.default cwd → GET /api/fs/default-cwd
```

**Requirements:**

- Preserve renderer result shapes exactly: `entries`, `path`, `text`, `binary`, `byteSize`, `mimeType`, `language`, and `truncated` where applicable.
- Do not read arbitrary browser/VPS paths directly from client code; every VPS filesystem read goes through authenticated Dashboard routes.
- Preserve server-side path sandboxing, sensitive-file checks, write-size limits, parent-directory requirements, and atomic write behavior.
- Do not use browser `FileSystemHandle` as a replacement for the VPS project tree.
- For browser-selected local files, use a browser file input and short-lived in-memory objects; do not fabricate OS paths.
- Use a bounded cleanup strategy for browser-selected file objects.
- Implement `/api/fs/download` for large downloads instead of forcing all files through data URLs when the renderer can consume a download path.

**Tests:**

- directory result shape;
- hidden directory behavior delegated to backend;
- text result shape;
- binary/truncated fields;
- write request shape;
- browser file selection cancellation;
- no fabricated local absolute paths;
- sensitive/path errors remain generic.

### Task 9: Implement Git/review bridge through existing Dashboard routes

**Objective:** Make Desktop’s coding rail, review pane, worktree, branch, and diff surfaces work against the VPS repository.

**Files:**

- Modify: `apps/desktop/web/src/preload.ts`
- Test: `apps/desktop/web/src/git.test.ts`

**Routes and bridge mappings:**

```text
GET  /api/git/status
GET  /api/git/worktrees
GET  /api/git/branches
GET  /api/git/base-branches
GET  /api/git/review/list
GET  /api/git/review/diff
GET  /api/git/file-diff
GET  /api/git/review/commit-context
GET  /api/git/review/rev-parse
GET  /api/git/review/ship-info
POST /api/git/review/pr-list
POST /api/git/review/stage
POST /api/git/review/unstage
POST /api/git/review/revert
POST /api/git/review/commit
POST /api/git/review/push
POST /api/git/review/create-pr
POST /api/git/worktree/add
POST /api/git/worktree/remove
POST /api/git/branch/switch
```

**Requirements:**

- Unwrap Dashboard envelopes to the exact Electron renderer contract:
  - `{ worktrees }` → `HermesGitWorktree[]`;
  - `{ branches }` → branch arrays;
  - `{ diff }` → string;
  - `{ sha }` → nullable SHA.
- Preserve body names exactly: `path`, `file`, `message`, `push`, `worktreePath`, `force`, `branches`, `numbers`, `name`, `branch`, `base`, and `existingBranch`.
- Keep mutations explicit and erroring; never silently downgrade a failed write to a read or alternate repository.
- Ensure Git paths are scoped to the authenticated backend and validated by Dashboard code.
- Do not use `simple-git` in the browser. It remains server-side.
- Keep PR-comment resolution disabled unless a real Dashboard route is added in a separately authorized backend change.

**Tests:**

- every supported endpoint’s method/path/body;
- envelope unwrapping;
- null/empty responses;
- mutation errors;
- no unsupported fake endpoint;
- path/profile/connection scope preservation.

### Task 10: Implement browser-safe host capabilities

**Objective:** Map safe Electron capabilities to browser APIs and make unsupported features non-disruptive.

**Mappings:**

| Desktop capability | Web behavior |
|---|---|
| clipboard read/write | `navigator.clipboard`, with permission errors returned normally |
| notifications | Notification API, permission-aware and non-blocking |
| external links | `window.open` with `noopener,noreferrer` |
| session window | browser tab/window or same-app route |
| image download | Blob/object URL download |
| microphone | browser `getUserMedia`, permission errors handled by UI |
| zoom | browser/local UI zoom state only |
| native titlebar/translucency | disabled/opaque browser mode |
| HUD/pet overlay | disabled or ordinary in-page surface; no repeated errors on startup |
| native dialogs | browser file input/download behavior |
| OS reveal/trash/rename | disabled until server route/browser UX exists |
| native updater/uninstall/bootstrap | disabled; deployment owns lifecycle |
| Electron find-in-page | browser-native find or renderer-local behavior; no Electron call |
| `readWindowBelow` | unsupported, controlled rejection only when explicitly invoked |
| PTY terminal | unsupported in this plan; no arbitrary command endpoint |

Use the existing Desktop renderer’s feature detection wherever it already supports optional capabilities. Do not make an optional native capability look available merely by defining a method that throws during startup.

---

## 7. Build-time browser compatibility

### Task 11: Remove Electron/Node leakage from the browser graph

**Objective:** Prove that the full imported Desktop UI can build without resolving native runtime modules.

**Files:**

- Modify: `apps/desktop/web/vite.config.ts`
- Modify: `apps/desktop/web/src/preload.ts`
- Do not modify: `apps/desktop/src/**`

**Procedure:**

1. Build the full graph.
2. Inspect Vite/Rolldown errors for `electron`, `node:fs`, `node:child_process`, `node-pty`, `simple-git`, and OS-only modules.
3. Trace each import to the renderer consumer.
4. Prefer an existing optional capability guard or build alias.
5. If a source module is unavoidably native-only, ensure it is not imported by the browser entry or is replaced by a web-only alias in Vite.
6. Do not add a broad fake Node polyfill; it can make native code appear runnable and conceal broken behavior.

**Verification:**

```bash
cd apps/desktop/web
npm run build
find dist -type f -name '*.js' -print0 | xargs -0 grep -Il "from ['\"]electron\|node:fs\|node:child_process\|node-pty" || true
```

Expected: build succeeds and browser chunks contain no native import dependency.

---

## 8. CLI/lifecycle integration contract

### Task 12: Connect the existing CLI glue to the web host

**Objective:** Make the existing `cmd_gui_web` path behave like Dashboard while keeping web implementation under `apps/desktop/web`.

**Scope note:** This requires the already-existing CLI integration outside the allowed web folder to pass runtime arguments or invoke the web package consistently. If changes to `hermes_cli/subcommands/gui_web.py` or `hermes_cli/main.py` are needed, stop and obtain explicit scope authorization; do not modify them under this plan’s current restriction.

**Required CLI behavior:**

- resolve the project’s existing Node/npm runtime exactly as Desktop/Dashboard does;
- run the web package’s existing dependency/build command unless `--skip-build` is set;
- use the same deterministic install/build behavior as Desktop;
- launch the web host with explicit `--host` and `--port` values;
- implement `--status`, `--stop`, and `--no-open` with the same process-safety rules as Dashboard;
- never kill a process solely because it happens to listen on the requested port without matching the Desktop Web command identity;
- never kill the existing Hermes backend at `13001` when stopping Desktop Web;
- open the browser only after the web listener is ready;
- preserve non-loopback authentication enforcement.

**Verification:**

```bash
hermes desktop-web --help
hermes desktop-web --status
hermes desktop-web --no-open --skip-build --host 127.0.0.1 --port 13043
hermes desktop-web --status
hermes desktop-web --stop
hermes desktop-web --status
```

Expected: help shows the requested Dashboard-style flags; start/status/stop affect only Desktop Web.

---

## 9. Testing strategy

### Task 13: Add web adapter contract tests

**Objective:** Test behavior rather than source text or snapshots.

**Files:**

- Create/modify: `apps/desktop/web/src/*.test.ts`
- Modify: `apps/desktop/web/package.json` only if a test script is missing

Required test groups:

1. **API transport**
   - JSON success/error;
   - HTML reverse-proxy failure;
   - 401 single-flight redirect;
   - timeout and abort;
   - credentials include;
   - profile/connection scope.
2. **Gateway URL/ticket**
   - `ws` vs `wss`;
   - path prefixes;
   - fresh ticket behavior;
   - auth vs transient errors.
3. **Bridge capabilities**
   - core methods exist;
   - unsupported native methods fail only when invoked;
   - startup-safe subscriptions are no-ops;
   - clipboard/notification/download browser paths are delegated.
4. **Filesystem**
   - exact route and response shape;
   - browser file inputs do not expose fake absolute paths.
5. **Git**
   - exact route/body mappings;
   - envelope unwrapping;
   - mutation failure behavior.
6. **Connection registry**
   - identity, scope, safe DTO, header filtering, fail-closed lookup.

Do not write tests that grep or read source files to prove implementation shape. Test exported pure helpers or mock `fetch`/WebSocket behavior.

### Task 14: Add real build and runtime smoke tests

**Objective:** Exercise the actual imported Desktop graph and command path.

Required checks:

```bash
cd apps/desktop/web
npm run typecheck
npm test
npm run build
```

Then from repository root:

```bash
hermes desktop-web --no-open --skip-build --host 127.0.0.1 --port 13043
curl -fsS http://127.0.0.1:13043/healthz
curl -I http://127.0.0.1:13043/
hermes desktop-web --stop
```

If the existing Dashboard does not expose `/healthz`, the web host must provide a minimal local health endpoint or the plan must use the exact Dashboard-compatible status probe; do not invent an unauthenticated endpoint that exposes configuration or secrets.

Browser smoke acceptance:

- page loads from `127.0.0.1:13043`;
- unauthenticated access reaches the Dashboard login boundary;
- no Electron/Node bundle error;
- no repeated startup `alert()`;
- no uncaught page exception before authentication;
- fonts/assets load;
- authenticated run verifies the real Desktop shell, profile/session list, Gateway WebSocket, chat send/stream, project tree, file read, and Git status.

A test using an authenticated browser context must obtain that context interactively or from an explicitly supplied test state. Never guess or persist credentials.

---

## 10. Security model

### Browser boundary

- Browser JavaScript receives only authenticated API results and safe DTOs.
- No arbitrary VPS filesystem access from client-side Node APIs.
- No arbitrary command execution endpoint.
- No token/API key in browser localStorage.
- No token/API key in logs, error messages, source maps, test fixtures, or screenshots.

### HTTP host

- Loopback default `127.0.0.1:13043`.
- Non-loopback binds retain Dashboard auth requirements.
- Same-origin state-changing request checks.
- No permissive CORS added for convenience.
- WebSocket upgrades restricted to the intended `/api` route and authenticated backend path.
- Static file paths normalized and prevented from escaping the web `dist` root.
- Request bodies bounded before proxying/uploading.
- Login redirect is single-flight and does not leak the original request body.

### Backend routes

- Continue using Dashboard’s `_fs_path` and file-size/sensitive-path checks.
- Continue using Dashboard Git route validation.
- Preserve profile and connection identity at every API/WebSocket boundary.
- Treat failed authoritative writes as errors; never retry them against another backend.
- Keep transient retry logic separate from 401/403 reauthentication logic.

### Secrets

The browser web host should not claim Electron `safeStorage` support. If multi-connection token authentication is implemented later, use a server-side secret manager or a deliberately documented encrypted store. Do not silently persist plaintext tokens merely because the browser cannot access an OS keychain.

---

## 11. Documentation and PR readiness

### Task 15: Document the contributor/deployment contract

**Files:**

- Modify: `apps/desktop/web/README.md`
- Modify: `apps/desktop/web/AGENTS.md` only if it is intentionally empty/incomplete

Document:

- why `apps/desktop/src` is imported directly;
- why Electron preload cannot be reused in the browser;
- bridge-first bootstrap ordering;
- supported/unsupported bridge methods;
- Dashboard HTTP/WebSocket routes used;
- default command and port `13043`;
- all CLI flags;
- `--skip-build` behavior;
- local development commands;
- authentication and reverse-proxy expectations;
- post-login browser verification procedure;
- upstream source update procedure;
- no-copy/no-edit source rule;
- how to run the web-specific typecheck and tests;
- how to avoid changing Desktop `src` to solve web-host issues.

### Task 16: Add source provenance and update workflow

**Objective:** Make future upstream changes reviewable and prevent silent contract drift.

**Requirements:**

- Record the tested Hermes source commit in the web README or a web-local compatibility metadata file.
- Add a check that the expected source roots exist and that the web build points at the live Desktop source.
- Do not freeze mutable model/catalog lists in tests.
- When upstream `global.d.ts` or `preload.ts` changes, compare the bridge contract before accepting the new revision.
- Require a fresh build, web typecheck, adapter tests, and authenticated smoke test before updating the compatibility record.

---

## 12. Acceptance matrix

| Area | Acceptance criterion | Evidence |
|---|---|---|
| Source reuse | Full UI imports from `apps/desktop/src` in place | Vite config + build graph; no copied renderer tree |
| Source safety | Desktop `src` and Electron files unchanged | `git diff -- apps/desktop/src apps/desktop/electron` empty |
| Toolchain | Uses existing Desktop Vite/Tailwind/React Compiler/TS/Vitest | package/config comparison and successful build |
| Bootstrap | Bridge installed before Desktop `main.tsx` evaluates | browser smoke and source-level runtime test seam |
| UI | Complete Desktop shell renders, not a reduced chat clone | authenticated browser smoke |
| Auth | Login/session cookies and WS ticket flow work | API + browser auth tests |
| Gateway | Chat/session streaming uses the real Hermes Gateway | authenticated WebSocket/chat run |
| Profiles | Profile/connection identity does not bleed across sources | scoped registry/Gateway tests |
| Files | Project tree/read/write uses validated Dashboard routes | filesystem integration tests |
| Git | Review/worktree/branch route shapes match Electron contract | Git adapter tests and real repo smoke |
| Native features | Unsupported features degrade without startup alerts/crashes | unsupported capability tests/browser console |
| Security | No arbitrary command route, unsafe header/path escape, or credential leak | security tests and manual audit |
| CLI | `hermes desktop-web` has Dashboard-style lifecycle flags | CLI help/status/start/stop evidence |
| Default port | Default listener is `127.0.0.1:13043` | listener and health output |
| Build skipping | `--skip-build` uses existing `dist` and fails clearly if absent | two CLI runs |
| Upstream | Tested revision is recorded and guarded | compatibility check |
| PR hygiene | Only allowed files changed for this plan | `git status`/diff review |

---

## 13. Final command sequence

Run the exact sequence before declaring the PR-ready implementation complete:

```bash
# Inspect scope first
git status --short
git diff -- apps/desktop/src apps/desktop/electron

# Existing Desktop workspace dependencies; do not add a second toolchain
npm install

# Web package checks
cd apps/desktop/web
npm run typecheck
npm test
npm run build
cd ../..

# CLI contract
hermes desktop-web --help
hermes desktop-web --status
hermes desktop-web --no-open --skip-build --host 127.0.0.1 --port 13043
hermes desktop-web --status
curl -fsS http://127.0.0.1:13043/healthz
hermes desktop-web --stop
hermes desktop-web --status

# Verify source boundaries and final state
git diff --check
git status --short
git diff --stat -- apps/desktop/web apps/desktop/tsconfig.web.json
```

Expected final result:

- web typecheck passes;
- web tests pass;
- full Desktop renderer build passes;
- CLI help exposes the required lifecycle flags;
- web listens on `127.0.0.1:13043` by default;
- status/stop do not affect the existing Hermes backend;
- unauthenticated browser flow reaches login cleanly;
- authenticated browser flow verifies the real Desktop shell and Gateway;
- `apps/desktop/src` and `apps/desktop/electron` remain untouched;
- no new dependency or unrelated file change is introduced.

---

## Known deliberate limitations for the first web milestone

These are not bugs unless the project scope explicitly expands:

- Native Electron HUD and pet overlay windows are unavailable.
- Global OS shortcuts are unavailable.
- Native OS file manager reveal/trash/rename dialogs are unavailable.
- Electron updater/bootstrap/uninstall are unavailable; VPS/service deployment owns lifecycle.
- Embedded PTY terminal is unavailable until a separate authenticated server-side PTY design is approved.
- SSH connection forwarding is unavailable until a server-side tunnel/service is designed.
- Electron-specific PR-comment `gh` operations are unavailable unless a matching authenticated Dashboard route exists.
- Browser-selected files are browser objects, not absolute OS paths.

The goal is complete Desktop UI reuse with honest browser-host capability boundaries, not a fake Electron runtime.
