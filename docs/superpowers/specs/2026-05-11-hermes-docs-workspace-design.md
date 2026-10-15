# Hermes Docs Workspace Design

Date: 2026-05-11

## Summary

Hermes Docs is a separate local document workspace app, paired with a lightweight Hermes Dashboard plugin. It gives users a Codex/Obsidian-style workspace for Markdown-first document work, live rich editing, document-aware side chat, comments, and document conversion through Kordoc.

The app is designed for users who want to pick any local folder, treat it as a document workspace, and keep durable workspace-specific chat, annotation, conversion, and editing history without writing application metadata into the selected business folder.

## Product Shape

The system has two surfaces:

- Hermes Docs app: the primary standalone local app for document work.
- Hermes Dashboard plugin: a lightweight launcher/status surface inside the existing dashboard.

The Dashboard plugin does not own the editing experience. It shows connection status, recent workspaces, and a shortcut to open the full Docs app. The standalone app owns the workspace drawer, editor, side chat, command bar, annotations, Kordoc actions, and onboarding/settings.

## Runtime And Distribution

Hermes Docs is a local-first web app, not a cloud-hosted document service.

Runtime shape:

- Web UI: browser-based app shell for the editor, side chat, workspace drawer, comments, and settings.
- Local backend: localhost service for file access, workspace metadata, Kordoc execution, Codex OAuth brokering, and Doc Agent execution.
- Dashboard plugin: a Hermes Dashboard tab that can show status and open the local web app.

Distribution shape:

- macOS users should receive a DMG installer for normal installation.
- The DMG installs the local Docs app, local backend launcher, Dashboard plugin assets, and any app-specific bundled resources.
- First launch runs onboarding for Codex OAuth, `docs` persona bootstrap, brainstorming availability, Kordoc verification, and first workspace selection.
- The app can still be developed as a web app, but production use is local-first and packaged for Mac users.

Pure cloud-web deployment is out of scope for the first implementation because the app needs trusted local folder access, local document conversion, and local OAuth/token brokering.

## Core Experience

Users can register any local folder as a workspace. Examples include:

- `급여대장`
- `월말결산`
- `Northpole Docs`

Each workspace shows a file tree and recent workspace session history. Workspace metadata is stored centrally under Hermes home, not inside the selected folder.

Storage root:

```text
~/.hermes/docs-workspaces/
  registry.json
  <workspace-id>/
    sessions/
    annotations/
    conversions/
    document-state.json
    preferences.json
```

The registered source folder remains clean unless the user explicitly saves or exports a document there.

## Main Editor Surface

The main editor surface is a live Markdown document surface, not a permanent split view.

Default behavior:

- A single rich document surface is shown.
- Markdown syntax is revealed only where helpful, such as heading tokens, source peek, or source mode.
- Source mode and reading mode are available as explicit toolbar modes.
- Text selection can create comments, send quoted text to side chat, or create a Doc Agent command.

The main editor surface must not show onboarding or install-flow content. Installation, OAuth, and setup status belong in onboarding/settings or the Dashboard plugin.

## Workspace Drawer

The left workspace UI has two states:

1. Unpinned overlay drawer
   - Only the narrow rail occupies layout space.
   - Hovering or clicking the rail opens the workspace drawer as an overlay above the editor.
   - The editor width does not change.
   - This is the default writing-focused state.

2. Pinned docked drawer
   - The user can pin the drawer open.
   - The drawer becomes a real docked column.
   - The editor width becomes narrower by the drawer width.
   - This is for file organization and reference-heavy work.

Pin state is persisted per user and may be overridden per workspace.

## Side Chat

The right side chat is a Codex-style document thinking space.

Its job is to help the user read, question, structure, and brainstorm around the current document. It should not directly mutate the document. It can produce:

- proposed comments
- suggested rewrites
- outline changes
- questions to ask
- claims to verify
- command drafts for the command bar

Side chat context includes:

- current workspace
- active document
- selected text
- linked file snippets or conversion outputs
- workspace session history
- unresolved annotations

The user can quote a selected block into side chat. Side chat results can be sent to annotations or the command bar.

## Command Bar

The bottom command bar is for document actions, not open-ended brainstorming.

Examples:

- rewrite selected block
- apply this suggested edit
- extract table 2 from the selected PDF
- convert this DOCX to Markdown
- add side-chat result as an annotation

Command bar actions may modify workspace state or create document edits. Risky actions should produce previewable changes before writing.

## Comments And Review

Users can select text and create comments. Comments are stored in the central workspace store and anchored to the document by a stable text-range strategy.

Minimum behavior:

- create comment from selection
- list unresolved comments
- resolve comment
- send comment context to side chat
- convert side-chat output into a suggested edit

Suggested edits should be previewed before applying to source files.

## Kordoc Integration

Kordoc is the document conversion and comparison layer.

Supported app actions:

- detect document format
- parse HWP/HWPX/PDF/DOCX/XLSX into Markdown
- extract page ranges
- extract tables
- compare documents
- parse forms
- fill supported forms where Kordoc can preserve the format

The Docs app should call Kordoc through a local trusted broker, not by exposing broad file access to browser code. Absolute file paths are used internally. Conversion artifacts are stored under the workspace's central `conversions/` folder.

Known caveat:

- Kordoc MCP support for `.xls` may be narrower than CLI/API support. The app should route `.xls` through a supported CLI/API path if MCP does not allow it.

## Authentication

The app should support one-time Codex OAuth setup during install or onboarding.

Target behavior on another Mac:

1. User installs Hermes Docs.
2. Installer checks for an existing Codex OAuth credential.
3. If missing, installer starts device-code OAuth.
4. The user authorizes once.
5. The local backend stores credentials in the local credential pool.
6. The Docs app can immediately run Doc Agent and side-chat actions.

Browser code must not receive or persist raw OAuth tokens. The local backend acts as the auth broker.

Provider priority:

1. OpenAI Codex OAuth
2. Hermes configured provider fallback
3. explicit API-key fallback when the user chooses it

The existing Hermes `openai-codex` provider and OAuth device-code flow are compatible with this direction.

## Docs Agent Persona

The install/onboarding flow prepares a `docs` agent persona.

The persona should specialize in:

- Markdown document editing
- comments and review
- brainstorming and structure
- document summarization
- claim checking prompts
- Kordoc conversion follow-up
- careful source-file edits with preview before apply

If the profile already exists, onboarding verifies it instead of overwriting user changes.

## Built-In Brainstorming

Brainstorming is a first-class workflow in the Docs app.

The app should expose brainstorming from:

- side chat mode selector
- new document flow
- selected text actions
- outline generation
- review question generation

Brainstorming results remain in workspace session history and can be converted to comments, outlines, or command-bar drafts.

## Design System

The final implementation must use the Mintlify design reference from getdesign.

Implementation setup must run from the app project root:

```bash
npx getdesign@latest add mintlify
```

The generated `DESIGN.md` is the source of truth for visual implementation.

Expected design direction:

- documentation-platform feel
- reading-optimized density
- strong light and dark theme parity
- restrained green accent for active states and primary actions
- clean typography suitable for long-form prose
- clear sidebar, rail, command bar, and side-chat hierarchy
- no marketing hero treatment inside the app

The existing mockups are only structural references. They are not final visual design.

## Error Handling

Workspace folder errors:

- Missing folder: show reconnect prompt and keep history.
- Permission denied: explain the blocked path and provide retry.
- File changed externally: show reload/merge choice.

Kordoc errors:

- Unsupported format: show supported alternatives.
- Parse warning: attach warnings to conversion result.
- Large file timeout: keep job record and allow retry.

Auth errors:

- No Codex OAuth: route to onboarding/login.
- Expired/invalid token: restart OAuth flow.
- Provider unavailable: offer configured fallback provider.

Write errors:

- Never silently overwrite source files.
- Preview edits before apply.
- Keep failed patch attempts in workspace session history.

## Testing And Verification

Targeted verification should cover:

- workspace registry create/list/remove
- central storage path isolation
- overlay drawer versus pinned drawer layout behavior
- file tree loading for allowed local folders
- live editor mode switching
- side chat quote-from-selection flow
- command bar action preview flow
- comment create/resolve persistence
- Kordoc conversion happy path and failure path
- Codex OAuth detection and onboarding state
- Dashboard plugin connection/status display
- light and dark theme rendering

Before implementation completion, verify the app visually in both light and dark themes and check that the unpinned drawer does not resize the editor.

## Out Of Scope For First Implementation

- collaborative multi-user editing
- remote cloud sync
- pure cloud-hosted web app deployment
- full Obsidian graph view
- arbitrary plugin marketplace
- direct browser access to raw OAuth tokens
- writing metadata into every registered workspace folder
- automatic destructive edits to source documents

## Open Design Decisions Resolved

- Use a separate Docs app plus Dashboard plugin, not a heavy built-in Dashboard page.
- Build it as a local-first web app and distribute it to Mac users with a DMG installer.
- Use central Hermes workspace metadata, not per-folder hidden metadata by default.
- Use live Markdown as the default editor, not permanent split view.
- Use side chat for thinking and command bar for actions.
- Use overlay drawer by default and docked drawer only when pinned.
- Keep install/auth UI out of the main editor surface.
