# Creative Tools

`{workflow.creative_tools}` is a registry of collaborative renderers invoked on demand when seeing options helps the user decide. Entries follow the standard prefix convention: `skill:NAME`, `file:PATH`, `tool:MCP_TOOL_NAME: <directive>`, or plain-text directive.

Defaults ship for HTML color themes, HTML design directions, Excalidraw wireframes (Discovery), and 1:1 HTML key-screen mocks (Finalize). Teams append more via override TOML — Figma MCP, custom skills, prompt-based mood boards.

## When to invoke

Decision moments where a visual beats more conversation: picking color tokens, picking a visual personality among directions, sketching IA, mocking a tricky flow. Fast-path users typically skip; coaching-path users typically lean in. Read the room.

## Artifact handling

Every renderer writes to `{doc_workspace}/.working/` with a descriptive filename. `.working/` is the audit trail and survives the run. At Finalize, the facilitator walks `.working/` with the user and promotes artifacts with lasting reference value to `{doc_workspace}/mockups/` (HTML anchoring a brand or layout decision) or `{doc_workspace}/wireframes/` (Excalidraw a dev would glance at). Bar for promotion: *would a future reader of `DESIGN.md` or `EXPERIENCE.md` open this?* Default is leave-in-`.working/`.

## Renderer contract

The parent passes the subagent: current `.memlog.md`, relevant prior `.working/` captures, the user's stated intent for this pass, the output path. The subagent writes its artifact under `.working/` and returns ONLY a compact summary (file path, one line per variant, mode coverage). Parent never holds the full payload.

For HTML, open in browser when interactive: `python3 -c "import webbrowser, pathlib; webbrowser.open(pathlib.Path('PATH').resolve().as_uri())"`. Skip in headless.
