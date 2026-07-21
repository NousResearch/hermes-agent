---
name: dogfood
description: "Exploratory QA of web apps: find bugs, evidence, reports."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [qa, testing, browser, web, dogfood]
    related_skills: []
---

# Dogfood: Systematic Web Application QA Testing

## Overview

This skill guides you through systematic exploratory QA testing of web applications using the browser toolset. You will navigate the application, interact with elements, capture evidence of issues, and produce a structured bug report.

## Prerequisites

- Browser toolset must be available (`browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_vision`, `browser_console`, `browser_scroll`, `browser_back`, `browser_press`)
- A target URL and testing scope from the user
- Terminal access is strongly recommended when dogfooding public endpoints, DNS,
  TLS, redirects, callbacks, or webhooks. Browser-only failures can be caused by
  resolver cache, proxy behavior, or client state rather than the app itself.

## Inputs

The user provides:
1. **Target URL** — the entry point for testing
2. **Scope** — what areas/features to focus on (or "full site" for comprehensive testing)
3. **Output directory** (optional) — where to save screenshots and the report (default: `./dogfood-output`)

## Workflow

Follow this 5-phase systematic workflow:

### Phase 1: Plan

1. Create the output directory structure:
   ```
   {output_dir}/
   ├── screenshots/       # Evidence screenshots
   └── report.md          # Final report (generated in Phase 5)
   ```
2. Identify the testing scope based on user input.
3. Build a rough sitemap by planning which pages and features to test:
   - Landing/home page
   - Navigation links (header, footer, sidebar)
   - Key user flows (sign up, login, search, checkout, etc.)
   - Forms and interactive elements
   - Edge cases (empty states, error pages, 404s)

### Phase 2: Explore

For each page or feature in your plan:

1. **Navigate** to the page:
   ```
   browser_navigate(url="https://example.com/page")
   ```

2. **Take a snapshot** to understand the DOM structure:
   ```
   browser_snapshot()
   ```

3. **Check the console** for JavaScript errors:
   ```
   browser_console(clear=true)
   ```
   Do this after every navigation and after every significant interaction. Silent JS errors are high-value findings.

4. **Take an annotated screenshot** to visually assess the page and identify interactive elements:
   ```
   browser_vision(question="Describe the page layout, identify any visual issues, broken elements, or accessibility concerns", annotate=true)
   ```
   The `annotate=true` flag overlays numbered `[N]` labels on interactive elements. Each `[N]` maps to ref `@eN` for subsequent browser commands.

5. **Test interactive elements** systematically:
   - Click buttons and links: `browser_click(ref="@eN")`
   - Fill forms: `browser_type(ref="@eN", text="test input")`
   - Test keyboard navigation: `browser_press(key="Tab")`, `browser_press(key="Enter")`
   - Scroll through content: `browser_scroll(direction="down")`
   - Test form validation with invalid inputs
   - Test empty submissions

6. **After each interaction**, check for:
   - Console errors: `browser_console()`
   - Visual changes: `browser_vision(question="What changed after the interaction?")`
   - Expected vs actual behavior

### Public Endpoint and DNS Checks

When the target is a public URL, generated hostname, callback URL, webhook URL,
or any feature where DNS/readiness matters, verify the network layer explicitly
before calling the app broken.

1. **Compare resolvers before reporting NXDOMAIN or DNS timeout.** The local
   system resolver can hold a negative cache after a record is created. Check a
   fresh public resolver and, when possible, authoritative nameservers:

   ```sh
   host=example.devopsellence.io
   date -u +%Y-%m-%dT%H:%M:%SZ
   getent ahosts "$host" || true
   dig "$host" A +noall +answer +authority +comments || true
   dig @1.1.1.1 "$host" A +noall +answer +authority +comments || true
   dig @8.8.8.8 "$host" A +noall +answer +authority +comments || true
   dig @9.9.9.9 "$host" A +noall +answer +authority +comments || true
   for ns in $(dig "${host#*.}" NS +short | sed 's/\.$//' | head -5); do
     dig @"$ns" "$host" A +noall +answer +authority +comments || true
   done
   ```

   Interpret carefully:
   - `NXDOMAIN` from only the local resolver, while `1.1.1.1`, `8.8.8.8`, or an
     authoritative nameserver returns an address, is usually local negative DNS
     caching. Record the SOA authority TTL and retry after it expires instead of
     filing a product/runtime failure.
   - `NXDOMAIN` from authoritative nameservers means the record is genuinely
     absent at the source.
   - `NOERROR` with no `A` but a valid `AAAA` (or vice versa) may still be enough
     depending on the expected address family; test the address family the app
     is supposed to support.

2. **Probe the endpoint using the resolved address when DNS evidence diverges.**
   If public resolvers or authoritative DNS return an address but the local
   resolver still fails, bypass the local cache to verify the app:

   ```sh
   host=example.devopsellence.io
   ip=203.0.113.10
   curl -fsS --resolve "$host:80:$ip" "http://$host/up"
   curl -fsS --resolve "$host:443:$ip" "https://$host/up"
   ```

   For HTTPS, `--resolve` preserves the original hostname for SNI/certificate
   validation while forcing the selected IP.

3. **Do not trust control-plane or app status alone.** A dashboard, CLI, or API
   may say a public URL is ready while DNS is still propagating or negatively
   cached. Conversely, a local browser may fail while public resolvers already
   work. Final readiness needs both sides: structured app/control-plane status
   and an external DNS+HTTP(S) probe.

### Phase 3: Collect Evidence

For every issue found:

1. **Take a screenshot** showing the issue:
   ```
   browser_vision(question="Capture and describe the issue visible on this page", annotate=false)
   ```
   Save the `screenshot_path` from the response — you will reference it in the report.

2. **Record the details**:
   - URL where the issue occurs
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Console errors (if any)
   - Screenshot path

3. **Classify the issue** using the issue taxonomy (see `references/issue-taxonomy.md`):
   - Severity: Critical / High / Medium / Low
   - Category: Functional / Visual / Accessibility / Console / UX / Content

### Phase 4: Categorize

1. Review all collected issues.
2. De-duplicate — merge issues that are the same bug manifesting in different places.
3. Assign final severity and category to each issue.
4. Sort by severity (Critical first, then High, Medium, Low).
5. Count issues by severity and category for the executive summary.

### Phase 5: Report

Generate the final report using the template at `templates/dogfood-report-template.md`.

The report must include:
1. **Executive summary** with total issue count, breakdown by severity, and testing scope
2. **Per-issue sections** with:
   - Issue number and title
   - Severity and category badges
   - URL where observed
   - Description of the issue
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshot references (use `MEDIA:<screenshot_path>` for inline images)
   - Console errors if relevant
3. **Summary table** of all issues
4. **Testing notes** — what was tested, what was not, any blockers

Save the report to `{output_dir}/report.md`.

## Tools Reference

| Tool | Purpose |
|------|---------|
| `browser_navigate` | Go to a URL |
| `browser_snapshot` | Get DOM text snapshot (accessibility tree) |
| `browser_click` | Click an element by ref (`@eN`) or text |
| `browser_type` | Type into an input field |
| `browser_scroll` | Scroll up/down on the page |
| `browser_back` | Go back in browser history |
| `browser_press` | Press a keyboard key |
| `browser_vision` | Screenshot + AI analysis; use `annotate=true` for element labels |
| `browser_console` | Get JS console output and errors |

## Tips

- **Always check `browser_console()` after navigating and after significant interactions.** Silent JS errors are among the most valuable findings.
- **Use `annotate=true` with `browser_vision`** when you need to reason about interactive element positions or when the snapshot refs are unclear.
- **Test with both valid and invalid inputs** — form validation bugs are common.
- **Scroll through long pages** — content below the fold may have rendering issues.
- **Test navigation flows** — click through multi-step processes end-to-end.
- **Check responsive behavior** by noting any layout issues visible in screenshots.
- **Don't forget edge cases**: empty states, very long text, special characters, rapid clicking.
- When reporting screenshots to the user, include `MEDIA:<screenshot_path>` so they can see the evidence inline.
