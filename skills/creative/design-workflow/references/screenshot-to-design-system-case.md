# Case Study: Screenshot-to-Design-System Product Redesign

This reference captures lessons from applying `design-workflow` to a product whose promise is: upload a screenshot or website URL, extract visual style/design-system rules, and output restoration prompts/config.

## Product Pattern

Use this pattern for tools that reverse-engineer visual style from references:

- **Input:** screenshot, image URL, or website URL.
- **Processing promise:** extract colors, typography, spacing, radius, component language, layout rhythm, and style confidence.
- **Outputs:** CSS variables, Tailwind config, shadcn/ui theme, design tokens, and restoration prompts.
- **Primary CTA:** make the first action concrete: `上传截图开始分析` / `Start analyzing`.

## Design Direction That Worked

- Treat the landing page as a **visual compiler / design DNA extraction tool**, not a generic AI assistant.
- Hero should show the input→output transformation immediately:
  - source screenshot preview
  - extracted DNA/token panel
  - copyable outputs such as CSS/Tailwind/Prompt
- Case library cards should show analysis artifacts, not just brand color swatches:
  - palette
  - type scale
  - radius
  - shadow style
  - keywords
  - `查看分析` / view analysis

## Information Architecture

Recommended landing sequence:

1. Hero: `screenshot/URL → design system` value proposition + concrete output preview.
2. Workflow: input → extract visual DNA → export restoration prompt/config.
3. Case library: recognizable styles as extracted rules, not inspiration-only cards.
4. Final CTA: repeat the concrete upload/analyze action.

Recommended analyze-page structure:

- Left rail: what the user will get, constraints, expected time, unsupported pages.
- Right workbench: mode switcher + current mode form.
- Modes:
  - image analysis: upload local image or image URL.
  - screenshot restoration: optimize for layout/component prompt extraction.
  - website analysis: public webpage URL only.
- CTA text should reflect mode and state:
  - empty: explain what is missing.
  - image: `开始分析图片`.
  - screenshot: `开始还原截图`.
  - website: `开始分析网页`.
  - loading: `分析中...`.

## UI Details Worth Reusing

- Use a real dropzone instead of native file input UI:
  - dashed border
  - upload icon
  - accepted formats and max size
  - selected-file name/size
  - remove/replace action
  - image preview
- Make disabled CTA explain the blocking condition.
- Keep primary action wording consistent across hero, header, and footer.
- Use `[word-break:keep-all]` or equivalent for large Chinese headings to avoid ugly mobile wrapping.
- Set `html lang="zh-CN"` for Chinese-first product pages.
- Add `focus-visible` and reduced-motion handling in global styles.

## Verification Lessons

- For Next.js projects, if visual inspection shows raw browser-default HTML, verify CSS responses before critiquing design.
- Running `next build` while `next dev` is active can leave stale dev-server asset paths (`/_next/static/... 404`) or `.next/server/... MODULE_NOT_FOUND` symptoms.
- Restart the dev server and reload before treating CSS/asset anomalies as design failures.

## Quality Gates Used

- `npm run lint`
- `npm run build`
- `npm test`
- Browser route checks for `/`, `/analyze`, and an example detail route.
- Browser visual verification for landing and analyze pages.
- Basic interaction check: tab switching and URL input enabling CTA.
