# Issue Taxonomy & Severity Guidelines

Use these guidelines to classify issues found during exploratory (Dogfooding) QA testing. Every issue in a Dogfood Report must have exactly one Category and one Severity Level.

## Severity Levels

| Severity | Definition | Examples |
|----------|------------|----------|
| **Critical** | Blocks a core user workflow entirely. Results in data loss, application crash, or a dead-end state with no recovery. | Payment button does not work. Login form crashes page. Navigation bar is entirely missing. |
| **High** | A major feature is broken or substantially degraded. The user can eventually complete their task but it is extremely frustrating or requires an obscure workaround. | Search filters do not apply. Mobile menu cannot be closed once opened. Form validation prevents valid input. |
| **Medium** | A feature behaves incorrectly but the core workflow is not blocked. A clear workaround exists. Noticeable UX/UI degradation. | Text overlaps slightly on small screens but is legible. Hover state stuck. Secondary link returns a 404. |
| **Low** | Minor cosmetic errors, lack of polish, or insignificant UX annoyances. | A button is misaligned by a few pixels. Typo in a tooltip. Color contrast slightly fails WCAG AAA but is mostly readable. |

## Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **Functional** | The logic, state, or interaction mechanism of the application is broken. | Broken links, non-submitting forms, infinite loading spinners, non-responsive buttons. |
| **Visual** | CSS, layout, responsiveness, or rendering issues. The application works but looks broken. | Text overlapping images, components breaking outside container bounds on mobile, missing icons. |
| **Accessibility** | Issues preventing users with disabilities from navigating the site. | Poor color contrast, missing alt text on meaningful images, focus trapping, unlabelled form inputs. |
| **Console/Network** | Hidden errors occurring behind the scenes that indicate system instability or broken integrations. | JavaScript exceptions in the console window, failed API calls (4xx, 5xx), unhandled promise rejections. |
| **UX** | confusing behavior that technically functions but violates user expectations. | Success message not appearing after form submission, confusing terminology, destructive actions without confirmation warnings. |
| **Content** | Typographical errors, broken placeholder copy, or factually incorrect static data. | `Lorem Ipsum` left in production, misspelled branding, outdated copyright years. |
