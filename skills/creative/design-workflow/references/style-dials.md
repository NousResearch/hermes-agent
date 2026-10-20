# Style Dials

Use style dials to translate vague taste into controllable decisions.

## Dial 1 — Design Variance

How much the design departs from conventional layout and components.

- **1-2:** very conventional, predictable, low risk.
- **3-4:** clean product UI with small distinctive touches.
- **5-6:** balanced originality; suitable for most launches.
- **7-8:** strong editorial/brand presence, asymmetric or memorable.
- **9-10:** experimental; use only when the brand/product can carry it.

## Dial 2 — Motion Intensity

How much animation and transition is part of the experience.

- **1-2:** nearly static; instant feedback only.
- **3-4:** subtle transitions for state/orientation.
- **5-6:** visible motion language but still utility-safe.
- **7-8:** expressive motion central to brand storytelling.
- **9-10:** cinematic/experimental; high performance and accessibility risk.

Always respect reduced motion.

## Dial 3 — Visual Density

How much information appears per screen.

- **1-2:** spacious, portfolio/editorial, few decisions.
- **3-4:** premium marketing, focused narrative.
- **5-6:** balanced product communication.
- **7-8:** operational/productivity UI with tables, filters, status.
- **9-10:** expert cockpit; only for trained users and high-frequency workflows.

## Recommended Defaults

| Scenario | Variance | Motion | Density |
|---|---:|---:|---:|
| B2B SaaS landing | 5 | 3 | 5 |
| Product website | 5 | 3 | 4 |
| Developer tool | 4 | 2 | 7 |
| Analytics dashboard | 2 | 1 | 8 |
| Admin console | 2 | 1 | 8 |
| Premium consumer landing | 7 | 4 | 3 |
| iOS productivity screen | 4 | 3 | 6 |
| Android utility screen | 3 | 2 | 7 |
| Onboarding | 5 | 4 | 4 |
| Settings/account | 2 | 1 | 7 |

## How to Use

When user says:
- "高级" → usually lower density, controlled color, better typography, less noise; not necessarily more animation.
- "更有设计感" → raise variance by 1-2, but keep hierarchy intact.
- "更稳重" → lower variance/motion, increase semantic clarity.
- "更像工具" → increase density, lower motion, make states and shortcuts clearer.
- "更像原生 app" → lower web-like decoration, follow platform primitives.

## Guardrails

- Do not raise all three dials at once unless the user asks for an experimental concept.
- Dashboards usually fail when variance/motion are too high.
- Landing pages usually fail when density is too low to build trust.
- Native mobile screens usually fail when density ignores thumb reach or text scaling.
