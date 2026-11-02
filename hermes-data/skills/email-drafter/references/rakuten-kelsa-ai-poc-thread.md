# Rakuten Symphony India — Kelsa AI POC (worked "find that email" example, 2026-09-14)

Recovery task: NDR asked "email about Rakuten's need for agentic AI in Kelsa — it's either in my ndraas.com or my O3 Infotech account."

## Where it lives
- ALL of it: **ndr@draas.com** inbox, forwarded by **Manohar Singh <msingh@o3infotech.com>** (O3/Kelsa staff — Aagney Singh, Dhanuja K also send from @kelsa.io). NOT in ndr@o3infotech.com — that mailbox has **no vault token** (only `google-draas` / `google-ahfl` / `google-gmail` exist in the gws vault; `build_service("gmail","v1",service_name="google-o3infotech")` fails "No token").
- The original Rakuten email also Cc's ndr@draas.com directly.
- When NDR says "check my O3 Infotech account", the mail is almost always sitting in the draas inbox via forwards from O3-sender addresses.

## Thread map (message ids on google-draas)
- `1a0323b248b3576a` (24-Aug) — Rakuten Zoom invite "Kelsa ( AI capability Demo )" (Raelekar, Satish Raju <satishraju.raelekar@rakuten.com>; Kanaglekar, Prajakta).
- `1a0323c484c66832` (24-Aug) — Manohar forwards invite to NDR + pavan/aagney/guna @kelsa.io.
- `1a046fa1c5d89e84` (28-Aug) — **"Kelsa - AI - POC"** — Anantha Prasanna (HR Transformation, **Rakuten Symphony India**, a Rakuten Group company) formalises POC scope after the demo showcase + asks for timeline and commercial model. This is the requirement-setting email.
- `1a0629b2d5c34b3f` / `1a066e259c8b7add` (2–3 Sep) — "Follow Up | Kelsa - AI Capabilities POC Demo": Aagney Singh (Kelsa BA) delivers the QMS dashboard wrapper, live at **https://rakuten-qms-dashboard.onrender.com/**; agentic POC deferred to a separate session with Anantha. Anantha replies: "looking forward to connect on the agentic use case POC."
- `1a0810f5cf9579c3` (8-Sep) — **"Demo Request | Kelsa - AI Wrapper"** — Andrea Leena (Rakuten HR Transformation Analyst) requests a demo call of the AI Wrapper, proposes 9-Sep 1:30 PM, asks who from Kelsa will present. No later follow-up reached NDR.

## Rakuten's requirement (as of 14-Sep)
1. **AI Agent (agentic layer) on R-Social**:
   - Queries on Stage/Status/Pending/Budget
   - Raising a request
   - Approving a request
   - Uploading bills
   - Budget Reports for Admin team
2. **AI Wrapper for dashboards**:
   - Query Management KPI/SLA (ticket) metrics dashboard
   - Attendance Regularisation dashboards
3. **Commercial / cost model** for extending the services (asked 28-Aug; not answered in the chain yet).

## Status
- Dashboard wrapper: delivered + live (rakuten-qms-dashboard.onrender.com), verified values on Kelsa side.
- Agentic use-case POC: NOT yet defined — Aagney promised a separate discussion with Anantha; Rakuten is explicitly waiting.
- 8-Sep demo request outcome unknown — no confirmation email in NDR's chain. Follow up with Manohar/Aagney, or draft a nudge to schedule the agentic-POC discussion.

## Search recipe that found it
Probe each vault service with `build_service` + `getProfile` (never guess service names — a wrong name errors identically to "not authorized"), then per account:
`q='"Rakuten" Kelsa'`, `q='"Rakuten" agentic'`, `q='"Rakuten" AI'`, `q='"Rakuten"'`.
The hits were almost all in google-draas; ahfl/gmail returned only newsletter noise. Also check `after:<date> (from:rakuten.com OR from:kelsa.io OR from:o3infotech.com)` to catch un-forwarded follow-ups.