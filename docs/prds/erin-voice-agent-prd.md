# PRD: Erin Voice Lead Intake Agent

## Problem Statement

Service businesses need a lightweight phone intake agent that can answer calls, capture lead details, and notify the team quickly without forcing humans to babysit every inbound call. Missed calls, incomplete notes, and delayed follow-up leak revenue like a corpse in a cheap tarp.

Erin should provide a warm, helpful automated intake experience, support English and basic Spanish, collect structured lead information, store every call in a new Google Sheet, and send an internal email alert so the team can follow up fast.

## Solution

Build Erin as an automated voice assistant for lead intake. Erin answers inbound calls, discloses that the call may be recorded/transcribed, identifies herself as automated, gathers lead details, confirms the captured request, writes the lead to a new Google Sheet named `Erin Leads`, and sends an internal alert email to `hello@alhambra.ai`.

Erin v1 is intentionally narrow: lead capture, logging, internal notification, spam soft-blocking, and voicemail-style human fallback. No live transfers, no caller SMS/email confirmations, and no operational heroics dressed up as an MVP.

## User Stories

1. As a caller, I want Erin to answer the phone with a warm greeting, so that I know I reached the service intake line.
2. As a caller, I want Erin to disclose that the call may be recorded and transcribed, so that I understand how my call is handled.
3. As a caller, I want Erin to identify herself as an automated assistant, so that I know I am speaking with AI.
4. As a caller, I want to describe what service I need in natural language, so that I do not have to navigate a rigid phone tree.
5. As a caller, I want Erin to ask for my name, so that the team knows who to contact.
6. As a caller, I want Erin to capture my phone number from caller ID or by asking, so that the team can call me back.
7. As a caller, I want Erin to capture my address or location, so that the team understands where service is needed.
8. As a caller, I want Erin to capture my preferred time, so that follow-up can be scheduled around my availability.
9. As a caller, I want Erin to capture urgency, so that urgent requests receive faster attention.
10. As a caller, I want Erin to ask how I heard about the service near the end of the call, so that I can optionally provide source information.
11. As a Spanish-speaking caller, I want Erin to handle basic intake in Spanish, so that I can submit my request without switching to English.
12. As a caller, I want Erin to summarize what she captured before submitting, so that mistakes can be corrected.
13. As a caller, I want Erin to only submit after I confirm the summary, so that incorrect leads are not created.
14. As a caller, I want Erin to say “Thanks. I’ve sent your request to our team, and someone will follow up as soon as possible,” so that I know what happens next.
15. As a caller who asks for a human, I want Erin to take a message for the team, so that my request is still captured without needing a live transfer.
16. As a caller who asks for a human, I want Erin to say “I can take a message and make sure the team gets it. Please tell me what you’d like them to know,” so that I understand the fallback path.
17. As a business owner, I want every lead saved to Google Sheets, so that leads are easy to review and manage.
18. As a business owner, I want a new spreadsheet named `Erin Leads`, so that v1 has a clean default storage location.
19. As a business owner, I want lead columns for timestamp, call ID, contact info, service need, urgency, location, preferred time, lead source, notes, alert status, and status, so that the team has a complete intake record.
20. As a business owner, I want lead statuses of `New`, `Contacted`, `Scheduled`, `Closed`, and `Spam`, so that leads can move through a simple follow-up workflow.
21. As a business owner, I want internal alert emails sent for new real leads, so that the team can respond quickly.
22. As a business owner, I want alert emails sent to `hello@alhambra.ai`, so that intake lands in the preferred inbox.
23. As a business owner, I want urgent leads to use subject `URGENT Erin Lead: {service_need}`, so that important requests stand out.
24. As a business owner, I want normal leads to use subject `New Erin Lead: {service_need}`, so that routine requests are still visible.
25. As a business owner, I want obvious spam or prank calls logged but marked as `Spam`, so that junk is auditable without polluting active work.
26. As a business owner, I want urgent alerts suppressed for spam calls, so that the inbox does not become a haunted clown cannon.
27. As an operator, I want the email alert to include caller name, phone, service need, urgency, address/location, preferred time, lead source if provided, and notes, so that I can follow up without opening the sheet first.
28. As an operator, I want the sheet row to indicate whether the alert email was sent, so that alert failures are visible.
29. As an operator, I want transcript summaries or notes stored, so that I can understand call context later.
30. As a developer, I want structured extraction for lead fields, so that downstream sheet/email logic does not parse freeform prose.
31. As a developer, I want a deterministic call state machine, so that voice behavior remains predictable and testable.
32. As a developer, I want separate adapters for telephony, lead persistence, and email notification, so that each can be tested without making real calls or sending real email.
33. As a developer, I want explicit spam classification behavior, so that prank calls are handled consistently.
34. As a developer, I want bilingual prompt/script support for English and basic Spanish, so that language behavior does not become a roulette wheel with a microphone.
35. As a developer, I want confirmation before side effects, so that a transcription error does not create bad records.

## Implementation Decisions

- Erin v1 is a lead intake voice assistant, not a general support agent.
- Erin’s voice/personality is warm and helpful.
- Erin must identify herself as automated in the greeting.
- Erin must disclose recording/transcription in the greeting: “This call may be recorded and transcribed so we can follow up accurately.”
- Erin supports English and basic Spanish intake in v1.
- Erin collects the following lead fields:
  - Caller phone
  - Caller name
  - Service need
  - Urgency
  - Address / location
  - Preferred time
  - Lead source
  - Notes / transcript summary
- Erin asks “how did you hear about us?” as an optional near-end question.
- Erin confirms the captured lead summary before submitting any lead record.
- Erin closes confirmed calls with: “Thanks. I’ve sent your request to our team, and someone will follow up as soon as possible.”
- If a caller asks for a human, Erin does not live-transfer in v1. She takes a message instead.
- Human fallback script: “I can take a message and make sure the team gets it. Please tell me what you’d like them to know.”
- Erin creates/uses a new Google Sheet named `Erin Leads`.
- The `Erin Leads` sheet columns are:
  - Timestamp
  - Call SID
  - Caller phone
  - Caller name
  - Service need
  - Urgency
  - Address / Location
  - Preferred time
  - Lead source
  - Notes / transcript summary
  - Email alert sent?
  - Status
- Sheet statuses are:
  - `New`
  - `Contacted`
  - `Scheduled`
  - `Closed`
  - `Spam`
- Erin sends an internal alert email for real leads.
- Internal alert recipient is `hello@alhambra.ai`.
- Caller confirmation by text or email is out of scope for v1.
- Alert subject rule:
  - High urgency: `URGENT Erin Lead: {service_need}`
  - Normal urgency: `New Erin Lead: {service_need}`
- Internal alert body should include caller name, phone, service need, urgency, address/location, preferred time, lead source if available, notes/summary, and ideally a Google Sheet row link when practical.
- Spam/prank handling is soft-block only: log the call, mark status `Spam`, and suppress urgent alert behavior when the call is obvious junk.
- Core call flow should use a deterministic state machine with states such as greeting, collect_request, collect_contact, collect_missing, optional_source, confirm, submit, human_message, spam, and done.
- Use structured extraction for lead fields and missing-field detection.
- Keep persistence and notification behind thin adapters so Sheets and email can be swapped later without rewriting the call brain.

## Testing Decisions

- Tests should verify external behavior, not implementation details. The system should be tested by feeding simulated call turns into the call flow and asserting the resulting prompts, extracted fields, side effects, and final states.
- Test the call state machine for:
  - normal English lead capture
  - basic Spanish lead capture
  - missing required fields
  - caller corrections during confirmation
  - high urgency lead subject selection
  - normal urgency lead subject selection
  - human request fallback
  - optional lead source capture
  - spam/prank soft-blocking
- Test structured extraction with representative utterances and ensure unknown/missing fields are explicit instead of guessed.
- Test Google Sheets adapter with mocked API/client calls to verify row shape, column ordering, status values, and alert-sent flag behavior.
- Test email adapter with mocked send calls to verify recipient, subject rules, and body contents.
- Test no caller SMS/email confirmation is attempted in v1.
- Test Spanish prompts remain basic and intake-focused rather than attempting unsupported conversational depth.
- Add integration smoke tests only after credentials/configuration are available; do not require live Google/Twilio/email credentials for unit tests.

## Out of Scope

- Live human transfer.
- Caller SMS confirmation.
- Caller email confirmation.
- Complex scheduling or calendar booking.
- Payment collection.
- Identity verification.
- Multi-location routing.
- CRM synchronization beyond Google Sheets.
- Full multilingual support beyond English plus basic Spanish intake.
- Sophisticated fraud detection beyond soft spam/prank classification.
- Replacing human follow-up; Erin only captures and notifies.

## Further Notes

- v1 should optimize for reliability and follow-up speed over clever conversation.
- The first production milestone should prove the full path: inbound call → structured lead → Google Sheets row → internal email alert.
- The system should log enough metadata to diagnose failures: timestamp, call SID, caller phone, final state, extracted fields, row creation result, email result, and error details.
- Any external side effect must be explicit and auditable. Voice agents hallucinate confidently enough already; no need to hand them a flamethrower.
