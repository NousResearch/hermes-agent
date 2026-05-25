# memd wake-up

- hermes-agent / main / hermes@session-501c9276 / none / all / auto / current_task

- recovery voice=caveman-lite | quality=partial:0.52 | dirty=10 | next=fix partial handoff quality before claiming native recovery ready | blocker=refresh recommended due to context pressure

## Instructions

- AGENTS.md: # Hermes Agent - Development Guide <!-- memd-managed:start --> These instructions are managed by memd. ## memd voice bootstrap - Treat `.memd/config.json` as the source of truth...

## Live

- resume_delta: working 4 -> 3

## Durable Truth

- id=a29b3bbb-370f-4d8f-8881-dd9d8b0b0732 | stage=canonical | scope=project | kind=fact | status=active | project=hermes-agent | ns=main | ...
- id=d9640289-7523-41bb-aea4-afd554f80964 | stage=canonical | scope=project | kind=fact | status=active | project=hermes-agent | ns=main | ...
## Wake Budget

- startup trimmed; use `memd lookup` or `memd resume` for deeper recall.

## Protocol

- Read first. Durable truth beats transcript recall. Promote stable truths.
- Lookup before answers on decisions, preferences, history, or prior user corrections.
- If a required fact is absent or unknown, ask a clarifying question or run lookup before acting.
- Recall: `memd lookup --output .memd --query "..."`.
- If the user corrects you, write the correction back instead of trusting the transcript.
- Writes: user-taught facts -> `memd teach --output .memd --content "..."`; decisions/preferences -> `memd remember`; short-term -> `memd checkpoint`; live/correction spill -> `memd hook capture --summary`.
- Default voice: caveman-lite. Reply in `caveman-lite` unless `.memd/config.json` changes it.
- If your draft is not in `caveman-lite`, stop and rewrite it before sending.
