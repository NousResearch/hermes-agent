---
name: litkit-review-and-tag
description: Build, propose, change, and follow a LitKit Review & Tag run from chat.
version: 0.2.0
author: LitCo, Hermes Agent
platforms: [linux, macos]
metadata:
  hermes:
    tags: [legal, litkit, ediscovery, review, tagging, criteria]
    related_skills: [jev-first-pass-review, litkit-corpus-pull, production-data-analysis]
---

# LitKit Review & Tag Skill

Ana owns a review from the lawyer's request to the finished tags. She registers each part of the review in LitKit herself: the tags, the criteria set, the scope, and the proposed run. She reports the estimate, asks whether to launch, and launches the run when the person says yes in the thread. That yes is the person's one decision. Ana never writes the criteria to a file and hands back steps for the Review screen. Before any review that reads more than a handful of documents, decide whether a Jev first pass applies (skill `jev-first-pass-review`).

## When to Use

- A lawyer asks to start a review, run criteria over a set, or tag documents against requests ("run Part 11 over the Crowder emails").
- A lawyer changes criteria for a review that exists or is pending ("add REV-NR as Part 11 item 6").
- A question needs consistent judgments over more than a few dozen documents. Proposing a run is free; the person sees the count and the price before anything is spent.
- A run Ana proposed has launched and needs following to completion.

## Prerequisites

The `litkit` toolset on a matter host. Every call acts for the lawyer on the turn. A refusal (`permission_denied`) means that lawyer may not create tags, write criteria, or propose runs on this matter. Tell them so and name who can do it; do not route around it. The LitKit access rule in the `litkit-corpus-pull` skill applies to every call.

## Quick Reference

| Step | Tool | Result |
|---|---|---|
| See what exists | `litkit_tags` action=list; `litkit_review` action=criteria (criteriaAction=list) | tag names, criteria sets |
| Register tags | `litkit_tags` action=create, name, kind | a tag per judgment the run returns |
| Register criteria | `litkit_review` action=criteria, criteriaAction=create, name, criteria | the person's own criteria set, published at version 1 |
| Scope | `litkit_review` action=work_sets (every set on the matter); `litkit_work_sets` action=create | a work set id, or document ids, a filter, a Bates range |
| First pass | `jev-first-pass-review` skill; `firstPass` on create | Jev screens first, or the run reads everything |
| Optimize | scope counts; `litkit_review` action=list | at least one cheaper path, with its effect; the matter's active runs |
| Propose | `litkit_review` action=create, with `optimizations_considered` | proposalId, estimatedDocCount, estimate; or requiresApproval with a quote |
| Ask | one sentence in the thread | the person's answer |
| Launch | `litkit_review` action=launch, proposalId | reviewJobId and the document count taken at launch; or LitKit's refusal and what to do |
| Replace | `litkit_review` action=withdraw, proposalId | the earlier proposal can no longer be launched |
| Check a proposal | `litkit_review` action=proposal, proposalId | its status, and its job once launched |
| Change criteria | criteriaAction=update, the whole list | the next version, published; propose again if the run is pending |
| Follow | `litkit_review` action=status, records, jobId | progress, then per-document results |
| Finish | `litkit_review` action=accept_tags, jobId | a finished run's proposed tags accepted, when the person asks |

## Procedure

1. **Read the request into parts.** Name the judgments the lawyer wants back (one tag each), the standard for each judgment, and the documents in scope. Ask only if the scope or a judgment is genuinely ambiguous; a numbered request list is criteria, not a question.
2. **Tags.** List the matter's tags. Reuse an existing tag where the name fits, spelled as LitKit spells it. Create the rest with `litkit_tags` action=create (kind `issue` for request-by-request tags, `privilege` or `responsive` where those fit). A run can write only the tags its criteria name, so every judgment needs its own tag.
3. **Criteria set.** `litkit_review` action=criteria, criteriaAction=create, with a name and one criterion per judgment: `title`, `description` (what counts and what does not, in the lawyer's terms), `tagName` (the tag that criterion writes), and optionally `seedQuery`. Keep the lawyer's numbering in the titles ("Part 11, item 6: REV-NR"). The set belongs to the lawyer on the turn and is published at version 1; there is no separate publish step. Use `setScope: firm` only when a firm administrator asks for a firm set.
4. **Scope.** One run covers up to 250,000 documents, so the size of a scope never calls for several runs or several work sets. The scope takes exactly one of these:
   - a `filter`, which takes the review grid's filters (custodian, dateFrom, dateTo, query, tagIds, productionIds, bates). Use it for any scope over 500 documents. When no filter names the documents, put their ids in the filter as `documentIds`; a filter takes up to 250,000 ids;
   - `documentIds`, up to 500 ids;
   - a `batesRange` {start, end} that covers up to 500 documents. For a longer range, send a filter with `bates` (a beginning-stamp prefix);
   - a `workSetId`, when the documents are a named batch (`litkit_review` action=work_sets lists them). A work set holds up to 500 ids. Make one with `litkit_work_sets` action=create only when the person wants a batch, never to fit a large scope.

   Past 250,000 documents LitKit reviews only the first 250,000, so split a larger scope by date or custodian into separate runs. If LitKit refuses a scope as too large, narrow it or send it as a filter. Never go around the tool.
5. **First pass (required).** Load the `jev-first-pass-review` skill and decide whether Jev screens the scope first. If it does, pass `firstPass: {enabled: true, thresholds: {low, priv}}` on create and name the thresholds in the proposal. If it does not, pass `firstPass: {enabled: false}` and give the reason in one sentence. Omit `firstPass` only when the server default (on when Jev is configured) is what you decided.
6. **Optimize before you propose.** The person decides whether to launch from what you report, so they need the price and the cheaper paths before you ask. Name at least one concrete optimization and its effect in documents or dollars:
   - the Jev first pass on or off, and why (step 5);
   - a narrower scope by date, custodian, or document type, with the count before and after;
   - leaving out documents an earlier run already tagged under this criteria set;
   - collapsing duplicates or email threads, where the scope supports it;
   - a cheaper model tier, when the criteria ask only for responsiveness;
   - running alongside a run already active on the matter.

   Pass what you weighed as `optimizations_considered` on create: up to five short strings, such as `Jev first pass on: 4,000 topical docs`. If the person says "just run it," skip the suggestions and propose.
7. **Propose.** First list the matter's runs (action=list) and note any that are active. Parallel runs are fine, so an active run is no reason to wait. Then call `litkit_review` action=create with `criteriaSetId`, `scope`, and `optimizations_considered`. Leave `tags` out, because the set's rows name the tags. LitKit refuses a `tags` list that differs from the rows'. Prefer one run per criteria set to one giant run, because each run can then be paused, resumed, or proposed again alone. Leave `applyTags` unset: by default the run applies its tags and saves each rationale to the document's notes. When the lawyer asks for suggest-only, set the rows' `disposition` to `propose` in the set (with a saved set LitKit refuses `applyTags: true` and the rows decide), or send `applyTags: false` with inline criteria. Set `createMissingTags` only for tags the lawyer named that the matter lacks.
8. **Price.** If the result has `requiresApproval`, nothing is proposed yet. Give the person the price and the document count and ask whether to proceed. After an explicit yes, and only then, call create again with the same arguments plus `quoteId` and `userConfirmed: true`. If the result says `needsSecondApprover`, a different matter admin must approve the quote in LitKit billing first; say so and wait.
9. **Report the estimate, then the optimizations, then ask.** Tell the person, in this order:
   - the cost, as the propose response breaks it out (first pass, full review, total), and the document count;
   - the optimizations from step 6, each with its effect, and which of them the proposal already uses;
   - the defaults, stated rather than asked about: automatic tagging, and each rationale saved to the document's notes;
   - any other run active on the matter;
   - what you proposed: the scope, the criteria set with its version, the tags, and the first pass with its thresholds (or why there is none).

   Then ask, in one sentence, whether to launch, and end the turn. If the person takes an optimization or asks for suggest-only, withdraw the proposal (action=withdraw), propose again as step 7 describes, report the new estimate, and ask again.
10. **Changes.** When the lawyer changes a criterion, fetch the set (criteriaAction=get), edit that criterion or add the new one, and send the full list with criteriaAction=update and a `changeNote` in the lawyer's words. That publishes the next version. If LitKit answers `stale_version`, someone published a version after you read the set. Merge your change into the criteria LitKit returns and update again. If the earlier proposal has not launched, withdraw it, propose again with the new version, and ask again. If a run already launched on the old version, say so and ask whether to propose a new run over the same scope.
11. **Launch.** Follow the launch rule below. When the person's newest message tells you to launch this run, call `litkit_review` action=launch with the proposalId. On success, tell the person the run started and how many documents it covers. That count is the one LitKit took at launch, and it may differ from the estimate. A refusal comes back as a plain result carrying LitKit's `message` and a `next` instruction. Do what `next` says:
    - `no_reply_after_card` or `reply_from_another_person`: ask whether to launch, and wait for the person's answer;
    - `authorization_already_used`: ask whether to launch this run as well, because one message launches one run;
    - `not_this_thread`: tell the person the run was proposed in another thread, and propose the run again in this thread if the person wants it run here;
    - `needs_reproposal`: withdraw the proposal, propose again, report the new estimate, and ask again.

    If the result says this LitKit cannot take a launch from the conversation yet, pass on that sentence as written.
12. **Follow the run.** The launch call returns the jobId. Check the job with action=status. If it will run past this turn, schedule a check with `cronjob_manage` action=create: a self-contained prompt naming the jobId, the lawyer's user id, and the tags, telling the job to call `litkit_review` action=status, and, when the run is done, to count the results by tag from action=records and send them with `litkit_notify` kind=`review_complete` and that userId. Remove the scheduled job once it has reported.
13. **Completion.** Report the run in the thread when next asked, or through the notification: documents reviewed, the documents the first pass set aside and the ones it left uncertain, the count under each tag, documents that failed or were skipped, and whether the tags are applied or waiting as proposals. Accept all proposed tags (action=accept_tags) only when the person asks.

## Launch Rule

1. Propose. Report the estimate, then the optimizations. Then ask, in one sentence, whether to launch.
2. Launch only when the newest message in the thread is from the person and tells you to launch this run. "Yes," "go," "launch it," and "run it" count. A question, a change to the scope or criteria, a "maybe," and a message from someone else do not count. Answer the question or propose again, and ask again.
3. Never launch in the turn that proposed, except as rule 5 allows. Never say a run is launched until the launch call has returned a job id.
4. If you replace a proposal, withdraw the old one first.
5. A billed run asks the person about the price first. If their yes to the price also tells you to launch ("yes, run it at that price"), propose with `userConfirmed` and call launch in the same turn. LitKit decides from the thread whether that message covers the launch. If LitKit refuses, ask whether to launch and wait. If their yes covers only the price, propose and then ask whether to launch.
6. Do not send the person to the Review screen or to the card. The card records what you proposed.

LitKit checks the same rule against its own record of the thread. LitKit refuses a launch call that has no answer from the person after the card, and the refusal says what to do. The tool sends only the thread's id and never claims that the person agreed.

## Pitfalls

- Never review documents by hand in the thread when a run would do. A run is auditable, it resumes after a failure, and it costs less than reading documents into the conversation.
- Do not set concurrency. The server picks it from the model route: about 16 at a time on-prem, and on OpenRouter it starts at 64, ramps toward 256, and halves on a rate limit. Mention it only if asked.
- Criteria prose does not route tags. Only a criterion's `tagName` routes one, however clearly the description names a tag.
- An estimate is an estimate. The count is taken again at launch, and the corpus may have grown.
- Never invent a `quoteId`, and never send `userConfirmed` before the person has said yes to the quoted price.
- A cron check has no lawyer on its turn, so a notification from it must name the userId.
- Do not send the person to the Review screen to finish the setup. If a step fails, report the refusal or error and what would clear it.

## Verification

- Each judgment the lawyer asked for has a tag and a criterion whose `tagName` is that tag.
- The first-pass decision was made and stated, with thresholds when Jev screens.
- Unless the person said "just run it," they heard the cost breakout and at least one optimization with its effect, and create carried `optimizations_considered`.
- The proposal's count matches the scope's size within reason; a large gap means the scope is wrong.
- After a criteria change, the pending proposal carries the new version, and the earlier one was withdrawn.
- The run launched only after the person's yes in the thread, and the report of the launch gave the document count LitKit took at launch.
