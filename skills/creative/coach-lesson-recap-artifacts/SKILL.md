---
name: coach-lesson-recap-artifacts
description: Create polished, student-safe lesson recap review artifacts from lesson source material (especially YouTube/video/audio transcripts), defaulting robust lesson media to the premium packet workflow and returning a real artifact path/link rather than chat-only summary text.
---

# Coach Lesson Recap Artifacts

## When to use
Use this when a coach, academy operator, Bryan, or Drew asks for:
- a lesson recap document
- a student-safe lesson review draft
- a review artifact from a lesson video, audio recording, or transcript
- a polished follow-up packet for student review

Especially relevant when the user provides a YouTube lesson link or a substantial lesson video/audio recording and expects a **real artifact** instead of a chat summary.

## Core rule
If the user asks for a **document**, **draft artifact**, **review artifact**, **packet**, **doc**, **Google Doc**, or asks for a link/path, do **not** stop at chat text.

Produce an actual artifact and return:
- the review link if a Google Doc is created
- the local file path(s) you generated
- the source draft path if helpful

A good chat summary can support the work, but it is **not** the finished deliverable when the user asked for a document.

## Routing default: robust lesson media -> premium packet
Treat robust lesson source material as the default trigger for the premium packet workflow.

Default to the premium packet path when most of the following are true:
- the input is a full or substantial lesson video
- the input is a full or substantial lesson audio recording
- the input is a long-form transcript or source package from a real lesson
- the user asked for a lesson recap, student-safe review, or follow-up packet
- there is enough source material to support a rich packet

This shared skill is the default premium path for coaches who have **not** completed a coach-specific brand design exercise.
In that pre-brand state, the generic **The System** premium template is an acceptable default branded output.

After a coach completes a brand exercise and coach-specific assets are staged locally:
- keep this skill as the **shared routing and artifact-discipline layer**
- load a **coach-specific branding skill** for final brand/render behavior
- treat the coach-specific brand as the default for that coach's normal lesson recap packets; do not wait for the coach to ask for their own branding every time
- do not treat this global skill as the whole post-brand solution forever

Natural-language operator usage rule:
- the workflow should remain discoverable from short ordinary coach/operator prompts, not only from highly structured procedural prompts
- when testing whether the default is really wired, prefer at least one realistic short prompt rather than relying only on explicit step-by-step forcing

Use lighter artifact paths only when one of these is true:
- the user explicitly asks for a quick/simple draft
- the source material is too thin for a premium packet
- the premium workflow is genuinely blocked
- the lighter artifact is an intermediate step rather than the final deliverable

1. **Shared class-level skill**
   - This skill owns the routing logic, artifact-vs-chat discipline, blocker behavior, and generic premium-packet default.
   - The shared fallback premium template is the right path for coaches who have **not** completed a brand design exercise.

2. **Coach-specific branding skill/config**
   - After a coach completes a brand design exercise, the actual premium packet branding/render behavior should become coach-agent-specific rather than remain only a global default.
   - The coach-specific layer should define the brand assets, preferred logos, voice/layout constraints, and any renderer/config specifics for that coach.

Practical rule:
- keep routing logic shared
- move post-brand-exercise branding/render decisions into the coach-specific lane
- use the shared/fallback The System template only when a coach-specific branded packet path is not yet prepared

## Brand Design Document source-truth rule
Before creating any branded lesson recap, premium packet, slide template, deck, or branded document, resolve the active Brand Design Document for the coach/profile.

Universal default model:
- The System Default Brand Design Document is the canonical baseline identity every coach agent should have available for general System artifacts and for coaches without a completed coach-specific brand ingestion exercise.
- This default gives unbranded/new coaches a premium template immediately; it is fallback source truth, not a replacement for coach identity.
- A coach/profile-specific Brand Design Document controls normal artifacts inside that coach lane once installed.
- A prompt- or registry-named venue/event Brand Design Document override controls only the current artifact and must not mutate the coach/profile default brand root.

Resolution order:
1. If the prompt explicitly names a temporary venue/event brand override, use that Brand Design Document only for this artifact.
2. Otherwise use the coach/profile default Brand Design Document if one is installed locally.
3. Otherwise use the generic The System premium brand fallback.

Use the active Brand Design Document as source truth for:
- identity/logo treatment
- colors and hierarchy
- typography direction
- visual style and page chrome
- member/student-facing tone
- packet header/footer and application guidance

## Premium packet Brand Token Map — paint-by-numbers requirement
Before rendering a branded premium lesson packet, convert the active Brand Design Document into a concrete packet token map. Do not rely on vague prose like “use the brand” or perform a logo-only swap.

Required token map shape:

```yaml
brand_source:
  registry_path: ~/.hermes/profiles/<profile>/home/brand-design-docs/BRAND_REGISTRY.yaml
  brand_id: <the-system | coach-brand | temporary-override-id>
  brand_design_document: <local PDF/HTML path>
  mode: <default | coach_default | temporary_override>
  reverts_to: <brand_id-or-null>
identity:
  brand_name: <visible club/coach/system name>
  logo_primary: <path or embedded asset>
  logo_mark: <path or null>
  logo_usage: <header/footer/cover guidance>
palette:
  primary: "#..."        # main headings, footer/header, major rules
  secondary: "#..."      # supporting panels, section labels, secondary rules
  accent: "#..."         # callouts, key numbers, subtle highlights
  background: "#..."     # page background / paper tone
  surface: "#..."        # cards / content panels
  text: "#..."           # main copy
  muted_text: "#..."     # captions, source labels, footer notes
typography:
  heading: <font family or direction>
  body: <font family or direction>
  label: <font family or direction>
  fallback: <safe system fallback>
packet_chrome:
  cover_style: <quiet club / bold academy / system premium / etc.>
  header_rule: <color + weight + placement>
  footer_rule: <brand name, page number, source label treatment>
  section_label_style: <caps/small label/quiet serif/etc.>
  card_style: <border/fill/radius/shadow guidance>
  motif: <crest line / monogram / stripe / none>
application_notes:
  do:
    - <concrete use rule>
  avoid:
    - <concrete avoid rule>
```

Paint-by-numbers application rules:
1. Use `palette.primary`, not the renderer's old default navy, for main headings, major rules, footer/header bars, and dominant accents.
2. Use `palette.secondary` and `palette.accent` for callouts, priority labels, stat chips, movement-map accents, and practice-plan highlights.
3. Use `background`, `surface`, `text`, and `muted_text` for page/card/body styling instead of inherited template defaults.
4. Apply typography direction to heading/body/label hierarchy even when exact fonts are unavailable; choose the closest safe fallback and record it.
5. Apply packet chrome tokens to page headers, footers, section labels, cards, dividers, and cover treatment.
6. Preserve the canonical premium packet backbone; brand tokens are render-layer inputs, not permission to fork the workflow or weaken the packet structure.
7. If a token cannot be extracted from the Brand Design Document, fill it with a labeled fallback from The System premium defaults and record that fallback in `application_notes`. Do not silently keep old coach-specific colors.

Acceptance gate:
- A packet is not fully Brand Design Document compliant if it only swaps logo/brand name while keeping the old color system.
- If logo/name are correct but palette/chrome remain inherited defaults, report `YELLOW: brand-token application incomplete`, not full GREEN.
- For temporary overrides such as Congressional, the output should visibly use the override's palette/colors and page chrome for that artifact only, then revert to the coach default brand root.

For local coach agents, prefer a profile-local brand registry such as `~/.hermes/profiles/<profile>/home/brand-design-docs/BRAND_REGISTRY.yaml` over Tailnet/public-proof URLs. Tailnet proof links are for operator review; local profile paths are the runtime source when available.

Temporary brand overrides must not mutate the coach's default brand root. For Bryan/Darin, if no override is named, Hermitage CC remains the default brand root.

## Lightweight path only when justified
Use a lighter markdown/docx/Google Doc path only when at least one of these is true:
- the user explicitly asked for a quick draft, simple recap, or plain doc
- the source is too thin for a premium packet (for example: a short clip, a single swing snippet, or sparse notes)
- the premium workflow is genuinely blocked after verification
- the lightweight artifact is an explicit intermediate draft step rather than the final deliverable

Do not silently downgrade a robust lesson recap request into a plain markdown/docx/Google Doc just because that path is easier.

## Best available workflow
1. **Ingest the source**
   - If the source is a YouTube link, load the `youtube-content` skill and fetch the transcript with timestamps.
   - Save the transcript locally if useful for provenance.
   - If the source is audio/video provided locally, extract or locate the best available transcript/source truth first.

2. **Anchor to source-backed claims**
   - Pull explicit lesson phrases and core diagnoses from the transcript.
   - Prefer exact lesson-backed themes over generic golf language.
   - Keep the output student-safe: no private metadata, no internal scoring, no invented academy facts.

3. **Choose the artifact class honestly**
   - For robust lesson media, prefer the premium packet workflow by default.
   - For lighter inputs or explicit quick-draft asks, a simpler artifact may be appropriate.
   - If the user names a specific local packet workflow, treat that as binding.

4. **Draft reusable source material**
   - Write or normalize the student-safe recap into a reusable local source file.
   - Reuse earlier artifacts you already created from the same lesson whenever possible instead of redoing the source work from scratch.

5. **If the user names a required local packet workflow, treat that as binding**
   - When the user specifies a repo-local workflow such as `GOLF-108`, `The System`, a named mapper/renderer, or a premium packet path, do not substitute a Google Doc or plain `.docx` just because those are easier.
   - Read the local mapper/renderer scripts and tests first so you know the expected input contract, output paths, and validation rules before attempting conversion.

6. **Run the required artifact pipeline, not a nearby substitute**
   - Normalize the earlier recap into whatever structured JSON the local mapper expects.
   - Execute the mapper and renderer that the user named.
   - If the environment fails, distinguish a real blocker from an invocation mistake. Verify PATH, working directory, and script path before concluding the workflow is unavailable.

7. **Create a real review artifact**
   - Preferred for robust lesson media: premium local packet artifact when available.
   - Preferred for collaborative review flows: create a Google Doc review draft when the Google Docs path is available and the user asked for a review doc workflow.
   - Also create a local `.docx` when practical so the user has a file path artifact.
   - But if the user explicitly requested a premium local packet workflow, the packet artifact is the deliverable and the doc path is not a substitute.
   - Return the artifact link/path, not just the prose.

8. **Verify before handoff**
   - Confirm the named pipeline actually produced files on disk.
   - If a render command times out, verify whether the expected output files were nonetheless written before reporting failure.
   - Confirm the local `.docx` exists / was written successfully when that path is part of the task.
   - Return exact source, intermediate, and final paths when the workflow involved a mapper/renderer chain.

## Default structure for the student-safe artifact
Use a clean, scannable structure such as:
- Title
- Short review-only subtitle
- Lesson Snapshot
- Main Priority
- What We Saw
- Practice Plan
- Setup Reminders
- Coach Note
- Next Touchpoint
- Source Anchors

This works better than a wall of text and aligns with the user's preference for readable, polished docs.

## Formatting rules
- Prefer polished markdown with real headings and bullets.
- Keep sections short and student-safe.
- Do not expose raw transcript dumps inside the student-facing artifact.
- Do not add internal system notes to the visible title area.
- Hero lines and section headers should read premium, warm, and student-facing rather than like swing-mechanics instruction.
- Prefer simple, human phrasing that leaves technical detail to the body copy.
- When choosing between an instructional/mechanical line and a cleaner premium recap line, choose the cleaner premium recap line.

### Hero-line standard
For recap hero lines, optimize for:
- personal feel
- warmth
- non-technical wording
- premium recap tone
- enough simplicity that the body can carry the detailed coaching

Example of the target shape:
- Better: "Keep the swing simple enough that the good contact can show up again."
- Worse: lines that sound like explicit mechanics coaching or internal instruction copy.

## Google Docs creation rule
When using the Google Workspace skill for Docs creation:
- Draft to a markdown file first.
- If `docs create --body-file` is unsupported in the runtime wrapper, fall back to passing the full markdown body directly through `--body` using a safe quoting path from Python/subprocess.
- Do not stop at the first CLI mismatch; retry with a working invocation.

## DOCX creation rule
When a local review artifact is useful:
- Generate a `.docx` from the markdown draft.
- Use a readable default layout: clean margins, simple heading hierarchy, scannable bullets.
- The `.docx` does not need heavy design polish, but it must be presentable and clearly readable.

## Pitfalls

- Do not route lesson-audio transcription through an ad hoc shell `whisper` CLI call unless the skill/task explicitly requires a local-command fallback and you have verified the binary path. Prefer Hermes built-in STT (`tools.transcription_tools.transcribe_audio()` or gateway auto-transcription) as the default path.
- If Hermes STT is configured with `stt.provider: local`, verify whether the failure is the transcription engine itself or a workflow bypass that shells out to `whisper`. A `whisper: command not found` error is a workflow-path issue, not proof that Hermes local STT is broken.
- For oversized lesson audio, distinguish between provider limits and local-engine capability. The current Hermes local STT path may reject files over the generic 25MB guard before transcription even when `faster-whisper` could handle them locally. Treat this as a product/code-path troubleshooting lead, not a reason to abandon built-in STT.
- For coach-agent YouTube lesson workflows, restore the known-good fallback in the actual runtime path, not just the skill text. If captions fail, the live processor should try direct audio-only acquisition first (`yt-dlp --extractor-args "youtube:player_client=android" -f "bestaudio[ext=m4a]/bestaudio"`), then Android-client muxed fallback (`-f 18`), then local `faster-whisper` transcription. A patched skill copy alone is not enough if the active processor script is still transcript-only.
- For scalable rollout across Darin/Sergio/future coach agents, update both layers together: (1) the shared/loaded lesson or YouTube fallback skill, and (2) the runtime processor/wrapper the agent actually executes. Also sync profile-local skill copies if those profiles keep their own copies, otherwise resets or stale profile-local skills will silently drop the learned fallback.
- **Do not confuse a recap request with a chat-summary request.** If the user asked for a lesson recap document, artifact, or draft, create the artifact.
- Do not respond with "here's the text" when the user asked for a review link or local path.
- Do not over-polish in chat while failing to produce the actual file/link.
- For robust lesson video/audio/transcript input, do not default to a lightweight draft when the premium path is available.
- When the user names a specific local premium workflow, do not downgrade to markdown/docx/Google Doc as a substitute deliverable.
- When a coach/operator attaches a specific packet PDF as the reference for future long-form lesson recaps, treat that reference as the acceptance target: record/copy it into the coach lane, use neutral product language if they dislike an internal label, and require the rendered packet/PDF as the primary deliverable before any companion Doc.
- Do not declare a workflow blocked until you have checked for simple invocation issues first: PATH visibility, working directory, and doubled path segments are common false blockers.
- If the first run fails for an environment visibility reason but the user provides the fix, retry the exact named workflow before reconsidering alternatives.
- Do not claim the artifact was shared externally.
- Do not invent coach-brand specifics unless they were provided or publicly confirmed.

## Handoff format
Good final handoff:
- premium packet path(s) when that workflow was used
- packet/PDF surfaced first when a premium packet workflow was the requested or selected deliverable
- Google Doc review link only when that path was requested/used as a companion surface
- local `.docx` path when created
- source/intermediate paths for mapper-renderer chains
- one-line note that it is review-only / not externally shared

## Delivery-surface rule for premium packets
When the artifact class is a premium lesson packet:
- the rendered packet/PDF is the primary deliverable surface
- a Google Doc may exist as an editable companion, but it is not the default first-class handoff unless the user explicitly makes Docs primary
- do not let an easier Google Docs/Drive path outrank the actual packet product
- if both exist, report the split truth explicitly: packet/PDF generated yes/no, packet/PDF surfaced yes/no, Google Doc companion created/shared yes/no
- if the coach asks for PDF links or shared PDF access, local packet paths plus shared Docs are still NO-GO; upload/share the PDF through the approved Workspace seam or state the exact seam blocker
- completion criteria must fail closed for premium packets: a local PDF alone is NOT complete, and a Google Doc companion is NOT complete
- when a canonical PDF is rendered and the approved Bryan/Darin Drive upload seam is available, immediately attempt the PDF upload/share step rather than stopping at local render success or surfacing a companion Doc
- for Bryan-facing premium packet automation, only mark the item complete when the upload returns a real Drive PDF URL and that URL is recorded in durable state (for example `driveLink`); if PDF render succeeded but upload/share did not produce a real file URL, mark the item blocked on PDF delivery with the exact blocker
- never use a Google Doc companion to mask missing PDF delivery; for Bryan premium packet automation, do not create, mention, or rely on a Doc companion at all unless Bryan explicitly asked for an editable Doc
- when Drew/Bryan says they are tired of the Doc-companion out or says Bryan does not care about a polished Google Doc output, treat that as a workflow correction, not a taste note: remove the Doc-companion escape hatch from the active path, patch the governing skill/prompt, and make the shared Drive PDF link the only default completion surface
- when the user asks why the agent created the premium PDF but did not share it, diagnose the completion gate first: if local PDF success was treated as completion without a real shared Drive PDF URL, classify that as a completion-criteria bug and patch the producer/skill so the item becomes `blocked_on_pdf_delivery` instead of surfacing any downgraded artifact
- follow `references/premium-packet-primary-delivery-gate.md` before saying DONE/GREEN: a rough/basic Doc or chat recap is NO-GO when the Connor/premium packet is expected, even if local packet PDFs were created later or prepared off to the side.
- follow `references/premium-packet-pdf-drive-delivery-seam.md` when local premium PDFs exist but the expected coach-facing deliverable is a clickable shared PDF link.
- when auditing a miss, do not hide behind vague language like "something drifted" if the actual failure is identifiable. Name the exact failure class directly: generation failure, upload/share failure, or delivery-contract/routing failure.
- if a local premium PDF already exists but the surfaced result is a Google Doc companion, classify the incident as a PDF-primary delivery-contract failure, not as a generic rendering failure and not as vague drift.
- for Drew/Bryan operational diagnosis, answer the concrete break first: what artifact existed, what artifact was surfaced, and which completion gate was incorrectly treated as satisfied.

## Runtime-path discipline for local transcription fallback
When transcript/caption-first extraction fails and the workflow must fall back to local transcription, do not treat "Whisper installed somewhere on the machine" as success. Verify the actual runtime path the coach agent will use.

Required checks before trusting the fallback lane:
- identify the exact Python/runtime environment the target profile's tools will invoke
- verify the transcription module imports in that same environment
- verify required external helpers on PATH in that same environment (especially `ffmpeg`)
- run a short bounded smoke transcription in the target profile context and verify a real output artifact is written

Important diagnosis lesson:
- a coach-agent may have `openai-whisper` installed in a profile-local Python user site while the Hermes main venv still lacks `whisper`
- this can produce a dangerous half-wired state: some retries say `whisper: command not found`, later retries partially run, and the agent burns long 600-second fallback attempts without closing the task cleanly
- the durable lesson is not "Whisper is broken"; it is that the fallback lane must use one explicit runtime + PATH contract and be health-checked in the target profile context

Operator truth rule for triage:
- if the gateway is alive and still receiving messages, classify the agent as alive even if a specific lesson request is stuck in a degraded transcription loop
- separate `service alive` from `job completed`

## References
- `references/reed-youtube-lesson-recap-pattern.md` — transcript -> markdown -> Google Doc + local DOCX lesson recap pattern.
- `references/golf108-premium-packet-retry-pattern.md` — premium local packet variant: reuse prior recap artifacts, normalize to mapper JSON, verify PATH/workdir before declaring a blocker, and return intermediate/final packet paths.
- `references/bryan-hermitage-premium-packet-branding-split-2026-06-20.md` — two-layer product model: keep lesson-recap routing shared, but move premium packet branding/render behavior into the coach-specific lane after brand-exercise completion.
- `references/coach-brand-packet-branding-assets.md` — how to stage coach/club brand assets into a coach workspace lane and treat them as available inputs for branded packet work without claiming renderer wiring is complete.
- `references/premium-packet-primary-delivery-gate.md` — gate for premium packet primary delivery; a Doc/chat recap is not success when the packet/PDF is expected.
- `references/premium-packet-pdf-drive-delivery-seam.md` — local premium PDFs are not enough when the user needs clickable shared PDF links; use or add a bounded Drive PDF upload/share seam.
- `references/premium-lesson-packet-default-routing.md` — portable class-level lesson: long-form lesson video/audio/transcript should default to the coach's approved premium packet artifact when available; rough/basic Docs are companion surfaces, not terminal success.
- `references/darin-local-transcription-runtime-path-2026-06-24.md` — profile-context lesson on Whisper fallback: check the exact runtime, PATH/ffmpeg visibility, and bounded smoke transcription before declaring the lane healthy.
- `references/darin-captionless-youtube-fallback-runtime-split-2026-06-25.md` — two-layer lesson: restoring captionless YouTube fallback requires both the shared/profile-local skill text and the live processor script; includes the verified direct-audio → Android muxed → local faster-whisper recovery path.
