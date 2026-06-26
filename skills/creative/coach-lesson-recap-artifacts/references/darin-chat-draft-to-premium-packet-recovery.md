# Darin chat draft -> premium packet recovery pattern

Use this when a coach-agent/Darin run produced a good lesson-recap draft in chat, but the requested/expected deliverable was a premium packet artifact.

## Lesson
A polished chat recap is not a terminal deliverable for robust lesson media when the user expects the Connor/premium packet path. Treat the chat draft as reusable source material, then run the canonical packet pipeline and verify real files exist.

## Recovery sequence
1. Recover the chat draft/source into a local markdown file for provenance.
2. Normalize the recovered draft and lesson source into the canonical packet JSON expected by the local Connor/premium renderer.
3. Render both HTML and PDF packet artifacts.
4. Verify the packet exists and is non-empty; when practical, capture PDF bytes/page count and SHA256.
5. Handoff with the packet/PDF first, then HTML/JSON/source draft paths.
6. State the split truth clearly: recovered from Darin's draft/source vs produced by Darin's original live response.

## GO / NO-GO language
- GREEN/GO only after the PDF/packet file exists and is non-empty.
- The original chat-only response is NO-GO for a premium packet request, even if the recap text is high quality.
- If renderer access is blocked, report the exact blocker; do not silently downgrade to chat text or a plain doc.

## Example artifact naming shape
Use deterministic source identifiers when available, e.g.:
- `<student-or-lesson>-<youtube_id>-<date>-connor-packet.pdf`
- `<student-or-lesson>-<youtube_id>-<date>-connor-packet.html`
- `<student-or-lesson>-<youtube_id>-<date>-connor-packet.json`
- `<student-or-lesson>-<youtube_id>-<date>-darin-chat-draft.md`
