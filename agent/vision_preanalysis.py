"""Shared prompt and handoff format for auxiliary image pre-analysis."""

from __future__ import annotations


IMAGE_PREANALYSIS_PROMPT = (
    "Concisely describe this image in 2-4 sentences "
    "(~200 Chinese characters or ~150 English words). Cover the main subject and overall context. "
    "For documents, receipts, reports, screenshots, labels, and medication packaging, transcribe "
    "identifying text verbatim, especially names, titles, labels, identifiers, medication names, doses, "
    "dates, and key values. Preserve original characters, punctuation, and ordering where visible; "
    "write [unclear] instead of guessing. Do not silently correct, normalize, or paraphrase visible text. "
    "For charts, diagrams, or scientific figures, include important labels, legends, and key values. "
    "Skip decorative details."
)


def build_preanalysis_note(
    *, description: str, image_path: str = "", role_label: str = "user",
) -> str:
    """Wrap vision output as source evidence that downstream models must not rewrite."""
    note = (
        f"[The {role_label} attached an image. Treat the vision extraction below as source text. "
        "When answering about visible names, titles, identifiers, medication names, doses, dates, or values, "
        "quote exact visible wording from the extraction; do not silently correct or paraphrase it. "
        "If the extraction marks text [unclear], preserve that uncertainty rather than guessing.\n"
        f"{description or 'Image analysis failed.'}]"
    )
    if image_path:
        note += f"\n[If you need a closer look, use vision_analyze with image_url: {image_path}]"
    return note