"""Choose whole summary records before falling back to the existing clipping path."""

from typing import Any


def prepare_summary_input(compressor: Any, turns: list[dict[str, Any]]) -> str:
    """Keep fitting windows intact; count clipping before aggregate bounding too."""
    records = compressor._serialize_records_for_summary(turns, clip=False)
    input_chars = sum(map(len, records))
    separators = 2 * max(len(records) - 1, 0)
    if input_chars + separators > compressor._SUMMARY_INPUT_MAX_CHARS:
        records = compressor._serialize_records_for_summary(turns)
    clipped_chars = input_chars - sum(map(len, records))
    if getattr(compressor, "tail_mode", "lean") == "lean":
        content, coverage = compressor._sample_summary_records(records)
        # Preserve the existing display-character convention: inserted markers occupy
        # input space, while the original record lengths now include pre-sampling clips.
        coverage["input_chars"] = input_chars
        coverage["omitted_chars"] += clipped_chars
        compressor._record_summary_input_coverage(coverage)
    else:
        content = compressor._bound_summary_input("\n\n".join(records))
        telemetry = getattr(compressor, "_active_compression_telemetry", None)
        if isinstance(telemetry, dict):
            sampled_chars = len(content) - separators
            telemetry.update({
                "summary_input_chars": input_chars,
                "summary_input_sampled_chars": sampled_chars,
                "summary_input_omitted_chars": input_chars - sampled_chars,
            })
    return content
