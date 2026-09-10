"""Gateway-owned STT evidence, separate from the text sent to the model."""

from copy import deepcopy


def media_source_message_ids(event):
    """Keep original message identities when pending attachment events merge."""
    paths = event.media_urls
    origins = list(event.media_source_message_ids)
    return origins + [event.message_id] * (len(paths) - len(origins))


def transcription_evidence(event, paths):
    origins = media_source_message_ids(event)
    records = []
    for path in paths:
        index = event.media_urls.index(path)
        records.append({
            "media_index": index,
            "source_path": path,
            "source_message_id": origins[index],
            "kind": "audio_transcript",
            "status": "failed",
            "method": "configured",
            # Completion is not a calibrated measure of recognition accuracy.
            "confidence": None,
        })
    event.metadata["audio_transcriptions"] = records
    return records


def user_display_metadata(event, *, persistence_owner=None):
    metadata = {}
    if persistence_owner:
        metadata["gateway_input_owner"] = persistence_owner
    if event is not None and event.metadata.get("audio_transcriptions"):
        metadata["audio_transcriptions"] = deepcopy(event.metadata["audio_transcriptions"])
    return metadata
