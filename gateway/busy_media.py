"""Attachment context for live-turn corrections, using the normal inbound file contracts."""

from gateway.run_inbound_media import rehome_inbound_media


async def prepare_redirect_text(runner, event):
    if not event.media_urls and not event.media_types:
        return (event.text or "").strip()
    rehome_inbound_media(event)
    text = await runner._prepare_busy_steer_text(event)
    images, voice, audio, video = runner._classify_inbound_media(event, False)
    names = runner._inbound_original_names(event)
    text = runner._prepend_inbound_media_file_notes(text, audio, video, names)
    text = runner._prepend_inbound_document_notes(event, text)
    for path in voice:
        _, agent_path = runner._inbound_attachment_display_name(path, names.get(path, ""))
        text = f"[Voice attachment saved at: {agent_path}]\n\n{text}"
    for path in images:
        display, agent_path = runner._inbound_attachment_display_name(path, names.get(path, ""))
        text = (f"[The user sent an image: '{display}'. It is saved at: {agent_path}. "
                "Inspect it with an image tool when its contents are relevant.]\n\n" + text)
    if not text.strip() and event.media_urls:
        text = "\n".join(f"[User sent an attachment: {path}]" for path in event.media_urls)
    return text.strip()
