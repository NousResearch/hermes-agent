from . import Copy

COPY = Copy(
    legacy='hermes_state_media',
    doc='Retired media and spillover candidates recorded inside the deletion fence.',
    names=(
        'PREFIX',
        'SPILLOVER_PREFIX',
        'retire_media',
        'retire_spillover',
    ),
    substitutions={
        # The caller's retained media root names the hosted documents a retired admission retires; without one
        # no hosted document is named.
        'retire_media': (
            ("def retire_media(conn, payload, request_id=''):", "def retire_media(conn, payload, request_id='', *, media_root):"),
            ("    references = [*admission_media_references(payload), *hosted_document_references(request_id, payload),\n",
             "    hosted = hosted_document_references(request_id, payload, media_root) if media_root is not None else []\n"
             "    references = [*admission_media_references(payload), *hosted,\n"),
        ),
    },
)
