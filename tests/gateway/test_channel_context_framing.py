from gateway.run import _format_channel_context_message


def test_channel_context_framing_marks_backfill_as_non_actionable():
    framed = _format_channel_context_message(
        channel_context="[Sarah] Hey @DrewBot can coupons apply to digital items?",
        trigger_message="[DrewBot] @Bugsy Cloudflare observability triage report",
    )

    assert framed == (
        "[Context only: recent channel messages, not addressed to you; "
        "do not answer or act on these unless the triggering message explicitly asks you to]\n"
        "[Sarah] Hey @DrewBot can coupons apply to digital items?\n\n"
        "[Triggering message: answer or act on this message]\n"
        "[DrewBot] @Bugsy Cloudflare observability triage report"
    )


def test_channel_context_framing_relabels_legacy_recent_header():
    framed = _format_channel_context_message(
        channel_context="[Recent channel messages]\n[Alice] side chatter",
        trigger_message="[DrewBot] @Bugsy handle this report",
    )

    assert framed.startswith(
        "[Context only: recent channel messages, not addressed to you; "
        "do not answer or act on these unless the triggering message explicitly asks you to]\n"
    )
    assert "[Recent channel messages]" not in framed
    assert "[Triggering message: answer or act on this message]\n[DrewBot]" in framed
