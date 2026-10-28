"""Enter arriving inside a fast input burst: pasted newline or scripted submit.

Paste without bracketed paste (tmux ``paste-buffer`` without ``-p``) and IME/voice dictation
deliver every newline as its own Enter key right behind the text, so an Enter within
``RAPID_INPUT_ENTER_WINDOW_S`` of the last buffer change is a line break, not a submit (#10994).
A script typing ``text`` + Enter in ONE write (``tmux send-keys 'text' Enter``, a PTY write of
``text\\r``) looks identical up to that Enter, except nothing follows it and no other Enter came
before it. So an Enter with more keys of the same read still queued stays a newline, and so does
the last Enter of a burst that already held one (the tail of a multi-line paste). The lone Enter
that ends a burst inserts a provisional newline and submits once input has stayed quiet for
``BURST_END_QUIET_S``; any input in that window (the next chunk of a paste split across reads)
keeps the newline.
"""

from __future__ import annotations

from time import monotonic

RAPID_INPUT_ENTER_WINDOW_S = 0.05
# Long enough for the next read of a paste split across reads (slow SSH links), short enough
# that a scripted submit still feels immediate. Humans never reach this path: their Enter comes
# more than RAPID_INPUT_ENTER_WINDOW_S after the last keystroke.
BURST_END_QUIET_S = 0.25


def handle_burst_enter(cli, event, buf) -> bool:
    """Handle an Enter that arrived mid-burst; False when it is an ordinary (human) Enter.

    ``cli._tui_submit_buffer(event, buf)`` is called later when this Enter turns out to end the
    burst.
    """
    now = monotonic()
    if now - getattr(cli, "_tui_last_text_change", 0.0) >= RAPID_INPUT_ENTER_WINDOW_S:
        return False
    pasted_newline_before = now - getattr(cli, "_tui_burst_newline_at", 0.0) < BURST_END_QUIET_S
    cli._tui_burst_newline_at = now
    before = buf.document  # the newline lands at the cursor, not necessarily at the end
    buf.insert_text("\n")
    app = event.app
    key_processor = getattr(app, "key_processor", None)
    loop = getattr(app, "loop", None)
    if key_processor is None or loop is None or key_processor.input_queue or pasted_newline_before:
        return True  # inside (or the tail of) a multi-line paste
    with_newline = buf.document
    changed_at = getattr(cli, "_tui_last_text_change", 0.0)

    def _settle() -> None:
        if not app.is_running or buf.document != with_newline or cli._tui_last_text_change != changed_at:
            return  # more input arrived (or the buffer was edited): the newline stays
        buf.document = before
        cli._tui_submit_buffer(event, buf)
        app.invalidate()

    loop.call_later(BURST_END_QUIET_S, _settle)
    return True
