"""Enter arriving inside a fast input burst: pasted newline or scripted submit.

Paste without bracketed paste (tmux ``paste-buffer`` without ``-p``) and IME/voice dictation
deliver every newline as its own Enter key right behind the text, so an Enter within
``RAPID_INPUT_ENTER_WINDOW_S`` of the last buffer change is a line break, not a submit (#10994).
A script typing ``text`` + Enter in ONE write (``tmux send-keys 'text' Enter``, a PTY write of
``text\\r``) looks identical up to that Enter, except nothing follows it and no other Enter came
before it. So an Enter with more keys of the same read still queued stays a newline, and so does
the last Enter of a burst that already held one (the tail of a multi-line paste: input never went
quiet for ``BURST_END_QUIET_S`` since that earlier Enter, however long the paste ran). The lone Enter
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


def note_text_change(cli) -> float:
    """Record a buffer change; input arriving inside the quiet window keeps a paste burst alive.

    So does input arriving while an Enter is held, however late a stalled loop reads it: it makes
    that Enter a pasted newline, so the paste's own trailing Enter stays one too.
    """
    now = monotonic()
    held = getattr(cli, "_tui_held_enter", None) is not None
    if held or now - getattr(cli, "_tui_paste_burst_at", float("-inf")) < BURST_END_QUIET_S:
        cli._tui_paste_burst_at = now
        cli._tui_held_enter = None
    return now


def handle_burst_enter(cli, event, buf) -> bool:
    """Handle an Enter that arrived mid-burst; False when it is an ordinary (human) Enter.

    ``cli._tui_submit_buffer(event, buf)`` is called later when this Enter turns out to end the
    burst.
    """
    now = monotonic()
    if now - getattr(cli, "_tui_last_text_change", 0.0) >= RAPID_INPUT_ENTER_WINDOW_S:
        return False
    pasted_newline_before = now - getattr(cli, "_tui_paste_burst_at", float("-inf")) < BURST_END_QUIET_S
    cli._tui_paste_burst_at = now
    before = buf.document  # the newline lands at the cursor, not necessarily at the end
    buf.insert_text("\n")
    app = event.app
    key_processor = getattr(app, "key_processor", None)
    loop = getattr(app, "loop", None)
    if key_processor is None or loop is None or key_processor.input_queue or pasted_newline_before:
        return True  # inside (or the tail of) a multi-line paste
    with_newline = buf.document
    changed_at = getattr(cli, "_tui_last_text_change", 0.0)
    held = cli._tui_held_enter = object()

    def _settle() -> None:
        if cli._tui_held_enter is held:
            cli._tui_held_enter = None
        if not app.is_running or buf.document != with_newline or cli._tui_last_text_change != changed_at:
            return  # more input arrived (or the buffer was edited): the newline stays
        buf.document = before
        cli._tui_submit_buffer(event, buf)
        app.invalidate()

    # Decide only after one more selector pass has run its reader callbacks: a loop stalled past
    # the window can reach this timer with the next chunk still unread (a signal handler that
    # interrupts the selector past its deadline returns no events). asyncio runs a call_soon
    # queued now ahead of the next pass's readers; the second hop runs after them.
    loop.call_later(BURST_END_QUIET_S, loop.call_soon, loop.call_soon, _settle)
    return True
