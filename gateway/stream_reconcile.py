"""Decide whether a streamed reply already covered the final response, or left a
partial message on screen that should be reconciled (edited) rather than re-sent.

Extracted from gateway/run.py's post-stream suppression block so the decision is
unit-testable. It fixes the Telegram flood-control duplication: when flood
control disables progressive edits mid-answer, the stream consumer leaves a
partial message on screen but never confirms final delivery, so none of the
suppression flags (streamed / previewed / content_delivered) are set and the
normal final send re-delivers the whole answer on top of the partial (a
duplicate). When this returns True, the caller edits the existing partial to the
final content (finalize=True splits on overflow) instead of sending a duplicate.
"""


def should_reconcile_partial(
    final_response,
    *,
    failed,
    transformed,
    streamed,
    previewed,
    content_delivered,
    already_sent,
    message_id,
):
    """True when streaming put a partial message on screen (``already_sent`` and a
    ``message_id``) but never confirmed final delivery, so the final response
    should be reconciled by editing that message rather than sent fresh.

    Returns False for the normal cases so existing suppress/send behavior is
    untouched:
      - empty/"(empty)" final, or a failed run (the final must reach the user fresh);
      - transformed responses (handled by the dedicated edit branch above it);
      - deliveries already confirmed (streamed / previewed / content_delivered);
      - nothing actually put on screen (no message to reconcile).
    """
    if not final_response or final_response == "(empty)" or failed:
        return False
    if transformed:
        return False
    if streamed or previewed or content_delivered:
        return False
    return bool(already_sent and message_id)
