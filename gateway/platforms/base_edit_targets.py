"""Edit capabilities a platform adapter declares, and the handle the next edit must target.

Adapters declare these as optional class attributes; callers read them with ``getattr`` so duck-typed
adapters (test fakes, minimal plugins) keep the defaults:

- ``SUPPORTS_MESSAGE_EDITING`` (default True): explicit ``edit_message`` calls work.
- ``SUPPORTS_STREAMING_EDITS`` (default: ``SUPPORTS_MESSAGE_EDITING``): high-frequency token-streaming
  edits are safe. A platform where every edit is a visible event (Signal) keeps explicit edits and
  opts out here.
- ``SUPPORTS_PROGRESS_EDITS`` (no default): whether the throttled, accumulated tool/thinking
  progress bubble may be edited. Separate because its cadence and cost differ from token streaming.
  Only a declared False opts out; an adapter that does not declare it keeps the progress gate's own
  rule (an ``edit_message`` override), whatever ``SUPPORTS_MESSAGE_EDITING`` says.
- ``EDIT_RESULT_ID_IS_NEXT_TARGET`` (default False): a successful edit's ``SendResult.message_id`` is
  the handle the NEXT edit must target. Signal mints a fresh timestamp per edit; replacement-event
  APIs such as Matrix return a fresh event id while later edits must keep targeting the original.
"""

from __future__ import annotations

from typing import Any, Optional


def adapter_supports_streaming_edits(adapter: Any) -> bool:
    """Whether edit-based token streaming is safe (narrower than explicit edit support)."""
    capability = getattr(adapter, "SUPPORTS_STREAMING_EDITS", None)
    if capability is not None:
        return bool(capability)
    return bool(getattr(adapter, "SUPPORTS_MESSAGE_EDITING", True))


def adapter_declines_progress_edits(adapter: Any) -> bool:
    """True only when the adapter declares ``SUPPORTS_PROGRESS_EDITS`` and it is False."""
    capability = getattr(adapter, "SUPPORTS_PROGRESS_EDITS", None)
    return capability is not None and not capability


def adapter_refuses_token_streaming(adapter: Any) -> bool:
    """True when the adapter explicitly opted out of edit-based token streaming and has no native
    stream transport. Legacy non-editable adapters (no ``SUPPORTS_STREAMING_EDITS``) are not
    refusals: the proxy path keeps streaming them with an empty cursor."""
    capability = getattr(adapter, "SUPPORTS_STREAMING_EDITS", None)
    native = bool(getattr(adapter, "SUPPORTS_NATIVE_STREAMING", False))
    return capability is not None and not capability and not native


def next_edit_target_message_id(adapter: Any, current_message_id: Optional[str], result: Any) -> Optional[str]:
    """The message id a subsequent edit must target after ``result``.

    Only adapters declaring ``EDIT_RESULT_ID_IS_NEXT_TARGET`` move the target, and only on a
    successful edit that returned an id; shared callers never infer the contract from
    ``SendResult.message_id`` alone.
    """
    if getattr(adapter, "EDIT_RESULT_ID_IS_NEXT_TARGET", False) is not True:
        return current_message_id
    if not (result and getattr(result, "success", False)):
        return current_message_id
    next_message_id = getattr(result, "message_id", None)
    if next_message_id is None or next_message_id == "":
        return current_message_id
    return str(next_message_id)
