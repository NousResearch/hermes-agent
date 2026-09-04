"""Small helpers for streaming response control."""


def _stop_spinner(thinking_spinner, thinking_callback):
    if thinking_spinner:
        thinking_spinner.stop("")
        thinking_spinner = None
    if thinking_callback:
        thinking_callback("")
    return thinking_spinner
