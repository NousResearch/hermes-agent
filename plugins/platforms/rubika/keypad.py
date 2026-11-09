"""Build Rubika chat_keypad / inline_keypad JSON from a simple button list.

v1 scope: one "Simple"-type button per row (id + label). Rubika's other
button types (Payment, Calendar, Location, CameraImage, GalleryImage, File,
Audio, RecordAudio, MyPhoneNumber, MyLocation, Textbox, Barcode, Link) are a
documented follow-up, not built here — see the design spec's Global
Constraints.
"""

from typing import Dict, List, Tuple


def _rows(buttons: List[Tuple[str, str]]) -> Dict:
    return {
        "rows": [
            {"buttons": [{"id": button_id, "type": "Simple", "button_text": button_text}]}
            for button_id, button_text in buttons
        ]
    }


def build_chat_keypad(buttons: List[Tuple[str, str]]) -> Dict:
    """Persistent chat keypad from a list of (button_id, button_text) pairs."""
    return _rows(buttons)


def build_inline_keypad(buttons: List[Tuple[str, str]]) -> Dict:
    """Per-message inline keypad from a list of (button_id, button_text) pairs."""
    return _rows(buttons)
