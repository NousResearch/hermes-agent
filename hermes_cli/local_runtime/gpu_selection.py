"""Physical GPU identity and selection, shared by budget probes and server launch.

An explicit choice is a resource reservation: never silently run on a different
card if it disappears. The caller may keep serving its previous configuration,
report the error, or fall back to a remote provider instead.
"""
from __future__ import annotations

from collections.abc import Mapping
import re


class GpuSelectionError(ValueError):
    """A requested accelerator is unavailable or cannot be identified safely."""


def normalize(value: object = 'auto') -> str:
    """Preserve YAML integer zero; booleans and empty values are not GPU indices."""
    if value is None:
        return 'auto'
    if isinstance(value, bool):
        raise GpuSelectionError('GPU must be auto, an index, or a device identifier')
    result = str(value).strip()
    return 'auto' if result.lower() == 'auto' else result


def find_card(cards: list[dict], identifier: str) -> dict | None:
    if re.fullmatch(r'[0-9]+', identifier):
        return next((c for c in cards if c.get('index') == int(identifier)), None)
    matches = [c for c in cards if identifier and (
        c.get('id') == identifier or c.get('name') == identifier
        or (identifier.startswith('GPU-') and c.get('uuid', '').startswith(identifier)))]
    return matches[0] if len(matches) == 1 else None


def visible_cards(cards: list[dict], env: Mapping[str, str], key: str) -> list[dict]:
    """Respect empty masks, UUIDs, lists and CUDA's stop-at-first-invalid rule."""
    if key not in env:
        return cards
    result = []
    for token in env[key].split(','):
        card = find_card(cards, token.strip())
        if card is None:
            break
        if card not in result:
            result.append(card)
    return result


def choose(cards: list[dict], value: object, *, env: Mapping[str, str] | None = None,
           visibility_key: str = 'CUDA_VISIBLE_DEVICES') -> dict | None:
    choice = normalize(value)
    if choice != 'auto':
        selected = find_card(cards, choice)
        if selected is None:
            raise GpuSelectionError(f'GPU {choice!r} is unavailable; select a detected GPU or Automatic')
        return selected
    available = visible_cards(cards, env or {}, visibility_key)
    # A stable input-order tie break preserves existing one-device classification.
    return max(available, key=lambda c: (c.get('type', 1) == 1, c['total_bytes']), default=None)
